"""Collect a fresh held-out (episode 19) sample set for moka_pots and
mug_in_microwave, this time also saving the TARGET OBJECT's own
clear-baseline segmentation footprint (`LiberoOccEnv.capture_clear_baseline`
-- the same mechanism already used for S_occ elsewhere in this project),
not just the arm mask.

Why a fresh collection instead of reusing arm_removal_pairs_policy/:
that data has no saved sim state, and pi0.5's action sampling isn't
bit-identical run to run (documented elsewhere in CLAUDE.md), so episode
19's exact frames can't be replayed from what's on disk -- only
regenerated. Same init_state_idx/seed as the original collection
(episode_idx=19 % len(init_states), SEED=7) reproduces the same initial
scene; the trajectory afterward will differ, which is fine -- this only
needs to be a valid, never-trained-on-by-the-LoRA sample, not a pixel
match to the earlier set.

Purpose: ground the "object region" mask used by
probe_pklp_mmada_collage.py in the target's real segmentation
(env.obj_of_interest) instead of the geometric `_gripper_end_bbox_token_mask`
heuristic already shown (2026-07-21) to invert object/background for
these two tasks' reach poses.

Requires pi05_worker running -- use
  scripts/run_with_gpu_cleanup.sh python3 scripts/collect_grounded_holdout_frames.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from openpi_client import image_tools
from PIL import Image
from robosuite.utils.transform_utils import quat2axisangle

_orig_torch_load = torch.load
torch.load = lambda *a, **k: _orig_torch_load(*a, **{**k, "weights_only": False})

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "third_party/openpi/third_party/libero"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "_workers"))

import rpc  # noqa: E402

from occ_vla.eval.libero_occ_env import AGENTVIEW_KEY, AGENTVIEW_SEGMENTATION_KEY, LiberoOccEnv, LiberoOccEnvConfig  # noqa: E402
from occ_vla.eval.metrics import Difficulty  # noqa: E402

PI05_RPC_DIR = str(_ROOT / ".rpc" / "pi05")
NUM_STEPS_WAIT = 10
SEED = 7
EPISODE_IDX = 19  # matches split_train_val's held-out (max) episode
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
SAMPLES_PER_EPISODE = 20
STEPS_BETWEEN_SAMPLES = 6
RESIZE_SIZE = 224
REPLAN_STEPS = 8
OUT_DIR = _ROOT / "grounded_holdout_frames"

TASKS = [
    {"suite": "libero_10", "task_id": 8, "label": "moka_pots"},
    {"suite": "libero_10", "task_id": 9, "label": "mug_in_microwave"},
]


def preprocess_image(raw_image: np.ndarray) -> np.ndarray:
    flipped = np.ascontiguousarray(raw_image[::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(flipped, RESIZE_SIZE, RESIZE_SIZE))


def state_vec(obs) -> np.ndarray:
    return np.concatenate(
        [obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
    ).astype(np.float32)


def call_pi05(base_image, wrist_image, state, prompt):
    arrays = {"base_image": base_image, "wrist_image": wrist_image, "state": state}
    resp_arrays, _ = rpc.call(PI05_RPC_DIR, arrays, {"prompt": prompt})
    return resp_arrays["actions"]


def robot_geom_ids(model) -> list[int]:
    body_names = [model.body_id2name(i) for i in range(model.nbody)]
    robot_body_ids = {
        i for i, n in enumerate(body_names) if n and any(k in n.lower() for k in ("robot", "panda", "gripper", "mount"))
    }
    geom_bodyid = model.geom_bodyid
    return [g for g in range(model.ngeom) if geom_bodyid[g] in robot_body_ids]


def render_pair(occ_env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    env = occ_env._env  # noqa: SLF001
    model = env.sim.model
    geom_ids = robot_geom_ids(model)

    state = env.get_sim_state()
    visible_obs = env.regenerate_obs_from_state(state)
    visible_img = visible_obs[AGENTVIEW_KEY].copy()

    seg_dict = env.get_segmentation_instances(visible_obs[AGENTVIEW_SEGMENTATION_KEY])
    arm_mask = (seg_dict["robot"].squeeze(-1) != 0) if "robot" in seg_dict else np.zeros(visible_img.shape[:2], dtype=bool)

    orig_rgba = model.geom_rgba[geom_ids].copy()
    model.geom_rgba[geom_ids, 3] = 0.0
    clear_obs = env.regenerate_obs_from_state(state)
    clear_img = clear_obs[AGENTVIEW_KEY].copy()
    model.geom_rgba[geom_ids] = orig_rgba

    return visible_img, clear_img, arm_mask


def collect_task(suite: str, task_id: int, label: str) -> list[dict]:
    from libero.libero import benchmark  # noqa: PLC0415

    bench = benchmark.get_benchmark(suite)()
    instruction = bench.get_task(task_id).language
    init_states = bench.get_task_init_states(task_id)

    config = LiberoOccEnvConfig(
        benchmark_suite=suite, task_id=task_id, difficulty=Difficulty.LIGHT,
        init_state_idx=EPISODE_IDX % len(init_states), seed=SEED, place_occluder=False,
    )
    occ_env = LiberoOccEnv(config, libero_root=str(_ROOT / "third_party/openpi/third_party/libero"))
    obs = occ_env.reset()
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = occ_env.step(LIBERO_DUMMY_ACTION)
    occ_env.capture_clear_baseline(obs)  # target's clear footprint, once, post-settle

    manifest = []
    action_queue = []
    n_saved = 0
    for sample_idx in range(SAMPLES_PER_EPISODE):
        done = False
        for _ in range(STEPS_BETWEEN_SAMPLES):
            base_image = preprocess_image(obs[AGENTVIEW_KEY])
            wrist_image = preprocess_image(obs["robot0_eye_in_hand_image"])
            if action_queue:
                action = action_queue.pop(0)
            else:
                actions = call_pi05(base_image, wrist_image, state_vec(obs), instruction)
                action = actions[0]
                action_queue.extend(actions[1:REPLAN_STEPS])
            obs, _, done, _ = occ_env.step(action.tolist())
            if done:
                break
        if done:
            break

        visible_img, clear_img, arm_mask = render_pair(occ_env)
        seg_dict = occ_env._env.get_segmentation_instances(obs[AGENTVIEW_SEGMENTATION_KEY])  # noqa: SLF001
        target_raw = seg_dict.get(occ_env.target_body_name)
        target_mask_now = (target_raw.squeeze(-1) != 0) if target_raw is not None else np.zeros(arm_mask.shape, dtype=bool)
        # object region actually needing restoration: the target's clear
        # (unoccluded) footprint, restricted to where the arm currently covers it
        object_occluded_mask = occ_env._target_mask_clear & arm_mask  # noqa: SLF001

        if not object_occluded_mask.any():
            continue  # arm doesn't actually occlude the target this sample -- not useful for this test

        stub = f"{label}_ep{EPISODE_IDX:02d}_s{sample_idx:03d}"
        Image.fromarray(visible_img).save(OUT_DIR / f"{stub}_armvis.png")
        Image.fromarray(clear_img).save(OUT_DIR / f"{stub}_armfree.png")
        Image.fromarray((arm_mask.astype(np.uint8) * 255)).save(OUT_DIR / f"{stub}_armmask.png")
        Image.fromarray((object_occluded_mask.astype(np.uint8) * 255)).save(OUT_DIR / f"{stub}_objmask.png")
        Image.fromarray((target_mask_now.astype(np.uint8) * 255)).save(OUT_DIR / f"{stub}_targetnowmask.png")
        manifest.append({
            "stub": stub, "label": label, "instruction": instruction,
            "arm_px": int(arm_mask.sum()), "object_occluded_px": int(object_occluded_mask.sum()),
        })
        n_saved += 1

    print(f"[{label}] {n_saved} pairs saved", flush=True)
    return manifest


def main():
    OUT_DIR.mkdir(exist_ok=True)
    manifest = []
    for task in TASKS:
        manifest.extend(collect_task(task["suite"], task["task_id"], task["label"]))
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"total: {len(manifest)} pairs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
