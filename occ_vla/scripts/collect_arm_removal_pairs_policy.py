"""Like collect_arm_removal_pairs.py, but drives the arm with REAL pi0.5
rollouts instead of random action deltas -- per user request
(2026-07-17, WMPO-style rationale): training data should reflect poses
the policy actually visits, not an arbitrary random distribution.
Writes to a separate directory (arm_removal_pairs_policy/) so it
doesn't collide with the earlier random-action pairs.

Same ground-truth-pair technique as collect_arm_removal_pairs.py
(confirmed working 2026-07-17): render the identical MuJoCo sim state
twice, once normally and once with the robot's geoms hidden (alpha=0).

Requires pi05_worker running (.rpc/pi05).
Run: python3 scripts/collect_arm_removal_pairs_policy.py \
    [--episodes-per-task N] [--samples-per-episode N] [--steps-between-samples N]
"""

import argparse
import collections
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
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
EPISODES_PER_TASK = 20
SAMPLES_PER_EPISODE = 20
STEPS_BETWEEN_SAMPLES = 6
RESIZE_SIZE = 224
REPLAN_STEPS = 8

TASKS = [
    {"suite": "libero_10", "task_id": 8, "label": "moka_pots"},
    {"suite": "libero_10", "task_id": 3, "label": "bowl_in_drawer"},
    {"suite": "libero_10", "task_id": 9, "label": "mug_in_microwave"},
    {"suite": "libero_spatial", "task_id": 4, "label": "bowl_top_drawer"},
    {"suite": "libero_10", "task_id": 4, "label": "mugs_on_plates"},
]

OUT_DIR = _ROOT / "arm_removal_pairs_policy"


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


def collect_task(suite: str, task_id: int, label: str, episodes: int, samples_per_episode: int, steps_between_samples: int) -> list[dict]:
    from libero.libero import benchmark  # noqa: PLC0415

    bench = benchmark.get_benchmark(suite)()
    instruction = bench.get_task(task_id).language
    init_states = bench.get_task_init_states(task_id)

    manifest = []
    for episode_idx in range(episodes):
        config = LiberoOccEnvConfig(
            benchmark_suite=suite, task_id=task_id, difficulty=Difficulty.LIGHT,
            init_state_idx=episode_idx % len(init_states), seed=SEED, place_occluder=False,
        )
        occ_env = LiberoOccEnv(config, libero_root=str(_ROOT / "third_party/openpi/third_party/libero"))
        obs = occ_env.reset()
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = occ_env.step(LIBERO_DUMMY_ACTION)

        action_queue = collections.deque()
        n_saved = 0
        for sample_idx in range(samples_per_episode):
            done = False
            for _ in range(steps_between_samples):
                base_image = preprocess_image(obs[AGENTVIEW_KEY])
                wrist_image = preprocess_image(obs["robot0_eye_in_hand_image"])
                if action_queue:
                    action = action_queue.popleft()
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
            if not arm_mask.any():
                continue

            stub = f"{label}_ep{episode_idx:02d}_s{sample_idx:03d}"
            Image.fromarray(visible_img).save(OUT_DIR / f"{stub}_armvis.png")
            Image.fromarray(clear_img).save(OUT_DIR / f"{stub}_armfree.png")
            Image.fromarray((arm_mask.astype(np.uint8) * 255)).save(OUT_DIR / f"{stub}_armmask.png")
            manifest.append({"stub": stub, "label": label, "instruction": instruction, "arm_px": int(arm_mask.sum())})
            n_saved += 1

        print(f"[{label}] ep{episode_idx}: {n_saved} pairs saved", flush=True)

    print(f"[{label}] {len(manifest)} total pairs saved", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-task", type=int, default=EPISODES_PER_TASK)
    parser.add_argument("--samples-per-episode", type=int, default=SAMPLES_PER_EPISODE)
    parser.add_argument("--steps-between-samples", type=int, default=STEPS_BETWEEN_SAMPLES)
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    manifest = []
    for task in TASKS:
        manifest.extend(
            collect_task(
                task["suite"], task["task_id"], task["label"],
                args.episodes_per_task, args.samples_per_episode, args.steps_between_samples,
            )
        )
        (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\ndone -- {len(manifest)} total pairs in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
