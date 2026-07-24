"""Collect ONE real occluded moment on bowl_top_drawer with everything
needed for the pi0.5 injection test (user request, 2026-07-22): the
earlier dense collections saved images + true camera poses + arm_s_occ,
but not the robot's proprioceptive state -- needed to build a real
Pi05Observation for the Welch's-t-test comparison
(subgoal_image=None vs. dust3r-recovered image), same methodology
already validated for the MMaDA subgoal slot ("N=20 calls, 3/7 action
dims differ at p<0.0001").

Stops as soon as arm_s_occ > ARM_OCC_THRESHOLD is reached (not a full
dense per-step scan like collect_dense_occlusion_frames.py -- this only
needs ONE good moment, not a whole occlusion-profile scan).

Requires pi05_worker running:
  scripts/run_with_gpu_cleanup.sh python3 scripts/collect_occlusion_moment_with_state.py
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

from occ_vla.eval.libero_occ_env import AGENTVIEW_KEY, LiberoOccEnv, LiberoOccEnvConfig  # noqa: E402
from occ_vla.eval.metrics import Difficulty  # noqa: E402

PI05_RPC_DIR = str(_ROOT / ".rpc" / "pi05")
NUM_STEPS_WAIT = 10
SEED = 0
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
MAX_STEPS = 220
RESIZE_SIZE = 224
REPLAN_STEPS = 8
ARM_OCC_THRESHOLD = 0.30
SUITE, TASK_ID, LABEL = "libero_spatial", 4, "bowl_top_drawer"
OUT_DIR = _ROOT / "texture_ceiling_probe" / "pi05_injection_test"


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


def main():
    from libero.libero import benchmark  # noqa: PLC0415

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bench = benchmark.get_benchmark(SUITE)()
    instruction = bench.get_task(TASK_ID).language

    config = LiberoOccEnvConfig(
        benchmark_suite=SUITE, task_id=TASK_ID, difficulty=Difficulty.LIGHT,
        init_state_idx=0, seed=SEED, place_occluder=False,
    )
    occ_env = LiberoOccEnv(config, libero_root=str(_ROOT / "third_party/openpi/third_party/libero"))
    obs = occ_env.reset()
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = occ_env.step(LIBERO_DUMMY_ACTION)
    occ_env.capture_clear_baseline(obs)

    action_queue = []
    step = 0
    found = False
    while step < MAX_STEPS:
        base_image = preprocess_image(obs[AGENTVIEW_KEY])
        wrist_image = preprocess_image(obs["robot0_eye_in_hand_image"])
        if action_queue:
            action = action_queue.pop(0)
        else:
            actions = call_pi05(base_image, wrist_image, state_vec(obs), instruction)
            action = actions[0]
            action_queue.extend(actions[1:REPLAN_STEPS])
        obs, _, done, _ = occ_env.step(action.tolist())
        step += 1

        arm_s_occ = occ_env.compute_arm_s_occ(obs)
        if step % 10 == 0:
            print(f"step {step}: arm_s_occ={arm_s_occ:.4f}", flush=True)

        if arm_s_occ > ARM_OCC_THRESHOLD:
            print(f"\nfound occlusion moment at step {step}: arm_s_occ={arm_s_occ:.4f}", flush=True)
            # recompute base_image/wrist_image from the POST-step obs --
            # the versions above (from before occ_env.step()) are stale
            # by one step and would mismatch arm_s_occ/cam_poses/state,
            # all of which reflect the post-step sim state (bug caught
            # before running, not after: it would have silently paired
            # a slightly-less-occluded image with a "found occlusion"
            # label).
            base_image = preprocess_image(obs[AGENTVIEW_KEY])
            wrist_image = preprocess_image(obs["robot0_eye_in_hand_image"])
            found = True
            break
        if done:
            break

    if not found:
        print("no occlusion moment found in this rollout -- try a different seed/episode", flush=True)
        return

    sim = occ_env._env.sim  # noqa: SLF001
    cam_poses = {}
    for cam_name in ("agentview", "robot0_eye_in_hand"):
        cam_id = sim.model.camera_name2id(cam_name)
        cam_poses[cam_name] = {
            "cam_pos": sim.data.cam_xpos[cam_id].tolist(),
            "cam_mat": sim.data.cam_xmat[cam_id].tolist(),
            "fovy_deg": float(sim.model.cam_fovy[cam_id]),
        }

    # GT-unoccluded condition (user request, 2026-07-22): same technique
    # as collect_arm_removal_pairs.py -- render the identical sim state
    # again with the robot's own geoms alpha-zeroed, so base_image shows
    # exactly what pi0.5 would see if the arm weren't in the way at all.
    # State/robot pose is otherwise unchanged -- this only edits the
    # agentview *render*, matching this project's established ground-
    # truth-pair technique, not a new hack.
    env = occ_env._env  # noqa: SLF001
    model = sim.model
    body_names = [model.body_id2name(i) for i in range(model.nbody)]
    robot_body_ids = {
        i for i, n in enumerate(body_names) if n and any(k in n.lower() for k in ("robot", "panda", "gripper", "mount"))
    }
    geom_bodyid = model.geom_bodyid
    robot_geom_ids = [g for g in range(model.ngeom) if geom_bodyid[g] in robot_body_ids]

    state = env.get_sim_state()
    orig_rgba = model.geom_rgba[robot_geom_ids].copy()
    model.geom_rgba[robot_geom_ids, 3] = 0.0
    gt_obs = env.regenerate_obs_from_state(state)
    gt_agentview_256 = gt_obs[AGENTVIEW_KEY].copy()
    model.geom_rgba[robot_geom_ids] = orig_rgba
    env.regenerate_obs_from_state(state)  # restore obs/render state before continuing
    gt_agentview_224 = preprocess_image(gt_agentview_256)

    Image.fromarray(obs[AGENTVIEW_KEY]).save(OUT_DIR / "occluded_agentview_256.png")
    Image.fromarray(obs["robot0_eye_in_hand_image"]).save(OUT_DIR / "occluded_wrist_256.png")
    Image.fromarray(base_image).save(OUT_DIR / "occluded_agentview_224.png")  # what pi0.5 actually sees
    Image.fromarray(wrist_image).save(OUT_DIR / "occluded_wrist_224.png")
    Image.fromarray(gt_agentview_256).save(OUT_DIR / "gt_agentview_256.png")
    Image.fromarray(gt_agentview_224).save(OUT_DIR / "gt_agentview_224.png")

    meta = {
        "instruction": instruction,
        "step": step,
        "arm_s_occ": arm_s_occ,
        "state": state_vec(obs).tolist(),
        "cam_poses": cam_poses,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved occluded observation + state + true cam poses to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
