"""Proposal-3 pre-validation (user request, 2026-07-23): before investing
in a real open-vocab detector (YOLO-World/SAM2 -- comparable setup cost
to today's SEINE integration), cheaply test whether the gate's target-
tracking logic degrades when given coarser (bounding-box-center)
spatial precision instead of pixel-mask centroid precision. Both
sources are still GT (get_segmentation_instances) -- only the
PRECISION is degraded, isolating "does bbox-level precision matter"
from "does swapping to a real detector matter" (a second, separate
question, not tested here).

mask_centroid_224: current method (see run_hybrid_gate_injection_pipeline.py)
  -- mean of all foreground pixel coordinates.
bbox_centroid_224: NEW -- bounding box of the same foreground pixels,
  center of that box used instead. For a compact, roughly-convex object
  these are similar; they diverge more for elongated/partially-visible
  (partially-occluded) shapes, which is exactly the regime that matters
  here.

Runs gate_only only (injection_only doesn't use target tracking at all,
so it's not a relevant comparison for this question) on
mug_in_microwave (task_id=9), the one task with verified real gate
engagement (see hybrid_gate_injection_results.json, 2026-07-23).

Requires pi05_worker running WITH occ_vla inputs enabled.
Run: python3 scripts/run_bbox_tracking_sanity.py --episodes N
"""

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from openpi_client import image_tools
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
from occ_vla.integration.runtime import OSC_POSE_MAX_DELTA_M, SCENE_BLEND_ALPHA, gated_blend_xy  # noqa: E402
from occ_vla.pklp.pixel_to_action import CameraProjector, pklp_pixel_delta_to_world_delta  # noqa: E402

PI05_RPC_DIR = os.environ.get("OCC_VLA_PI05_RPC_DIR", str(_ROOT / ".rpc" / "pi05"))
BENCHMARK_SUITE = "libero_10"
TASK_ID = 9  # mug_in_microwave -- verified real gate engagement
RESIZE_SIZE = 224
NUM_STEPS_WAIT = 10
SEED = 7
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
REPLAN_STEPS = 8
MAX_STEPS = 520
N_EPISODES = 10
GATE_THRESHOLD = 0.30
CLEAR_UPDATE_THRESHOLD = 0.05
RESULTS_PATH = _ROOT / "bbox_tracking_sanity_results.json"

TRACKING_MODES = ["mask_centroid", "bbox_centroid"]


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


def target_position_224(occ_env, obs, mode: str) -> np.ndarray | None:
    seg_dict = occ_env._env.get_segmentation_instances(obs[AGENTVIEW_SEGMENTATION_KEY])  # noqa: SLF001
    target_mask_raw = seg_dict.get(occ_env.target_body_name)
    if target_mask_raw is None:
        return None
    target_mask = target_mask_raw.squeeze(-1) != 0
    if not target_mask.any():
        return None
    h, w = target_mask.shape
    ys, xs = np.where(target_mask)
    flipped_ys = h - 1 - ys
    flipped_xs = w - 1 - xs
    scale = RESIZE_SIZE / h
    if mode == "mask_centroid":
        return np.array([flipped_xs.mean() * scale, flipped_ys.mean() * scale])
    # bbox_centroid: center of the bounding box of the same foreground
    # pixels, not their mean -- what a real detector's bbox output
    # would give instead of pixel-precise segmentation.
    cx = (flipped_xs.min() + flipped_xs.max()) / 2.0
    cy = (flipped_ys.min() + flipped_ys.max()) / 2.0
    return np.array([cx * scale, cy * scale])


def run_episode(tracking_mode: str, episode_idx: int, max_steps: int) -> dict:
    from libero.libero import benchmark  # noqa: PLC0415

    bench = benchmark.get_benchmark(BENCHMARK_SUITE)()
    init_states = bench.get_task_init_states(TASK_ID)
    instruction = bench.get_task(TASK_ID).language

    config = LiberoOccEnvConfig(
        benchmark_suite=BENCHMARK_SUITE, task_id=TASK_ID, difficulty=Difficulty.LIGHT,
        init_state_idx=episode_idx % len(init_states), seed=SEED, place_occluder=False,
    )
    occ_env = LiberoOccEnv(config, libero_root=str(_ROOT / "third_party/openpi/third_party/libero"))
    obs = occ_env.reset()
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = occ_env.step(LIBERO_DUMMY_ACTION)
    occ_env.capture_clear_baseline(obs)

    projector = CameraProjector.from_sim(occ_env._env.sim, "agentview", resolution=RESIZE_SIZE)  # noqa: SLF001
    last_known_position = target_position_224(occ_env, obs, tracking_mode)

    action_queue = collections.deque()
    blend_engaged_steps = 0
    max_arm_s_occ = 0.0
    for step in range(max_steps):
        arm_s_occ = occ_env.compute_arm_s_occ(obs)
        max_arm_s_occ = max(max_arm_s_occ, arm_s_occ)
        pos_now = target_position_224(occ_env, obs, tracking_mode)
        if arm_s_occ < CLEAR_UPDATE_THRESHOLD and pos_now is not None:
            last_known_position = pos_now
        occluded = arm_s_occ >= GATE_THRESHOLD

        base_image = preprocess_image(obs[AGENTVIEW_KEY])
        wrist_image = preprocess_image(obs["robot0_eye_in_hand_image"])

        if action_queue:
            action = action_queue.popleft()
        else:
            actions = call_pi05(base_image, wrist_image, state_vec(obs), instruction)
            action = actions[0]
            action_queue.extend(actions[1:REPLAN_STEPS])

        if occluded and last_known_position is not None:
            eef_pos_world = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
            eef_pixel = projector.project(eef_pos_world)
            world_delta = pklp_pixel_delta_to_world_delta(projector, eef_pos_world, eef_pixel, last_known_position)
            pklp_delta_xy = world_delta[:2] / OSC_POSE_MAX_DELTA_M
            action = action.copy()
            blended = gated_blend_xy(action[:2].astype(np.float64), pklp_delta_xy, SCENE_BLEND_ALPHA)
            action[:2] = np.clip(blended, -1.0, 1.0)
            blend_engaged_steps += 1

        obs, _, done, _ = occ_env.step(action.tolist())
        if done:
            return {"tracking_mode": tracking_mode, "episode": episode_idx, "done_step": step,
                    "max_arm_s_occ": max_arm_s_occ, "blend_engaged_steps": blend_engaged_steps}

    return {"tracking_mode": tracking_mode, "episode": episode_idx, "done_step": None,
            "max_arm_s_occ": max_arm_s_occ, "blend_engaged_steps": blend_engaged_steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--modes", nargs="+", default=TRACKING_MODES, choices=TRACKING_MODES)
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    results = json.loads(args.results_path.read_text()) if args.results_path.exists() else []

    def already_done(mode, episode_idx):
        return any(r["tracking_mode"] == mode and r["episode"] == episode_idx for r in results)

    for mode in args.modes:
        for episode_idx in range(args.episodes):
            if already_done(mode, episode_idx):
                continue
            t0 = time.time()
            result = run_episode(mode, episode_idx, args.max_steps)
            result["wall_s"] = time.time() - t0
            results.append(result)
            print(f"[bbox_tracking_sanity {mode} ep{episode_idx}] {result}", flush=True)
            args.results_path.write_text(json.dumps(results, indent=2))

    print("\n=== BBOX TRACKING SANITY REPORT ===")
    for mode in TRACKING_MODES:
        rows = [r for r in results if r["tracking_mode"] == mode]
        if not rows:
            continue
        steps = [r["done_step"] for r in rows if r["done_step"] is not None]
        blend_total = sum(r.get("blend_engaged_steps", 0) for r in rows)
        n_engaged = sum(1 for r in rows if r.get("blend_engaged_steps", 0) > 0)
        print(f"{mode}: {len(steps)}/{len(rows)} success, steps={steps}, engaged={n_engaged}/{len(rows)}, blend_total={blend_total}")


if __name__ == "__main__":
    main()
