"""n=3 comparison on T08 (moka pots): does the non-generative pipeline
(Phase 4 soft gating + Phase 2 PKLP visual overlay + text-only
cot_anchor -- see integration/runtime.py, enable_subgoal_image_generation
now defaults to False) reduce the ~15-25% step-count overhead previously
measured on T08's clean pixel-mask occlusion test (collect_multiview_data.py
--pixel-mask, 2026-07-15 session: 385-466 steps occluded vs 356-374
unoccluded, 3/3 success, with NO countermeasure at all)?

That occlusion is a full, permanent blackout of the target's own clear
footprint, applied from the start of policy control and never moving --
so this script's own baseline_pixelmask condition re-measures the same
"no countermeasure" case fresh (fair same-session comparison, since
pi0.5's flow-matching sampler is unseeded/stochastic run to run -- see
CLAUDE.md's validation status notes), and soft_pipeline adds:

- Phase 4 soft gating on the real (uncorrupted-by-the-pixel-mask)
  arm_s_occ -- expected to rarely engage here (arm_s_occ has topped out
  ~0.07-0.15 on this task, under the 0.30 gate threshold); this
  occlusion scenario doesn't really exercise Phase 4, and that's
  reported as such rather than glossed over.
- Phase 2's PKLP kinematic estimate, computed ONCE from the last 3
  settle-wait frames (before the mask is ever applied -- target fully
  visible, effectively stationary) and then held fixed for the rest of
  the episode. Continuing to re-estimate from RAFT flow over the
  blacked-out region afterward would be meaningless (no real motion
  signal), so this is a deliberate "freeze the last real observation"
  choice, not an oversight -- see kinematic_extrapolator usage below.
  The frozen predicted position is drawn on the frame (visual_overlay)
  and also stated in a plain-text cot_anchor (no MMaDA call -- Option 2
  decision, this experiment shouldn't depend on MMaDA at all).

Requires already running:
  - pi0.5 RPC worker on .rpc/pi05    (scripts/_workers/pi05_worker.py)
  - RAFT RPC worker on .rpc/raft     (scripts/_workers/raft_worker.py)

Run: python3 scripts/run_t08_soft_pipeline.py [--max-steps N] [--episodes N]
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import cv2
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

from occ_vla.control.occlusion_gating import apply_soft_gate, gate_scale  # noqa: E402
from occ_vla.eval.libero_occ_env import AGENTVIEW_KEY, AGENTVIEW_SEGMENTATION_KEY, LiberoOccEnv, LiberoOccEnvConfig  # noqa: E402
from occ_vla.eval.metrics import Difficulty  # noqa: E402
from occ_vla.pklp.kinematics import KinematicExtrapolator  # noqa: E402
from occ_vla.pklp.optical_flow import PatchFlow  # noqa: E402
from occ_vla.pklp.visual_overlay import draw_kinematic_overlay  # noqa: E402

import os

# Overridable so run_parallel_conditions.py can point each condition's
# process at its own worker/RPC dir when running conditions on
# separate GPUs concurrently, instead of sharing one worker serially.
PI05_RPC_DIR = os.environ.get("OCC_VLA_PI05_RPC_DIR", str(_ROOT / ".rpc" / "pi05"))
RAFT_RPC_DIR = os.environ.get("OCC_VLA_RAFT_RPC_DIR", str(_ROOT / ".rpc" / "raft"))

BENCHMARK_SUITE = "libero_10"
TASK_ID = 8
TARGET_BODY_NAME = "moka_pot_1"
INSTRUCTION = "put both moka pots on the stove"
MAX_STEPS = 520
NUM_STEPS_WAIT = 10
RESIZE_SIZE = 224
SEED = 7
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
REPLAN_STEPS = 8
N_EPISODES = 3

RESULTS_PATH = _ROOT / "t08_soft_pipeline_results.json"


def preprocess_image(raw_image: np.ndarray) -> np.ndarray:
    flipped = np.ascontiguousarray(raw_image[::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(flipped, RESIZE_SIZE, RESIZE_SIZE))


def preprocess_mask(raw_mask: np.ndarray) -> np.ndarray:
    """Same flip the raw camera frames get, then a plain resize to
    RESIZE_SIZE -- resize_with_pad degenerates to a plain resize for
    LIBERO's square renders anyway (see occ_vla/CLAUDE.md item 9), and a
    boolean mask doesn't need convert_to_uint8's photo-specific handling."""
    flipped = np.ascontiguousarray(raw_mask[::-1, ::-1])
    return cv2.resize(flipped.astype(np.uint8), (RESIZE_SIZE, RESIZE_SIZE), interpolation=cv2.INTER_NEAREST) > 0


def state_vec(obs) -> np.ndarray:
    return np.concatenate(
        [obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
    ).astype(np.float32)


def call_pi05(base_image, wrist_image, state, prompt, cot_anchor=None):
    arrays = {"base_image": base_image, "wrist_image": wrist_image, "state": state}
    fields = {"prompt": prompt}
    if cot_anchor is not None:
        fields["cot_anchor"] = cot_anchor
    resp_arrays, _ = rpc.call(PI05_RPC_DIR, arrays, fields)
    return resp_arrays["actions"]


def call_raft(frame_t2, frame_t1, frame_t0):
    resp_arrays, resp_fields = rpc.call(
        RAFT_RPC_DIR,
        {"frame_t2": frame_t2, "frame_t1": frame_t1, "frame_t0": frame_t0},
        {},
        timeout_s=120,
    )
    grid_shape = (resp_fields["grid_rows"], resp_fields["grid_cols"])
    flow_earlier = PatchFlow(patch_centers=resp_arrays["patch_centers"], flow=resp_arrays["flow_earlier"], grid_shape=grid_shape)
    flow_latest = PatchFlow(patch_centers=resp_arrays["patch_centers"], flow=resp_arrays["flow_latest"], grid_shape=grid_shape)
    return flow_earlier, flow_latest


def nearest_patch_idx(patch_centers: np.ndarray, point_xy: np.ndarray) -> int:
    return int(np.argmin(np.sum((patch_centers - point_xy) ** 2, axis=1)))


def estimate_frozen_target_state(settle_frames_224: list, target_mask_224: np.ndarray):
    """Seeds PKLP's kinematic state from the last 3 settle-wait frames
    (target fully visible, effectively stationary) -- see module
    docstring for why this is frozen rather than re-estimated after the
    pixel mask is applied. Returns (predicted_position | None)."""
    if target_mask_224.sum() == 0:
        return None
    ys, xs = np.where(target_mask_224)
    centroid = np.array([xs.mean(), ys.mean()])

    t2, t1, t0 = settle_frames_224[-3:]
    flow_earlier, flow_latest = call_raft(t2, t1, t0)
    idx = nearest_patch_idx(flow_latest.patch_centers, centroid)

    extrapolator = KinematicExtrapolator()
    state = extrapolator.estimate_state([flow_earlier, flow_latest], idx)
    return extrapolator.extrapolate(state)


def run_episode(condition: str, episode_idx: int, max_steps: int) -> dict:
    from libero.libero import benchmark  # noqa: PLC0415

    bench = benchmark.get_benchmark(BENCHMARK_SUITE)()
    init_states = bench.get_task_init_states(TASK_ID)

    config = LiberoOccEnvConfig(
        benchmark_suite=BENCHMARK_SUITE,
        task_id=TASK_ID,
        difficulty=Difficulty.LIGHT,
        init_state_idx=episode_idx % len(init_states),
        seed=SEED,
        place_occluder=False,
        pixel_mask=True,
    )
    occ_env = LiberoOccEnv(config, libero_root=str(_ROOT / "third_party/openpi/third_party/libero"))
    obs = occ_env.reset()

    settle_frames_224 = []
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = occ_env.step(LIBERO_DUMMY_ACTION)
        settle_frames_224.append(preprocess_image(obs[AGENTVIEW_KEY]))

    occ_env.capture_clear_baseline(obs)  # also activates automatic pixel-masking in occ_env.step() from here on

    predicted_position = None
    if condition == "soft_pipeline":
        target_mask_clear_224 = preprocess_mask(occ_env._target_mask_clear)  # noqa: SLF001
        predicted_position = estimate_frozen_target_state(settle_frames_224, target_mask_clear_224)
        print(f"  [{condition} ep{episode_idx}] frozen predicted target position (224x224 px): {predicted_position}", flush=True)

    action_queue = collections.deque()
    gate_engaged_steps = 0
    for step in range(max_steps):
        base_image = preprocess_image(obs[AGENTVIEW_KEY])  # pixel-mask already baked in by occ_env.step()
        wrist_image = preprocess_image(obs["robot0_eye_in_hand_image"])
        arm_s_occ = occ_env.compute_arm_s_occ(obs)
        cot_anchor = None

        if condition == "soft_pipeline":
            scale = gate_scale(arm_s_occ)
            if scale < 1.0:
                gate_engaged_steps += 1
                base_image = apply_soft_gate(base_image, scale)
            if predicted_position is not None:
                base_image = draw_kinematic_overlay(base_image, predicted_position, predicted_position)
                cot_anchor = (
                    f"{INSTRUCTION}. The target moka pot is currently occluded from the main camera; "
                    f"its last known position (marked in the image) was approximately "
                    f"({predicted_position[0]:.0f}, {predicted_position[1]:.0f}) in image pixel coordinates."
                )

        if action_queue:
            action = action_queue.popleft()
        else:
            actions = call_pi05(base_image, wrist_image, state_vec(obs), INSTRUCTION, cot_anchor)
            action = actions[0]
            action_queue.extend(actions[1:REPLAN_STEPS])

        obs, _, done, _ = occ_env.step(action.tolist())
        if done:
            print(f"  [{condition} ep{episode_idx}] success at step {step}", flush=True)
            return {"condition": condition, "episode": episode_idx, "done_step": step, "gate_engaged_steps": gate_engaged_steps}

    print(f"  [{condition} ep{episode_idx}] did not finish within {max_steps} steps", flush=True)
    return {"condition": condition, "episode": episode_idx, "done_step": None, "gate_engaged_steps": gate_engaged_steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["baseline_pixelmask", "soft_pipeline"],
        choices=["baseline_pixelmask", "soft_pipeline"],
        help="Run only these conditions (e.g. after a partial crash) instead of both.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=RESULTS_PATH,
        help="Overridable so run_parallel_conditions.py can give each concurrent per-condition process its own file to avoid clobbering.",
    )
    args = parser.parse_args()

    # Load and keep any existing results (e.g. a condition that already
    # finished before a crash) instead of clobbering them -- only
    # (condition, episode) pairs about to be (re-)run are dropped first.
    results = json.loads(args.results_path.read_text()) if args.results_path.exists() else []
    results = [
        r for r in results if not (r["condition"] in args.conditions and r["episode"] < args.episodes)
    ]

    for condition in args.conditions:
        for episode_idx in range(args.episodes):
            t0 = time.time()
            result = run_episode(condition, episode_idx, args.max_steps)
            result["wall_s"] = time.time() - t0
            results.append(result)
            print(f"[{condition} ep{episode_idx}] {result}", flush=True)
            args.results_path.write_text(json.dumps(results, indent=2))

    print("=== T08 SOFT-PIPELINE REPORT ===")
    for condition in ("baseline_pixelmask", "soft_pipeline"):
        steps = [r["done_step"] for r in results if r["condition"] == condition and r["done_step"] is not None]
        n = len([r for r in results if r["condition"] == condition])
        print(f"{condition}: {len(steps)}/{n} success, steps={steps}")


if __name__ == "__main__":
    main()
