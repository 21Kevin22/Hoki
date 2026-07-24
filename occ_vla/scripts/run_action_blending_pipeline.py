"""Plan 3 (post-hoc action blending) empirical test: baseline pi0.5 vs.
pi0.5 with its XY translation output blended toward a geometry-derived
correction while the target is arm-occluded.

Modeled closely on run_self_occlusion_pipeline.py (same tasks, same
target-tracking approach, same done_step/occluded_jitter_mean/
max_arm_s_occ metrics -- reused as-is, not reimplemented), but this is a
genuinely different intervention (edits the action vector itself, not
the image/text the policy conditions on) so it gets its own script
rather than another ABLATION_CONDITIONS entry there.

IMPORTANT ARCHITECTURE NOTE (found 2026-07-18 while setting this up):
integration/runtime.py::ControlLoop only runs its PKLP+action-blending
path in the SCENE branch (OcclusionRouter routes there only when
scene_dyn_occ=True and arm_s_occ is LOW -- i.e. a *different* object
occluding the target). moka_pots/bowl_top_drawer are ARM self-occlusion
(arm_s_occ HIGH -> SELF branch), which under the default config
(enable_subgoal_image_generation=False) never falls through to SCENE --
so running this experiment through ControlLoop as currently wired would
silently never blend anything. Rather than reworking ControlLoop's
routing (SCENE was deliberately built for a different occlusion source),
this script computes the blend directly and inline, exactly like
run_self_occlusion_pipeline.py already does for gate/overlay/cot --
using the *same* SCENE_BLEND_ALPHA and OSC_POSE_MAX_DELTA_M constants
imported from runtime.py (not duplicated), and the same
CameraProjector/pklp_pixel_delta_to_world_delta building blocks
validated in scripts/prototype_pixel_to_action.py. "predicted_pixel"
here is last_known_position (ground-truth segmentation centroid, same
mechanism run_self_occlusion_pipeline.py already uses and already
validated) rather than PKLP's RAFT-based constant-acceleration
extrapolation -- the target isn't a moving external object here, it's a
fixed grasp target hidden behind the arm, so "where was it last clearly
seen" is a strictly more reliable predicted_pixel than a velocity/
acceleration extrapolation designed for a different scenario.

Requires only the pi0.5 RPC worker (.rpc/pi05) running.
Run: python3 scripts/run_action_blending_pipeline.py [--episodes N] [--max-steps N]
"""

import argparse
import collections
import json
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

import os  # noqa: E402

PI05_RPC_DIR = os.environ.get("OCC_VLA_PI05_RPC_DIR", str(_ROOT / ".rpc" / "pi05"))
BENCHMARK_SUITE = "libero_spatial"
TASK_ID = 4
RESIZE_SIZE = 224
NUM_STEPS_WAIT = 10
SEED = 7
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
REPLAN_STEPS = 8
MAX_STEPS = 300
N_EPISODES = 10
GATE_THRESHOLD = 0.30  # matches control/occlusion_gating.DEFAULT_GATE_THRESHOLD
CLEAR_UPDATE_THRESHOLD = 0.05

RESULTS_PATH = _ROOT / "action_blending_pipeline_results.json"

CONDITIONS = ["baseline", "action_blending", "action_blending_gated"]


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


def target_centroid_224(occ_env, obs) -> np.ndarray | None:
    """Ground-truth target centroid, in the flipped 224x224 space
    preprocess_image() produces -- same convention CameraProjector's
    project() was validated against (CLAUDE.md, 2026-07-18)."""
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
    return np.array([flipped_xs.mean() * scale, flipped_ys.mean() * scale])


def run_episode(condition: str, episode_idx: int, max_steps: int, suite: str = None, task_id: int = None) -> dict:
    from libero.libero import benchmark  # noqa: PLC0415

    suite = suite or BENCHMARK_SUITE
    task_id = TASK_ID if task_id is None else task_id
    bench = benchmark.get_benchmark(suite)()
    init_states = bench.get_task_init_states(task_id)
    instruction = bench.get_task(task_id).language

    config = LiberoOccEnvConfig(
        benchmark_suite=suite, task_id=task_id, difficulty=Difficulty.LIGHT,
        init_state_idx=episode_idx % len(init_states), seed=SEED, place_occluder=False,
    )
    occ_env = LiberoOccEnv(config, libero_root=str(_ROOT / "third_party/openpi/third_party/libero"))
    obs = occ_env.reset()
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = occ_env.step(LIBERO_DUMMY_ACTION)
    occ_env.capture_clear_baseline(obs)

    projector = CameraProjector.from_sim(occ_env._env.sim, "agentview", resolution=RESIZE_SIZE)  # noqa: SLF001

    last_known_position = target_centroid_224(occ_env, obs)
    action_queue = collections.deque()
    blend_engaged_steps = 0
    max_arm_s_occ = 0.0
    prev_action = None
    occluded_jitter = []

    for step in range(max_steps):
        arm_s_occ = occ_env.compute_arm_s_occ(obs)
        max_arm_s_occ = max(max_arm_s_occ, arm_s_occ)
        centroid_now = target_centroid_224(occ_env, obs)
        if arm_s_occ < CLEAR_UPDATE_THRESHOLD and centroid_now is not None:
            last_known_position = centroid_now

        base_image = preprocess_image(obs[AGENTVIEW_KEY])
        wrist_image = preprocess_image(obs["robot0_eye_in_hand_image"])
        occluded = arm_s_occ >= GATE_THRESHOLD

        if action_queue:
            action = action_queue.popleft()
        else:
            actions = call_pi05(base_image, wrist_image, state_vec(obs), instruction)
            action = actions[0]
            action_queue.extend(actions[1:REPLAN_STEPS])

        if condition in ("action_blending", "action_blending_gated") and occluded and last_known_position is not None:
            eef_pos_world = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
            eef_pixel = projector.project(eef_pos_world)
            world_delta = pklp_pixel_delta_to_world_delta(projector, eef_pos_world, eef_pixel, last_known_position)
            pklp_delta_xy = world_delta[:2] / OSC_POSE_MAX_DELTA_M
            action = action.copy()
            if condition == "action_blending_gated":
                # Conflict-avoidance gate (2026-07-18, see
                # runtime.gated_blend_xy docstring): only blend when
                # pklp_delta_xy still agrees with pi0.5's own direction.
                blended = gated_blend_xy(action[:2].astype(np.float64), pklp_delta_xy, SCENE_BLEND_ALPHA)
            else:
                blended = (1.0 - SCENE_BLEND_ALPHA) * action[:2] + SCENE_BLEND_ALPHA * pklp_delta_xy
            action[:2] = np.clip(blended, -1.0, 1.0)
            blend_engaged_steps += 1

        if occluded and prev_action is not None:
            occluded_jitter.append(float(np.linalg.norm(action[:6] - prev_action[:6])))
        prev_action = action

        obs, _, done, _ = occ_env.step(action.tolist())
        if done:
            break

    return {
        "condition": condition,
        "episode": episode_idx,
        "done_step": step if done else None,
        "blend_engaged_steps": blend_engaged_steps,
        "max_arm_s_occ": max_arm_s_occ,
        "occluded_jitter_mean": float(np.mean(occluded_jitter)) if occluded_jitter else None,
        "occluded_jitter_n": len(occluded_jitter),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--suite", type=str, default=BENCHMARK_SUITE)
    parser.add_argument("--task-id", type=int, default=TASK_ID)
    args = parser.parse_args()

    results = json.loads(args.results_path.read_text()) if args.results_path.exists() else []

    def already_done(condition, episode_idx):
        return any(r["condition"] == condition and r["episode"] == episode_idx for r in results)

    for condition in args.conditions:
        for episode_idx in range(args.episodes):
            if already_done(condition, episode_idx):
                continue
            t0 = time.time()
            result = run_episode(condition, episode_idx, args.max_steps, suite=args.suite, task_id=args.task_id)
            result["wall_s"] = time.time() - t0
            results.append(result)
            print(f"[{condition} ep{episode_idx}] {result}", flush=True)
            args.results_path.write_text(json.dumps(results, indent=2))

    print("=== ACTION BLENDING PIPELINE REPORT ===")
    for condition in CONDITIONS:
        rows = [r for r in results if r["condition"] == condition]
        if not rows:
            continue
        steps = [r["done_step"] for r in rows if r["done_step"] is not None]
        blend_frac = np.mean([r["blend_engaged_steps"] > 0 for r in rows])
        jitters = [r["occluded_jitter_mean"] for r in rows if r.get("occluded_jitter_mean") is not None]
        jitter_str = f", mean_occluded_jitter={np.mean(jitters):.4f}" if jitters else ""
        # Easy/Hard post-hoc split by peak occlusion severity, per rollout
        # -- no engineered init states, just bucketing what actually
        # happened (median split within this condition's own episodes).
        occ_vals = [r["max_arm_s_occ"] for r in rows]
        median_occ = float(np.median(occ_vals)) if occ_vals else 0.0
        easy = [r for r in rows if r["max_arm_s_occ"] <= median_occ]
        hard = [r for r in rows if r["max_arm_s_occ"] > median_occ]
        easy_sr = np.mean([r["done_step"] is not None for r in easy]) if easy else float("nan")
        hard_sr = np.mean([r["done_step"] is not None for r in hard]) if hard else float("nan")
        print(
            f"{condition}: {len(steps)}/{len(rows)} success, steps={steps}, "
            f"episodes_with_blend_engaged={blend_frac:.0%}{jitter_str}, "
            f"easy(n={len(easy)})_sr={easy_sr:.0%}, hard(n={len(hard)})_sr={hard_sr:.0%}, "
            f"median_max_arm_s_occ={median_occ:.2f}"
        )


if __name__ == "__main__":
    main()
