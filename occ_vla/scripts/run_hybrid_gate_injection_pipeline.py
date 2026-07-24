"""Hybrid experiment (user request, 2026-07-23): combine the two
previously-independent intervention systems in one rollout loop.

  - geometric gate (Plan 3 / action_blending_gated, see
    run_action_blending_pipeline.py): post-hoc, blends a geometry-
    derived XY correction into pi0.5's OUTPUT action, gated by
    cos_angle agreement with pi0.5's own predicted direction. No image
    generation, no change to pi0.5's input.
  - generative injection (sd_styled, see
    run_wm_subgoal_rollout_pipeline.py): pre-hoc, injects a style-
    adapted SD-inpainted image into pi0.5's INPUT (subgoal_image slot,
    repurposed right_wrist_0_rgb).

These operate on different ends of the pipeline (input vs. output) so
they are structurally independent and can run in the same step without
conflicting code-wise. The open question is *interaction*, not
compatibility: does the gate also catch/filter bad actions caused by
injection-driven OOD disruption (today's finding: even oracle-content
injection failed 2/10 on libero_10 task 0), the same way it already
catches ungated-blending's own bad corrections (mug_in_microwave
100%->40%->100%)?

Four conditions: baseline, gate_only (=action_blending_gated),
injection_only (=sd_styled), both.

Requires:
  - pi05_worker running WITH occ_vla inputs enabled
    (PI05_WORKER_USE_OCC_VLA_INPUTS=1)
  - sd_worker running (scripts/_workers/sd_worker.py, .venv_mmada)

Run: python3 scripts/run_hybrid_gate_injection_pipeline.py --suite libero_10 --task-id 8 --episodes N
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
SD_RPC_DIR = os.environ.get("SD_WORKER_RPC_DIR", str(_ROOT / ".rpc" / "sd_worker"))
BENCHMARK_SUITE = "libero_10"
TASK_ID = 8  # moka_pots
RESIZE_SIZE = 224
NUM_STEPS_WAIT = 10
SEED = 7
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
REPLAN_STEPS = 8
MAX_STEPS = 520
N_EPISODES = 10
GATE_THRESHOLD = 0.30
CLEAR_UPDATE_THRESHOLD = 0.05
RESULTS_PATH = _ROOT / "hybrid_gate_injection_results.json"
# 2026-07-23: validated fix for moka_pots episode 7 (continuous gate
# engagement over a long unbroken occlusion window blocked task
# completion; baseline solved it fine). Applying here too to confirm
# it doesn't weaken the safety property established on this task
# (ungated blending 100%->40%, gated blending back to 100%).
DECAY_START_STEPS = 30
DECAY_WINDOW = 50

CONDITIONS = ["baseline", "gate_only", "injection_only", "both"]


def decayed_alpha(consecutive_occluded_steps: int) -> float:
    if consecutive_occluded_steps <= DECAY_START_STEPS:
        return SCENE_BLEND_ALPHA
    frac_into_decay = (consecutive_occluded_steps - DECAY_START_STEPS) / DECAY_WINDOW
    return SCENE_BLEND_ALPHA * max(0.0, 1.0 - frac_into_decay)


def preprocess_image(raw_image: np.ndarray) -> np.ndarray:
    flipped = np.ascontiguousarray(raw_image[::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(flipped, RESIZE_SIZE, RESIZE_SIZE))


def state_vec(obs) -> np.ndarray:
    return np.concatenate(
        [obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
    ).astype(np.float32)


def call_pi05(base_image, wrist_image, state, prompt, subgoal_image=None):
    arrays = {"base_image": base_image, "wrist_image": wrist_image, "state": state}
    if subgoal_image is not None:
        arrays["subgoal_image"] = subgoal_image
    resp_arrays, _ = rpc.call(PI05_RPC_DIR, arrays, {"prompt": prompt})
    return resp_arrays["actions"]


def call_sd(image_raw, mask_raw, instruction):
    resp_arrays, _ = rpc.call(SD_RPC_DIR, {"image": image_raw, "mask": mask_raw.astype(np.uint8)}, {"instruction": instruction}, timeout_s=60)
    return resp_arrays["image"]


def arm_pixel_mask(occ_env, obs) -> np.ndarray:
    seg_dict = occ_env._env.get_segmentation_instances(obs[AGENTVIEW_SEGMENTATION_KEY])  # noqa: SLF001
    if "robot" not in seg_dict:
        return np.zeros(obs[AGENTVIEW_KEY].shape[:2], dtype=bool)
    return seg_dict["robot"].squeeze(-1) != 0


def target_centroid_224(occ_env, obs) -> np.ndarray | None:
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


def run_episode(suite: str, task_id: int, condition: str, episode_idx: int, max_steps: int, use_decay: bool = False) -> dict:
    from libero.libero import benchmark  # noqa: PLC0415

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

    use_gate = condition in ("gate_only", "both")
    use_injection = condition in ("injection_only", "both")

    action_queue = collections.deque()
    blend_engaged_steps = 0
    sd_calls = 0
    max_arm_s_occ = 0.0
    consecutive_occluded_steps = 0
    for step in range(max_steps):
        arm_s_occ = occ_env.compute_arm_s_occ(obs)
        max_arm_s_occ = max(max_arm_s_occ, arm_s_occ)
        centroid_now = target_centroid_224(occ_env, obs)
        if arm_s_occ < CLEAR_UPDATE_THRESHOLD and centroid_now is not None:
            last_known_position = centroid_now
        occluded = arm_s_occ >= GATE_THRESHOLD
        consecutive_occluded_steps = consecutive_occluded_steps + 1 if occluded else 0

        base_image = preprocess_image(obs[AGENTVIEW_KEY])
        wrist_image = preprocess_image(obs["robot0_eye_in_hand_image"])

        if action_queue:
            action = action_queue.popleft()
        else:
            subgoal_image = None
            if use_injection and occluded:
                mask_raw = arm_pixel_mask(occ_env, obs)
                recovered_raw = call_sd(obs[AGENTVIEW_KEY], mask_raw, instruction)
                subgoal_image = preprocess_image(recovered_raw)
                sd_calls += 1
            actions = call_pi05(base_image, wrist_image, state_vec(obs), instruction, subgoal_image=subgoal_image)
            action = actions[0]
            action_queue.extend(actions[1:REPLAN_STEPS])

        alpha = decayed_alpha(consecutive_occluded_steps) if use_decay else SCENE_BLEND_ALPHA
        if use_gate and occluded and last_known_position is not None and alpha > 0:
            eef_pos_world = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
            eef_pixel = projector.project(eef_pos_world)
            world_delta = pklp_pixel_delta_to_world_delta(projector, eef_pos_world, eef_pixel, last_known_position)
            pklp_delta_xy = world_delta[:2] / OSC_POSE_MAX_DELTA_M
            action = action.copy()
            blended = gated_blend_xy(action[:2].astype(np.float64), pklp_delta_xy, alpha)
            action[:2] = np.clip(blended, -1.0, 1.0)
            blend_engaged_steps += 1

        obs, _, done, _ = occ_env.step(action.tolist())
        if done:
            return {"suite": suite, "task_id": task_id, "instruction": instruction, "condition": condition, "episode": episode_idx, "use_decay": use_decay,
                    "done_step": step, "max_arm_s_occ": max_arm_s_occ, "sd_calls": sd_calls, "blend_engaged_steps": blend_engaged_steps}

    return {"suite": suite, "task_id": task_id, "instruction": instruction, "condition": condition, "episode": episode_idx, "use_decay": use_decay,
            "done_step": None, "max_arm_s_occ": max_arm_s_occ, "sd_calls": sd_calls, "blend_engaged_steps": blend_engaged_steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, default=BENCHMARK_SUITE)
    parser.add_argument("--task-id", type=int, default=TASK_ID)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    parser.add_argument("--use-decay", action="store_true")
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    results = json.loads(args.results_path.read_text()) if args.results_path.exists() else []

    def already_done(condition, episode_idx):
        return any(
            r.get("suite") == args.suite and r.get("task_id") == args.task_id
            and r["condition"] == condition and r.get("use_decay", False) == args.use_decay and r["episode"] == episode_idx
            for r in results
        )

    for condition in args.conditions:
        for episode_idx in range(args.episodes):
            if already_done(condition, episode_idx):
                continue
            t0 = time.time()
            result = run_episode(args.suite, args.task_id, condition, episode_idx, args.max_steps, args.use_decay)
            result["wall_s"] = time.time() - t0
            results.append(result)
            print(f"[{args.suite}:{args.task_id} {condition} ep{episode_idx}] {result}", flush=True)
            args.results_path.write_text(json.dumps(results, indent=2))

    print("\n=== HYBRID GATE+INJECTION REPORT ===")
    for condition in CONDITIONS:
        rows = [r for r in results if r.get("suite") == args.suite and r.get("task_id") == args.task_id and r["condition"] == condition
                and r.get("use_decay", False) == args.use_decay]
        if not rows:
            continue
        steps = [r["done_step"] for r in rows if r["done_step"] is not None]
        sd_calls_total = sum(r.get("sd_calls", 0) for r in rows)
        blend_total = sum(r.get("blend_engaged_steps", 0) for r in rows)
        n_engaged = sum(1 for r in rows if r.get("sd_calls", 0) > 0 or r.get("blend_engaged_steps", 0) > 0)
        print(f"{condition}: {len(steps)}/{len(rows)} success, steps={steps}")
        if condition != "baseline":
            # 2026-07-23: moka_pots produced a fully invalid comparison
            # (0 engaged episodes across 40+30 episodes, 3 separate
            # runs) that looked like a real effect in raw success rate
            # -- always surface engagement rate so this can't happen
            # silently again.
            flag = "  <-- WARNING: mechanism never engaged, this comparison is meaningless" if n_engaged == 0 else ""
            print(f"    engagement: {n_engaged}/{len(rows)} episodes had sd_calls>0 or blend_engaged_steps>0 "
                  f"(sd_calls_total={sd_calls_total}, blend_engaged_steps_total={blend_total}){flag}")


if __name__ == "__main__":
    main()
