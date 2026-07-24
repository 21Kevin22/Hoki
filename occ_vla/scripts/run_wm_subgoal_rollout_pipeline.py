"""Real multi-episode rollout success-rate comparison for the "WM +
subgoal" (style-adapted SD inpainting) direction explored 2026-07-23 --
per user request, the first actual task-completion-rate test of this
whole day's image-injection work (everything before this was a single-
frame, N=20-calls action-distance-to-GT proxy, not a rollout success
measure -- see CLAUDE.md's own caution about conflating the two).

Bug caught while building this (2026-07-23): earlier single-frame tests
(recovered_224.png, raw_sd_224.png, styled_smoothed_sd_224.png) were
derived from occluded_agentview_256.png, which is the RAW (unflipped)
render -- but base_image/wrist_image both go through preprocess_image's
`[::-1, ::-1]` flip (CLAUDE.md item 9: getting this wrong took a working
checkpoint to 0% success for base_image specifically). Those earlier
subgoal images were therefore NOT flip-consistent with base_image. This
script applies the same preprocess_image() flip to the generated
subgoal image before injection, fixing that inconsistency going
forward -- today's earlier single-frame results should be read with
that caveat (the effect found was still real and directionally
sensible, but was measured with a subtly wrong-oriented injected image).

Two conditions: baseline (occluded, no injection) vs. sd_styled
(occluded + style-adapted SD inpainting injected into subgoal_image
whenever arm_s_occ > ARM_OCC_THRESHOLD). SD regenerated only when pi0.5
is queried fresh (not every single env step), matching the existing
REPLAN_STEPS action-requery cadence, to keep wall-clock reasonable
(~5s/SD-call, matching test_sd_style_adapted.py's timing).

Extended 2026-07-23 (user request) to sweep across arbitrary
suite/task_id, since libero_spatial:4 (bowl_top_drawer) turned out to
be a ceiling case (baseline already 10/10 -- see CLAUDE.md/prior
session note): both conditions hit 10/10 there, so the method's effect
was unmeasurable on that task specifically. libero_10 was picked next
since several of its tasks involve reaching into enclosed spaces
(drawers/microwave/caddy) and are generally harder than libero_spatial,
making a baseline failure (and therefore a measurable gap) more likely.

Requires:
  - pi05_worker running WITH occ_vla inputs enabled
    (PI05_WORKER_USE_OCC_VLA_INPUTS=1)
  - sd_worker running (scripts/_workers/sd_worker.py, .venv_mmada)

Extended again 2026-07-23 (user request) with a third condition,
gt_unoccluded, and per-step EEF trajectory logging, because success
rate saturated (10/10 or 9/10) on every libero_10 task tried so far --
too coarse to detect the effect the single-frame proxy already found.
gt_unoccluded injects the TRUE arm-alpha-zeroed render (same technique
as collect_occlusion_moment_with_state.py: zero the robot geoms' rgba
alpha, re-render via env.regenerate_obs_from_state, restore) into the
same subgoal_image slot sd_styled uses -- i.e. the content-quality
ceiling for this injection mechanism, holding the mechanism itself
fixed. Its trajectory is used as the reference for ADE/DTW/FDE: does
sd_styled's real trajectory move closer to this oracle-injected
trajectory than baseline's (no-injection) trajectory does?

Run: python3 scripts/run_wm_subgoal_rollout_pipeline.py [--suite libero_10] [--task-id N] [--episodes N] [--conditions ...]
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

PI05_RPC_DIR = os.environ.get("OCC_VLA_PI05_RPC_DIR", str(_ROOT / ".rpc" / "pi05"))
SD_RPC_DIR = os.environ.get("SD_WORKER_RPC_DIR", str(_ROOT / ".rpc" / "sd_worker"))
BENCHMARK_SUITE = "libero_spatial"
TASK_ID = 4
RESIZE_SIZE = 224
NUM_STEPS_WAIT = 10
SEED = 7
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
REPLAN_STEPS = 8
MAX_STEPS = 300
N_EPISODES = 10
ARM_OCC_THRESHOLD = 0.30
RESULTS_PATH = _ROOT / "wm_subgoal_rollout_results.json"

CONDITIONS = ["baseline", "sd_styled", "gt_unoccluded"]


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


def robot_geom_ids_for(occ_env) -> list:
    model = occ_env._env.sim.model  # noqa: SLF001
    body_names = [model.body_id2name(i) for i in range(model.nbody)]
    robot_body_ids = {
        i for i, n in enumerate(body_names) if n and any(k in n.lower() for k in ("robot", "panda", "gripper", "mount"))
    }
    geom_bodyid = model.geom_bodyid
    return [g for g in range(model.ngeom) if geom_bodyid[g] in robot_body_ids]


def render_gt_unoccluded(occ_env, robot_geom_ids) -> np.ndarray:
    """Same sim state, robot geoms alpha-zeroed -- see collect_occlusion_moment_with_state.py."""
    env = occ_env._env  # noqa: SLF001
    model = env.sim.model
    state = env.get_sim_state()
    orig_rgba = model.geom_rgba[robot_geom_ids].copy()
    model.geom_rgba[robot_geom_ids, 3] = 0.0
    gt_obs = env.regenerate_obs_from_state(state)
    gt_agentview = gt_obs[AGENTVIEW_KEY].copy()
    model.geom_rgba[robot_geom_ids] = orig_rgba
    env.regenerate_obs_from_state(state)  # restore render state before continuing the rollout
    return gt_agentview


def run_episode(suite: str, task_id: int, condition: str, episode_idx: int, max_steps: int) -> dict:
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

    robot_geom_ids = robot_geom_ids_for(occ_env) if condition == "gt_unoccluded" else None

    action_queue = collections.deque()
    max_arm_s_occ = 0.0
    sd_calls = 0
    eef_traj = []
    for step in range(max_steps):
        arm_s_occ = occ_env.compute_arm_s_occ(obs)
        max_arm_s_occ = max(max_arm_s_occ, arm_s_occ)
        occluded = arm_s_occ > ARM_OCC_THRESHOLD
        eef_traj.append(obs["robot0_eef_pos"].tolist())

        base_image = preprocess_image(obs[AGENTVIEW_KEY])
        wrist_image = preprocess_image(obs["robot0_eye_in_hand_image"])

        if action_queue:
            action = action_queue.popleft()
        else:
            subgoal_image = None
            if condition == "sd_styled" and occluded:
                mask_raw = arm_pixel_mask(occ_env, obs)
                recovered_raw = call_sd(obs[AGENTVIEW_KEY], mask_raw, instruction)
                subgoal_image = preprocess_image(recovered_raw)  # same flip as base_image -- bug fix, see module docstring
                sd_calls += 1
            elif condition == "gt_unoccluded" and occluded:
                recovered_raw = render_gt_unoccluded(occ_env, robot_geom_ids)
                subgoal_image = preprocess_image(recovered_raw)
                sd_calls += 1
            actions = call_pi05(base_image, wrist_image, state_vec(obs), instruction, subgoal_image=subgoal_image)
            action = actions[0]
            action_queue.extend(actions[1:REPLAN_STEPS])

        obs, _, done, _ = occ_env.step(action.tolist())
        if done:
            return {"suite": suite, "task_id": task_id, "instruction": instruction, "condition": condition, "episode": episode_idx, "done_step": step, "max_arm_s_occ": max_arm_s_occ, "sd_calls": sd_calls, "eef_traj": eef_traj}

    return {"suite": suite, "task_id": task_id, "instruction": instruction, "condition": condition, "episode": episode_idx, "done_step": None, "max_arm_s_occ": max_arm_s_occ, "sd_calls": sd_calls, "eef_traj": eef_traj}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, default=BENCHMARK_SUITE)
    parser.add_argument("--task-ids", type=int, nargs="+", default=[TASK_ID])
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    results = json.loads(args.results_path.read_text()) if args.results_path.exists() else []

    def already_done(suite, task_id, condition, episode_idx):
        return any(
            r.get("suite") == suite and r.get("task_id") == task_id
            and r["condition"] == condition and r["episode"] == episode_idx
            for r in results
        )

    for task_id in args.task_ids:
        for condition in args.conditions:
            for episode_idx in range(args.episodes):
                if already_done(args.suite, task_id, condition, episode_idx):
                    continue
                t0 = time.time()
                result = run_episode(args.suite, task_id, condition, episode_idx, args.max_steps)
                result["wall_s"] = time.time() - t0
                results.append(result)
                print(f"[{args.suite}:{task_id} {condition} ep{episode_idx}] {result}", flush=True)
                args.results_path.write_text(json.dumps(results, indent=2))

    print("\n=== WM+SUBGOAL ROLLOUT REPORT ===")
    for task_id in args.task_ids:
        for condition in CONDITIONS:
            rows = [r for r in results if r.get("suite") == args.suite and r.get("task_id") == task_id and r["condition"] == condition]
            if not rows:
                continue
            steps = [r["done_step"] for r in rows if r["done_step"] is not None]
            print(f"{args.suite}:{task_id} {condition}: {len(steps)}/{len(rows)} success, steps={steps}")


if __name__ == "__main__":
    main()
