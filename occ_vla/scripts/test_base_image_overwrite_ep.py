"""Test (user request, 2026-07-23): overwrite base_image itself with
the GT-unoccluded render when occluded, instead of injecting a separate
subgoal_image (the design tested all day so far). Rationale raised by
the user: base_image is a slot pi0.5 was actually trained on, unlike
subgoal_image (repurposed right_wrist_0_rgb, never meaningfully filled
during training) -- so replacing occluded content in the KNOWN channel
might avoid the OOD-injection-slot cost already established (oracle
subgoal_image injection still failed 2/10 on libero_10 task 0).

This uses GT-unoccluded content (oracle upper bound, same technique as
render_gt_unoccluded in run_wm_subgoal_rollout_pipeline.py) rather than
a real generative model -- cheapest possible test of whether this
DIFFERENT injection point (known slot vs. new slot) changes the
outcome at all, before spending effort on a real generator for it.

Two conditions on mug_in_microwave (task_id=9, well-characterized):
  baseline: occluded base_image as-is, no intervention
  base_overwrite_gt: base_image replaced with GT-unoccluded render
    whenever arm_s_occ > threshold; wrist_image and everything else
    unchanged; no subgoal_image used at all

Run: python3 scripts/test_base_image_overwrite_ep.py --episodes N
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

from occ_vla.eval.libero_occ_env import AGENTVIEW_KEY, LiberoOccEnv, LiberoOccEnvConfig  # noqa: E402
from occ_vla.eval.metrics import Difficulty  # noqa: E402

PI05_RPC_DIR = str(_ROOT / ".rpc" / "pi05")
BENCHMARK_SUITE = "libero_10"
TASK_ID = 9  # mug_in_microwave
RESIZE_SIZE = 224
NUM_STEPS_WAIT = 10
SEED = 7
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
REPLAN_STEPS = 8
MAX_STEPS = 520
N_EPISODES = 10
ARM_OCC_THRESHOLD = 0.30
RESULTS_PATH = _ROOT / "base_image_overwrite_results.json"

CONDITIONS = ["baseline", "base_overwrite_gt"]


def preprocess_image(raw_image):
    flipped = np.ascontiguousarray(raw_image[::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(flipped, RESIZE_SIZE, RESIZE_SIZE))


def state_vec(obs):
    return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]).astype(np.float32)


def call_pi05(base_image, wrist_image, state, prompt):
    resp_arrays, _ = rpc.call(PI05_RPC_DIR, {"base_image": base_image, "wrist_image": wrist_image, "state": state}, {"prompt": prompt})
    return resp_arrays["actions"]


def robot_geom_ids_for(occ_env):
    model = occ_env._env.sim.model  # noqa: SLF001
    body_names = [model.body_id2name(i) for i in range(model.nbody)]
    robot_body_ids = {i for i, n in enumerate(body_names) if n and any(k in n.lower() for k in ("robot", "panda", "gripper", "mount"))}
    geom_bodyid = model.geom_bodyid
    return [g for g in range(model.ngeom) if geom_bodyid[g] in robot_body_ids]


def render_gt_unoccluded(occ_env, robot_geom_ids):
    env = occ_env._env  # noqa: SLF001
    model = env.sim.model
    state = env.get_sim_state()
    orig_rgba = model.geom_rgba[robot_geom_ids].copy()
    model.geom_rgba[robot_geom_ids, 3] = 0.0
    gt_obs = env.regenerate_obs_from_state(state)
    gt_agentview = gt_obs[AGENTVIEW_KEY].copy()
    model.geom_rgba[robot_geom_ids] = orig_rgba
    env.regenerate_obs_from_state(state)
    return gt_agentview


def run_episode(condition: str, episode_idx: int, max_steps: int) -> dict:
    from libero.libero import benchmark  # noqa: PLC0415

    bench = benchmark.get_benchmark(BENCHMARK_SUITE)()
    init_states = bench.get_task_init_states(TASK_ID)
    instruction = bench.get_task(TASK_ID).language

    config = LiberoOccEnvConfig(benchmark_suite=BENCHMARK_SUITE, task_id=TASK_ID, difficulty=Difficulty.LIGHT,
                                 init_state_idx=episode_idx % len(init_states), seed=SEED, place_occluder=False)
    occ_env = LiberoOccEnv(config, libero_root=str(_ROOT / "third_party/openpi/third_party/libero"))
    obs = occ_env.reset()
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = occ_env.step(LIBERO_DUMMY_ACTION)
    occ_env.capture_clear_baseline(obs)
    robot_geom_ids = robot_geom_ids_for(occ_env)

    action_queue = collections.deque()
    max_arm_s_occ = 0.0
    overwrite_steps = 0
    for step in range(max_steps):
        arm_s_occ = occ_env.compute_arm_s_occ(obs)
        max_arm_s_occ = max(max_arm_s_occ, arm_s_occ)
        occluded = arm_s_occ > ARM_OCC_THRESHOLD

        if condition == "base_overwrite_gt" and occluded:
            base_raw = render_gt_unoccluded(occ_env, robot_geom_ids)
            overwrite_steps += 1
        else:
            base_raw = obs[AGENTVIEW_KEY]
        base_image = preprocess_image(base_raw)
        wrist_image = preprocess_image(obs["robot0_eye_in_hand_image"])

        if action_queue:
            action = action_queue.popleft()
        else:
            actions = call_pi05(base_image, wrist_image, state_vec(obs), instruction)
            action = actions[0]
            action_queue.extend(actions[1:REPLAN_STEPS])

        obs, _, done, _ = occ_env.step(action.tolist())
        if done:
            return {"condition": condition, "episode": episode_idx, "done_step": step, "max_arm_s_occ": max_arm_s_occ, "overwrite_steps": overwrite_steps}

    return {"condition": condition, "episode": episode_idx, "done_step": None, "max_arm_s_occ": max_arm_s_occ, "overwrite_steps": overwrite_steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    results = json.loads(args.results_path.read_text()) if args.results_path.exists() else []

    def already_done(condition, episode_idx):
        return any(r["condition"] == condition and r["episode"] == episode_idx for r in results)

    for condition in args.conditions:
        for episode_idx in range(args.episodes):
            if already_done(condition, episode_idx):
                continue
            t0 = time.time()
            result = run_episode(condition, episode_idx, args.max_steps)
            result["wall_s"] = time.time() - t0
            results.append(result)
            print(f"[{condition} ep{episode_idx}] {result}", flush=True)
            args.results_path.write_text(json.dumps(results, indent=2))

    print("\n=== BASE IMAGE OVERWRITE REPORT ===")
    for condition in CONDITIONS:
        rows = [r for r in results if r["condition"] == condition]
        if not rows:
            continue
        steps = [r["done_step"] for r in rows if r["done_step"] is not None]
        print(f"{condition}: {len(steps)}/{len(rows)} success, steps={steps}")


if __name__ == "__main__":
    main()
