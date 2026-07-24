"""Per-suite success-rate sweep (2026-07-19): n=20 episodes/task, gated
action_blending only, across the 35 tasks not yet individually tested
this session (libero_spatial minus task 4, libero_object minus task 0,
libero_goal minus task 1, libero_10 minus tasks 8/9 -- those 3 already
have full n=10 baseline+gated data, see
action_blending_{bowl_top_drawer,moka_pots,mug_in_microwave}_results.json).
libero_90 excluded per explicit user request (90 tasks, disproportionate
cost).

Runs everything under ONE long-lived pi0.5 worker (unlike the earlier
per-task wrapper invocations) since this is ~9 hours of wall time and
reloading the checkpoint 35 times would waste GPU time for no benefit --
reuses run_episode from run_action_blending_pipeline.py rather than
reimplementing the control loop.

Requires only the pi0.5 RPC worker (.rpc/pi05 or override via
OCC_VLA_PI05_RPC_DIR). Run:
  scripts/run_with_gpu_cleanup.sh python3 scripts/run_suite_success_rate_sweep.py
"""

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_action_blending_pipeline import run_episode  # noqa: E402

N_EPISODES = 20
CONDITION = "action_blending_gated"
RESULTS_PATH = _ROOT / "suite_success_rate_sweep_results.json"

# (suite, task_id, max_steps) -- max_steps matches this session's
# established per-suite convention (see CLAUDE.md), not left at a
# single default, to avoid the earlier MAX_STEPS/suite mismatch bug.
SPATIAL_MAX_STEPS = 300
OBJECT_MAX_STEPS = 280
GOAL_MAX_STEPS = 300
LIBERO10_MAX_STEPS = 520

TASKS = (
    [("libero_spatial", tid, SPATIAL_MAX_STEPS) for tid in [0, 1, 2, 3, 5, 6, 7, 8, 9]]
    + [("libero_object", tid, OBJECT_MAX_STEPS) for tid in [1, 2, 3, 4, 5, 6, 7, 8, 9]]
    + [("libero_goal", tid, GOAL_MAX_STEPS) for tid in [0, 2, 3, 4, 5, 6, 7, 8, 9]]
    + [("libero_10", tid, LIBERO10_MAX_STEPS) for tid in [0, 1, 2, 3, 4, 5, 6, 7]]
)


def load_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return {}


def task_key(suite: str, task_id: int) -> str:
    return f"{suite}:{task_id}"


def main():
    results = load_results()
    print(f"=== SWEEP START: {len(TASKS)} tasks x {N_EPISODES} episodes, condition={CONDITION} ===", flush=True)

    for suite, task_id, max_steps in TASKS:
        key = task_key(suite, task_id)
        episodes = results.setdefault(key, {"suite": suite, "task_id": task_id, "max_steps": max_steps, "episodes": []})["episodes"]
        already_done = {e["episode"] for e in episodes}

        for episode_idx in range(N_EPISODES):
            if episode_idx in already_done:
                continue
            t0 = time.time()
            result = run_episode(CONDITION, episode_idx, max_steps, suite=suite, task_id=task_id)
            result["wall_s"] = time.time() - t0
            episodes.append(result)
            RESULTS_PATH.write_text(json.dumps(results, indent=2))

        n_success = sum(1 for e in episodes if e["done_step"] is not None)
        print(f"[TASK DONE] {key}: {n_success}/{len(episodes)} success", flush=True)

    print("=== SWEEP REPORT (per suite) ===", flush=True)
    by_suite: dict[str, list] = {}
    for key, data in results.items():
        by_suite.setdefault(data["suite"], []).extend(data["episodes"])
    for suite, episodes in by_suite.items():
        n_success = sum(1 for e in episodes if e["done_step"] is not None)
        print(f"{suite}: {n_success}/{len(episodes)} = {n_success / len(episodes):.1%}", flush=True)

    print("SWEEP_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
