"""ADE / DTW / FDE analysis (user request, 2026-07-23) for the
baseline / sd_styled / gt_unoccluded EEF trajectories logged by
run_wm_subgoal_rollout_pipeline.py, since binary success rate
saturated (10/10 or 9/10) on every libero_10 task tried so far and
can't detect the effect the single-frame proxy already found.

gt_unoccluded is the reference trajectory (oracle-content injection --
see that script's docstring). For each episode_idx, compares baseline
and sd_styled against the SAME episode's gt_unoccluded trajectory
(identical start state; the three runs diverge only once actions
differ, since each condition is its own independent closed-loop
rollout -- not a replay of a shared trajectory).

  ADE: mean pointwise L2 distance over the common index prefix
       (min(len_a, len_gt)) -- "how far off is the path, on average,
       while both are still running".
  FDE: L2 distance between each trajectory's OWN final EEF position
       and the reference's own final EEF position -- standard
       final-displacement-error definition, independent of length.
  DTW: dynamic-time-warping distance (handles the fact that the two
       closed-loop rollouts run at different effective speeds/timing
       once they diverge), normalized by (len_a + len_gt) so it's
       comparable across episodes of different lengths.

Run: python3 scripts/compute_trajectory_metrics.py <results.json> [--task-id N]
"""

import argparse
import json
from pathlib import Path

import numpy as np


def ade(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    return float(np.linalg.norm(a[:n] - b[:n], axis=1).mean())


def fde(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[-1] - b[-1]))


def dtw(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        row_costs = np.linalg.norm(b - ai, axis=1)  # cost to every b[j-1]
        for j in range(1, m + 1):
            D[i, j] = row_costs[j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


def mean_ci(values):
    values = np.array(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, 0.0
    sem = float(values.std(ddof=1) / np.sqrt(n))
    return mean, 1.96 * sem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path", type=Path)
    parser.add_argument("--task-id", type=int, default=None)
    args = parser.parse_args()

    results = json.loads(args.results_path.read_text())
    if args.task_id is not None:
        results = [r for r in results if r.get("task_id") == args.task_id]

    by_key = {}
    for r in results:
        if "eef_traj" not in r or not r["eef_traj"]:
            continue
        key = (r["task_id"], r["condition"], r["episode"])
        by_key[key] = np.array(r["eef_traj"], dtype=np.float64)

    task_ids = sorted({k[0] for k in by_key})
    for task_id in task_ids:
        episodes = sorted({k[2] for k in by_key if k[0] == task_id and k[1] == "gt_unoccluded"})
        print(f"\n=== task_id {task_id} (n_episodes={len(episodes)}) ===")

        metrics = {"baseline": {"ade": [], "fde": [], "dtw": []}, "sd_styled": {"ade": [], "fde": [], "dtw": []}}
        for ep in episodes:
            gt_key = (task_id, "gt_unoccluded", ep)
            if gt_key not in by_key:
                continue
            traj_gt = by_key[gt_key]
            for cond in ["baseline", "sd_styled"]:
                cond_key = (task_id, cond, ep)
                if cond_key not in by_key:
                    continue
                traj = by_key[cond_key]
                a = ade(traj, traj_gt)
                f = fde(traj, traj_gt)
                d = dtw(traj, traj_gt)
                metrics[cond]["ade"].append(a)
                metrics[cond]["fde"].append(f)
                metrics[cond]["dtw"].append(d)
                print(f"  ep{ep} {cond:<10} vs gt_unoccluded: ADE={a:.4f}  FDE={f:.4f}  DTW={d:.5f}  (len={len(traj)}, gt_len={len(traj_gt)})")

        print(f"\n  --- task {task_id} summary (mean +/- 95% CI over episodes) ---")
        for cond in ["baseline", "sd_styled"]:
            if not metrics[cond]["ade"]:
                continue
            am, ac = mean_ci(metrics[cond]["ade"])
            fm, fc = mean_ci(metrics[cond]["fde"])
            dm, dc = mean_ci(metrics[cond]["dtw"])
            print(f"  {cond:<10} ADE={am:.4f}+/-{ac:.4f}  FDE={fm:.4f}+/-{fc:.4f}  DTW={dm:.5f}+/-{dc:.5f}  (n={len(metrics[cond]['ade'])})")


if __name__ == "__main__":
    main()
