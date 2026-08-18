"""
compute_occlusion_trajectory_metrics.py

Post-hoc trajectory analysis over the .npz files run_libero_eval_occlusion.py
saves when --log_trajectories is set (eef_pos (T,3), gripper_qpos (T,2),
engaged (T,) bool, success, done_step, correction_mode, task_id, episode_idx).

NAMING NOTE (2026-08-08): originally written as compute_trajectory_metrics.py,
which collided with a pre-existing, unrelated script of the same name
already committed to this repo (ADE/DTW/FDE analysis for
run_wm_subgoal_rollout_pipeline.py's pi0.5/MMaDA trajectories, a
completely different investigation sharing this same scripts/ directory --
see occ_vla/CLAUDE.md's own explicit warning about not conflating the two
projects). That collision briefly overwrote the original file in the
working copy before being caught and reverted -- renamed here to avoid
any repeat.

Computes, per (task_id, correction_mode):
  - n episodes, success rate (sanity cross-check against the eval log)
  - mean/median done_step (all episodes, and successes-only)
  - mean path length: sum of ||eef_pos[t+1]-eef_pos[t]|| over the episode
    (a directness/efficiency proxy -- longer path for the same task = more
    wasted/wandering motion, not necessarily more steps if some steps move
    very little)
  - mean per-step displacement ("jitter"): path_length / done_step -- how
    much the eef moves per control step on average; compared ACROSS
    conditions this flags whether correction makes motion visibly smoother
    or choppier, independent of raw episode length
  - engaged_frac: fraction of control steps where correction was actually
    active (0.0 for correction_mode=none by construction; for
    oracle/dynamic this should be ~1.0 given delay_steps=0 occludes the
    whole episode -- included as a sanity check, not a headline number,
    but turned out to be the decisive diagnostic for why libero_goal's
    dynamic trigger underperforms oracle: engaged_frac=15.6% there vs
    ~99% for libero_10's tasks, 2026-08-08)

No comparison to a true "clean" (unoccluded) reference trajectory is
possible from this data alone -- log_trajectories was not run under a
clean condition in this session, only none/oracle/dynamic (all under real
occlusion from t=0). This script only ever compares those three against
each other.

Run with the openvla-oft conda env or plain python3 (numpy only):
  python scripts/compute_occlusion_trajectory_metrics.py \
    --trajectory-dirs thirdparty/openvla-oft/trajectory_logs_prod_libero10 \
                       thirdparty/openvla-oft/trajectory_logs_prod_goal
"""

import argparse
import glob
import os

import numpy as np


def load_group(traj_dir):
    """Returns {(task_id, correction_mode): [episode_dict, ...]}."""
    groups = {}
    for path in sorted(glob.glob(os.path.join(traj_dir, "*.npz"))):
        d = np.load(path, allow_pickle=True)
        task_id = int(d["task_id"])
        mode = str(d["correction_mode"])
        key = (task_id, mode)
        groups.setdefault(key, []).append(
            {
                "eef_pos": d["eef_pos"],
                "engaged": d["engaged"],
                "success": bool(d["success"]),
                "done_step": int(d["done_step"]),
                "episode_idx": int(d["episode_idx"]),
                "path": path,
            }
        )
    return groups


def path_length(eef_pos):
    if len(eef_pos) < 2:
        return 0.0
    deltas = np.diff(eef_pos, axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def summarize(episodes):
    n = len(episodes)
    n_succ = sum(e["success"] for e in episodes)
    done_steps = np.array([e["done_step"] for e in episodes], dtype=float)
    succ_done_steps = np.array([e["done_step"] for e in episodes if e["success"]], dtype=float)
    path_lengths = np.array([path_length(e["eef_pos"]) for e in episodes])
    jitter = np.array([
        path_length(e["eef_pos"]) / max(e["done_step"], 1) for e in episodes
    ])
    engaged_fracs = np.array([
        float(e["engaged"].mean()) if len(e["engaged"]) > 0 else 0.0 for e in episodes
    ])
    return {
        "n": n,
        "success_rate": n_succ / n if n else 0.0,
        "mean_done_step": float(done_steps.mean()) if n else float("nan"),
        "mean_done_step_success_only": float(succ_done_steps.mean()) if len(succ_done_steps) else float("nan"),
        "mean_path_length": float(path_lengths.mean()) if n else float("nan"),
        "mean_jitter_per_step": float(jitter.mean()) if n else float("nan"),
        "mean_engaged_frac": float(engaged_fracs.mean()) if n else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dirs", nargs="+", required=True)
    args = parser.parse_args()

    all_groups = {}
    for d in args.trajectory_dirs:
        g = load_group(d)
        for k, v in g.items():
            all_groups.setdefault(k, []).extend(v)

    task_ids = sorted({k[0] for k in all_groups})
    for task_id in task_ids:
        print(f"\n=== task_id={task_id} ===")
        modes_here = sorted({k[1] for k in all_groups if k[0] == task_id})
        for mode in modes_here:
            eps = all_groups[(task_id, mode)]
            s = summarize(eps)
            print(
                f"  {mode:10s} n={s['n']:3d} success_rate={s['success_rate']*100:5.1f}%  "
                f"mean_done_step={s['mean_done_step']:6.1f}  "
                f"mean_done_step(success)={s['mean_done_step_success_only']:6.1f}  "
                f"mean_path_len={s['mean_path_length']:6.3f}  "
                f"mean_jitter/step={s['mean_jitter_per_step']:7.5f}  "
                f"mean_engaged_frac={s['mean_engaged_frac']*100:5.1f}%"
            )


if __name__ == "__main__":
    main()
