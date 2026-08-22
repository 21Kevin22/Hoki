"""occ_vla addition (2026-08-22), per user's "Representation Alignment"
proposal (Option 1 of their 3 elegant-fine-tuning ideas): collects paired
(I_clean, I_occ) agentview frames at the SAME sim state for
representation-alignment training data. No rollout/policy needed --
reuses the exact alpha-hide-and-reveal technique already validated
throughout this session (find_occluder_body_names + geom_rgba alpha=0)
to render the identical state twice, once with the occluder visible and
once without. Small random actions (not a real policy) drive state
diversity across steps, matching the sibling project's own established
"random-action-delta" convention for this kind of ground-truth-pair
collection (its `collect_arm_removal_pairs.py`).
"""
import argparse
import json
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import register_libero_occ_suites  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from run_libero_occluded_oracle_headroom import (  # noqa: E402
    get_libero_env_seg, find_occluder_body_names, geom_ids_for_bodies, get_agentview_frames,
)


def collect_for_task(task_id, occ_suite, stock_suite, out_dir, n_episodes, n_samples_per_episode, resolution, seed_base):
    task = occ_suite.get_task(task_id)
    occluder_names = find_occluder_body_names(task, stock_suite)
    if not occluder_names:
        print(f"  task{task_id}: no occluder found, skipping")
        return []

    env = get_libero_env_seg(task, resolution=resolution)
    env.reset()
    sim = env.env.sim
    occluder_geom_ids = geom_ids_for_bodies(sim, set(occluder_names))
    if not occluder_geom_ids:
        print(f"  task{task_id}: no occluder geoms resolved, skipping")
        env.close()
        return []

    init_states = occ_suite.get_task_init_states(task_id)
    manifest = []
    rng = np.random.RandomState(seed_base + task_id)
    for ep in range(min(n_episodes, len(init_states))):
        env.seed(seed_base + task_id * 100 + ep)
        env.reset()
        # occ_vla bug fix (2026-08-22, same "stale sim reference after
        # env.reset()" lesson already documented repeatedly elsewhere in
        # this project): env.reset() reloads a fresh mjModel/mjData --
        # must re-fetch `sim` AND `occluder_geom_ids` are still valid
        # (geom indices are stable across reset since the XML doesn't
        # change) but `sim` itself must be re-bound every episode.
        sim = env.env.sim
        env.set_init_state(init_states[ep])
        for step_i in range(n_samples_per_episode):
            # small random action to diversify arm pose (not a real policy)
            action = rng.uniform(-0.3, 0.3, size=7)
            action[6] = 1.0 if rng.rand() > 0.5 else -1.0
            env.step(action.tolist())

            occ_color, _ = get_agentview_frames(env, resolution)
            orig_alpha = sim.model.geom_rgba[occluder_geom_ids, 3].copy()
            sim.model.geom_rgba[occluder_geom_ids, 3] = 0.0
            sim.forward()
            clean_color, _ = get_agentview_frames(env, resolution)
            sim.model.geom_rgba[occluder_geom_ids, 3] = orig_alpha
            sim.forward()

            uid = f"task{task_id}_ep{ep}_s{step_i}"
            occ_path = os.path.join(out_dir, f"{uid}_occ.png")
            clean_path = os.path.join(out_dir, f"{uid}_clean.png")
            Image.fromarray(occ_color).save(occ_path)
            Image.fromarray(clean_color).save(clean_path)
            manifest.append({"uid": uid, "task_id": task_id, "episode": ep, "step": step_i,
                              "occ_path": os.path.basename(occ_path), "clean_path": os.path.basename(clean_path)})
    env.close()
    print(f"  task{task_id}: collected {len(manifest)} pairs")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-ids", type=int, nargs="+", default=[1, 6, 8])
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--n-samples-per-episode", type=int, default=10)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--out-dir", default="clean_occluded_pairs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bm = benchmark.get_benchmark_dict()
    occ_suite = bm["libero_10_occluded"]()
    stock_suite = bm["libero_10"]()

    all_manifest = []
    for task_id in args.task_ids:
        all_manifest.extend(collect_for_task(
            task_id, occ_suite, stock_suite, args.out_dir,
            args.n_episodes, args.n_samples_per_episode, args.resolution, args.seed_base,
        ))

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(all_manifest, f, indent=2)
    print(f"\ntotal pairs: {len(all_manifest)}, saved to {args.out_dir}/manifest.json")


if __name__ == "__main__":
    main()
