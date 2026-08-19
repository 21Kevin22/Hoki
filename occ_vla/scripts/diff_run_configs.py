"""
diff_run_configs.py

Per user request (2026-08-19), after a real incident: today's "current"-
depth comparisons silently used a different RESOLVED layer (dino=15,
siglip=17) than intended (dino=16, siglip=18) across different runs,
because a formula change wasn't reflected in the CLI default, and nothing
compared the two runs' actual resolved config before treating them as
"the same condition." Systematic fix: always diff two runs' run_config.json
(written by run_libero_occluded_oracle_headroom.py as of 2026-08-19) BEFORE
comparing their results.

Usage: python diff_run_configs.py <results_dir_1> <results_dir_2>
Exits non-zero (and prints a loud warning) if the resolved layers, task
ids, episode range, or conditions differ -- the cases that would silently
invalidate a "same condition, different run" comparison.
"""
import argparse
import json
import os
import sys

CRITICAL_KEYS = ["resolved_layers", "task_ids", "n_episodes", "episode_offset", "conditions"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir1")
    parser.add_argument("dir2")
    args = parser.parse_args()

    c1 = json.load(open(os.path.join(args.dir1, "run_config.json")))
    c2 = json.load(open(os.path.join(args.dir2, "run_config.json")))

    print(f"comparing {args.dir1} vs {args.dir2}\n")
    mismatch = False
    all_keys = sorted(set(c1) | set(c2))
    for k in all_keys:
        v1, v2 = c1.get(k, "<missing>"), c2.get(k, "<missing>")
        same = v1 == v2
        flag = "" if same else "  <-- DIFFERS"
        critical = " [CRITICAL]" if k in CRITICAL_KEYS else ""
        print(f"{k:<28}{str(v1):<35}{str(v2):<35}{flag}{critical}")
        if not same and k in CRITICAL_KEYS:
            mismatch = True

    print()
    if mismatch:
        print("FAILED: critical config fields differ -- these runs are NOT directly comparable.")
        print("Do not treat their results as the same condition until this is resolved.")
        sys.exit(1)
    else:
        print("OK: all critical config fields match -- safe to compare results directly.")


if __name__ == "__main__":
    main()
