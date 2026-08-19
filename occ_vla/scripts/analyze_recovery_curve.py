"""
analyze_recovery_curve.py

Per user request (2026-08-19): "occluded_run_length vs action error" --
does the mid-layer correction's effect on the ACTION (not just features)
degrade the longer occlusion has persisted? Pure re-aggregation of
already-saved action_diff_log entries (occluded_run_length, delta_a_first,
delta_a_chunk_mean) -- zero new GPU compute, per the user's own framing
("ログの再集計のみ").

Usage:
  python analyze_recovery_curve.py --results-json <path to taskN.json>...
"""
import argparse
import json

import numpy as np


def load_entries(paths):
    entries = []
    for path in paths:
        d = json.load(open(path))
        oracle = d["results"]["oracle"] if isinstance(d.get("results"), dict) else d.get("oracle", [])
        for ep_idx, r in enumerate(oracle):
            for e in r.get("action_diff_log", []):
                entries.append(e)
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", nargs="+", required=True)
    parser.add_argument("--bucket-size", type=int, default=20,
                         help="occluded_run_length bucket width (env steps)")
    args = parser.parse_args()

    entries = load_entries(args.results_json)
    print(f"loaded {len(entries)} action_diff_log entries from {len(args.results_json)} file(s)\n")
    if not entries:
        print("no entries -- nothing to analyze")
        return

    run_lengths = np.array([e["occluded_run_length"] for e in entries])
    delta_first = np.array([e["delta_a_norm_first"] for e in entries])
    delta_chunk = np.array([e["delta_a_norm_chunk_mean"] for e in entries])

    print(f"occluded_run_length range: {run_lengths.min()}-{run_lengths.max()}")
    print(f"delta_a_first: mean={delta_first.mean():.4f} median={np.median(delta_first):.4f}")
    print(f"delta_a_chunk_mean: mean={delta_chunk.mean():.4f} median={np.median(delta_chunk):.4f}\n")

    max_len = run_lengths.max()
    buckets = list(range(0, int(max_len) + args.bucket_size, args.bucket_size))
    print(f"{'run_length bucket':<20}{'n':>6}{'delta_a_first (mean±sem)':>30}{'delta_a_chunk_mean (mean±sem)':>32}")
    bucket_means_first = []
    bucket_centers = []
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        mask = (run_lengths >= lo) & (run_lengths < hi)
        n = mask.sum()
        if n == 0:
            continue
        f_mean, f_sem = delta_first[mask].mean(), delta_first[mask].std(ddof=1) / max(np.sqrt(n), 1)
        c_mean, c_sem = delta_chunk[mask].mean(), delta_chunk[mask].std(ddof=1) / max(np.sqrt(n), 1)
        print(f"[{lo:>4}, {hi:>4})       {n:>6}{f_mean:>16.4f} ± {f_sem:<10.4f}{c_mean:>18.4f} ± {c_sem:<10.4f}")
        bucket_means_first.append(f_mean)
        bucket_centers.append((lo + hi) / 2)

    if len(bucket_centers) >= 3:
        rho = np.corrcoef(bucket_centers, bucket_means_first)[0, 1]
        # also raw (unbucketed) Spearman-style rank correlation, cheap via numpy
        order_x = run_lengths.argsort().argsort()
        order_y = delta_first.argsort().argsort()
        spearman_raw = np.corrcoef(order_x, order_y)[0, 1]
        print(f"\nbucket-mean Pearson r (run_length vs delta_a_first): {rho:.3f}")
        print(f"raw (unbucketed) Spearman rho (run_length vs delta_a_first): {spearman_raw:.3f}")


if __name__ == "__main__":
    main()
