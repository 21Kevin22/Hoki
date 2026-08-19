"""
analyze_shift_deltaa_success.py

Per user request (2026-08-19): d(spliced, reference) x ||Delta-a|| x
success -- no trained predictor needed, oracle features + action-diff logs
already exist. If "larger representation shift -> larger delta_a -> lower
success" all hold together, that's a step from a purely descriptive
finding toward a causal chain for the mechanism.

Matches per-step oracle feature .npz files (task{task}_ep{ep}_t{t}_features.npz,
key "dino"/"siglip" = injected patch_clean, mean-pooled -> Mahalanobis
distance to the reference distribution, same method as
analyze_correction_distribution_shift.py) against the corresponding
action_diff_log entry (same task/ep/t) for delta_a_norm_first, and the
episode's own success flag.

Usage:
  python analyze_shift_deltaa_success.py \\
      --reference-dir feature_distribution_reference_task1 \\
      --features-dir oracle_features_task1_replication \\
      --results-json libero_occluded_oracle_task1_replication_offset20_n20/task1.json
"""
import argparse
import glob
import json
import os
import re

import numpy as np
from scipy import stats


def load_pooled_ref(npz_dir, key):
    vecs = []
    for path in sorted(glob.glob(os.path.join(npz_dir, "*.npz"))):
        d = np.load(path)
        if key not in d:
            continue
        arr = d[key]
        vecs.append(arr.reshape(-1, arr.shape[-1]).mean(axis=0))
    return np.stack(vecs, axis=0)


def fit_mahalanobis(ref, n_components=30):
    n_ref, d = ref.shape
    n_components = max(1, min(n_components, n_ref - 1, d))
    mu = ref.mean(axis=0)
    ref_c = ref - mu
    U, S, Vt = np.linalg.svd(ref_c, full_matrices=False)
    comps = Vt[:n_components]
    ref_proj = ref_c @ comps.T
    var = np.maximum(ref_proj.var(axis=0, ddof=1), 1e-8)

    def score(x):
        x_proj = (x - mu) @ comps.T
        return float(np.sqrt(((x_proj ** 2) / var).sum()))

    return score


def partial_corr(x, y, z):
    rxy = np.corrcoef(x, y)[0, 1]
    rxz = np.corrcoef(x, z)[0, 1]
    ryz = np.corrcoef(y, z)[0, 1]
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return (rxy - rxz * ryz) / denom if denom > 1e-12 else np.nan


FNAME_RE = re.compile(r"task(\d+)_ep(\d+)_t(\d+)_features\.npz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    # action_diff_log lookup: (episode, t) -> delta_a_norm_first
    d = json.load(open(args.results_json))
    oracle = d["results"]["oracle"] if isinstance(d.get("results"), dict) else d.get("oracle", [])
    delta_lookup = {}
    success_lookup = {}
    for r in oracle:
        ep = r["episode"]
        success_lookup[ep] = bool(r["success"])
        for e in r.get("action_diff_log", []):
            delta_lookup[(ep, e["t"])] = e["delta_a_norm_first"]

    scorers = {}
    for key in ["dino", "siglip"]:
        ref = load_pooled_ref(args.reference_dir, key)
        scorers[key] = (fit_mahalanobis(ref), ref.shape[0])
        print(f"reference '{key}': n={ref.shape[0]}")

    rows = []  # (episode, t, d_dino, d_siglip, delta_a, success)
    for path in sorted(glob.glob(os.path.join(args.features_dir, "*.npz"))):
        m = FNAME_RE.match(os.path.basename(path))
        if not m:
            continue
        _, ep, t = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if (ep, t) not in delta_lookup:
            continue
        npz = np.load(path)
        d_dino = scorers["dino"][0](npz["dino"].reshape(-1, npz["dino"].shape[-1]).mean(axis=0))
        d_siglip = scorers["siglip"][0](npz["siglip"].reshape(-1, npz["siglip"].shape[-1]).mean(axis=0))
        rows.append((ep, t, d_dino, d_siglip, delta_lookup[(ep, t)], success_lookup[ep]))

    print(f"\nmatched {len(rows)} (feature, action_diff) pairs\n")
    eps = np.array([r[0] for r in rows])
    d_dino = np.array([r[2] for r in rows])
    d_siglip = np.array([r[3] for r in rows])
    delta_a = np.array([r[4] for r in rows])
    success = np.array([r[5] for r in rows])

    for name, d_arr in [("dino", d_dino), ("siglip", d_siglip)]:
        rho, p = stats.spearmanr(d_arr, delta_a)
        print(f"[{name}] Spearman rho(shift, delta_a) = {rho:.3f} (naive p={p:.2e}, step-level -- see cluster bootstrap)")
        # episode-cluster bootstrap
        unique_eps = np.unique(eps)
        boots = []
        rng = np.random.default_rng(0)
        for _ in range(args.n_bootstrap):
            sampled = rng.choice(unique_eps, size=len(unique_eps), replace=True)
            idx = np.concatenate([np.where(eps == e)[0] for e in sampled])
            if len(idx) < 10:
                continue
            r, _ = stats.spearmanr(d_arr[idx], delta_a[idx])
            if not np.isnan(r):
                boots.append(r)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"       episode-cluster bootstrap 95% CI: [{lo:.3f}, {hi:.3f}]")

        # episode-level: mean shift vs success (point-biserial, genuinely independent units)
        ep_mean_d, ep_succ = [], []
        for e in unique_eps:
            mask = eps == e
            ep_mean_d.append(d_arr[mask].mean())
            ep_succ.append(success[eps == e][0])
        ep_mean_d, ep_succ = np.array(ep_mean_d), np.array(ep_succ, dtype=bool)
        if ep_succ.sum() >= 2 and (~ep_succ).sum() >= 2:
            pb_r, pb_p = stats.pointbiserialr(ep_succ.astype(float), ep_mean_d)
            print(f"       episode-level mean shift vs success: point-biserial r={pb_r:.3f}, p={pb_p:.4f} "
                  f"(sign: {'higher shift in FAILURES' if pb_r < 0 else 'higher shift in SUCCESSES'})")
        print()


if __name__ == "__main__":
    main()
