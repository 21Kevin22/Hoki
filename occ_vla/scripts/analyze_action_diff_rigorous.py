"""
analyze_action_diff_rigorous.py

Per user's statistical critique (2026-08-19) of the first recovery-curve
pass: addresses 4 specific problems before any correlation number is
trusted.

  (1) Episode-level ||Delta-a|| vs success/failure -- cheap circumstantial
      evidence for "is this a perturbation" (existing logs only).
  (2) Partial correlation of occluded_run_length vs delta_a CONTROLLING
      FOR absolute step t (the confound: representation/policy state also
      drifts with t regardless of occlusion length; a raw correlation
      can't distinguish "occlusion duration" from "just later in the
      episode"). Reported both on pooled steps AND via an episode-cluster
      bootstrap (steps within one episode are not independent -- a raw
      n=947 p-value is invalid).
  (3) Report Spearman (rank, robust to the funnel-shaped variance already
      visible in the bucket table) as the PRIMARY number; bucket-mean
      Pearson r is for the figure only, never quoted as the headline
      statistic (inflated by within-bucket averaging).
  (4) Per-bucket episode counts, so small-n tail buckets (>500 steps) are
      visually distinguishable, not silently pooled with well-powered ones.

Usage:
  python analyze_action_diff_rigorous.py --results-json <path>...
"""
import argparse
import json

import numpy as np
from scipy import stats


def load_episodes(paths):
    """Returns list of dicts: {success, entries: [action_diff_log dicts]}."""
    episodes = []
    for path in paths:
        d = json.load(open(path))
        oracle = d["results"]["oracle"] if isinstance(d.get("results"), dict) else d.get("oracle", [])
        for r in oracle:
            episodes.append({"success": bool(r["success"]), "entries": r.get("action_diff_log", [])})
    return episodes


def partial_corr(x, y, z):
    """Standard partial correlation of x,y controlling for z (Pearson-based,
    on ranks if x/y/z are pre-rank-transformed by the caller for a
    Spearman-style partial correlation)."""
    rxy = np.corrcoef(x, y)[0, 1]
    rxz = np.corrcoef(x, z)[0, 1]
    ryz = np.corrcoef(y, z)[0, 1]
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return (rxy - rxz * ryz) / denom if denom > 1e-12 else np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", nargs="+", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--bucket-size", type=int, default=20)
    args = parser.parse_args()

    episodes = load_episodes(args.results_json)
    n_ep = len(episodes)
    print(f"loaded {n_ep} episodes from {len(args.results_json)} file(s)\n")

    # ---------------------------------------------------------------
    # (1) Episode-level mean ||Delta-a|| vs success/failure
    # ---------------------------------------------------------------
    print("=" * 70)
    print("(1) Episode-level mean delta_a_first vs success/failure")
    print("=" * 70)
    ep_mean_delta = []
    ep_success = []
    for ep in episodes:
        if not ep["entries"]:
            continue
        vals = [e["delta_a_norm_first"] for e in ep["entries"]]
        ep_mean_delta.append(np.mean(vals))
        ep_success.append(ep["success"])
    ep_mean_delta = np.array(ep_mean_delta)
    ep_success = np.array(ep_success, dtype=bool)
    n_succ, n_fail = ep_success.sum(), (~ep_success).sum()
    print(f"episodes with >=1 action_diff_log entry: {len(ep_mean_delta)} "
          f"(success={n_succ}, fail={n_fail})")
    if n_succ >= 2 and n_fail >= 2:
        succ_vals, fail_vals = ep_mean_delta[ep_success], ep_mean_delta[~ep_success]
        print(f"  success episodes: mean={succ_vals.mean():.4f} median={np.median(succ_vals):.4f}")
        print(f"  failure episodes: mean={fail_vals.mean():.4f} median={np.median(fail_vals):.4f}")
        u_stat, p_mwu = stats.mannwhitneyu(succ_vals, fail_vals, alternative="two-sided")
        # point-biserial (episode-level, n=n_ep, genuinely independent unit here)
        pb_r, pb_p = stats.pointbiserialr(ep_success.astype(float), ep_mean_delta)
        print(f"  Mann-Whitney U={u_stat:.1f}, p={p_mwu:.4f}")
        print(f"  point-biserial r={pb_r:.3f}, p={pb_p:.4f}  "
              f"(sign: {'higher delta_a in FAILURES' if pb_r < 0 else 'higher delta_a in SUCCESSES'})")
    else:
        print("  too few success or failure episodes for a meaningful test")
    print()

    # ---------------------------------------------------------------
    # (2) Partial correlation of run_length vs delta_a, controlling for t
    # ---------------------------------------------------------------
    print("=" * 70)
    print("(2) occluded_run_length vs delta_a_first, controlling for step t")
    print("=" * 70)
    all_entries, entry_ep_id = [], []
    for i, ep in enumerate(episodes):
        for e in ep["entries"]:
            all_entries.append(e)
            entry_ep_id.append(i)
    run_len = np.array([e["occluded_run_length"] for e in all_entries], dtype=float)
    t_arr = np.array([e["t"] for e in all_entries], dtype=float)
    delta = np.array([e["delta_a_norm_first"] for e in all_entries], dtype=float)
    entry_ep_id = np.array(entry_ep_id)
    n = len(all_entries)
    print(f"pooled steps: n={n} across {len(episodes)} episodes")

    raw_rho, raw_p = stats.spearmanr(run_len, delta)
    t_rho, _ = stats.spearmanr(t_arr, delta)
    rl_t_rho, _ = stats.spearmanr(run_len, t_arr)
    print(f"raw Spearman rho(run_length, delta_a): {raw_rho:.3f} (naive p={raw_p:.2e}, INVALID -- steps not independent, see below)")
    print(f"Spearman rho(t, delta_a): {t_rho:.3f}")
    print(f"Spearman rho(run_length, t): {rl_t_rho:.3f}  <- how confounded run_length and t actually are")

    # rank-transform then partial-correlate (Spearman-style partial corr)
    rank_rl = stats.rankdata(run_len)
    rank_t = stats.rankdata(t_arr)
    rank_delta = stats.rankdata(delta)
    partial_rho = partial_corr(rank_rl, rank_delta, rank_t)
    print(f"PARTIAL Spearman rho(run_length, delta_a | t): {partial_rho:.3f}  <- primary number, controls for the t confound")

    # Episode-cluster bootstrap for a valid CI (resample EPISODES with
    # replacement, not individual steps -- steps within an episode are not
    # independent, so a naive pooled p-value/CI is invalid per the user's
    # point (3))
    unique_ep_ids = np.unique(entry_ep_id)
    boot_partials = []
    rng = np.random.default_rng(0)
    for _ in range(args.n_bootstrap):
        sampled_eps = rng.choice(unique_ep_ids, size=len(unique_ep_ids), replace=True)
        idx = np.concatenate([np.where(entry_ep_id == e)[0] for e in sampled_eps])
        if len(idx) < 10:
            continue
        rl_b, t_b, d_b = run_len[idx], t_arr[idx], delta[idx]
        pr = partial_corr(stats.rankdata(rl_b), stats.rankdata(d_b), stats.rankdata(t_b))
        if not np.isnan(pr):
            boot_partials.append(pr)
    boot_partials = np.array(boot_partials)
    ci_lo, ci_hi = np.percentile(boot_partials, [2.5, 97.5])
    print(f"episode-cluster bootstrap (n={args.n_bootstrap} resamples of {len(unique_ep_ids)} episodes): "
          f"95% CI for partial rho = [{ci_lo:.3f}, {ci_hi:.3f}]")
    print()

    # ---------------------------------------------------------------
    # (4) Per-bucket episode counts (tail-buckets flagged)
    # ---------------------------------------------------------------
    print("=" * 70)
    print("(4) Per-bucket sample composition (episode count, not just step count)")
    print("=" * 70)
    max_len = run_len.max()
    buckets = list(range(0, int(max_len) + args.bucket_size, args.bucket_size))
    print(f"{'bucket':<16}{'n_steps':>9}{'n_episodes':>12}{'flag':>10}")
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        mask = (run_len >= lo) & (run_len < hi)
        n_steps = mask.sum()
        if n_steps == 0:
            continue
        n_eps_in_bucket = len(np.unique(entry_ep_id[mask]))
        flag = "LOW-N" if n_eps_in_bucket < 5 else ""
        print(f"[{lo:>4}, {hi:>4})   {n_steps:>9}{n_eps_in_bucket:>12}{flag:>10}")


if __name__ == "__main__":
    main()
