"""
analyze_oft_experiment_logs.py

Turns the raw step-log JSONL files + episode results.json written by
run_oft_camera_dropout_eval.py into the actual A1/A2/A4/B1/B3 numbers from
the experiment plan (2026-08-17 discussion). Deliberately GPU/torch-free
(stdlib + optional numpy) -- pure log analysis, runs anywhere, unit-tested
without the openvla-oft environment (see scripts/tests/test_analyze_oft_experiment_logs.py).

Usage (after a real run on the GPU machine):
    python scripts/analyze_oft_experiment_logs.py \
        --results-path smoketest_results.json \
        --log-steps-dir steplogs_smoketest \
        --baseline-condition baseline \
        --report-path report.json

Terminology matches oft_step_logger.py's schema exactly -- see that
module for what each field means. Every function here takes plain lists
of dicts (already-parsed JSONL rows) or the parsed results.json dict, not
file paths, so they're trivially testable with synthetic fixtures.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

from oft_occlusion_gate import recompute_correction_applied_from_log  # noqa: E402
from oft_step_logger import read_jsonl  # noqa: E402


def _step_rows(rows):
    """Filter a parsed JSONL file down to record_type=='step' rows,
    dropping the episode_summary trailer."""
    return [r for r in rows if r.get("record_type") == "step"]


def _percentile(values, pct):
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(pct / 100 * (len(s) - 1))))
    return s[idx]


# ---------------------------------------------------------------------------
# A1: latency
# ---------------------------------------------------------------------------

def compute_latency_stats(rows):
    """A1. Restricted to rows where a real VLA call actually happened this
    step (t_vla_ms > 0 -- every other logged step reused a queued action
    from the prior call and has t_vla_ms/t_predictor_ms == 0.0 by
    construction, see run_oft_camera_dropout_eval.py's own comment on
    this). Returns None fields (not zeros/exceptions) when there's no
    data, so a condition where the predictor never engaged still reports
    cleanly rather than looking like an error."""
    steps = _step_rows(rows)
    vla_ms = [r["t_vla_ms"] for r in steps if r["t_vla_ms"] > 0]
    predictor_ms = [r["t_predictor_ms"] for r in steps if r["t_vla_ms"] > 0]

    def _stats(values):
        if not values:
            return {"n": 0, "mean_ms": None, "median_ms": None, "p95_ms": None}
        return {
            "n": len(values),
            "mean_ms": statistics.mean(values),
            "median_ms": statistics.median(values),
            "p95_ms": _percentile(values, 95),
        }

    vla_stats = _stats(vla_ms)
    result = {
        "vla_call": vla_stats,
        "predictor": _stats(predictor_ms),
        "hz": (1000.0 / vla_stats["mean_ms"]) if vla_stats["mean_ms"] else None,
    }
    if vla_stats["mean_ms"] and result["predictor"]["mean_ms"] is not None:
        result["predictor_frac_of_vla_call"] = result["predictor"]["mean_ms"] / vla_stats["mean_ms"]
    else:
        result["predictor_frac_of_vla_call"] = None
    return result


# ---------------------------------------------------------------------------
# A2: engagement rate
# ---------------------------------------------------------------------------

def compute_engagement_rate(rows):
    """A2. Per-step fraction with occ_flag / correction_applied True, over
    ALL logged steps (not just VLA-call steps -- occlusion state is
    per-env-step, independent of whether a fresh VLA call happened)."""
    steps = _step_rows(rows)
    n = len(steps)
    if n == 0:
        return {"n_steps": 0, "occ_flag_rate": None, "correction_applied_rate": None}
    return {
        "n_steps": n,
        "occ_flag_steps": sum(1 for r in steps if r["occ_flag"]),
        "occ_flag_rate": sum(1 for r in steps if r["occ_flag"]) / n,
        "correction_applied_steps": sum(1 for r in steps if r["correction_applied"]),
        "correction_applied_rate": sum(1 for r in steps if r["correction_applied"]) / n,
    }


# ---------------------------------------------------------------------------
# A4: occluded-step-only trajectory error vs. a baseline (paired) episode
# ---------------------------------------------------------------------------

def compute_occluded_ade(baseline_rows, condition_rows, gate_key="occ_flag"):
    """A4. Average Displacement Error (mean Euclidean distance between
    ee_position at matching step indices) between `condition_rows` and a
    PAIRED `baseline_rows` episode (same init_state -- same episode index,
    same task), restricted to steps where `condition_rows`'s `gate_key`
    (default occ_flag; pass "correction_applied" to score only steps the
    gate actually intervened on) is True.

    CAVEAT, stated plainly: this is an approximation, not a rigorous
    trajectory alignment. OpenVLA-OFT's own action sampling is NOT
    bit-identical run to run (this project's own repeated finding, see
    occ_vla's CLAUDE.md) -- "same step index" across two SEPARATE episode
    runs is not guaranteed to mean "same point in the task," especially
    later in a trajectory where stochastic divergence compounds. Treat
    this as a rough signal, not a precise metric, and prefer looking at
    EARLY-occlusion-onset steps (where divergence has had less time to
    accumulate) if the two trajectories' lengths differ substantially."""
    baseline_by_step = {r["step"]: r for r in _step_rows(baseline_rows)}
    dists = []
    for r in _step_rows(condition_rows):
        if not r.get(gate_key):
            continue
        b = baseline_by_step.get(r["step"])
        if b is None:
            continue
        dx = r["ee_position"][0] - b["ee_position"][0]
        dy = r["ee_position"][1] - b["ee_position"][1]
        dz = r["ee_position"][2] - b["ee_position"][2]
        dists.append(math.sqrt(dx * dx + dy * dy + dz * dz))
    if not dists:
        return {"n": 0, "mean_ade_m": None, "median_ade_m": None, "caveat": "no overlapping occluded/gate-key steps found between the two episodes"}
    return {
        "n": len(dists),
        "mean_ade_m": statistics.mean(dists),
        "median_ade_m": statistics.median(dists),
        "max_ade_m": max(dists),
    }


# ---------------------------------------------------------------------------
# B1: debounce-k sweep, recomputed from the logged S_occ sequence -- no rerun
# ---------------------------------------------------------------------------

def compute_k_sweep(rows, k_values=(1, 3, 5, 10), threshold=0.3, latch=True):
    """B1. The entire point of always logging S_occ raw: this recomputes
    correction_applied for every k in `k_values` from the SAME logged
    S_occ sequence, no rerun of the actual rollout needed. Also includes
    "always"/"never" (mode-based, not k-based) as the advisor plan's own
    named endpoints ("無条件"/"補正なし")."""
    s_occ_seq = [r["s_occ"] for r in _step_rows(rows)]
    n = len(s_occ_seq)
    out = {}
    for k in k_values:
        replayed = recompute_correction_applied_from_log(s_occ_seq, threshold=threshold, k=k, latch=latch)
        engaged = sum(1 for r in replayed if r.correction_applied)
        out[f"k={k}"] = {"engaged_steps": engaged, "engagement_rate": engaged / n if n else None}
    for mode in ("always", "never"):
        replayed = recompute_correction_applied_from_log(s_occ_seq, threshold=threshold, k=1, mode=mode, latch=latch)
        engaged = sum(1 for r in replayed if r.correction_applied)
        out[mode] = {"engaged_steps": engaged, "engagement_rate": engaged / n if n else None}
    return out


# ---------------------------------------------------------------------------
# B3 (and general): compare two conditions' success rate / step count
# ---------------------------------------------------------------------------

def compare_conditions(results_json, condition_a, condition_b):
    """B3 (and general-purpose): success rate + mean/median steps-to-success
    for two conditions from a parsed results.json, plus a naive paired
    win/loss/tie breakdown by episode index (same init_state across
    conditions, per this project's own established convention)."""
    a = results_json["results"].get(condition_a, [])
    b = results_json["results"].get(condition_b, [])
    n = min(len(a), len(b))

    def _summary(episodes):
        n_ep = len(episodes)
        n_success = sum(1 for e in episodes if e["success"])
        success_steps = [e["done_step"] for e in episodes if e["success"]]
        return {
            "n_episodes": n_ep,
            "success_rate": n_success / n_ep if n_ep else None,
            "n_success": n_success,
            "mean_steps_among_success": statistics.mean(success_steps) if success_steps else None,
            "median_steps_among_success": statistics.median(success_steps) if success_steps else None,
        }

    a_by_ep = {e["episode"]: e for e in a}
    b_by_ep = {e["episode"]: e for e in b}
    both_success = a_only_success = b_only_success = both_fail = 0
    for ep in sorted(set(a_by_ep) & set(b_by_ep)):
        sa, sb = a_by_ep[ep]["success"], b_by_ep[ep]["success"]
        if sa and sb:
            both_success += 1
        elif sa and not sb:
            a_only_success += 1
        elif sb and not sa:
            b_only_success += 1
        else:
            both_fail += 1

    return {
        condition_a: _summary(a),
        condition_b: _summary(b),
        "paired": {
            "n_pairs": n,
            "both_success": both_success,
            f"{condition_a}_only_success": a_only_success,
            f"{condition_b}_only_success": b_only_success,
            "both_fail": both_fail,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_condition_step_logs(log_steps_dir, condition):
    """All episodes' step rows for one condition, concatenated in episode
    order (used for A1/A2 aggregate stats across a whole condition, not
    just one episode)."""
    paths = sorted(glob.glob(os.path.join(log_steps_dir, f"{condition}_ep*.jsonl")))
    all_rows = []
    for path in paths:
        all_rows.extend(read_jsonl(path))
    return all_rows, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-path", required=True, help="results.json written by run_oft_camera_dropout_eval.py")
    parser.add_argument("--log-steps-dir", required=True, help="--log-steps-dir used for that same run")
    parser.add_argument("--baseline-condition", default="baseline", help="condition to use as the A4 reference trajectory")
    parser.add_argument("--gated-condition", default="wrist_partial_vjepa_gated")
    parser.add_argument("--no-correction-condition", default="wrist_partial")
    parser.add_argument("--prevframe-condition", default="wrist_partial_prevframe")
    parser.add_argument("--report-path", default=None, help="write the full report as JSON here (also always printed to stdout)")
    args = parser.parse_args()

    with open(args.results_path) as f:
        results_json = json.load(f)
    conditions = list(results_json["results"].keys())
    print(f"Conditions found in {args.results_path}: {conditions}")

    report = {"source_results_path": args.results_path, "source_log_steps_dir": args.log_steps_dir, "per_condition": {}}

    for condition in conditions:
        rows, paths = _load_condition_step_logs(args.log_steps_dir, condition)
        if not rows:
            print(f"  [skip] no step logs found for condition={condition} (looked for {condition}_ep*.jsonl under {args.log_steps_dir})")
            continue
        report["per_condition"][condition] = {
            "n_episode_logs": len(paths),
            "a1_latency": compute_latency_stats(rows),
            "a2_engagement": compute_engagement_rate(rows),
        }

    # A4: pair each non-baseline condition against the baseline, per matching episode index
    if args.baseline_condition in results_json["results"]:
        baseline_paths = sorted(glob.glob(os.path.join(args.log_steps_dir, f"{args.baseline_condition}_ep*.jsonl")))
        baseline_by_ep = {}
        for p in baseline_paths:
            ep_rows = read_jsonl(p)
            if ep_rows:
                baseline_by_ep[ep_rows[0]["episode"]] = ep_rows

        for condition in conditions:
            if condition == args.baseline_condition:
                continue
            cond_paths = sorted(glob.glob(os.path.join(args.log_steps_dir, f"{condition}_ep*.jsonl")))
            per_ep_ade = []
            for p in cond_paths:
                ep_rows = read_jsonl(p)
                if not ep_rows:
                    continue
                ep_idx = ep_rows[0]["episode"]
                if ep_idx not in baseline_by_ep:
                    continue
                per_ep_ade.append(compute_occluded_ade(baseline_by_ep[ep_idx], ep_rows))
            valid = [a["mean_ade_m"] for a in per_ep_ade if a["mean_ade_m"] is not None]
            if condition in report["per_condition"]:
                report["per_condition"][condition]["a4_occluded_ade_vs_baseline"] = {
                    "per_episode": per_ep_ade,
                    "mean_ade_m_across_episodes": statistics.mean(valid) if valid else None,
                }

    # B1: k-sweep, using the first episode's log per gated-style condition as a demonstration
    # (real B1 analysis should run this per-episode and aggregate; kept simple here since
    # the point is showing the mechanism works on real data, not a full aggregate yet).
    for condition in (args.gated_condition,):
        if condition not in results_json["results"]:
            continue
        rows, paths = _load_condition_step_logs(args.log_steps_dir, condition)
        if rows:
            report.setdefault("b1_k_sweep", {})[condition] = compute_k_sweep(rows)

    # B3: prevframe vs. learned-predictor comparison
    if args.prevframe_condition in results_json["results"] and args.gated_condition in results_json["results"]:
        report["b3_prevframe_vs_gated"] = compare_conditions(results_json, args.prevframe_condition, args.gated_condition)

    print(json.dumps(report, indent=2))
    if args.report_path:
        with open(args.report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report to {args.report_path}")


if __name__ == "__main__":
    main()
