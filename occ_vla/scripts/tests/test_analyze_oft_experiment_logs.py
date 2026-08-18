import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze_oft_experiment_logs import (
    compare_conditions,
    compute_engagement_rate,
    compute_k_sweep,
    compute_latency_stats,
    compute_occluded_ade,
)


def _step(step, s_occ=0.0, occ_flag=False, correction_applied=False, ee_position=(0.0, 0.0, 0.0),
          t_vla_ms=0.0, t_predictor_ms=0.0):
    return {
        "record_type": "step", "step": step, "episode": 0, "task_id": 8, "seed": 7,
        "s_occ": s_occ, "occ_flag": occ_flag, "debounce_counter": 0, "correction_applied": correction_applied,
        "occ_gt": s_occ, "ee_position": list(ee_position), "action": [0.0] * 7,
        "t_vla_ms": t_vla_ms, "t_predictor_ms": t_predictor_ms, "t_total_ms": t_vla_ms,
    }


def _summary(success=True, steps_to_success=100):
    return {"record_type": "episode_summary", "episode": 0, "task_id": 8, "seed": 7, "success": success, "steps_to_success": steps_to_success}


# ---------------------------------------------------------------------------
# A1
# ---------------------------------------------------------------------------

def test_latency_stats_only_counts_real_vla_calls():
    rows = [
        _step(0, t_vla_ms=100.0, t_predictor_ms=10.0),
        _step(1, t_vla_ms=0.0, t_predictor_ms=0.0),  # reused action, not a real call
        _step(2, t_vla_ms=0.0, t_predictor_ms=0.0),
        _step(3, t_vla_ms=200.0, t_predictor_ms=20.0),
        _summary(),
    ]
    stats = compute_latency_stats(rows)
    assert stats["vla_call"]["n"] == 2
    assert stats["vla_call"]["mean_ms"] == 150.0
    assert stats["predictor"]["n"] == 2
    assert stats["predictor"]["mean_ms"] == 15.0
    assert abs(stats["hz"] - 1000.0 / 150.0) < 1e-9
    assert abs(stats["predictor_frac_of_vla_call"] - 15.0 / 150.0) < 1e-9


def test_latency_stats_empty_when_no_real_calls():
    rows = [_step(0), _step(1), _summary()]
    stats = compute_latency_stats(rows)
    assert stats["vla_call"]["n"] == 0
    assert stats["vla_call"]["mean_ms"] is None
    assert stats["hz"] is None


# ---------------------------------------------------------------------------
# A2
# ---------------------------------------------------------------------------

def test_engagement_rate_counts_correctly():
    rows = [
        _step(0, occ_flag=False, correction_applied=False),
        _step(1, occ_flag=True, correction_applied=False),
        _step(2, occ_flag=True, correction_applied=True),
        _step(3, occ_flag=True, correction_applied=True),
        _summary(),
    ]
    stats = compute_engagement_rate(rows)
    assert stats["n_steps"] == 4
    assert stats["occ_flag_steps"] == 3
    assert stats["occ_flag_rate"] == 0.75
    assert stats["correction_applied_steps"] == 2
    assert stats["correction_applied_rate"] == 0.5


def test_engagement_rate_handles_empty_log():
    stats = compute_engagement_rate([_summary(success=False, steps_to_success=None)])
    assert stats["n_steps"] == 0
    assert stats["occ_flag_rate"] is None


# ---------------------------------------------------------------------------
# A4
# ---------------------------------------------------------------------------

def test_occluded_ade_only_scores_gated_steps_and_matching_indices():
    baseline = [
        _step(0, ee_position=(0.0, 0.0, 0.0)),
        _step(1, ee_position=(1.0, 0.0, 0.0)),
        _step(2, ee_position=(2.0, 0.0, 0.0)),
        _summary(),
    ]
    condition = [
        _step(0, occ_flag=False, ee_position=(0.0, 0.0, 0.0)),   # not occluded -> excluded
        _step(1, occ_flag=True, ee_position=(1.0, 3.0, 4.0)),    # occluded, distance = 5.0 (3-4-5 triangle)
        _step(2, occ_flag=True, ee_position=(2.0, 0.0, 1.0)),    # occluded, distance = 1.0
        _summary(),
    ]
    result = compute_occluded_ade(baseline, condition)
    assert result["n"] == 2
    assert abs(result["mean_ade_m"] - 3.0) < 1e-9  # mean(5.0, 1.0)
    assert result["max_ade_m"] == 5.0


def test_occluded_ade_uses_correction_applied_when_requested():
    baseline = [_step(0, ee_position=(0, 0, 0)), _summary()]
    condition = [_step(0, occ_flag=True, correction_applied=False, ee_position=(10, 0, 0)), _summary()]
    # occ_flag=True but correction_applied=False -- should be excluded when gate_key="correction_applied"
    result = compute_occluded_ade(baseline, condition, gate_key="correction_applied")
    assert result["n"] == 0
    assert result["mean_ade_m"] is None


def test_occluded_ade_no_overlap_returns_none_not_a_crash():
    baseline = [_step(5, ee_position=(0, 0, 0)), _summary()]  # only step 5 exists
    condition = [_step(0, occ_flag=True, ee_position=(1, 1, 1)), _summary()]  # only step 0, no match
    result = compute_occluded_ade(baseline, condition)
    assert result["n"] == 0
    assert result["mean_ade_m"] is None


# ---------------------------------------------------------------------------
# B1
# ---------------------------------------------------------------------------

def test_k_sweep_matches_direct_gate_recomputation():
    # Same pattern as oft_occlusion_gate.py's own B1 test: a short occluded
    # run (2 steps, never reaches k=3) then a long one (6 steps, does).
    rows = [_step(i, s_occ=(0.5 if i in (0, 1, 4, 5, 6, 7, 8, 9) else 0.1)) for i in range(10)]
    sweep = compute_k_sweep(rows, k_values=(1, 3, 5, 10), latch=False)
    assert sweep["k=1"]["engaged_steps"] >= sweep["k=3"]["engaged_steps"] >= sweep["k=5"]["engaged_steps"] >= sweep["k=10"]["engaged_steps"]
    assert sweep["always"]["engaged_steps"] == 10  # every one of the 10 rows, regardless of s_occ
    assert sweep["never"]["engaged_steps"] == 0
    assert sweep["k=1"]["engagement_rate"] == sweep["k=1"]["engaged_steps"] / 10


# ---------------------------------------------------------------------------
# B3
# ---------------------------------------------------------------------------

def test_compare_conditions_success_rate_and_pairing():
    results_json = {
        "results": {
            "prevframe": [
                {"episode": 0, "success": True, "done_step": 100, "n_calls": 12, "wall_s": 10.0},
                {"episode": 1, "success": False, "done_step": 300, "n_calls": 30, "wall_s": 20.0},
                {"episode": 2, "success": True, "done_step": 150, "n_calls": 15, "wall_s": 12.0},
            ],
            "gated": [
                {"episode": 0, "success": True, "done_step": 120, "n_calls": 14, "wall_s": 11.0},
                {"episode": 1, "success": True, "done_step": 310, "n_calls": 31, "wall_s": 21.0},  # recovered
                {"episode": 2, "success": False, "done_step": 300, "n_calls": 30, "wall_s": 20.0},  # regressed
            ],
        }
    }
    cmp = compare_conditions(results_json, "prevframe", "gated")
    assert cmp["prevframe"]["success_rate"] == 2 / 3
    assert cmp["gated"]["success_rate"] == 2 / 3
    assert cmp["prevframe"]["mean_steps_among_success"] == 125.0  # mean(100, 150)
    assert cmp["paired"]["n_pairs"] == 3
    assert cmp["paired"]["both_success"] == 1  # episode 0
    assert cmp["paired"]["gated_only_success"] == 1  # episode 1
    assert cmp["paired"]["prevframe_only_success"] == 1  # episode 2
    assert cmp["paired"]["both_fail"] == 0


def test_compare_conditions_handles_missing_condition_gracefully():
    results_json = {"results": {"a": [{"episode": 0, "success": True, "done_step": 10, "n_calls": 1, "wall_s": 1.0}]}}
    cmp = compare_conditions(results_json, "a", "does_not_exist")
    assert cmp["does_not_exist"]["n_episodes"] == 0
    assert cmp["does_not_exist"]["success_rate"] is None
    assert cmp["paired"]["n_pairs"] == 0
