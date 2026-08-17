import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oft_occlusion_gate import OcclusionGate, recompute_correction_applied_from_log


def test_below_threshold_never_fires():
    gate = OcclusionGate(threshold=0.3, k=3, mode="threshold")
    for _ in range(20):
        r = gate.step(0.1)
        assert not r.occ_flag
        assert not r.correction_applied
        assert r.debounce_counter == 0


def test_debounce_requires_k_consecutive_steps():
    gate = OcclusionGate(threshold=0.3, k=3, mode="threshold", latch=False)
    results = [gate.step(s) for s in [0.5, 0.5, 0.5, 0.5]]
    # steps 0,1,2 are occ_flag=True but debounce_counter only reaches 3 on step 2 (1-indexed count)
    assert [r.debounce_counter for r in results] == [1, 2, 3, 4]
    assert [r.correction_applied for r in results] == [False, False, True, True]


def test_single_dip_resets_debounce_counter_without_latch():
    gate = OcclusionGate(threshold=0.3, k=3, mode="threshold", latch=False)
    seq = [0.5, 0.5, 0.1, 0.5, 0.5, 0.5]  # dip at index 2 restarts the run
    results = [gate.step(s) for s in seq]
    assert [r.debounce_counter for r in results] == [1, 2, 0, 1, 2, 3]
    assert [r.correction_applied for r in results] == [False, False, False, False, False, True]


def test_latch_stays_true_after_a_single_dip():
    gate = OcclusionGate(threshold=0.3, k=2, mode="threshold", latch=True)
    seq = [0.5, 0.5, 0.1, 0.1, 0.1]  # fires at index 1, then occlusion clears
    results = [gate.step(s) for s in seq]
    assert [r.correction_applied for r in results] == [False, True, True, True, True]
    # occ_flag/debounce_counter still reflect the RAW instantaneous signal,
    # independent of the latch -- latching only affects correction_applied
    assert [r.occ_flag for r in results] == [True, True, False, False, False]
    assert [r.debounce_counter for r in results] == [1, 2, 0, 0, 0]


def test_k1_is_plain_unsmoothed_threshold():
    gate = OcclusionGate(threshold=0.3, k=1, mode="threshold", latch=False)
    seq = [0.1, 0.5, 0.1, 0.5]
    results = [gate.step(s) for s in seq]
    assert [r.correction_applied for r in results] == [False, True, False, True]


def test_mode_always_ignores_s_occ():
    gate = OcclusionGate(threshold=0.3, k=3, mode="always")
    for s in [0.0, 0.01, 0.99]:
        r = gate.step(s)
        assert r.correction_applied is True
        # occ_flag/debounce_counter are still computed honestly even though
        # they don't drive correction_applied in this mode -- needed so the
        # step log is directly comparable across arms
    assert gate.step(0.0).occ_flag is False


def test_mode_never_ignores_s_occ():
    gate = OcclusionGate(threshold=0.3, k=1, mode="never")
    for s in [0.0, 0.99, 1.0]:
        assert gate.step(s).correction_applied is False


def test_reset_clears_state_between_episodes():
    gate = OcclusionGate(threshold=0.3, k=2, mode="threshold", latch=True)
    gate.step(0.9)
    r = gate.step(0.9)
    assert r.correction_applied is True
    gate.reset()
    r2 = gate.step(0.1)
    assert r2.correction_applied is False
    assert r2.debounce_counter == 0


def test_recompute_from_log_matches_live_stepping():
    s_occ_sequence = [0.1, 0.5, 0.5, 0.5, 0.1, 0.9, 0.9, 0.9, 0.9]
    live_gate = OcclusionGate(threshold=0.3, k=3, mode="threshold", latch=True)
    live_results = [live_gate.step(s) for s in s_occ_sequence]
    replayed_results = recompute_correction_applied_from_log(s_occ_sequence, threshold=0.3, k=3, latch=True)
    assert [r.correction_applied for r in live_results] == [r.correction_applied for r in replayed_results]
    assert [r.debounce_counter for r in live_results] == [r.debounce_counter for r in replayed_results]


def test_b1_sweep_produces_different_engagement_rates_for_different_k():
    # Directly exercises B1's premise: same S_occ log, different k, no rerun.
    s_occ_sequence = [0.5] * 2 + [0.1] * 2 + [0.5] * 6  # two separate occluded runs, lengths 2 and 6
    engagement_by_k = {}
    for k in [1, 3, 5, 10]:
        results = recompute_correction_applied_from_log(s_occ_sequence, threshold=0.3, k=k, latch=False)
        engagement_by_k[k] = sum(r.correction_applied for r in results)
    # monotonically non-increasing engagement as k grows (stricter debounce
    # can only delay/shrink firing, never bring it forward)
    assert engagement_by_k[1] >= engagement_by_k[3] >= engagement_by_k[5] >= engagement_by_k[10]
    assert engagement_by_k[1] > 0
    assert engagement_by_k[10] == 0  # neither run in this sequence is 10 steps long


def test_invalid_threshold_and_k_raise():
    import pytest

    with pytest.raises(ValueError):
        OcclusionGate(threshold=1.5)
    with pytest.raises(ValueError):
        OcclusionGate(k=-1)
