import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from oft_step_logger import StepLogWriter, read_jsonl


def test_writes_one_json_line_per_step_and_a_trailer(tmp_path):
    path = str(tmp_path / "log.jsonl")
    writer = StepLogWriter(path, episode=0, task_id=8, seed=42)
    for t in range(3):
        writer.log_step(
            step=t,
            s_occ=0.1 * t,
            occ_flag=t > 1,
            debounce_counter=max(0, t - 1),
            correction_applied=t > 1,
            occ_gt=0.2 * t,
            ee_position=[float(t), 0.0, 1.0],
            action=[0.0] * 7,
            t_vla_ms=12.3,
            t_predictor_ms=1.5,
            t_total_ms=13.8,
        )
    writer.close(success=True, steps_to_success=3)

    records = read_jsonl(path)
    assert len(records) == 4  # 3 steps + 1 episode_summary trailer
    steps = [r for r in records if r["record_type"] == "step"]
    summary = [r for r in records if r["record_type"] == "episode_summary"]
    assert len(steps) == 3
    assert len(summary) == 1
    assert summary[0] == {
        "record_type": "episode_summary",
        "episode": 0,
        "task_id": 8,
        "seed": 42,
        "success": True,
        "steps_to_success": 3,
    }
    # every requested logging-spec column is present on step rows
    expected_keys = {
        "step", "episode", "task_id", "seed", "s_occ", "occ_flag",
        "debounce_counter", "correction_applied", "occ_gt", "ee_position",
        "action", "t_vla_ms", "t_predictor_ms", "t_total_ms", "record_type",
    }
    assert set(steps[0].keys()) == expected_keys
    assert steps[1]["step"] == 1
    assert steps[1]["ee_position"] == [1.0, 0.0, 1.0]


def test_numpy_values_are_coerced_to_plain_json_types(tmp_path):
    path = str(tmp_path / "log.jsonl")
    writer = StepLogWriter(path, episode=1, task_id=3, seed=7)
    writer.log_step(
        step=0,
        s_occ=np.float32(0.42),
        occ_flag=np.bool_(True),
        debounce_counter=np.int64(2),
        correction_applied=np.bool_(False),
        occ_gt=np.float64(0.5),
        ee_position=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        action=np.zeros(7, dtype=np.float32),
        t_vla_ms=np.float32(10.0),
        t_predictor_ms=np.float32(1.0),
        t_total_ms=np.float32(11.0),
    )
    writer.close(success=False, steps_to_success=None)

    records = read_jsonl(path)  # would raise on non-JSON-serializable content if coercion failed
    step = records[0]
    assert step["ee_position"] == [1.0, 2.0, 3.0]
    assert isinstance(step["s_occ"], float)
    assert step["occ_flag"] is True
    assert records[1]["steps_to_success"] is None


def test_default_mode_does_not_leak_stale_content_from_a_prior_run(tmp_path):
    # Regression test for a real bug found on Kaggle infra (2026-08-18):
    # mode="a" used to be the default, so a script rerun (e.g. retrying
    # after an earlier crash) silently appended onto whatever a PRIOR,
    # unrelated attempt had already written at the same path -- a real
    # run's log ended up with a stale episode_summary(success=False) row
    # from an old failed attempt sitting before its own real data.
    path = str(tmp_path / "log.jsonl")
    stale_writer = StepLogWriter(path, episode=0, task_id=0, seed=0)
    stale_writer.log_step(
        step=0, s_occ=0.9, occ_flag=True, debounce_counter=1, correction_applied=False,
        occ_gt=0.9, ee_position=[9, 9, 9], action=[9] * 7, t_vla_ms=1.0, t_predictor_ms=0.0, t_total_ms=1.0,
    )
    stale_writer.close(success=False, steps_to_success=None)  # simulates an earlier failed attempt

    fresh_writer = StepLogWriter(path, episode=0, task_id=0, seed=0)  # default mode -- must NOT see the stale content
    fresh_writer.log_step(
        step=0, s_occ=0.0, occ_flag=False, debounce_counter=0, correction_applied=False,
        occ_gt=0.0, ee_position=[0, 0, 0], action=[0] * 7, t_vla_ms=1.0, t_predictor_ms=0.0, t_total_ms=1.0,
    )
    fresh_writer.close(success=True, steps_to_success=0)

    records = read_jsonl(path)
    assert len(records) == 2  # exactly this run's step + trailer, no stale leftovers
    assert records[0]["s_occ"] == 0.0  # NOT the stale 0.9
    assert records[1]["success"] is True  # NOT the stale False


def test_append_mode_accumulates_across_multiple_writer_instances_when_explicitly_requested(tmp_path):
    path = str(tmp_path / "log.jsonl")
    for ep in range(2):
        writer = StepLogWriter(path, episode=ep, task_id=0, seed=0, mode="a")
        writer.log_step(
            step=0, s_occ=0.0, occ_flag=False, debounce_counter=0, correction_applied=False,
            occ_gt=0.0, ee_position=[0, 0, 0], action=[0] * 7, t_vla_ms=1.0, t_predictor_ms=0.0, t_total_ms=1.0,
        )
        writer.close(success=True, steps_to_success=0)

    records = read_jsonl(path)
    episodes_seen = sorted({r["episode"] for r in records})
    assert episodes_seen == [0, 1]


def test_creates_missing_parent_directory(tmp_path):
    nested_path = str(tmp_path / "a" / "b" / "log.jsonl")
    writer = StepLogWriter(nested_path, episode=0, task_id=0, seed=0)
    writer.close(success=True, steps_to_success=0)
    assert os.path.exists(nested_path)


def test_context_manager_flushes_partial_log_on_exception(tmp_path):
    path = str(tmp_path / "log.jsonl")
    with pytest.raises(RuntimeError):
        with StepLogWriter(path, episode=0, task_id=0, seed=0) as writer:
            writer.log_step(
                step=0, s_occ=0.9, occ_flag=True, debounce_counter=1, correction_applied=False,
                occ_gt=0.9, ee_position=[0, 0, 0], action=[0] * 7, t_vla_ms=1.0, t_predictor_ms=0.0, t_total_ms=1.0,
            )
            raise RuntimeError("simulated episode crash")

    records = read_jsonl(path)
    # the step logged before the crash must survive -- crash-safety is the
    # whole point of JSONL-append over a single end-of-episode JSON dump
    assert len(records) == 1
    assert records[0]["step"] == 0
    assert not any(r.get("record_type") == "episode_summary" for r in records)
