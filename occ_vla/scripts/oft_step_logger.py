"""
oft_step_logger.py

Per-step + per-episode logging for the OFT occlusion-recovery eval harness,
matching the schema agreed with the user (2026-08-17):

  step, episode, task_id, seed          -- keys
  s_occ (raw)                           -- for post-hoc threshold/k sweeps
  occ_flag, debounce_counter,
    correction_applied                  -- from OcclusionGate (oft_occlusion_gate.py)
  occ_gt                                -- ground-truth occlusion label/fraction
  ee_position, action                   -- for ADE/FDE/DTW/path length
  t_vla_ms, t_predictor_ms, t_total_ms  -- latency (A1)
  success, steps_to_success             -- SR (episode-level, written once)

Deliberately GPU/torch-free (stdlib + optional numpy for array coercion) so
it can be unit-tested without the openvla-oft environment. Writes JSON Lines
(one JSON object per step) rather than a single JSON blob per episode --
appendable, crash-safe (a killed episode still leaves every step logged so
far on disk), and trivially concatenable across episodes/conditions/tasks
for pandas-style post-hoc analysis (`pd.read_json(path, lines=True)`).

One StepLogWriter instance == one episode. Episode-level fields (success,
steps_to_success) are written as a final trailer record with
`record_type="episode_summary"` so a single file mixes step + summary rows
without needing a second file -- filter on `record_type` when loading.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any


def _to_plain(value: Any) -> Any:
    """Coerce numpy scalars/arrays (and torch tensors, if present) to plain
    JSON-serializable Python types. Avoids importing numpy/torch at module
    level -- only touches them if the value actually looks like one."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    # numpy scalar/array or torch tensor -- both expose .tolist()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (ValueError, TypeError):
            pass
    raise TypeError(f"Don't know how to make {type(value)} JSON-serializable")


@dataclasses.dataclass
class StepLogRecord:
    step: int
    episode: int
    task_id: int
    seed: int
    s_occ: float
    occ_flag: bool
    debounce_counter: int
    correction_applied: bool
    occ_gt: float  # fraction in [0, 1]; 0.0/1.0 for a hard boolean label too
    ee_position: list[float]  # [x, y, z]
    action: list[float]
    t_vla_ms: float
    t_predictor_ms: float
    t_total_ms: float
    record_type: str = "step"

    def to_json_dict(self) -> dict[str, Any]:
        return {f.name: _to_plain(getattr(self, f.name)) for f in dataclasses.fields(self)}


class StepLogWriter:
    """JSONL writer for one episode's step log, appending one line per
    `log_step()` call within its own lifetime (open -> log_step* -> close).

    Usage:
        writer = StepLogWriter(path, episode=ep, task_id=task_id, seed=seed)
        for t in range(max_steps):
            ...
            writer.log_step(step=t, s_occ=..., occ_flag=..., ...)
        writer.close(success=success, steps_to_success=done_step)

    `mode="w"` (default): truncates `path` on open. Each fresh
    StepLogWriter instance represents ONE run's data for that (condition,
    episode) -- re-running the exact same script invocation against the
    same `--log-steps-dir` (a very normal thing to do while iterating,
    e.g. retrying after a crash) must NOT silently concatenate onto
    whatever a PRIOR, unrelated attempt left behind at that same path.
    Confirmed as a real bug, not a hypothetical: `mode="a"` used to be the
    default, and a real Kaggle debugging session's finally-successful run
    appended its data after a stale `episode_summary` (success=False) row
    from an earlier failed attempt at the same path, silently corrupting
    the log's first row. Pass `mode="a"` explicitly if you genuinely want
    to accumulate multiple runs into one file (e.g. a controlled multi-
    invocation batch script that manages this deliberately) -- that is
    NOT the safe default for casual reruns.
    """

    def __init__(self, path: str, episode: int, task_id: int, seed: int, mode: str = "w"):
        self.path = path
        self.episode = episode
        self.task_id = task_id
        self.seed = seed
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, mode, encoding="utf-8")
        self._closed = False

    def log_step(
        self,
        step: int,
        s_occ: float,
        occ_flag: bool,
        debounce_counter: int,
        correction_applied: bool,
        occ_gt: float,
        ee_position,
        action,
        t_vla_ms: float,
        t_predictor_ms: float,
        t_total_ms: float,
    ) -> None:
        record = StepLogRecord(
            step=step,
            episode=self.episode,
            task_id=self.task_id,
            seed=self.seed,
            s_occ=float(s_occ),
            occ_flag=bool(occ_flag),
            debounce_counter=int(debounce_counter),
            correction_applied=bool(correction_applied),
            occ_gt=float(occ_gt),
            ee_position=_to_plain(ee_position),
            action=_to_plain(action),
            t_vla_ms=float(t_vla_ms),
            t_predictor_ms=float(t_predictor_ms),
            t_total_ms=float(t_total_ms),
        )
        self._fh.write(json.dumps(record.to_json_dict()) + "\n")

    def close(self, success: bool, steps_to_success: int | None) -> None:
        summary = {
            "record_type": "episode_summary",
            "episode": self.episode,
            "task_id": self.task_id,
            "seed": self.seed,
            "success": bool(success),
            "steps_to_success": steps_to_success,
        }
        self._fh.write(json.dumps(summary) + "\n")
        self._fh.close()
        self._closed = True

    def __enter__(self) -> "StepLogWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._closed:
            # Episode errored out before a normal close() -- still flush
            # what's on disk (per-step rows already written are kept; a
            # crashed episode's steps are NOT silently lost, matching the
            # "crash-safe" design goal above), but don't fabricate a
            # success/steps_to_success we don't have.
            self._fh.close()


def read_jsonl(path: str) -> list[dict[str, Any]]:
    """Minimal reader for tests / quick inspection -- real analysis should
    use `pandas.read_json(path, lines=True)` instead."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
