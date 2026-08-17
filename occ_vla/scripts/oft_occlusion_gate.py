"""
oft_occlusion_gate.py

Debounced threshold gate that turns a continuous per-step occlusion severity
signal (S_occ, in [0, 1]) into a boolean "should the mid-layer vjepa
correction fire this step" decision.

Deliberately GPU/torch-free (stdlib only) so it can be unit-tested and
iterated on without the openvla-oft environment, and so the *exact same*
gating logic can be reused for the offline B1 ablation (log recompute, no
rerun) and the live eval loop (real-time gating) without drift between the
two -- see run_oft_camera_dropout_eval.py's `--debounce-k` wiring.

Terminology (matches the logging spec discussed with the user, 2026-08-17):
  - S_occ: raw estimated occlusion severity for the current step, any
    source (oracle ground truth, or later a trained detector/probe).
    Always logged raw -- this is what lets k be swept post-hoc.
  - occ_flag: S_occ > threshold, BEFORE debouncing (single-step trigger).
  - debounce_counter: consecutive steps occ_flag has been True.
  - correction_applied: the actual gated decision -- True once
    debounce_counter reaches k (and stays True until occ_flag drops,
    depending on `latch` -- see OcclusionGate docstring).

Two special k values are NOT "debounce" in the literal sense but are useful
sweep endpoints for B1 (this project's own advisor plan calls them out
explicitly: "補正なし / 無条件 / 1 / 3 / 5 / 10"):
  - k=0 with `mode="always"`: unconditional correction (ignore S_occ
    entirely) -- the "無条件" arm.
  - `mode="never"`: correction never applied -- the "補正なし" arm.
k=1 (mode="threshold") is a plain 1-step threshold, no debounce smoothing.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

GateMode = Literal["threshold", "always", "never"]


@dataclasses.dataclass
class GateStepResult:
    s_occ: float
    occ_flag: bool
    debounce_counter: int
    correction_applied: bool


class OcclusionGate:
    """Stateful, one-step-at-a-time debounced threshold gate.

    Call `.step(s_occ)` once per control step, in order, within a single
    episode. Call `.reset()` at the start of each new episode (mirrors
    `model.reset_vjepa_state()`'s per-episode reset in run_oft_camera_dropout_eval.py
    -- debounce state must not leak across episodes any more than the
    predictor's own temporal state should).

    Args:
        threshold: S_occ strictly above this counts as a single-step
            "occ_flag" trigger. Ignored when mode != "threshold".
        k: number of CONSECUTIVE occ_flag=True steps required before
            correction_applied flips True. k=1 is a plain threshold with
            no smoothing. Ignored when mode != "threshold".
        mode: "threshold" (normal debounced-threshold gating), "always"
            (unconditional correction -- the "無条件" arm), or "never"
            (correction never applied -- the "補正なし" arm). In both
            "always" and "never", S_occ/occ_flag are still computed and
            logged (so the same step log covers all arms uniformly), only
            correction_applied is forced.
        latch: if True (default), once correction_applied fires it stays
            True for the rest of the episode (matches "occlusion has begun,
            keep correcting until the episode ends" semantics used
            elsewhere in this project, e.g. the pi0.5-track soft gate).
            If False, correction_applied drops back to False as soon as
            debounce_counter falls below k again (occ_flag went False) --
            useful for measuring how often the gate would flap on/off.
    """

    def __init__(
        self,
        threshold: float = 0.3,
        k: int = 3,
        mode: GateMode = "threshold",
        latch: bool = True,
    ) -> None:
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold
        self.k = k
        self.mode = mode
        self.latch = latch
        self._debounce_counter = 0
        self._latched = False

    def reset(self) -> None:
        """Clear all per-episode state. Call at the start of every episode."""
        self._debounce_counter = 0
        self._latched = False

    def step(self, s_occ: float) -> GateStepResult:
        occ_flag = s_occ > self.threshold

        if occ_flag:
            self._debounce_counter += 1
        else:
            self._debounce_counter = 0

        threshold_fires = self.k == 0 or self._debounce_counter >= max(self.k, 1)
        # k=0 under mode="threshold" degenerates to "fire on the very first
        # occ_flag=True step" (debounce_counter >= 1) rather than being a
        # divide-by-zero/always-on special case -- k=0 is not one of the
        # advisor's named sweep points (0 is expressed via mode="always"
        # instead), but this keeps the arithmetic well-defined if someone
        # passes it anyway.

        if self.mode == "always":
            correction_applied = True
        elif self.mode == "never":
            correction_applied = False
        else:
            if self.latch:
                self._latched = self._latched or threshold_fires
                correction_applied = self._latched
            else:
                correction_applied = threshold_fires

        return GateStepResult(
            s_occ=s_occ,
            occ_flag=occ_flag,
            debounce_counter=self._debounce_counter,
            correction_applied=correction_applied,
        )


def recompute_correction_applied_from_log(
    s_occ_sequence: list[float],
    threshold: float = 0.3,
    k: int = 3,
    mode: GateMode = "threshold",
    latch: bool = True,
) -> list[GateStepResult]:
    """Re-run the gate over an already-logged S_occ sequence (one episode's
    worth, in order) with a NEW (threshold, k, mode) -- this is the whole
    point of always logging S_occ raw: B1's k-sweep is this function called
    once per k value, no rerun of the actual rollout needed."""
    gate = OcclusionGate(threshold=threshold, k=k, mode=mode, latch=latch)
    return [gate.step(s) for s in s_occ_sequence]
