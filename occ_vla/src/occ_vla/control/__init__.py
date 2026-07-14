"""pi0.5 control backbone.

Thin adapter over openpi.policies.policy.Policy (see pi05_policy.py for
why). Hierarchical subtask planning and 50Hz action decoding both
happen inside the vendored pi0.5 checkpoint's forward pass — occ_vla
does not reimplement them, only supplies observations (optionally
augmented with a world-model subgoal image or PKLP-derived state) and
consumes the resulting action chunk.
"""

from occ_vla.control.pi05_policy import Pi05Observation, Pi05Policy

__all__ = ["Pi05Observation", "Pi05Policy"]
