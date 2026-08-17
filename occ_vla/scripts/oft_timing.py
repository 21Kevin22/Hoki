"""
oft_timing.py

Millisecond-precision timing for A1 (latency): splits one VLA call into
`t_vla_ms` (whole call, unmodified) and `t_predictor_ms` (time spent inside
`vjepa_predictor_dino`/`_siglip`'s forward, a strict subset of t_vla_ms) so
the three baseline / correction-off / correction-on conditions in A1's plan
can be told apart from a single instrumented run instead of three separate
profiling passes.

Non-invasive by design: wraps a module's `.forward` in place (same
monkeypatch pattern `run_oft_camera_dropout_eval.py`'s
`wrist_partial_midlayer_oracle` condition already uses for
`model.vision_backbone.forward` -- see that file) rather than editing the
vendored `modeling_prismatic.py`/`vjepa_latent_predictor.py`. `unwrap()`
restores the original, so this can be toggled on only when timing is
actually wanted (it does add a small constant overhead -- a CUDA sync per
wrapped call when `sync_cuda=True` -- so leave it off for throughput-
sensitive rollouts that don't need per-step latency numbers).

CUDA-synchronization note: without `torch.cuda.synchronize()`, elapsed time
around an async CUDA call measures kernel-launch time, not actual compute
time -- misleadingly near-zero. `sync_cuda=True` (default when a CUDA
device is available) makes the number honest at the cost of serializing
the GPU pipeline around every wrapped call; that's the correct tradeoff for
an A1 latency *measurement* run, not something you'd want in a normal
production/other-ablation rollout.
"""

from __future__ import annotations

import time
from typing import Callable

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - only exercised outside the openvla-oft env
    _HAS_TORCH = False


class CallTimer:
    """Accumulates elapsed time across possibly-multiple calls between
    resets (e.g. the mid-layer predictor is invoked once per camera per
    step -- DINO+SigLIP featurizers each call it -- and A1 wants one
    per-step total, not per-call)."""

    def __init__(self) -> None:
        self.total_ms = 0.0
        self.n_calls = 0

    def reset(self) -> None:
        self.total_ms = 0.0
        self.n_calls = 0

    def record(self, elapsed_ms: float) -> None:
        self.total_ms += elapsed_ms
        self.n_calls += 1


def wrap_forward_with_timer(module, timer: CallTimer, sync_cuda: bool | None = None) -> Callable:
    """Monkeypatch `module.forward` to record elapsed ms into `timer` on
    every call, then delegate to the real forward. Returns the original
    (bound) forward so the caller can restore it later:

        original = wrap_forward_with_timer(model.vision_backbone.vjepa_predictor_dino, timer)
        ...
        model.vision_backbone.vjepa_predictor_dino.forward = original  # unwrap
    """
    original_forward = module.forward
    if sync_cuda is None:
        sync_cuda = _HAS_TORCH and torch.cuda.is_available()

    def timed_forward(*args, **kwargs):
        if sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = original_forward(*args, **kwargs)
        if sync_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        timer.record((t1 - t0) * 1000.0)
        return out

    module.forward = timed_forward
    return original_forward


def unwrap_forward(module, original_forward: Callable) -> None:
    module.forward = original_forward


class StepTimer:
    """Simple wall-clock stopwatch for the OUTER `t_vla_ms`/`t_total_ms`
    measurements (no CUDA sync needed here -- the outer call already blocks
    on the GPU work finishing via its own return value, e.g. `.cpu()`/
    `.numpy()` inside `get_action`, so an explicit sync would be redundant,
    not wrong -- kept separate from CallTimer to avoid conflating "time
    spent inside one specific submodule" with "wall time around a whole
    call")."""

    def __init__(self) -> None:
        self._t0: float | None = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop_ms(self) -> float:
        if self._t0 is None:
            raise RuntimeError("StepTimer.stop_ms() called before start()")
        elapsed = (time.perf_counter() - self._t0) * 1000.0
        self._t0 = None
        return elapsed
