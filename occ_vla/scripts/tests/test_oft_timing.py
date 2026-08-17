import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from oft_timing import CallTimer, StepTimer, unwrap_forward, wrap_forward_with_timer


class _SleepyModule(nn.Module):
    """Stand-in for vjepa_predictor_dino/_siglip: a real nn.Module (so
    `.forward` monkeypatching behaves exactly as it would on the real
    predictor) with a controllable, measurable amount of work per call."""

    def __init__(self, sleep_s: float = 0.02):
        super().__init__()
        self.sleep_s = sleep_s
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        time.sleep(self.sleep_s)
        return self.linear(x)


def test_call_timer_accumulates_and_resets():
    timer = CallTimer()
    timer.record(5.0)
    timer.record(3.0)
    assert timer.total_ms == 8.0
    assert timer.n_calls == 2
    timer.reset()
    assert timer.total_ms == 0.0
    assert timer.n_calls == 0


def test_wrap_forward_measures_real_elapsed_time_and_preserves_output():
    module = _SleepyModule(sleep_s=0.02)
    timer = CallTimer()
    original = wrap_forward_with_timer(module, timer, sync_cuda=False)

    x = torch.randn(1, 4)
    out = module(x)

    assert torch.allclose(out, module.linear(x))  # output unaffected by wrapping
    assert timer.n_calls == 1
    # real sleep was ~20ms; allow generous slack for CI/scheduler jitter,
    # but this must be far above zero (an unwrapped/no-op timer would read 0)
    assert timer.total_ms > 10.0

    unwrap_forward(module, original)
    assert module.forward is original


def test_wrap_forward_accumulates_across_multiple_calls_before_reset():
    module = _SleepyModule(sleep_s=0.01)
    timer = CallTimer()
    wrap_forward_with_timer(module, timer, sync_cuda=False)

    x = torch.randn(1, 4)
    module(x)  # e.g. DINO featurizer's predictor call
    module(x)  # e.g. SigLIP featurizer's predictor call, same step

    assert timer.n_calls == 2
    assert timer.total_ms > 15.0  # two ~10ms calls, well above one call's worth


def test_unwrap_restores_original_forward_behavior():
    module = _SleepyModule(sleep_s=0.0)
    timer = CallTimer()
    original = wrap_forward_with_timer(module, timer, sync_cuda=False)
    unwrap_forward(module, original)

    x = torch.randn(1, 4)
    module(x)
    assert timer.n_calls == 0  # no longer being timed after unwrap


def test_step_timer_measures_positive_elapsed_ms():
    st = StepTimer()
    st.start()
    time.sleep(0.01)
    elapsed = st.stop_ms()
    assert elapsed > 5.0


def test_step_timer_raises_if_stopped_before_started():
    st = StepTimer()
    try:
        st.stop_ms()
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
