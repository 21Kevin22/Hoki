"""Conditional Flow Matching latent-space extrapolation for PKLP: when
an object goes behind an occluder, predict its future latent state by
integrating an ODE from noise to the target distribution in a handful
of Euler steps — avoiding a full diffusion-model pixel-generation pass
so the estimate stays fast enough for the control loop.

Euler-integration convention mirrored from openpi's own action decoder
(third_party/openpi/src/openpi/models/pi0.py::Pi0.sample_actions,
`step()` closure): t=1 is noise, t=0 is the target distribution,
`dt = -1.0 / num_steps`, and at each step `x_t = x_t + dt * v_t` where
`v_t` is the (here: caller-supplied) velocity field evaluated at the
current latent and timestep. pi0 uses num_steps=10 for actions; PKLP
uses num_steps=5 (DEFAULT_EULER_STEPS below) since it's predicting a
much lower-entropy quantity (a near-linear kinematic continuation, not
a full action distribution) and needs to stay near the ~40ms budget.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_EULER_STEPS = 5

VelocityFn = Callable[[np.ndarray, float, Any], np.ndarray]


@dataclass
class FlowMatchingConfig:
    num_steps: int = DEFAULT_EULER_STEPS


class ConditionalFlowMatcher:
    """velocity_fn(x_t, t, condition) -> v_t is supplied by the caller —
    in the real system this is a small trained network conditioned on
    the PKLP kinematic state (see kinematics.py); here it's kept
    generic so the integration logic itself is testable independent of
    that network."""

    def __init__(self, velocity_fn: VelocityFn, config: FlowMatchingConfig | None = None):
        self.velocity_fn = velocity_fn
        self.config = config or FlowMatchingConfig()

    def integrate(self, noise: np.ndarray, condition: Any) -> np.ndarray:
        dt = -1.0 / self.config.num_steps
        x_t = noise
        t = 1.0
        for _ in range(self.config.num_steps):
            v_t = self.velocity_fn(x_t, t, condition)
            x_t = x_t + dt * v_t
            t += dt
        return x_t

    def integrate_trajectory(self, noise: np.ndarray, condition: Any) -> list[np.ndarray]:
        """Same as integrate(), but returns every intermediate x_t
        (length num_steps + 1, starting with `noise`) — useful for the
        adaptive-horizon variance check in adaptive_horizon.py, which
        needs the trajectory, not just the endpoint."""
        dt = -1.0 / self.config.num_steps
        x_t = noise
        t = 1.0
        trajectory = [x_t]
        for _ in range(self.config.num_steps):
            v_t = self.velocity_fn(x_t, t, condition)
            x_t = x_t + dt * v_t
            t += dt
            trajectory.append(x_t)
        return trajectory
