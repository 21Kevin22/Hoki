"""Adaptive horizon stopping: run S parallel stochastic predictions per
candidate horizon K (each a fresh conditional-flow-matching sample, see
latent_predictor.py), and stop extending K once the S samples disagree
too much — "the physical prediction can no longer be trusted past this
point" — falling back on the last horizon where they agreed.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

DEFAULT_NUM_SAMPLES = 5

PredictFn = Callable[[int], np.ndarray]  # horizon K -> one stochastic sample of the predicted state


@dataclass
class AdaptiveHorizonResult:
    trusted_horizon: int  # largest K for which the S-sample variance stayed <= tau_u; 0 if even the first failed
    mean_prediction: np.ndarray | None  # mean over S samples at trusted_horizon; None if trusted_horizon == 0
    variance: float | None
    stopped_early: bool  # True if some larger horizon was requested but rejected on variance


class AdaptiveHorizonStopper:
    def __init__(self, tau_u: float, num_samples: int = DEFAULT_NUM_SAMPLES):
        self.tau_u = tau_u
        self.num_samples = num_samples

    def run(self, predict_fn: PredictFn, horizons: list[int]) -> AdaptiveHorizonResult:
        trusted_horizon = 0
        mean_prediction = None
        variance = None

        for horizon in horizons:
            samples = np.stack([predict_fn(horizon) for _ in range(self.num_samples)])
            sample_variance = float(samples.var(axis=0).mean())
            if sample_variance > self.tau_u:
                return AdaptiveHorizonResult(
                    trusted_horizon=trusted_horizon,
                    mean_prediction=mean_prediction,
                    variance=variance,
                    stopped_early=True,
                )
            trusted_horizon = horizon
            mean_prediction = samples.mean(axis=0)
            variance = sample_variance

        return AdaptiveHorizonResult(
            trusted_horizon=trusted_horizon,
            mean_prediction=mean_prediction,
            variance=variance,
            stopped_early=False,
        )
