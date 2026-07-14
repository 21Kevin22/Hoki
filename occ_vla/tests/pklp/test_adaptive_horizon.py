import numpy as np

from occ_vla.pklp.adaptive_horizon import AdaptiveHorizonStopper


def test_stops_at_first_horizon_exceeding_tau_u():
    rng = np.random.default_rng(0)

    def predict_fn(horizon):
        # variance grows with horizon: noise scale = horizon * 0.1
        return np.array([float(horizon)]) + rng.normal(scale=horizon * 0.1, size=1)

    stopper = AdaptiveHorizonStopper(tau_u=0.05, num_samples=20)
    result = stopper.run(predict_fn, horizons=[1, 2, 3, 4, 5])

    assert result.stopped_early
    assert result.trusted_horizon < 5
    assert result.mean_prediction is not None


def test_never_exceeds_tau_u_runs_all_horizons():
    def predict_fn(horizon):
        return np.array([float(horizon)])  # zero variance, deterministic

    stopper = AdaptiveHorizonStopper(tau_u=1e-9, num_samples=5)
    result = stopper.run(predict_fn, horizons=[1, 2, 3])

    assert not result.stopped_early
    assert result.trusted_horizon == 3
    np.testing.assert_allclose(result.mean_prediction, [3.0])


def test_fails_immediately_on_first_horizon_gives_trusted_horizon_zero():
    rng = np.random.default_rng(1)

    def predict_fn(horizon):
        return rng.normal(scale=10.0, size=1)

    stopper = AdaptiveHorizonStopper(tau_u=1e-6, num_samples=10)
    result = stopper.run(predict_fn, horizons=[1, 2, 3])

    assert result.stopped_early
    assert result.trusted_horizon == 0
    assert result.mean_prediction is None
