import numpy as np

from occ_vla.pklp.latent_predictor import ConditionalFlowMatcher, FlowMatchingConfig


def _straight_line_velocity(x_t, t, condition):
    # dx/dt = noise - target for the OT-CFM straight-line path
    # x_t = t*noise + (1-t)*target, constant along the path.
    return condition["noise"] - condition["target"]


def test_integrate_recovers_target_for_straight_line_field():
    noise = np.array([5.0, -3.0])
    target = np.array([1.0, 2.0])
    matcher = ConditionalFlowMatcher(_straight_line_velocity, FlowMatchingConfig(num_steps=5))
    result = matcher.integrate(noise, {"noise": noise, "target": target})
    np.testing.assert_allclose(result, target, atol=1e-8)


def test_integrate_is_step_count_invariant_for_linear_field():
    noise = np.array([5.0, -3.0])
    target = np.array([1.0, 2.0])
    condition = {"noise": noise, "target": target}
    for num_steps in (1, 5, 20):
        matcher = ConditionalFlowMatcher(_straight_line_velocity, FlowMatchingConfig(num_steps=num_steps))
        result = matcher.integrate(noise, condition)
        np.testing.assert_allclose(result, target, atol=1e-8)


def test_integrate_trajectory_starts_at_noise_and_ends_at_integrate_result():
    noise = np.array([5.0, -3.0])
    target = np.array([1.0, 2.0])
    matcher = ConditionalFlowMatcher(_straight_line_velocity, FlowMatchingConfig(num_steps=5))
    trajectory = matcher.integrate_trajectory(noise, {"noise": noise, "target": target})
    assert len(trajectory) == 6
    np.testing.assert_allclose(trajectory[0], noise)
    np.testing.assert_allclose(trajectory[-1], matcher.integrate(noise, {"noise": noise, "target": target}))


def test_integrate_trajectory_progresses_monotonically_toward_target_for_linear_field():
    noise = np.array([0.0])
    target = np.array([10.0])
    matcher = ConditionalFlowMatcher(_straight_line_velocity, FlowMatchingConfig(num_steps=5))
    trajectory = matcher.integrate_trajectory(noise, {"noise": noise, "target": target})
    distances = [abs(x[0] - target[0]) for x in trajectory]
    assert distances == sorted(distances, reverse=True)
