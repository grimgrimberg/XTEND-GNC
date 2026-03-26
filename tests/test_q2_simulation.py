import numpy as np

from helpers.q2_simulation import (
    InterceptorConstraints,
    SimulationConfig,
    apply_constraints,
    make_nominal_target_velocity,
    simulate_interception,
)


def test_constraints_clip_axes_and_speed():
    constraints = InterceptorConstraints(max_speed=5.0, max_accel_x=2.0, max_accel_y=1.0)
    next_velocity, applied_accel = apply_constraints(
        current_velocity=np.array([4.5, 0.0, 0.0]),
        commanded_accel=np.array([3.0, 2.0, 4.0]),
        dt=0.5,
        constraints=constraints,
    )
    assert applied_accel[0] <= 2.0 + 1e-9
    assert applied_accel[1] <= 1.0 + 1e-9
    assert np.linalg.norm(next_velocity) <= 5.0 + 1e-9


def test_nominal_target_velocity_points_away_from_origin():
    velocity = make_nominal_target_velocity(np.array([8.0, 2.0, 3.0]), speed=13.0)
    assert np.isclose(np.linalg.norm(velocity), 13.0, atol=1e-9)
    assert np.dot(velocity, np.array([8.0, 2.0, 3.0])) > 0.0


def test_predictive_guidance_intercepts_easy_target_case():
    config = SimulationConfig(
        target_initial_position=np.array([20.0, 0.0, 0.0]),
        target_velocity=np.array([0.0, 0.0, 0.0]),
        interceptor_initial_position=np.array([0.0, 0.0, 0.0]),
        interceptor_initial_velocity=np.array([0.0, 0.0, 0.0]),
        constraints=InterceptorConstraints(max_speed=10.0, max_accel_x=5.0, max_accel_y=5.0),
        dt=0.05,
        horizon_s=10.0,
        intercept_radius_m=0.5,
    )
    result = simulate_interception(config, guidance="predictive")
    assert result.intercepted
    assert result.intercept_time_s is not None
    assert result.distance_m.min() <= 0.5 + 1e-9
