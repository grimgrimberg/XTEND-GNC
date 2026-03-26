import numpy as np

from helpers.q2_simulation import (
    GuidanceConfig,
    InterceptorConstraints,
    SimulationConfig,
    TargetBehaviorConfig,
    analytic_lead_intercept_time,
    proportional_navigation_command,
    simulate_interception,
    target_velocity_for_behavior,
)


def test_analytic_lead_intercept_time_handles_stationary_target():
    intercept_time = analytic_lead_intercept_time(
        relative_position_m=np.array([100.0, 0.0, 0.0]),
        target_velocity_mps=np.zeros(3),
        pursuer_speed_mps=20.0,
    )

    assert np.isclose(intercept_time, 5.0, atol=1e-9)


def test_proportional_navigation_command_is_zero_when_los_rate_is_zero():
    command = proportional_navigation_command(
        relative_position_m=np.array([150.0, 0.0, 0.0]),
        relative_velocity_mps=np.array([-12.0, 0.0, 0.0]),
        interceptor_velocity_mps=np.array([18.0, 0.0, 0.0]),
        navigation_constant=3.0,
    )

    assert np.allclose(command, np.zeros(3), atol=1e-9)


def test_target_velocity_for_behavior_preserves_speed_during_weave():
    config = TargetBehaviorConfig(mode="weave", seed=11, weave_amplitude_deg=18.0, weave_period_s=5.0)

    velocity = target_velocity_for_behavior(
        time_s=2.5,
        current_position_m=np.array([800.0, 200.0, 300.0]),
        interceptor_position_m=np.zeros(3),
        base_speed_mps=13.0,
        base_heading_rad=0.35,
        base_vertical_speed_mps=2.0,
        config=config,
    )

    assert np.isclose(np.linalg.norm(velocity), 13.0, atol=1e-9)


def test_reactive_behavior_changes_heading_when_interceptor_is_close():
    config = TargetBehaviorConfig(
        mode="reactive",
        seed=3,
        reactive_trigger_distance_m=250.0,
        reactive_turn_amplitude_deg=35.0,
    )
    far_velocity = target_velocity_for_behavior(
        time_s=4.0,
        current_position_m=np.array([800.0, 200.0, 300.0]),
        interceptor_position_m=np.array([0.0, 0.0, 0.0]),
        base_speed_mps=13.0,
        base_heading_rad=0.35,
        base_vertical_speed_mps=2.0,
        config=config,
    )
    near_velocity = target_velocity_for_behavior(
        time_s=4.0,
        current_position_m=np.array([120.0, 20.0, 300.0]),
        interceptor_position_m=np.array([0.0, 0.0, 0.0]),
        base_speed_mps=13.0,
        base_heading_rad=0.35,
        base_vertical_speed_mps=2.0,
        config=config,
    )

    far_heading = np.arctan2(far_velocity[1], far_velocity[0])
    near_heading = np.arctan2(near_velocity[1], near_velocity[0])
    assert abs(near_heading - far_heading) > np.deg2rad(5.0)


def test_simulate_interception_supports_proportional_navigation():
    simulation = SimulationConfig(
        target_initial_position=np.array([60.0, 0.0, 0.0]),
        interceptor_initial_position=np.zeros(3),
        interceptor_initial_velocity=np.zeros(3),
        constraints=InterceptorConstraints(max_speed=22.0, max_accel_x=8.0, max_accel_y=8.0, max_accel_z=8.0),
        target_behavior=TargetBehaviorConfig(mode="straight"),
        guidance=GuidanceConfig(mode="pn", navigation_constant=3.5, response_time_s=0.2),
        dt=0.02,
        horizon_s=12.0,
        intercept_radius_m=0.75,
        target_speed_mps=6.0,
        target_heading_rad=0.0,
        target_vertical_speed_mps=0.0,
    )

    result = simulate_interception(simulation)

    assert result.intercepted
    assert result.intercept_time_s is not None
    assert result.commanded_accel_mps2.shape == result.applied_accel_mps2.shape
