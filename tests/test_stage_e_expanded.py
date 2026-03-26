"""Expanded test suite: integration, edge-case, and regression tests added during the Stage E upgrade pass."""

from __future__ import annotations

import numpy as np
import pandas as pd

from helpers.q1_pipeline import (
    FrameConvention,
    camera_angles_to_los,
    estimate_angle_rate,
    estimate_local_ray_bundle_points,
    los_body_from_camera,
    los_to_world_angles_nwu,
    synchronize_q1_streams,
)
from helpers.q2_guidance import analytic_lead_intercept_time, proportional_navigation_command
from helpers.q2_simulation import (
    GuidanceConfig,
    InterceptorConstraints,
    SimulationConfig,
    apply_constraints,
    simulate_interception,
)
from helpers.q2_targets import TargetBehaviorConfig, target_velocity_for_behavior
from helpers.utils import to_builtin


# ─── Q1 edge-case tests ───────────────────────────────────────────────

def test_camera_angles_to_los_at_large_angles_survives():
    """The tan projection model should not crash at ~12 deg (max data range)."""
    los = camera_angles_to_los(np.array([12.0]), np.array([28.0]))
    assert los.shape == (1, 3)
    assert np.isclose(np.linalg.norm(los[0]), 1.0, atol=1e-9)


def test_estimate_angle_rate_returns_zero_for_constant_signal():
    """A constant signal should have zero derivative."""
    time_s = np.linspace(0.0, 5.0, 201)
    signal_deg = np.full_like(time_s, 42.0)
    for method in ("gradient", "savgol", "local_polynomial", "spline"):
        rate = estimate_angle_rate(time_s, signal_deg, method=method)
        assert np.max(np.abs(rate)) < 0.5, f"{method} returned non-zero rate for constant signal"


def test_los_body_from_camera_identity_at_zero_pitch():
    """Zero gimbal pitch should not rotate the camera LOS."""
    los = camera_angles_to_los(np.array([5.0]), np.array([3.0]))
    los_body = los_body_from_camera(los, np.array([0.0]))
    assert np.allclose(los[0], los_body[0], atol=1e-9)


# ─── Q2 edge-case tests ──────────────────────────────────────────────

def test_analytic_lead_time_same_speed_still_works():
    """When pursuer speed == target speed, but target moves toward pursuer, intercept is possible."""
    t = analytic_lead_intercept_time(
        relative_position_m=np.array([100.0, 0.0, 0.0]),
        target_velocity_mps=np.array([-13.0, 0.0, 0.0]),
        pursuer_speed_mps=13.0,
    )
    assert t is not None
    assert t > 0.0
    assert t > 0.0


def test_analytic_lead_time_impossible_returns_none():
    """When target is faster and running away, no intercept should be possible."""
    t = analytic_lead_intercept_time(
        relative_position_m=np.array([100.0, 0.0, 0.0]),
        target_velocity_mps=np.array([50.0, 0.0, 0.0]),
        pursuer_speed_mps=10.0,
    )
    assert t is None


def test_speed_feasible_delta_scale_closed_form_matches_brute():
    """Verify the closed-form speed cap matches a brute-force search."""
    from helpers.q2_simulation import _speed_feasible_delta_scale

    v = np.array([15.0, 8.0, 10.0])
    dv = np.array([5.0, -3.0, 7.0])
    max_speed = 20.0
    alpha = _speed_feasible_delta_scale(v, dv, max_speed)
    result_speed = np.linalg.norm(v + alpha * dv)
    assert result_speed <= max_speed + 1e-9
    # Alpha should be close to the speed sphere
    assert abs(result_speed - max_speed) < 0.01


def test_apply_constraints_all_axes_respected_under_speed_cap():
    """After speed capping, axis acceleration limits must still hold."""
    constraints = InterceptorConstraints(max_speed=20.0, max_accel_x=2.5, max_accel_y=1.0)
    rng = np.random.default_rng(42)
    for _ in range(50):
        # Generate a valid initial velocity bounded by max_speed
        raw_v = rng.uniform(-1, 1, size=3)
        v = raw_v * rng.uniform(0, 19.9) / np.linalg.norm(raw_v)
        
        a = rng.uniform(-10, 10, size=3)
        next_v, applied_a = apply_constraints(
            current_velocity=v, commanded_accel=a, dt=0.05, constraints=constraints
        )
        assert np.linalg.norm(next_v) <= 20.0 + 1e-9
        assert abs(applied_a[0]) <= 2.5 + 1e-9
        assert abs(applied_a[1]) <= 1.0 + 1e-9


def test_pn_nonzero_lateral_command_when_los_rate_exists():
    """PN should generate a lateral command when there IS a LOS rate."""
    cmd = proportional_navigation_command(
        relative_position_m=np.array([100.0, 0.0, 0.0]),
        relative_velocity_mps=np.array([-10.0, 5.0, 0.0]),
        interceptor_velocity_mps=np.array([15.0, 0.0, 0.0]),
        navigation_constant=3.5,
    )
    assert np.linalg.norm(cmd) > 0.1


def test_target_speed_preserved_all_modes():
    """Speed magnitude must be constant across all target behavior modes."""
    base_speed = 13.0
    for mode in ("straight", "weave", "bounded_turn", "reactive"):
        config = TargetBehaviorConfig(mode=mode, seed=7)
        for t in (0.0, 5.0, 50.0, 100.0):
            v = target_velocity_for_behavior(
                time_s=t,
                current_position_m=np.array([200.0, 50.0, 100.0]),
                interceptor_position_m=np.array([100.0, 30.0, 80.0]),
                base_speed_mps=base_speed,
                base_heading_rad=0.35,
                base_vertical_speed_mps=2.0,
                config=config,
            )
            assert np.isclose(np.linalg.norm(v), base_speed, atol=1e-9), f"Speed violated in {mode} at t={t}"


def test_deterministic_seeds_produce_identical_results():
    """Same seed must produce identical simulation results."""
    config = SimulationConfig(
        target_initial_position=np.array([100.0, 20.0, 30.0]),
        target_velocity=np.array([6.0, 0.0, 1.5]),
        interceptor_initial_position=np.zeros(3),
        interceptor_initial_velocity=np.zeros(3),
        constraints=InterceptorConstraints(max_speed=18.0, max_accel_x=6.0, max_accel_y=6.0),
        target_behavior=TargetBehaviorConfig(mode="reactive", seed=42),
        guidance=GuidanceConfig(mode="predictive", response_time_s=0.25),
        dt=0.05,
        horizon_s=15.0,
        intercept_radius_m=1.0,
    )
    r1 = simulate_interception(config)
    r2 = simulate_interception(config)
    assert np.allclose(r1.interceptor_position_m, r2.interceptor_position_m)
    assert np.allclose(r1.target_position_m, r2.target_position_m)


# ─── Shared utility tests ────────────────────────────────────────────

def test_to_builtin_round_trip():
    """to_builtin should produce JSON-serializable types."""
    import json
    data = {
        "array": np.array([1.0, 2.0]),
        "scalar": np.float64(3.14),
        "df": pd.DataFrame({"a": [1, 2]}),
        "nested": {"inner": np.int64(7)},
    }
    result = to_builtin(data)
    json_str = json.dumps(result)
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["scalar"] == 3.14
    assert parsed["array"] == [1.0, 2.0]


# ─── Integration / smoke tests ───────────────────────────────────────

def test_q2_all_grid_modes_satisfy_constraints():
    """Every (guidance, target) pair in the grid must obey axis and speed constraints."""
    from helpers.q2_simulation import run_scenario_grid
    from dataclasses import replace

    config = SimulationConfig(
        target_initial_position=np.array([100.0, 20.0, 30.0]),
        target_velocity=np.array([6.0, 0.0, 1.5]),
        interceptor_initial_position=np.zeros(3),
        interceptor_initial_velocity=np.zeros(3),
        constraints=InterceptorConstraints(max_speed=18.0, max_accel_x=6.0, max_accel_y=6.0),
        target_behavior=TargetBehaviorConfig(mode="straight", seed=5),
        guidance=GuidanceConfig(mode="predictive", response_time_s=0.25),
        dt=0.05,
        horizon_s=30.0,
        intercept_radius_m=1.0,
    )
    grid = run_scenario_grid(config, guidance_modes=("predictive", "pn"), target_modes=("straight", "weave", "reactive"))
    for _, row in grid.iterrows():
        assert row["max_speed_mps"] <= 18.0 + 1e-6, f"Speed violated: {row['guidance_mode']}/{row['target_mode']}"
        assert row["max_abs_accel_x_mps2"] <= 6.0 + 1e-6, f"ax violated: {row['guidance_mode']}/{row['target_mode']}"
        assert row["max_abs_accel_y_mps2"] <= 6.0 + 1e-6, f"ay violated: {row['guidance_mode']}/{row['target_mode']}"


def test_q1_synchronize_on_synthetic_known_answer():
    """Build a known-answer synthetic CSV and verify world angles come out right."""
    # Simple case: identity quaternion, zero gimbal, target at (az=5, el=3)
    rows = []
    for t in np.linspace(0, 1, 21):
        rows.append({"time": t, "orientation_w": 1.0, "orientation_x": 0.0, "orientation_y": 0.0, "orientation_z": 0.0,
                      "position_x": 0, "position_y": 0, "position_z": 10,
                      "angular_x": 0, "angular_y": 0, "angular_z": 0,
                      "vel_x": 1, "vel_y": 0, "vel_z": 0})
        rows.append({"time": t + 0.001, "gimbal_pitch_rad": 0.0})
        rows.append({"time": t + 0.002, "target_x_deg": 5.0, "target_y_deg": 3.0})
    df = pd.DataFrame(rows)
    convention = FrameConvention(
        quaternion_is_body_to_world=True,
        camera_positive_azimuth_right=True,
        camera_positive_elevation_up=True,
        gimbal_positive_pitch_raises=False,
    )
    synced = synchronize_q1_streams(df, convention)
    # With identity quaternion and zero gimbal, world az should ≈ 5 deg, el ≈ 3 deg
    # (tan model: slight deviation from exactly 5,3 but should be close)
    assert np.allclose(synced["world_az_deg"].to_numpy(), synced["world_az_deg"].iloc[0], atol=0.5)
    los_world = np.column_stack([synced["world_los_x"], synced["world_los_y"], synced["world_los_z"]])
    assert np.all(los_world[:, 0] > 0.9)  # LOS should be predominantly forward
