import numpy as np
from pathlib import Path

from helpers.q2_simulation import (
    GuidanceConfig,
    InterceptorConstraints,
    SimulationConfig,
    TargetBehaviorConfig,
    apply_constraints,
    run_q2_analysis,
    run_scenario_grid,
    sweep_horizontal_headings,
)


def make_small_config() -> SimulationConfig:
    return SimulationConfig(
        target_initial_position=np.array([100.0, 20.0, 30.0]),
        target_velocity=np.array([6.0, 0.0, 1.5]),
        interceptor_initial_position=np.zeros(3),
        interceptor_initial_velocity=np.zeros(3),
        constraints=InterceptorConstraints(max_speed=18.0, max_accel_x=6.0, max_accel_y=6.0, max_accel_z=6.0),
        target_behavior=TargetBehaviorConfig(mode="straight", seed=5),
        guidance=GuidanceConfig(mode="predictive", response_time_s=0.25, navigation_constant=3.0),
        dt=0.05,
        horizon_s=30.0,
        intercept_radius_m=1.0,
    )


def test_sweep_horizontal_headings_includes_pn_columns():
    heading_sweep = sweep_horizontal_headings(make_small_config(), heading_deg_step=90.0)

    assert {"pure_intercept_time_s", "predictive_intercept_time_s", "pn_intercept_time_s"}.issubset(heading_sweep.columns)
    assert len(heading_sweep) == 4


def test_run_scenario_grid_reports_every_mode_pair():
    scenario_grid = run_scenario_grid(
        make_small_config(),
        guidance_modes=("predictive", "pn"),
        target_modes=("straight", "weave", "reactive"),
    )

    assert len(scenario_grid) == 6
    assert set(scenario_grid["guidance_mode"]) == {"predictive", "pn"}
    assert set(scenario_grid["target_mode"]) == {"straight", "weave", "reactive"}


def test_apply_constraints_keeps_axis_limits_after_speed_cap():
    constraints = InterceptorConstraints(max_speed=20.0, max_accel_x=2.5, max_accel_y=1.0, max_accel_z=None)

    next_velocity, applied_accel = apply_constraints(
        current_velocity=np.array([9.17986244, 1.74499966, 17.40289695]),
        commanded_accel=np.array([6.31707108, -9.94523, 7.14808553]),
        dt=0.08,
        constraints=constraints,
    )

    assert np.linalg.norm(next_velocity) <= 20.0 + 1e-9
    assert abs(applied_accel[0]) <= 2.5 + 1e-9
    assert abs(applied_accel[1]) <= 1.0 + 1e-9


def test_run_q2_analysis_emits_all_evasion_scenario_plots(tmp_path):
    summary = run_q2_analysis(
        tmp_path,
        horizon_s=40.0,
        render_animation=False,
    )

    scenario_results = summary["evasive_scenario_guidance_results"]
    assert set(scenario_results) == {"weave", "bounded_turn", "reactive"}
    for guidance_map in scenario_results.values():
        assert set(guidance_map) == {"predictive", "pure", "pn"}

    artifact_names = {Path(path).name for path in summary["artifact_paths"]}
    assert {"q2_overview_matrix.png", "q2_summary.json"}.issubset(
        artifact_names | {Path(summary["summary_path"]).name}
    )
    assert len(summary["scenario_bundles"]) == 12
