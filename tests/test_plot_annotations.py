from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers.plotting import PlotConfig
from helpers.q1_pipeline import FrameConvention, FrameConventionSelection
from helpers.q1_visualization import (
    plot_camera_fov_reticle,
    plot_convention_diagnostics,
    plot_world_angles,
)
from helpers.q2_simulation import InterceptorConstraints, SimulationConfig, SimulationResult
from helpers.q2_targets import TargetBehaviorConfig
from helpers.q2_guidance import GuidanceConfig
from helpers.q2_visualization import plot_bundle_engagement, plot_overview_matrix


def _capture_figure(monkeypatch, module):
    captured: dict[str, plt.Figure] = {}

    def fake_close_and_save(figure, output_path, *, dpi=None):
        captured["figure"] = figure
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(module, "close_and_save", fake_close_and_save)
    return captured


def test_q1_annotation_plots_have_legends_and_axis_labels(monkeypatch, tmp_path):
    import helpers.q1_visualization as q1_viz

    captured = _capture_figure(monkeypatch, q1_viz)

    selection = FrameConventionSelection(
        best_convention=FrameConvention(),
        candidate_table=pd.DataFrame(
            [
                {
                    "quaternion_is_body_to_world": False,
                    "camera_positive_azimuth_right": True,
                    "camera_positive_elevation_up": False,
                    "gimbal_positive_pitch_raises": False,
                    "pair_miss_median_m": 0.04,
                    "combined_rate_std_deg_s": 29.0,
                    "heading_error_median_deg": 4.1,
                },
                {
                    "quaternion_is_body_to_world": True,
                    "camera_positive_azimuth_right": True,
                    "camera_positive_elevation_up": True,
                    "gimbal_positive_pitch_raises": False,
                    "pair_miss_median_m": 0.30,
                    "combined_rate_std_deg_s": 35.0,
                    "heading_error_median_deg": 48.0,
                },
            ]
        ),
    )

    plot_convention_diagnostics(tmp_path / "frame.png", selection, PlotConfig())
    figure = captured["figure"]
    for axis in figure.axes:
        assert axis.get_xlabel()
        assert axis.get_ylabel()
    assert any(axis.get_legend() is not None for axis in figure.axes)
    plt.close(figure)

    synchronized = pd.DataFrame(
        {
            "time_s": np.linspace(0.0, 1.0, 5),
            "world_az_deg": np.linspace(0.0, 5.0, 5),
            "world_el_deg": np.linspace(-3.0, 2.0, 5),
            "target_x_deg": np.linspace(-2.0, 3.0, 5),
            "target_y_deg": np.linspace(4.0, -1.0, 5),
        }
    )

    plot_world_angles(tmp_path / "angles.png", synchronized, FrameConvention(), PlotConfig())
    figure = captured["figure"]
    assert all(axis.get_legend() is not None for axis in figure.axes)
    plt.close(figure)

    plot_camera_fov_reticle(tmp_path / "reticle.png", synchronized, PlotConfig())
    figure = captured["figure"]
    plotted_axes = [axis for axis in figure.axes if axis.get_title()]
    assert all(axis.get_legend() is not None for axis in plotted_axes)
    plt.close(figure)


def test_q2_annotation_plots_have_legends_and_axis_labels(monkeypatch, tmp_path):
    import helpers.q2_visualization as q2_viz

    captured = _capture_figure(monkeypatch, q2_viz)

    scenario_grid = pd.DataFrame(
        [
            {"target_mode": target_mode, "guidance_mode": guidance_mode, "intercept_time_s": 10.0, "minimum_distance_m": 1.0}
            for target_mode in ("straight", "weave")
            for guidance_mode in ("predictive", "pure")
        ]
    )
    plot_overview_matrix(tmp_path / "overview.png", scenario_grid, PlotConfig())
    figure = captured["figure"]
    heatmap_axes = [axis for axis in figure.axes if axis.get_title()]
    assert all(axis.get_xlabel() for axis in heatmap_axes)
    assert all(axis.get_ylabel() for axis in heatmap_axes)
    plt.close(figure)

    config = SimulationConfig(
        target_initial_position=np.array([100.0, 20.0, 30.0]),
        target_velocity=np.array([6.0, 0.0, 1.5]),
        interceptor_initial_position=np.zeros(3),
        interceptor_initial_velocity=np.zeros(3),
        constraints=InterceptorConstraints(max_speed=18.0, max_accel_x=6.0, max_accel_y=6.0),
        target_behavior=TargetBehaviorConfig(mode="straight"),
        guidance=GuidanceConfig(mode="predictive", response_time_s=0.25),
        dt=0.05,
        horizon_s=2.0,
        intercept_radius_m=1.0,
    )
    time_s = np.linspace(0.0, 2.0, 5)
    target_position = np.column_stack([100.0 - 2.0 * time_s, 20.0 * np.ones_like(time_s), 30.0 * np.ones_like(time_s)])
    interceptor_position = np.column_stack([10.0 * time_s, np.zeros_like(time_s), np.zeros_like(time_s)])
    result = SimulationResult(
        time_s=time_s,
        target_position_m=target_position,
        target_velocity_mps=np.tile(np.array([6.0, 0.0, 0.0]), (len(time_s), 1)),
        interceptor_position_m=interceptor_position,
        interceptor_velocity_mps=np.tile(np.array([8.0, 0.0, 0.0]), (len(time_s), 1)),
        commanded_accel_mps2=np.tile(np.array([1.0, 0.5, 0.0]), (len(time_s), 1)),
        applied_accel_mps2=np.tile(np.array([1.0, 0.5, 0.0]), (len(time_s), 1)),
        distance_m=np.linspace(100.0, 5.0, len(time_s)),
        closing_speed_mps=np.linspace(2.0, 8.0, len(time_s)),
        los_rate_norm_radps=np.linspace(0.1, 0.02, len(time_s)),
        guidance_mode="predictive",
        target_mode="straight",
        intercepted=False,
        intercept_time_s=None,
    )

    plot_bundle_engagement(tmp_path / "engagement.png", config, result, PlotConfig())
    figure = captured["figure"]
    plotted_axes = [axis for axis in figure.axes if axis.get_title()]
    assert all(axis.get_xlabel() for axis in plotted_axes)
    assert all(axis.get_ylabel() for axis in plotted_axes)
    assert all(axis.get_legend() is not None for axis in plotted_axes)
    plt.close(figure)
