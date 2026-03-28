from __future__ import annotations

import json
from pathlib import Path
from dataclasses import replace

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd

from .plotting import PALETTE, PlotConfig, add_note, close_and_save, set_equal_3d_axes
from .q2_simulation import SimulationConfig, SimulationResult, _result_summary
from .utils import to_builtin


def create_q2_visuals(
    output_dir: Path,
    base_config: SimulationConfig,
    scenario_results: dict[str, dict[str, SimulationResult]],
    heading_sweep: pd.DataFrame,
    scenario_grid: pd.DataFrame,
    selected_guidance: str,
    selected_target: str,
    render_animation: bool,
    plot_config: PlotConfig,
) -> dict[str, object]:
    baseline_results = scenario_results["straight"]
    artifacts: list[Path] = []
    bundle_records: list[dict[str, object]] = []

    baseline_bundle_config = _bundle_config(base_config, target_mode="straight", guidance_mode="predictive")
    artifacts.append(
        plot_nominal_trajectory(
            output_dir / "q2_nominal_trajectory.png",
            baseline_bundle_config,
            baseline_results["predictive"],
            plot_config,
        )
    )
    artifacts.append(
        plot_constraint_traces(
            output_dir / "q2_constraint_traces.png",
            baseline_bundle_config,
            baseline_results["predictive"],
            plot_config,
        )
    )
    artifacts.append(plot_los_geometry_traces(output_dir / "q2_los_geometry.png", baseline_results, plot_config))
    artifacts.append(plot_guidance_comparison(output_dir / "q2_guidance_comparison.png", baseline_results, heading_sweep, plot_config))
    artifacts.append(plot_guidance_3d_comparison(output_dir / "q2_guidance_3d_comparison.png", baseline_results, plot_config))
    overview_path = plot_overview_matrix(output_dir / "q2_overview_matrix.png", scenario_grid, plot_config)
    artifacts.append(overview_path)

    for target_mode, guidance_results in scenario_results.items():
        for guidance_mode, result in guidance_results.items():
            bundle_dir = output_dir / target_mode / guidance_mode
            bundle_config = _bundle_config(base_config, target_mode=target_mode, guidance_mode=guidance_mode)
            bundle_record = write_q2_bundle_artifacts(
                bundle_dir=bundle_dir,
                config=bundle_config,
                result=result,
                render_animation=render_animation,
                plot_config=plot_config,
            )
            bundle_records.append(bundle_record)
            artifacts.extend(Path(path) for path in bundle_record["artifact_paths"])
            artifacts.append(Path(bundle_record["summary_path"]))

    return {
        "artifact_paths": [str(path) for path in artifacts],
        "scenario_bundles": bundle_records,
        "overview_matrix_path": str(overview_path),
    }


def plot_nominal_trajectory(
    output_path: Path,
    config: SimulationConfig,
    result: SimulationResult,
    plot_config: PlotConfig,
) -> Path:
    figure = plt.figure(figsize=(15.5, 10.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    axis_3d = figure.add_subplot(grid[:, 0], projection="3d")
    axis_xy = figure.add_subplot(grid[0, 1])
    axis_xz = figure.add_subplot(grid[1, 1])

    axis_3d.plot(result.target_position_m[:, 0], result.target_position_m[:, 1], result.target_position_m[:, 2], color=PALETTE["secondary"], linewidth=2.0, label="target")
    axis_3d.plot(result.interceptor_position_m[:, 0], result.interceptor_position_m[:, 1], result.interceptor_position_m[:, 2], color=PALETTE["primary"], linewidth=2.2, label="interceptor")
    if result.intercepted:
        intercept_index = int(np.argmin(result.distance_m))
        axis_3d.scatter(*result.interceptor_position_m[intercept_index], color=PALETTE["success"], marker="*", s=160, label="intercept")
    point_cloud = np.vstack([result.target_position_m, result.interceptor_position_m])
    set_equal_3d_axes(axis_3d, point_cloud)
    axis_3d.set_xlabel("North X [m]")
    axis_3d.set_ylabel("West Y [m]")
    axis_3d.set_zlabel("Up Z [m]")
    axis_3d.set_title("3D Trajectory")
    axis_3d.legend(loc="upper left")

    axis_xy.plot(result.target_position_m[:, 0], result.target_position_m[:, 1], color=PALETTE["secondary"], label="target")
    axis_xy.plot(result.interceptor_position_m[:, 0], result.interceptor_position_m[:, 1], color=PALETTE["primary"], label="interceptor")
    axis_xy.set_xlabel("North X [m]")
    axis_xy.set_ylabel("West Y [m]")
    axis_xy.set_title("XY Projection")
    axis_xy.legend(loc="best")

    axis_xz.plot(result.target_position_m[:, 0], result.target_position_m[:, 2], color=PALETTE["secondary"], label="target")
    axis_xz.plot(result.interceptor_position_m[:, 0], result.interceptor_position_m[:, 2], color=PALETTE["primary"], label="interceptor")
    axis_xz.set_xlabel("North X [m]")
    axis_xz.set_ylabel("Up Z [m]")
    axis_xz.set_title("XZ Projection")
    axis_xz.legend(loc="best")
    add_note(
        axis_xz,
        "The baseline still uses straight, constant-speed target motion.\n"
        "Only the unspecified heading is fixed to a mild crossing case so the 3D plot is readable.",
    )
    figure.suptitle("Q2 Nominal Interception Geometry", fontsize=15)
    close_and_save(figure, output_path, dpi=plot_config.dpi)
    return output_path


def plot_bundle_trajectory(
    output_path: Path,
    config: SimulationConfig,
    result: SimulationResult,
    *,
    plot_config: PlotConfig,
) -> Path:
    figure = plt.figure(figsize=(15.5, 10.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    axis_3d = figure.add_subplot(grid[:, 0], projection="3d")
    axis_xy = figure.add_subplot(grid[0, 1])
    axis_xz = figure.add_subplot(grid[1, 1])

    axis_3d.plot(result.target_position_m[:, 0], result.target_position_m[:, 1], result.target_position_m[:, 2], color=PALETTE["secondary"], linewidth=2.0, label="target")
    axis_3d.plot(result.interceptor_position_m[:, 0], result.interceptor_position_m[:, 1], result.interceptor_position_m[:, 2], color=PALETTE["primary"], linewidth=2.2, label="interceptor")
    if result.intercepted:
        intercept_index = int(np.argmin(result.distance_m))
        axis_3d.scatter(*result.interceptor_position_m[intercept_index], color=PALETTE["success"], marker="*", s=160, label="intercept")
    point_cloud = np.vstack([result.target_position_m, result.interceptor_position_m])
    set_equal_3d_axes(axis_3d, point_cloud)
    axis_3d.set_xlabel("North X [m]")
    axis_3d.set_ylabel("West Y [m]")
    axis_3d.set_zlabel("Up Z [m]")
    axis_3d.set_title("3D Trajectory")
    axis_3d.legend(loc="upper left")

    axis_xy.plot(result.target_position_m[:, 0], result.target_position_m[:, 1], color=PALETTE["secondary"], label="target")
    axis_xy.plot(result.interceptor_position_m[:, 0], result.interceptor_position_m[:, 1], color=PALETTE["primary"], label="interceptor")
    axis_xy.set_xlabel("North X [m]")
    axis_xy.set_ylabel("West Y [m]")
    axis_xy.set_title("XY Projection")
    axis_xy.legend(loc="best")

    axis_xz.plot(result.target_position_m[:, 0], result.target_position_m[:, 2], color=PALETTE["secondary"], label="target")
    axis_xz.plot(result.interceptor_position_m[:, 0], result.interceptor_position_m[:, 2], color=PALETTE["primary"], label="interceptor")
    axis_xz.set_xlabel("North X [m]")
    axis_xz.set_ylabel("Up Z [m]")
    axis_xz.set_title("XZ Projection")
    axis_xz.legend(loc="best")
    add_note(
        axis_xz,
        f"Bundle: {result.target_mode} target + {result.guidance_mode} guidance.\n"
        "The geometry and constraints are shown for the exact bundle that produced this summary.",
    )
    figure.suptitle(f"Q2 Bundle Geometry | {result.target_mode} / {result.guidance_mode}", fontsize=15)
    close_and_save(figure, output_path, dpi=plot_config.dpi)
    return output_path


def plot_constraint_traces(output_path: Path, config: SimulationConfig, result: SimulationResult, plot_config: PlotConfig) -> Path:
    figure, axes = plt.subplots(5, 1, figsize=(15.0, 12.0), sharex=True, constrained_layout=True)
    interceptor_speed = np.linalg.norm(result.interceptor_velocity_mps, axis=1)
    target_speed = np.linalg.norm(result.target_velocity_mps, axis=1)
    axes[0].plot(result.time_s, result.distance_m, color=PALETTE["primary"], label="distance")
    axes[0].axhline(config.intercept_radius_m, color=PALETTE["success"], linestyle="--", label="intercept radius")
    axes[0].set_ylabel("Distance [m]")
    axes[0].set_title("Closure And Constraint Traces")
    axes[0].legend(loc="upper right")

    axes[1].plot(result.time_s, interceptor_speed, color=PALETTE["primary"], label="interceptor speed")
    axes[1].plot(result.time_s, target_speed, color=PALETTE["secondary"], label="target speed")
    axes[1].axhline(config.constraints.max_speed, color=PALETTE["warning"], linestyle="--", label="max interceptor speed")
    axes[1].set_ylabel("Speed [m/s]")
    axes[1].legend(loc="upper right")

    axes[2].plot(result.time_s, result.commanded_accel_mps2[:, 0], color=PALETTE["soft"], label="commanded ax")
    axes[2].plot(result.time_s, result.applied_accel_mps2[:, 0], color=PALETTE["primary"], label="applied ax")
    axes[2].plot(result.time_s, result.commanded_accel_mps2[:, 1], color=PALETTE["danger"], alpha=0.4, label="commanded ay")
    axes[2].plot(result.time_s, result.applied_accel_mps2[:, 1], color=PALETTE["secondary"], label="applied ay")
    axes[2].axhline(config.constraints.max_accel_x, color=PALETTE["primary"], linestyle="--", alpha=0.6)
    axes[2].axhline(-config.constraints.max_accel_x, color=PALETTE["primary"], linestyle="--", alpha=0.6)
    axes[2].axhline(config.constraints.max_accel_y, color=PALETTE["secondary"], linestyle=":", alpha=0.7)
    axes[2].axhline(-config.constraints.max_accel_y, color=PALETTE["secondary"], linestyle=":", alpha=0.7)
    axes[2].set_ylabel("Accel [m/s²]")
    axes[2].legend(loc="upper right", ncols=2)

    axes[3].plot(result.time_s, result.closing_speed_mps, color=PALETTE["success"], label="closing speed")
    axes[3].set_ylabel("Closing Speed [m/s]")
    axes[3].legend(loc="upper right")

    # Closing speed and LOS rate use different physical units, so keep them on
    # separate panels instead of asking the reader to decode a mixed axis.
    axes[4].plot(result.time_s, result.los_rate_norm_radps, color=PALETTE["accent"], label="LOS rate norm")
    axes[4].set_ylabel("LOS Rate [rad/s]")
    axes[4].set_xlabel("Time [s]")
    axes[4].legend(loc="upper right")
    add_note(axes[4], "Commanded vs applied acceleration exposes clipping directly.\nThis keeps the X/Y constraint enforcement visible rather than implicit.", y=1.18)
    figure.suptitle("Q2 Constraint Verification", fontsize=15)
    close_and_save(figure, output_path, dpi=plot_config.dpi)
    return output_path


def plot_bundle_engagement(
    output_path: Path,
    config: SimulationConfig,
    result: SimulationResult,
    plot_config: PlotConfig,
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(15.2, 10.0), sharex=True, constrained_layout=True)
    axes = axes.ravel()
    interceptor_speed = np.linalg.norm(result.interceptor_velocity_mps, axis=1)
    relative_position = result.target_position_m - result.interceptor_position_m
    range_m = np.linalg.norm(relative_position, axis=1)
    lead_angle_deg = np.rad2deg(
        np.arccos(
            np.clip(
                np.sum(result.interceptor_velocity_mps * relative_position, axis=1)
                / np.maximum(interceptor_speed * range_m, 1e-9),
                -1.0,
                1.0,
            )
        )
    )

    axes[0].plot(result.time_s, result.distance_m, color=PALETTE["primary"], linewidth=2.0, label="distance")
    axes[0].axhline(config.intercept_radius_m, color=PALETTE["success"], linestyle="--", alpha=0.7, label="intercept radius")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Distance [m]")
    axes[0].set_title("Range Closure")
    axes[0].legend(loc="upper right")

    axes[1].plot(result.time_s, result.closing_speed_mps, color=PALETTE["accent"], linewidth=2.0, label="closing speed")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Closing Speed [m/s]")
    axes[1].set_title("Closing Speed")
    axes[1].legend(loc="upper right")

    axes[2].plot(result.time_s, result.los_rate_norm_radps, color=PALETTE["secondary"], linewidth=2.0, label="LOS rate")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("LOS Rate [rad/s]")
    axes[2].set_title("Line-Of-Sight Rate")
    axes[2].legend(loc="upper right")

    axes[3].plot(result.time_s, lead_angle_deg, color=PALETTE["danger"], linewidth=2.0, label="lead angle")
    axes[3].set_ylabel("Lead Angle [deg]")
    axes[3].set_xlabel("Time [s]")
    axes[3].set_title("Lead Angle")
    axes[3].legend(loc="upper right")

    for axis in axes:
        axis.grid(True, alpha=0.4)
    add_note(
        axes[3],
        f"Bundle: {result.target_mode} target + {result.guidance_mode} guidance.\n"
        "This panel is bundle-specific and is used inside each scenario package.",
        y=1.18,
    )
    figure.suptitle(f"Q2 Engagement Summary | {result.target_mode} / {result.guidance_mode}", fontsize=15)
    close_and_save(figure, output_path, dpi=plot_config.dpi)
    return output_path


def plot_guidance_comparison(output_path: Path, nominal_results: dict[str, SimulationResult], heading_sweep: pd.DataFrame, plot_config: PlotConfig) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(15.0, 5.8), constrained_layout=True)
    for mode, color in (("pure", PALETTE["secondary"]), ("predictive", PALETTE["primary"]), ("pn", PALETTE["accent"])):
        result = nominal_results[mode]
        axes[0].plot(result.time_s, result.distance_m, color=color, label=mode)
        axes[1].plot(heading_sweep["heading_deg"], heading_sweep[f"{mode}_intercept_time_s"], marker="o", markersize=4, color=color, label=mode)
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Distance [m]")
    axes[0].set_title("Nominal Miss-Distance Comparison")
    axes[0].legend(loc="upper right")
    axes[1].set_xlabel("Horizontal Target Heading [deg]")
    axes[1].set_ylabel("Intercept Time [s]")
    axes[1].set_title("Heading Sensitivity Sweep")
    axes[1].legend(loc="upper right")
    add_note(axes[1], "PN is intentionally exposed as a selectable extension mode.\nThe comparison keeps the original baseline visible while showing why the extra mode matters.")
    figure.suptitle("Q2 Guidance-Mode Comparison", fontsize=15)
    close_and_save(figure, output_path, dpi=plot_config.dpi)
    return output_path


def plot_evasion_comparison(output_path: Path, scenario_grid: pd.DataFrame, plot_config: PlotConfig) -> Path:
    return plot_overview_matrix(output_path, scenario_grid, plot_config)


def plot_overview_matrix(output_path: Path, scenario_grid: pd.DataFrame, plot_config: PlotConfig) -> Path:
    pivot_time = scenario_grid.pivot(index="target_mode", columns="guidance_mode", values="intercept_time_s")
    pivot_distance = scenario_grid.pivot(index="target_mode", columns="guidance_mode", values="minimum_distance_m")
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), constrained_layout=True)
    time_cmap = plt.colormaps["viridis"].copy()
    time_cmap.set_bad(color="#e7e7e7")
    time_values = pivot_time.to_numpy(dtype=float)
    distance_values = pivot_distance.to_numpy(dtype=float)
    time_image = axes[0].imshow(time_values, aspect="auto", cmap=time_cmap)
    distance_image = axes[1].imshow(distance_values, aspect="auto", cmap="magma_r")
    for axis, pivot, title in (
        (axes[0], pivot_time, "Intercept Time [s]"),
        (axes[1], pivot_distance, "Minimum Distance [m]"),
    ):
        axis.set_xlabel("Guidance Law")
        axis.set_ylabel("Target Mode")
        axis.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
        axis.set_yticks(np.arange(len(pivot.index)), pivot.index)
        axis.set_title(title)
        for row_index in range(len(pivot.index)):
            for column_index in range(len(pivot.columns)):
                value = pivot.to_numpy(dtype=float)[row_index, column_index]
                if title == "Intercept Time [s]" and np.isnan(value):
                    label = "MISS"
                    text_color = PALETTE["ink"]
                else:
                    label = f"{value:.1f}"
                    text_color = "white"
                axis.text(column_index, row_index, label, ha="center", va="center", color=text_color, fontsize=9.5)
    figure.colorbar(time_image, ax=axes[0], shrink=0.85, label="Seconds")
    figure.colorbar(distance_image, ax=axes[1], shrink=0.85, label="Meters")
    add_note(axes[1], "The baseline `straight` mode remains the assignment answer.\nThe evasive modes are clearly labeled extensions so a reviewer can separate core solution from extras.", y=1.16)
    figure.suptitle("Q2 Evasive-Target Extension Matrix", fontsize=15)
    close_and_save(figure, output_path, dpi=plot_config.dpi)
    return output_path


def plot_evasive_trajectories(output_path: Path, extension_results: dict[str, SimulationResult], plot_config: PlotConfig) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(15.0, 6.0), constrained_layout=True)
    colors = {
        "straight": PALETTE["primary"],
        "weave": PALETTE["secondary"],
        "bounded_turn": PALETTE["accent"],
        "reactive": PALETTE["success"],
    }
    for mode, result in extension_results.items():
        color = colors[mode]
        axes[0].plot(result.target_position_m[:, 0], result.target_position_m[:, 1], color=color, linewidth=2.0, label=mode)
        axes[1].plot(result.interceptor_position_m[:, 0], result.interceptor_position_m[:, 1], color=color, linewidth=2.0, label=mode)
    axes[0].set_xlabel("North X [m]")
    axes[0].set_ylabel("West Y [m]")
    axes[0].set_title("Target Paths By Mode")
    axes[1].set_xlabel("North X [m]")
    axes[1].set_ylabel("West Y [m]")
    axes[1].set_title("Interceptor Response By Mode")
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    add_note(axes[1], "These are extension scenarios under the selected guidance law.\nThey make the evasive behaviors visually concrete, while the straight case remains the assignment baseline.")
    figure.suptitle("Q2 Spatial Effect Of Evasive Target Modes", fontsize=15)
    close_and_save(figure, output_path, dpi=plot_config.dpi)
    return output_path


def plot_all_evasion_scenarios(
    output_dir: Path,
    evasive_scenario_results: dict[str, dict[str, SimulationResult]],
    plot_config: PlotConfig,
) -> list[Path]:
    artifact_paths: list[Path] = []
    for scenario_name, guidance_results in evasive_scenario_results.items():
        artifact_paths.append(
            plot_evasion_scenario_guidance_comparison(
                output_dir / f"q2_evasion_{scenario_name}_guidance.png",
                scenario_name,
                guidance_results,
                plot_config,
            )
        )
    return artifact_paths


def write_q2_bundle_artifacts(
    *,
    bundle_dir: Path,
    config: SimulationConfig,
    result: SimulationResult,
    render_animation: bool,
    plot_config: PlotConfig,
) -> dict[str, object]:
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths: list[Path] = []
    artifact_paths.append(plot_bundle_trajectory(bundle_dir / "trajectory.png", config, result, plot_config=plot_config))
    artifact_paths.append(plot_constraint_traces(bundle_dir / "constraints.png", config, result, plot_config))
    artifact_paths.append(plot_bundle_engagement(bundle_dir / "engagement.png", config, result, plot_config))
    animation_path = bundle_dir / "animation.gif"
    if render_animation:
        create_q2_animation(animation_path, result, f"{result.target_mode} / {result.guidance_mode}", plot_config)
        artifact_paths.append(animation_path)

    summary = {
        "target_mode": result.target_mode,
        "guidance_mode": result.guidance_mode,
        "bundle_dir": str(bundle_dir),
        "scenario": {
            "target_initial_position_m": to_builtin(config.target_initial_position),
            "interceptor_initial_position_m": to_builtin(config.interceptor_initial_position),
            "interceptor_initial_velocity_mps": to_builtin(config.interceptor_initial_velocity),
            "target_speed_mps": float(config.target_speed_mps),
            "target_heading_rad": None if config.target_heading_rad is None else float(config.target_heading_rad),
            "target_vertical_speed_mps": None if config.target_vertical_speed_mps is None else float(config.target_vertical_speed_mps),
            "intercept_radius_m": float(config.intercept_radius_m),
            "dt_s": float(config.dt),
            "horizon_s": float(config.horizon_s),
        },
        "metrics": _result_summary(result),
        "artifact_paths": [str(path) for path in artifact_paths],
    }
    summary_path = bundle_dir / "summary.json"
    summary_path.write_text(json.dumps(to_builtin(summary), indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def _bundle_config(base_config: SimulationConfig, *, target_mode: str, guidance_mode: str) -> SimulationConfig:
    return SimulationConfig(
        target_initial_position=base_config.target_initial_position,
        interceptor_initial_position=base_config.interceptor_initial_position,
        interceptor_initial_velocity=base_config.interceptor_initial_velocity,
        constraints=base_config.constraints,
        target_velocity=base_config.target_velocity,
        target_speed_mps=base_config.target_speed_mps,
        target_heading_rad=base_config.target_heading_rad,
        target_vertical_speed_mps=base_config.target_vertical_speed_mps,
        target_behavior=replace(base_config.target_behavior, mode=target_mode),
        guidance=replace(base_config.guidance, mode=guidance_mode),
        dt=base_config.dt,
        horizon_s=base_config.horizon_s,
        intercept_radius_m=base_config.intercept_radius_m,
    )


def plot_evasion_scenario_guidance_comparison(
    output_path: Path,
    scenario_name: str,
    guidance_results: dict[str, SimulationResult],
    plot_config: PlotConfig,
) -> Path:
    colors = {"pure": PALETTE["secondary"], "predictive": PALETTE["primary"], "pn": PALETTE["accent"]}
    ordered_modes = ("predictive", "pure", "pn")
    figure = plt.figure(figsize=(16.0, 9.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(2.1, 1.0))

    all_points = np.vstack(
        [
            np.vstack([result.target_position_m[:, :2], result.interceptor_position_m[:, :2]])
            for result in guidance_results.values()
        ]
    )
    x_limits = (float(np.min(all_points[:, 0])), float(np.max(all_points[:, 0])))
    y_limits = (float(np.min(all_points[:, 1])), float(np.max(all_points[:, 1])))

    distance_axis = figure.add_subplot(grid[1, :])
    for column_index, mode in enumerate(ordered_modes):
        result = guidance_results[mode]
        axis = figure.add_subplot(grid[0, column_index])
        axis.plot(
            result.target_position_m[:, 0],
            result.target_position_m[:, 1],
            color=PALETTE["danger"],
            linewidth=1.8,
            label="target",
        )
        axis.plot(
            result.interceptor_position_m[:, 0],
            result.interceptor_position_m[:, 1],
            color=colors[mode],
            linewidth=2.1,
            label="interceptor",
        )
        if result.intercepted:
            intercept_index = int(np.argmin(result.distance_m))
            axis.scatter(
                result.interceptor_position_m[intercept_index, 0],
                result.interceptor_position_m[intercept_index, 1],
                color=PALETTE["success"],
                marker="*",
                s=150,
                zorder=5,
                label="intercept",
            )
            title_suffix = f"intercept {result.intercept_time_s:.1f}s"
        else:
            title_suffix = f"miss {np.min(result.distance_m):.1f}m"
        axis.set_title(f"{mode.upper()} | {title_suffix}", fontsize=11)
        axis.set_xlabel("North X [m]")
        axis.set_ylabel("West Y [m]")
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)
        axis.legend(loc="best", fontsize=8)

        distance_axis.plot(result.time_s, result.distance_m, color=colors[mode], linewidth=2.0, label=mode)

    distance_axis.set_xlabel("Time [s]")
    distance_axis.set_ylabel("Distance [m]")
    distance_axis.set_title(f"{scenario_name.replace('_', ' ').title()} Distance Trace")
    distance_axis.legend(loc="upper right", ncols=3)
    add_note(
        distance_axis,
        "Each top panel keeps the same target behavior but swaps the guidance law.\n"
        "This makes the success or miss mechanism visible without hiding the baseline comparison plots.",
        y=1.16,
    )
    figure.suptitle(f"Q2 {scenario_name.replace('_', ' ').title()} Scenario Across Guidance Modes", fontsize=15)
    close_and_save(figure, output_path, dpi=plot_config.dpi)
    return output_path


def plot_los_geometry_traces(output_path: Path, nominal_results: dict[str, SimulationResult], plot_config: PlotConfig) -> Path:
    """Show LOS rate, closing speed, and lead angle over time for each guidance mode."""

    colors = {"pure": PALETTE["secondary"], "predictive": PALETTE["primary"], "pn": PALETTE["accent"]}
    figure, axes = plt.subplots(3, 1, figsize=(15.5, 10.5), sharex=True, constrained_layout=True)

    for mode, result in nominal_results.items():
        color = colors.get(mode, PALETTE["neutral"])
        axes[0].plot(result.time_s, result.closing_speed_mps, color=color, linewidth=1.5, label=mode)
        axes[1].plot(result.time_s, result.los_rate_norm_radps, color=color, linewidth=1.5, label=mode)
        # Lead angle: angle between interceptor velocity and LOS
        relative_pos = result.target_position_m - result.interceptor_position_m
        interceptor_speed = np.linalg.norm(result.interceptor_velocity_mps, axis=1)
        range_m = np.linalg.norm(relative_pos, axis=1)
        cos_lead = np.sum(result.interceptor_velocity_mps * relative_pos, axis=1) / np.maximum(interceptor_speed * range_m, 1e-9)
        lead_angle_deg = np.rad2deg(np.arccos(np.clip(cos_lead, -1, 1)))
        axes[2].plot(result.time_s, lead_angle_deg, color=color, linewidth=1.5, label=mode)

    axes[0].set_ylabel("Closing Speed [m/s]")
    axes[0].set_title("LOS Geometry Traces")
    axes[0].set_xlabel("Time [s]")
    axes[0].legend(loc="upper right")
    axes[1].set_ylabel("LOS Rate [rad/s]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(loc="upper right")
    axes[2].set_ylabel("Lead Angle [deg]")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend(loc="upper right")
    add_note(axes[2], "Lead angle = angle between interceptor velocity and LOS to target.\n"
             "Lower closing → approach phase. Higher LOS rate → more steering needed.")
    figure.suptitle("Q2 LOS Geometry Comparison Across Guidance Modes", fontsize=15)
    close_and_save(figure, output_path, dpi=plot_config.dpi)
    return output_path


def plot_guidance_3d_comparison(output_path: Path, nominal_results: dict[str, SimulationResult], plot_config: PlotConfig) -> Path:
    """Side-by-side 3D trajectory comparison for all three guidance modes."""

    colors_i = {"pure": PALETTE["secondary"], "predictive": PALETTE["primary"], "pn": PALETTE["accent"]}
    modes = list(nominal_results.keys())
    figure = plt.figure(figsize=(17, 6.5), constrained_layout=True)

    all_points = np.vstack([np.vstack([r.target_position_m, r.interceptor_position_m]) for r in nominal_results.values()])
    for idx, mode in enumerate(modes):
        axis = figure.add_subplot(1, len(modes), idx + 1, projection="3d")
        result = nominal_results[mode]
        axis.plot(result.target_position_m[:, 0], result.target_position_m[:, 1], result.target_position_m[:, 2],
                  color=PALETTE["danger"], linewidth=1.4, alpha=0.7, label="target")
        axis.plot(result.interceptor_position_m[:, 0], result.interceptor_position_m[:, 1], result.interceptor_position_m[:, 2],
                  color=colors_i[mode], linewidth=2.0, label="interceptor")
        if result.intercepted:
            idx_min = int(np.argmin(result.distance_m))
            axis.scatter(*result.interceptor_position_m[idx_min], color=PALETTE["success"], marker="*", s=150)
        set_equal_3d_axes(axis, all_points)
        axis.set_xlabel("North X [m]")
        axis.set_ylabel("West Y [m]")
        axis.set_zlabel("Up Z [m]")
        title_extra = f" | t={result.intercept_time_s:.1f}s" if result.intercepted else " | MISS"
        axis.set_title(f"{mode.upper()}{title_extra}", fontsize=11)
        axis.legend(loc="upper left", fontsize=8)
    figure.suptitle("Q2 3D Trajectory Comparison", fontsize=15)
    close_and_save(figure, output_path, dpi=plot_config.dpi)
    return output_path


def create_q2_animation(output_path: Path, result: SimulationResult, mode_name: str, plot_config: PlotConfig) -> None:
    """Render a 3D animated GIF of the interception geometry."""

    figure = plt.figure(figsize=(9, 7), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    
    t_m = result.target_position_m
    i_m = result.interceptor_position_m
    all_points = np.vstack([t_m, i_m])
    
    # Static weak ghost trails
    axis.plot(t_m[:, 0], t_m[:, 1], t_m[:, 2], color=PALETTE["danger"], alpha=0.25, linewidth=1, label="target trail")
    axis.plot(i_m[:, 0], i_m[:, 1], i_m[:, 2], color=PALETTE["primary"], alpha=0.25, linewidth=1, label="interceptor trail")
    
    # Active markers
    target_marker, = axis.plot([], [], [], marker="o", color=PALETTE["danger"], markersize=7, linestyle="None", label="Target")
    interceptor_marker, = axis.plot([], [], [], marker="v", color=PALETTE["primary"], markersize=7, linestyle="None", label="Interceptor")
    los_line, = axis.plot([], [], [], color=PALETTE["accent"], linewidth=1.5, linestyle="--", alpha=0.8, label="LOS line")
    
    set_equal_3d_axes(axis, all_points)
    axis.set_xlabel("North X [m]")
    axis.set_ylabel("West Y [m]")
    axis.set_zlabel("Up Z [m]")
    axis.legend(loc="upper left")
    title = figure.suptitle(f"Q2 Interception: {mode_name.upper()}", fontsize=14)
    
    stride = max(1, len(result.time_s) // 80)
    indices = list(range(0, len(result.time_s), stride))
    if indices[-1] != len(result.time_s) - 1:
        indices.append(len(result.time_s) - 1)
        
    def update(frame_idx: int):
        idx = indices[frame_idx]
        tx, ty, tz = t_m[idx]
        ix, iy, iz = i_m[idx]
        
        target_marker.set_data_3d([tx], [ty], [tz])
        interceptor_marker.set_data_3d([ix], [iy], [iz])
        los_line.set_data_3d([ix, tx], [iy, ty], [iz, tz])
        
        dist = result.distance_m[idx]
        status = "INTERCEPTED!" if (idx == indices[-1] and result.intercepted) else f"Dist: {dist:.1f}m"
        title.set_text(f"Q2 Integration ({mode_name.upper()}) | t = {result.time_s[idx]:.1f} s | {status}")
        return target_marker, interceptor_marker, los_line, title

    animation = FuncAnimation(figure, update, frames=len(indices), interval=60, blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=12), dpi=plot_config.animation_dpi)
    plt.close(figure)
