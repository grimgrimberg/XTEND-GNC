from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

from .plotting import PALETTE, PlotConfig, add_note, close_and_save, set_equal_3d_axes
from .q1_pipeline import FrameConvention, FrameConventionSelection, GuidanceStreams


def create_q1_visuals(
    *,
    output_dir: Path,
    streams: GuidanceStreams,
    stream_summary: pd.DataFrame,
    sync_offset_summary: pd.DataFrame,
    synchronized: pd.DataFrame,
    convention_selection: FrameConventionSelection,
    selected_convention: FrameConvention,
    rate_candidates: dict[str, dict[str, np.ndarray]],
    rate_metrics: pd.DataFrame,
    selected_rate_method: str,
    bundle_points: pd.DataFrame,
    ground_footprint: pd.DataFrame,
    render_animation: bool,
    plot_config: PlotConfig,
) -> list[Path]:
    artifacts = [
        plot_stream_overview(output_dir / "q1_sync_diagnostics.png", streams, stream_summary, sync_offset_summary, plot_config),
        plot_convention_diagnostics(output_dir / "q1_frame_diagnostics.png", convention_selection, plot_config),
        plot_sync_quality(output_dir / "q1_sync_quality.png", synchronized, plot_config),
        plot_world_angles(output_dir / "q1_world_angles.png", synchronized, selected_convention, plot_config),
        plot_rate_estimates(
            output_dir / "q1_rate_estimates.png",
            synchronized,
            rate_candidates,
            rate_metrics,
            selected_rate_method,
            plot_config,
        ),
        plot_rate_comparison_overlay(
            output_dir / "q1_rate_raw_vs_clean.png",
            synchronized,
            rate_candidates,
            selected_rate_method,
            plot_config,
        ),
        plot_kalman_tracking(
            output_dir / "q1_kalman_tracking.png",
            synchronized,
            rate_candidates,
            plot_config,
        ),
        plot_camera_fov_reticle(output_dir / "q1_camera_fov.png", synchronized, plot_config),
        plot_bundle_residual_histogram(output_dir / "q1_bundle_residuals.png", bundle_points, plot_config),
        plot_geometry_topdown(output_dir / "q1_geometry_topdown.png", synchronized, bundle_points, ground_footprint, plot_config),
        plot_geometry_3d(output_dir / "q1_geometry_3d.png", synchronized, bundle_points, plot_config),
    ]
    if render_animation:
        animation_path = output_dir / "q1_geometry_animation.gif"
        create_geometry_animation(animation_path, synchronized, ground_footprint, plot_config)
        artifacts.append(animation_path)
    return artifacts


def plot_stream_overview(
    output_path: Path,
    streams: GuidanceStreams,
    stream_summary: pd.DataFrame,
    sync_offset_summary: pd.DataFrame,
    plot_config: PlotConfig,
) -> Path:
    reference_start = float(streams.raw["time"].iloc[0])
    figure = plt.figure(figsize=(15.5, 9.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=[1.2, 1.0])
    axis_raster = figure.add_subplot(grid[0, :])
    axis_offsets = figure.add_subplot(grid[1, 0])
    axis_summary = figure.add_subplot(grid[1, 1])

    stream_offsets = {
        "camera": streams.camera["time"].to_numpy() - reference_start,
        "gimbal": streams.gimbal["time"].to_numpy() - reference_start,
        "nav": streams.nav["time"].to_numpy() - reference_start,
        "empty": streams.empty_rows["time"].to_numpy() - reference_start,
    }
    y_positions = {"camera": 0, "gimbal": 1, "nav": 2, "empty": 3}
    colors = {"camera": PALETTE["primary"], "gimbal": PALETTE["accent"], "nav": PALETTE["success"], "empty": PALETTE["soft"]}
    for name, offsets in stream_offsets.items():
        axis_raster.scatter(offsets, np.full_like(offsets, y_positions[name], dtype=float), s=12 if name != "empty" else 7, alpha=0.9 if name != "empty" else 0.4, color=colors[name], label=name)
    axis_raster.set_yticks(list(y_positions.values()), list(y_positions.keys()))
    axis_raster.set_xlabel("Time Since First Sample [s]")
    axis_raster.set_title("Asynchronous Event Log Structure")
    add_note(axis_raster, "The CSV is not a row-aligned table.\nCamera, gimbal, nav, and placeholder rows are interleaved.\nThat forces an explicit synchronization stage.")
    axis_raster.legend(loc="upper right", ncols=4)

    for source_name, group in sync_offset_summary.groupby("source"):
        axis_offsets.hist(group["offset_ms"], bins=30, alpha=0.7, label=source_name, color=colors.get(source_name, PALETTE["secondary"]))
    axis_offsets.set_xlabel("Nearest Offset To Camera Timestamp [ms]")
    axis_offsets.set_ylabel("Count")
    axis_offsets.set_title("Nearest-Sample Offsets Before Interpolation")
    axis_offsets.legend(loc="upper right")

    axis_summary.axis("off")
    summary_lines = []
    for _, row in stream_summary.iterrows():
        if np.isnan(row.get("median_dt_s", np.nan)):
            summary_lines.append(f"{row['stream']}: {int(row['samples'])} rows")
            continue
        summary_lines.append(
            f"{row['stream']:<12} samples={int(row['samples']):4d}  "
            f"median_dt={1e3 * row['median_dt_s']:.2f} ms  "
            f"nominal_rate={row['nominal_rate_hz']:.2f} Hz"
        )
    axis_summary.text(0.02, 0.98, "\n".join(summary_lines), va="top", ha="left", family="monospace", fontsize=10.0)
    figure.suptitle("Q1 Synchronization Diagnostics", fontsize=15)
    close_and_save(figure, output_path, dpi=plot_config.dpi)
    return output_path


def plot_convention_diagnostics(output_path: Path, selection: FrameConventionSelection, plot_config: PlotConfig) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.8), constrained_layout=True)
    top = selection.candidate_table.head(8).copy()
    labels = [
        "\n".join(
            [
                "quat B->W" if row["quaternion_is_body_to_world"] else "quat W->B",
                "az->R" if row["camera_positive_azimuth_right"] else "az->L",
                "el->U" if row["camera_positive_elevation_up"] else "el->D",
                "g->U" if row["gimbal_positive_pitch_raises"] else "g->D",
            ]
        )
        for _, row in top.iterrows()
    ]
    best_color = [PALETTE["success"]] + [PALETTE["soft"]] * (len(top) - 1)
    axes[0].bar(labels, top["pair_miss_median_m"], color=best_color)
    axes[0].set_xlabel("Convention Candidate")
    axes[0].set_ylabel("Median Ray-Bundle Residual [m]")
    axes[0].set_title("Geometry Coherence")
    axes[1].bar(labels, top["combined_rate_std_deg_s"], color=best_color)
    axes[1].set_xlabel("Convention Candidate")
    axes[1].set_ylabel("Combined Rate Std [deg/s]")
    axes[1].set_title("Angular Roughness")
    axes[2].bar(labels, top["heading_error_median_deg"], color=best_color)
    axes[2].set_xlabel("Convention Candidate")
    axes[2].set_ylabel("Median Forward/Velocity Error [deg]")
    axes[2].set_title("Quaternion Plausibility")
    for axis in axes:
        axis.tick_params(axis="x", labelsize=9)
    axes[0].bar([], [], color=PALETTE["success"], label="best-ranked candidate")
    axes[0].bar([], [], color=PALETTE["soft"], label="other candidates")
    axes[0].legend(loc="upper right")
    add_note(axes[2], "Selection rule:\n1) lowest bundle residual\n2) lowest p90 residual\n3) lowest rate roughness\n4) lowest forward/velocity error")
    figure.suptitle("Frame Convention Comparison", fontsize=15)
    close_and_save(figure, output_path, dpi=plot_config.dpi)
    return output_path


def plot_sync_quality(output_path: Path, synchronized: pd.DataFrame, plot_config: PlotConfig) -> Path:
    figure, axes = plt.subplots(3, 1, figsize=(15.5, 10.0), sharex=True, constrained_layout=True)
    time_s = synchronized["time_s"].to_numpy()
    axes[0].plot(time_s, np.rad2deg(synchronized["held_gimbal_pitch_rad"]), color=PALETTE["soft"], label="zero-order hold")
    axes[0].plot(time_s, np.rad2deg(synchronized["gimbal_pitch_rad"]), color=PALETTE["accent"], label="linear interpolation")
    axes[0].set_ylabel("Pitch [deg]")
    axes[0].set_title("Gimbal Synchronization")
    axes[0].legend(loc="upper right")

    axes[1].plot(time_s, synchronized["held_world_az_deg"], color=PALETTE["soft"], label="nearest-sample fusion")
    axes[1].plot(time_s, synchronized["world_az_deg"], color=PALETTE["primary"], label="interpolated fusion")
    axes[1].set_ylabel("Azimuth [deg]")
    axes[1].set_title("World Azimuth: Hold vs Interpolate")
    axes[1].legend(loc="upper right")

    axes[2].plot(time_s, synchronized["held_world_el_deg"], color=PALETTE["soft"], label="nearest-sample fusion")
    axes[2].plot(time_s, synchronized["world_el_deg"], color=PALETTE["secondary"], label="interpolated fusion")
    axes[2].set_ylabel("Elevation [deg]")
    axes[2].set_xlabel("Time Since First Camera Sample [s]")
    axes[2].set_title("World Elevation: Hold vs Interpolate")
    axes[2].legend(loc="upper right")
    add_note(axes[2], "Only the synchronization policy changes here.\nThis plot makes the interpolation cleanup visible.", y=1.18)
    figure.suptitle("Synchronization Quality Comparison", fontsize=15)
    close_and_save(figure, output_path, dpi=plot_config.dpi)
    return output_path


def plot_world_angles(output_path: Path, synchronized: pd.DataFrame, convention: FrameConvention, plot_config: PlotConfig) -> Path:
    figure, axes = plt.subplots(2, 1, figsize=(15.5, 8.5), sharex=True, constrained_layout=True)
    time_s = synchronized["time_s"].to_numpy()
    axes[0].plot(time_s, synchronized["world_az_deg"], color=PALETTE["primary"], linewidth=2.1, label="world azimuth")
    axes[0].set_ylabel("Azimuth [deg]")
    axes[0].set_title("World-Frame Target Azimuth")
    axes[0].legend(loc="upper right")
    add_note(axes[0], f"Selected convention:\n{convention.label()}")

    axes[1].plot(time_s, synchronized["world_el_deg"], color=PALETTE["secondary"], linewidth=2.1, label="world elevation")
    axes[1].set_ylabel("Elevation [deg]")
    axes[1].set_xlabel("Time Since First Camera Sample [s]")
    axes[1].set_title("World-Frame Target Elevation")
    axes[1].legend(loc="upper right")
    add_note(axes[1], "Time base choice: camera timestamps.\nReason: LOS is only directly observed when the camera reports angles.", y=1.16)
    figure.suptitle("Q1 World Angles", fontsize=15)
    close_and_save(figure, output_path, dpi=plot_config.dpi)
    return output_path


def plot_rate_estimates(
    output_path: Path,
    synchronized: pd.DataFrame,
    rate_candidates: dict[str, dict[str, np.ndarray]],
    rate_metrics: pd.DataFrame,
    selected_rate_method: str,
    plot_config: PlotConfig,
) -> Path:
    figure, axes = plt.subplots(2, 1, figsize=(15.5, 9.5), sharex=True, constrained_layout=True)
    styles = {
        "gradient": {"color": PALETTE["soft"], "linewidth": 1.1},
        "savgol": {"color": PALETTE["accent"], "linewidth": 1.4},
        "local_polynomial": {"color": PALETTE["success"], "linewidth": 2.0},
        "spline": {"color": PALETTE["danger"], "linewidth": 1.2, "alpha": 0.75},
        "kalman_cv": {"color": PALETTE["secondary"], "linewidth": 1.8, "linestyle": "--"},
    }
    time_s = synchronized["time_s"].to_numpy()
    for method_name, rates in rate_candidates.items():
        label = method_name.replace("_", " ").title()
        axes[0].plot(time_s, rates["azimuth"], label=label, **styles[method_name])
        axes[1].plot(time_s, rates["elevation"], label=label, **styles[method_name])
    axes[0].plot(time_s, synchronized["az_rate_kalman_cv_deg_s"], label="Kalman Cv", **styles["kalman_cv"])
    axes[1].plot(time_s, synchronized["el_rate_kalman_cv_deg_s"], label="Kalman Cv", **styles["kalman_cv"])
    axes[0].set_ylabel("Azimuth Rate [deg/s]")
    axes[0].set_title("Azimuth Rate Estimators")
    axes[0].legend(loc="upper right", ncols=2)
    axes[1].set_ylabel("Elevation Rate [deg/s]")
    axes[1].set_xlabel("Time Since First Camera Sample [s]")
    axes[1].set_title("Elevation Rate Estimators")
    axes[1].legend(loc="upper right", ncols=2)
    for axis, channel in zip(axes, ("azimuth", "elevation")):
        channel_metrics = rate_metrics.loc[rate_metrics["channel"] == channel]
        summary_lines = []
        for _, row in channel_metrics.iterrows():
            marker = "*" if row["method"] == selected_rate_method else "-"
            summary_lines.append(
                f"{marker} {row['method']:<17} noise={row['noise_proxy_deg_per_s']:.2f}  "
                f"lag={1e3 * row['lag_proxy_seconds']:.1f} ms  "
                f"rmse={row['reconstruction_rmse_deg']:.2f}"
            )
        add_note(axis, "\n".join(summary_lines), x=0.01, y=0.98)
    figure.suptitle("Q1 Rate Estimator Comparison", fontsize=15)
    close_and_save(figure, output_path, dpi=plot_config.dpi)
    return output_path


def plot_kalman_tracking(
    output_path: Path,
    synchronized: pd.DataFrame,
    rate_candidates: dict[str, dict[str, np.ndarray]],
    plot_config: PlotConfig,
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(16.0, 9.5), sharex="col", constrained_layout=True)
    time_s = synchronized["time_s"].to_numpy()

    axes[0, 0].plot(time_s, synchronized["world_az_deg"], color=PALETTE["soft"], linewidth=0.9, label="raw azimuth")
    axes[0, 0].plot(time_s, synchronized["world_az_kalman_deg"], color=PALETTE["secondary"], linewidth=1.8, label="Kalman azimuth")
    axes[0, 0].set_ylabel("Angle [deg]")
    axes[0, 0].set_title("Azimuth Tracking")
    axes[0, 0].legend(loc="upper right")

    axes[1, 0].plot(time_s, rate_candidates["gradient"]["azimuth"], color=PALETTE["soft"], linewidth=0.9, label="gradient")
    axes[1, 0].plot(time_s, synchronized["az_rate_kalman_cv_deg_s"], color=PALETTE["secondary"], linewidth=1.8, label="Kalman rate")
    axes[1, 0].set_ylabel("Rate [deg/s]")
    axes[1, 0].set_xlabel("Time Since First Camera Sample [s]")
    axes[1, 0].legend(loc="upper right")

    axes[0, 1].plot(time_s, synchronized["world_el_deg"], color=PALETTE["soft"], linewidth=0.9, label="raw elevation")
    axes[0, 1].plot(time_s, synchronized["world_el_kalman_deg"], color=PALETTE["secondary"], linewidth=1.8, label="Kalman elevation")
    axes[0, 1].set_ylabel("Angle [deg]")
    axes[0, 1].set_title("Elevation Tracking")
    axes[0, 1].legend(loc="upper right")

    axes[1, 1].plot(time_s, rate_candidates["gradient"]["elevation"], color=PALETTE["soft"], linewidth=0.9, label="gradient")
    axes[1, 1].plot(time_s, synchronized["el_rate_kalman_cv_deg_s"], color=PALETTE["secondary"], linewidth=1.8, label="Kalman rate")
    axes[1, 1].set_ylabel("Rate [deg/s]")
    axes[1, 1].set_xlabel("Time Since First Camera Sample [s]")
    axes[1, 1].legend(loc="upper right")

    add_note(
        axes[1, 1],
        "Kalman candidate uses a causal constant-velocity state model.\n"
        "Raw angles/rates stay visible so the noise-versus-latency trade is inspectable.",
    )
    figure.suptitle("Q1 Kalman Tracking Diagnostic", fontsize=15)
    close_and_save(figure, output_path, dpi=plot_config.dpi)
    return output_path


def plot_geometry_topdown(
    output_path: Path,
    synchronized: pd.DataFrame,
    bundle_points: pd.DataFrame,
    ground_footprint: pd.DataFrame,
    plot_config: PlotConfig,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(16.5, 6.8), constrained_layout=True)
    positions = synchronized[["position_x_m", "position_y_m"]].to_numpy()
    axes[0].plot(positions[:, 0], positions[:, 1], color=PALETTE["primary"], linewidth=2.0, label="drone path")
    if len(ground_footprint):
        axes[0].scatter(ground_footprint["x_m"], ground_footprint["y_m"], c=ground_footprint["time_s"], cmap="viridis", s=16, alpha=0.75, label="optical-axis ground hit")
    sample_indices = np.linspace(0, len(synchronized) - 1, 9, dtype=int)
    for index in sample_indices:
        origin = positions[index]
        los = synchronized.loc[index, ["world_los_x", "world_los_y"]].to_numpy(dtype=float)
        los = los / max(np.linalg.norm(los), 1e-9)
        body = synchronized.loc[index, ["forward_world_x", "forward_world_y"]].to_numpy(dtype=float)
        body = body / max(np.linalg.norm(body), 1e-9)
        axes[0].arrow(origin[0], origin[1], 18.0 * body[0], 18.0 * body[1], width=0.6, color=PALETTE["neutral"], alpha=0.45)
        axes[0].arrow(origin[0], origin[1], 40.0 * los[0], 40.0 * los[1], width=0.8, color=PALETTE["accent"], alpha=0.55)
    axes[0].plot([], [], color=PALETTE["neutral"], linewidth=2.0, label="body forward axis")
    axes[0].plot([], [], color=PALETTE["accent"], linewidth=2.0, label="sample LOS rays")
    axes[0].set_xlabel("North [m]")
    axes[0].set_ylabel("West [m]")
    axes[0].set_title("Top-Down Geometry")
    axes[0].legend(loc="best")
    add_note(axes[0], "Gray arrows: body forward axis.\nPurple arrows: LOS direction.\nGround hits use the z=0 plane only as a diagnostic reference.")

    axes[1].plot(synchronized["time_s"], synchronized["position_z_m"], color=PALETTE["primary"], label="drone altitude")
    if len(bundle_points):
        axes[1].scatter(bundle_points["time_s"], bundle_points["z_m"], c=bundle_points["residual_rms_m"], cmap="magma_r", s=26, label="bundle-estimated point altitude")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Up [m]")
    axes[1].set_title("Altitude And Bundle Geometry")
    axes[1].legend(loc="best")
    figure.suptitle("Q1 Geometry Diagnostics", fontsize=15)
    close_and_save(figure, output_path, dpi=plot_config.dpi)
    return output_path


def plot_geometry_3d(output_path: Path, synchronized: pd.DataFrame, bundle_points: pd.DataFrame, plot_config: PlotConfig) -> Path:
    figure = plt.figure(figsize=(12.0, 9.5), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    trajectory = synchronized[["position_x_m", "position_y_m", "position_z_m"]].to_numpy()
    axis.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color=PALETTE["primary"], linewidth=2.2, label="drone")
    if len(bundle_points):
        scatter = axis.scatter(bundle_points["x_m"], bundle_points["y_m"], bundle_points["z_m"], c=bundle_points["residual_rms_m"], cmap="viridis_r", s=32, label="local ray-bundle points")
        figure.colorbar(scatter, ax=axis, shrink=0.7, pad=0.08, label="Bundle Residual [m]")
    sample_indices = np.linspace(0, len(synchronized) - 1, 8, dtype=int)
    for index in sample_indices:
        origin = trajectory[index]
        los = synchronized.loc[index, ["world_los_x", "world_los_y", "world_los_z"]].to_numpy(dtype=float)
        axis.plot([origin[0], origin[0] + 55.0 * los[0]], [origin[1], origin[1] + 55.0 * los[1]], [origin[2], origin[2] + 55.0 * los[2]], color=PALETTE["accent"], alpha=0.55)
    axis.plot([], [], [], color=PALETTE["accent"], linewidth=2.0, label="sample LOS rays")
    point_cloud = np.vstack([trajectory, bundle_points[["x_m", "y_m", "z_m"]].to_numpy()]) if len(bundle_points) else trajectory
    set_equal_3d_axes(axis, point_cloud)
    axis.set_xlabel("North [m]")
    axis.set_ylabel("West [m]")
    axis.set_zlabel("Up [m]")
    axis.set_title("3D LOS Geometry")
    axis.legend(loc="upper left")
    figure.suptitle("Q1 3D View Of What The Drone Is Looking At", fontsize=15)
    close_and_save(figure, output_path, dpi=plot_config.dpi)
    return output_path


def plot_rate_comparison_overlay(
    output_path: Path,
    synchronized: pd.DataFrame,
    rate_candidates: dict[str, dict[str, np.ndarray]],
    selected_rate_method: str,
    plot_config: PlotConfig,
) -> Path:
    """Overlay the raw gradient estimate against the selected rate estimate."""

    figure, axes = plt.subplots(2, 1, figsize=(15.5, 8.5), sharex=True, constrained_layout=True)
    time_s = synchronized["time_s"].to_numpy()
    selected_label = selected_rate_method.replace("_", " ")
    for axis, channel, ylabel in (
        (axes[0], "azimuth", "Azimuth Rate [deg/s]"),
        (axes[1], "elevation", "Elevation Rate [deg/s]"),
    ):
        raw = rate_candidates["gradient"][channel]
        clean = rate_candidates[selected_rate_method][channel]
        axis.fill_between(time_s, raw, clean, alpha=0.18, color=PALETTE["danger"], label="noise removed")
        axis.plot(time_s, raw, color=PALETTE["soft"], linewidth=0.8, label="gradient (raw)")
        axis.plot(time_s, clean, color=PALETTE["success"], linewidth=1.6, label=f"{selected_label} (selected)")
        axis.set_ylabel(ylabel)
        axis.legend(loc="upper right")
    axes[0].set_title("Raw vs Cleaned Rate — Direct Noise Comparison")
    axes[1].set_xlabel("Time Since First Camera Sample [s]")
    add_note(
        axes[1],
        "The shaded region shows the gap between the raw gradient\n"
        f"and the selected {selected_label} estimate.",
    )
    figure.suptitle("Q1 Rate Estimator: Noise Reduction Overlay", fontsize=15)
    close_and_save(figure, output_path, dpi=plot_config.dpi)
    return output_path


def plot_camera_fov_reticle(output_path: Path, synchronized: pd.DataFrame, plot_config: PlotConfig) -> Path:
    """Show camera-frame target position as a reticle view ('what the drone sees')."""

    figure, axes = plt.subplots(1, 2, figsize=(15.5, 7.0), constrained_layout=True)
    target_x = synchronized["target_x_deg"].to_numpy()
    target_y = synchronized["target_y_deg"].to_numpy()
    time_s = synchronized["time_s"].to_numpy()

    # Left panel: scatter in camera frame with time coloring
    scatter = axes[0].scatter(target_x, target_y, c=time_s, cmap="viridis", s=10, alpha=0.8)
    axes[0].axhline(0.0, color=PALETTE["neutral"], linewidth=0.6, linestyle="--", alpha=0.5, label="boresight crosshair")
    axes[0].axvline(0.0, color=PALETTE["neutral"], linewidth=0.6, linestyle="--", alpha=0.5)
    # Draw concentric FOV rings
    for radius_index, radius in enumerate((5, 10, 15, 20, 25)):
        circle = plt.Circle((0, 0), radius, fill=False, color=PALETTE["grid"], linewidth=0.5, linestyle=":")
        axes[0].add_patch(circle)
        if radius_index == 0:
            circle.set_label("FOV guide rings")
    axes[0].set_xlabel("Camera Azimuth [deg]")
    axes[0].set_ylabel("Camera Elevation [deg]")
    axes[0].set_title("Camera-Frame Target Tracking")
    axes[0].set_aspect("equal", adjustable="datalim")
    figure.colorbar(scatter, ax=axes[0], label="Time [s]", shrink=0.85)
    axes[0].plot([], [], marker="o", linestyle="None", color=PALETTE["primary"], label="target observations")
    axes[0].legend(loc="upper right")
    add_note(axes[0], "Each dot is a camera observation.\nColor shows time progression.\nCrosshairs mark the optical axis center.")

    # Right panel: target offset magnitude over time
    offset_magnitude = np.sqrt(target_x**2 + target_y**2)
    axes[1].plot(time_s, offset_magnitude, color=PALETTE["primary"], linewidth=1.4, label="off-axis magnitude")
    axes[1].fill_between(time_s, 0, offset_magnitude, color=PALETTE["primary"], alpha=0.12, label="offset envelope")
    axes[1].set_xlabel("Time Since First Camera Sample [s]")
    axes[1].set_ylabel("Off-Axis Angle [deg]")
    axes[1].set_title("Target Offset From Boresight")
    axes[1].legend(loc="upper right")
    add_note(axes[1], "How far the target is from the camera center over time.\nLarge offsets may indicate rapid slew or target at FOV edge.")
    figure.suptitle("Q1 Camera-Frame View: What The Drone Sees", fontsize=15)
    close_and_save(figure, output_path, dpi=plot_config.dpi)
    return output_path


def plot_bundle_residual_histogram(output_path: Path, bundle_points: pd.DataFrame, plot_config: PlotConfig) -> Path:
    """Show the distribution of ray-bundle residuals as a quality diagnostic."""

    if len(bundle_points) == 0:
        figure, axis = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
        axis.text(0.5, 0.5, "No bundle points available", transform=axis.transAxes, ha="center", va="center")
        close_and_save(figure, output_path, dpi=plot_config.dpi)
        return output_path

    residuals = bundle_points["residual_rms_m"].to_numpy()
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), constrained_layout=True)

    axes[0].hist(residuals, bins=35, color=PALETTE["primary"], edgecolor=PALETTE["ink"], alpha=0.85)
    median_val = float(np.median(residuals))
    p90_val = float(np.percentile(residuals, 90))
    axes[0].axvline(median_val, color=PALETTE["success"], linewidth=2, linestyle="--", label=f"median = {median_val:.3f} m")
    axes[0].axvline(p90_val, color=PALETTE["warning"], linewidth=2, linestyle=":", label=f"p90 = {p90_val:.3f} m")
    axes[0].set_xlabel("Bundle Residual RMS [m]")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Ray-Bundle Fit Quality Distribution")
    axes[0].legend(loc="upper right")

    axes[1].scatter(bundle_points["time_s"], residuals, c=bundle_points["condition_number"], cmap="magma_r", s=18, alpha=0.8, label="bundle residual samples")
    axes[1].axhline(median_val, color=PALETTE["success"], linewidth=1.2, linestyle="--", alpha=0.6, label="median residual")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Bundle Residual [m]")
    axes[1].set_title("Residuals Over Time")
    axes[1].legend(loc="upper right")
    add_note(axes[1], "Color = condition number of the normal matrix.\nHigher condition → less reliable intersection point.")
    figure.suptitle("Q1 Bundle Geometry Quality", fontsize=15)
    close_and_save(figure, output_path, dpi=plot_config.dpi)
    return output_path


def create_geometry_animation(output_path: Path, synchronized: pd.DataFrame, ground_footprint: pd.DataFrame, plot_config: PlotConfig) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), constrained_layout=True)
    positions = synchronized[["position_x_m", "position_y_m"]].to_numpy()
    time_s = synchronized["time_s"].to_numpy()
    frame_indices = np.linspace(0, len(synchronized) - 1, min(80, len(synchronized)), dtype=int)

    axes[0].plot(positions[:, 0], positions[:, 1], color=PALETTE["soft"], alpha=0.4, label="drone path")
    if len(ground_footprint):
        axes[0].scatter(ground_footprint["x_m"], ground_footprint["y_m"], color=PALETTE["soft"], s=6, alpha=0.25, label="ground-hit trace")
    drone_marker, = axes[0].plot([], [], marker="o", color=PALETTE["primary"], markersize=7, label="drone")
    los_line, = axes[0].plot([], [], color=PALETTE["accent"], linewidth=2.0, label="LOS ray")
    axes[0].set_xlabel("North [m]")
    axes[0].set_ylabel("West [m]")
    axes[0].set_title("Top-Down LOS Sweep")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend(loc="upper right")

    axes[1].plot(time_s, synchronized["world_az_deg"], color=PALETTE["primary"], label="azimuth")
    axes[1].plot(time_s, synchronized["world_el_deg"], color=PALETTE["secondary"], label="elevation")
    time_cursor = axes[1].axvline(time_s[0], color=PALETTE["danger"], linestyle="--")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Angle [deg]")
    axes[1].set_title("Angle Traces")
    axes[1].legend(loc="upper right")
    title = figure.suptitle("Q1 Geometry Animation", fontsize=14)

    def update(frame_number: int):
        index = frame_indices[frame_number]
        origin = positions[index]
        los = synchronized.loc[index, ["world_los_x", "world_los_y"]].to_numpy(dtype=float)
        los = los / max(np.linalg.norm(los), 1e-9)
        endpoint = origin + 55.0 * los
        drone_marker.set_data([origin[0]], [origin[1]])
        los_line.set_data([origin[0], endpoint[0]], [origin[1], endpoint[1]])
        time_cursor.set_xdata([time_s[index], time_s[index]])
        title.set_text(f"Q1 Geometry Animation | t = {time_s[index]:.2f} s")
        return drone_marker, los_line, time_cursor, title

    animation = FuncAnimation(figure, update, frames=len(frame_indices), interval=60, blit=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=12), dpi=plot_config.animation_dpi)
    plt.close(figure)
