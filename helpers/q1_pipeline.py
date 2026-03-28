from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

from .paths import clean_output_dir
from .plotting import PlotConfig, configure_matplotlib
from .utils import to_builtin


NAV_COLUMNS = [
    "orientation_w",
    "orientation_x",
    "orientation_y",
    "orientation_z",
    "position_x",
    "position_y",
    "position_z",
    "angular_x",
    "angular_y",
    "angular_z",
    "vel_x",
    "vel_y",
    "vel_z",
]
ORIENTATION_COLUMNS = ["orientation_w", "orientation_x", "orientation_y", "orientation_z"]
# The prompt asks for rate estimates with low noise and low latency. Keep those
# two terms dominant, then use reconstruction / edge / holdout metrics only as
# secondary guardrails so the selected derivative still behaves plausibly.
RATE_SELECTION_WEIGHTS = {
    "lag_abs_mean_s": 0.40,
    "noise_proxy_mean_deg_per_s": 0.40,
    "reconstruction_rmse_mean_deg": 0.10,
    "edge_proxy_mean_deg_per_s": 0.05,
    "holdout_rmse_mean_deg": 0.05,
}


@dataclass(frozen=True)
class FrameConvention:
    """Camera/gimbal/body/world sign choices for the Q1 rotation chain."""

    quaternion_is_body_to_world: bool = True
    camera_positive_azimuth_right: bool = True
    camera_positive_elevation_up: bool = True
    gimbal_positive_pitch_raises: bool = False

    def label(self) -> str:
        quaternion = "quat:B->W" if self.quaternion_is_body_to_world else "quat:W->B"
        azimuth = "az:right" if self.camera_positive_azimuth_right else "az:left"
        elevation = "el:up" if self.camera_positive_elevation_up else "el:down"
        gimbal = "gimbal:up" if self.gimbal_positive_pitch_raises else "gimbal:down"
        return f"{quaternion} | {azimuth} | {elevation} | {gimbal}"


@dataclass(frozen=True)
class GuidanceStreams:
    """Classified views of the sparse event-log CSV."""

    raw: pd.DataFrame
    camera: pd.DataFrame
    gimbal: pd.DataFrame
    nav: pd.DataFrame
    empty_rows: pd.DataFrame
    unclassified_rows: pd.DataFrame


@dataclass(frozen=True)
class FrameConventionSelection:
    """Ranking of plausible frame/sign conventions."""

    best_convention: FrameConvention
    candidate_table: pd.DataFrame


def split_guidance_streams(dataframe: pd.DataFrame) -> GuidanceStreams:
    """Split the sparse CSV into camera, gimbal, navigation, and empty-row streams."""

    dataframe = dataframe.copy()
    for column_name in ["target_x_deg", "target_y_deg", "gimbal_pitch_rad", *NAV_COLUMNS]:
        if column_name not in dataframe.columns:
            dataframe[column_name] = np.nan
    dataframe = dataframe.sort_values("time").reset_index(drop=True)
    camera_mask = dataframe[["target_x_deg", "target_y_deg"]].notna().all(axis=1)
    gimbal_mask = dataframe[["gimbal_pitch_rad"]].notna().all(axis=1)
    nav_mask = dataframe[ORIENTATION_COLUMNS].notna().all(axis=1)
    empty_mask = dataframe.drop(columns=["time"]).isna().all(axis=1)
    classified_mask = camera_mask | gimbal_mask | nav_mask | empty_mask
    unclassified_mask = ~classified_mask
    return GuidanceStreams(
        raw=dataframe,
        camera=_sorted_unique(dataframe.loc[camera_mask, ["time", "target_x_deg", "target_y_deg"]]),
        gimbal=_sorted_unique(dataframe.loc[gimbal_mask, ["time", "gimbal_pitch_rad"]]),
        nav=_sorted_unique(dataframe.loc[nav_mask, ["time", *NAV_COLUMNS]]),
        empty_rows=dataframe.loc[empty_mask, ["time"]].reset_index(drop=True),
        unclassified_rows=dataframe.loc[unclassified_mask].reset_index(drop=True),
    )


def camera_angles_to_los(
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    convention: FrameConvention | None = None,
) -> np.ndarray:
    """Convert camera-frame az/el observations into unit LOS vectors."""

    convention = convention or FrameConvention()
    azimuth_sign = 1.0 if convention.camera_positive_azimuth_right else -1.0
    elevation_sign = 1.0 if convention.camera_positive_elevation_up else -1.0
    azimuth_rad = np.deg2rad(np.asarray(azimuth_deg, dtype=float) * azimuth_sign)
    elevation_rad = np.deg2rad(np.asarray(elevation_deg, dtype=float) * elevation_sign)
    los_camera = np.column_stack(
        [
            np.cos(elevation_rad) * np.cos(azimuth_rad),
            np.cos(elevation_rad) * np.sin(azimuth_rad),
            np.sin(elevation_rad),
        ]
    )
    return los_camera / np.linalg.norm(los_camera, axis=1, keepdims=True)


def los_body_from_camera(
    los_camera: np.ndarray,
    gimbal_pitch_rad: np.ndarray,
    convention: FrameConvention | None = None,
) -> np.ndarray:
    """Rotate camera LOS vectors through the gimbal pitch into the body frame."""

    convention = convention or FrameConvention()
    pitch_sign = -1.0 if convention.gimbal_positive_pitch_raises else 1.0
    rotations = Rotation.from_rotvec(
        np.column_stack(
            [
                np.zeros_like(gimbal_pitch_rad, dtype=float),
                pitch_sign * np.asarray(gimbal_pitch_rad, dtype=float),
                np.zeros_like(gimbal_pitch_rad, dtype=float),
            ]
        )
    )
    return rotations.apply(np.asarray(los_camera, dtype=float))


def los_to_world_angles_nwu(los_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world-frame LOS vectors into NWU azimuth/elevation angles."""

    los_world = np.asarray(los_world, dtype=float)
    azimuth_deg = np.rad2deg(np.arctan2(los_world[:, 1], los_world[:, 0]))
    elevation_deg = np.rad2deg(np.arctan2(los_world[:, 2], np.linalg.norm(los_world[:, :2], axis=1)))
    return azimuth_deg, elevation_deg


def estimate_angle_rate(
    time_s: np.ndarray,
    angle_deg: np.ndarray,
    *,
    method: str = "local_polynomial",
    unwrap: bool = False,
) -> np.ndarray:
    """Estimate angular rates with several comparison methods."""

    time_s = np.asarray(time_s, dtype=float)
    angle_deg = np.asarray(angle_deg, dtype=float)
    angle_work = np.rad2deg(np.unwrap(np.deg2rad(angle_deg))) if unwrap else angle_deg.copy()
    if len(time_s) < 5:
        return np.gradient(angle_work, time_s)
    if method == "gradient":
        return np.gradient(angle_work, time_s)
    if method == "savgol":
        return _estimate_savgol_rate(time_s, angle_work)
    if method == "spline":
        return _estimate_spline_rate(time_s, angle_work)
    if method == "local_polynomial":
        return _estimate_local_polynomial_rate(time_s, angle_work)
    raise ValueError(f"Unsupported method: {method}")


def estimate_local_ray_bundle_points(
    *,
    time_s: np.ndarray,
    positions_world_m: np.ndarray,
    los_world: np.ndarray,
    window_size: int = 9,
    stride: int = 5,
    max_condition_number: float = 1e6,
) -> pd.DataFrame:
    """Estimate locally coherent target points by intersecting short LOS bundles."""

    positions_world_m = np.asarray(positions_world_m, dtype=float)
    los_world = np.asarray(los_world, dtype=float)
    if len(time_s) != len(positions_world_m) or len(time_s) != len(los_world):
        raise ValueError("Ray bundle inputs must have matching lengths.")
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3.")

    half_window = window_size // 2
    identity = np.eye(3)
    time_s = np.asarray(time_s, dtype=float)
    rows: list[dict[str, float]] = []
    for center_index in range(half_window, len(time_s) - half_window, stride):
        indices = slice(center_index - half_window, center_index + half_window + 1)
        positions = positions_world_m[indices]
        directions = los_world[indices]
        projectors = identity - directions[:, :, None] * directions[:, None, :]
        normal_matrix = projectors.sum(axis=0)
        condition_number = float(np.linalg.cond(normal_matrix))
        if not np.isfinite(condition_number) or condition_number > max_condition_number:
            continue
        rhs = np.einsum("nij,nj->i", projectors, positions)
        point = np.linalg.solve(normal_matrix, rhs)
        offsets = point[None, :] - positions
        along_ray = np.einsum("ij,ij->i", offsets, directions)
        residuals = np.linalg.norm(np.cross(offsets, directions), axis=1)
        positive_ratio = float(np.mean(along_ray > 0.0))
        if positive_ratio < 0.7:
            continue
        rows.append(
            {
                "time_s": float(time_s[center_index]),
                "x_m": float(point[0]),
                "y_m": float(point[1]),
                "z_m": float(point[2]),
                "range_median_m": float(np.median(along_ray)),
                "residual_rms_m": float(np.sqrt(np.mean(residuals**2))),
                "positive_range_ratio": positive_ratio,
                "condition_number": condition_number,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "time_s",
            "x_m",
            "y_m",
            "z_m",
            "range_median_m",
            "residual_rms_m",
            "positive_range_ratio",
            "condition_number",
        ],
    )


def synchronize_q1_streams(dataframe: pd.DataFrame, convention: FrameConvention | None = None) -> pd.DataFrame:
    """Synchronize gimbal and navigation streams onto the camera timeline.

    Returns a DataFrame with one row per camera sample, containing:

    **Time columns:**
    - ``time``: original camera timestamp (unix epoch)
    - ``time_s``: seconds since first camera sample

    **Raw camera observations:**
    - ``target_x_deg``, ``target_y_deg``: camera-frame target angles

    **Gimbal (interpolated onto camera time):**
    - ``gimbal_pitch_rad``: linearly interpolated gimbal pitch
    - ``held_gimbal_pitch_rad``: zero-order-hold gimbal pitch (diagnostic)

    **Attitude quaternion (SLERP-interpolated onto camera time):**
    - ``orientation_x/y/z/w``: quaternion components (scipy xyzw order)

    **Navigation (interpolated onto camera time):**
    - ``position_x/y/z_m``: world-frame position [m]
    - ``velocity_x/y/z_mps``: world-frame velocity [m/s]
    - ``angular_x/y/z_radps``: body-frame angular rates [rad/s]

    **Computed world-frame LOS (from interpolated pipeline):**
    - ``world_los_x/y/z``: unit world LOS vector components
    - ``world_los``: full 3-element LOS array per row
    - ``world_az_deg``, ``world_el_deg``: NWU azimuth/elevation [deg]

    **Computed world-frame LOS (from held pipeline, diagnostic):**
    - ``held_world_az_deg``, ``held_world_el_deg``: same, zero-order hold

    **Body forward axis in world frame:**
    - ``forward_world_x/y/z``: body +x axis rotated to world

    **Synchronization quality:**
    - ``gimbal_offset_s``: abs nearest-sample offset to gimbal stream
    - ``attitude_offset_s``: abs nearest-sample offset to nav stream

    **Ground-plane footprint (diagnostic):**
    - ``ground_hit_valid``: True if LOS intersects z=0 ahead of drone
    - ``ground_hit_x/y/z_m``: intersection point (NaN if invalid)
    """

    convention = convention or FrameConvention()
    streams = split_guidance_streams(dataframe)
    if len(streams.camera) == 0 or len(streams.gimbal) == 0 or len(streams.nav) == 0:
        raise ValueError("Camera, gimbal, and navigation streams are all required for Q1 analysis.")

    camera_time = streams.camera["time"].to_numpy()
    gimbal_time = streams.gimbal["time"].to_numpy()
    nav_time = streams.nav["time"].to_numpy()
    gimbal_pitch_rad = interpolate_scalar(camera_time, gimbal_time, streams.gimbal["gimbal_pitch_rad"].to_numpy())
    held_pitch_rad = zero_order_hold(camera_time, gimbal_time, streams.gimbal["gimbal_pitch_rad"].to_numpy())
    nav_orientation = interpolate_rotations(camera_time, nav_time, navigation_rotations(streams.nav))
    held_orientation = hold_rotations(camera_time, nav_time, navigation_rotations(streams.nav))

    position_world_m = np.column_stack(
        [
            interpolate_scalar_or_zero(camera_time, nav_time, streams.nav["position_x"].to_numpy()),
            interpolate_scalar_or_zero(camera_time, nav_time, streams.nav["position_y"].to_numpy()),
            interpolate_scalar_or_zero(camera_time, nav_time, streams.nav["position_z"].to_numpy()),
        ]
    )
    velocity_world_mps = np.column_stack(
        [
            interpolate_scalar_or_zero(camera_time, nav_time, streams.nav["vel_x"].to_numpy()),
            interpolate_scalar_or_zero(camera_time, nav_time, streams.nav["vel_y"].to_numpy()),
            interpolate_scalar_or_zero(camera_time, nav_time, streams.nav["vel_z"].to_numpy()),
        ]
    )
    angular_body_rad_per_s = np.column_stack(
        [
            interpolate_scalar_or_zero(camera_time, nav_time, streams.nav["angular_x"].to_numpy()),
            interpolate_scalar_or_zero(camera_time, nav_time, streams.nav["angular_y"].to_numpy()),
            interpolate_scalar_or_zero(camera_time, nav_time, streams.nav["angular_z"].to_numpy()),
        ]
    )

    los_camera = camera_angles_to_los(
        streams.camera["target_x_deg"].to_numpy(),
        streams.camera["target_y_deg"].to_numpy(),
        convention,
    )
    los_body = los_body_from_camera(los_camera, gimbal_pitch_rad, convention)
    held_los_body = los_body_from_camera(los_camera, held_pitch_rad, convention)
    los_world = world_los_from_body(los_body, nav_orientation, convention)
    held_los_world = world_los_from_body(held_los_body, held_orientation, convention)
    world_az_deg, world_el_deg = los_to_world_angles_nwu(los_world)
    held_world_az_deg, held_world_el_deg = los_to_world_angles_nwu(held_los_world)
    aligned_quaternion = nav_orientation.as_quat()
    forward_world = body_axis_world(nav_orientation, axis_body=np.array([1.0, 0.0, 0.0]), convention=convention)
    footprint = estimate_ground_footprint(position_world_m, los_world)

    return pd.DataFrame(
        {
            "time": camera_time,
            "time_s": camera_time - camera_time[0],
            "target_x_deg": streams.camera["target_x_deg"].to_numpy(),
            "target_y_deg": streams.camera["target_y_deg"].to_numpy(),
            "gimbal_pitch_rad": gimbal_pitch_rad,
            "held_gimbal_pitch_rad": held_pitch_rad,
            "orientation_x": aligned_quaternion[:, 0],
            "orientation_y": aligned_quaternion[:, 1],
            "orientation_z": aligned_quaternion[:, 2],
            "orientation_w": aligned_quaternion[:, 3],
            "position_x_m": position_world_m[:, 0],
            "position_y_m": position_world_m[:, 1],
            "position_z_m": position_world_m[:, 2],
            "velocity_x_mps": velocity_world_mps[:, 0],
            "velocity_y_mps": velocity_world_mps[:, 1],
            "velocity_z_mps": velocity_world_mps[:, 2],
            "angular_x_radps": angular_body_rad_per_s[:, 0],
            "angular_y_radps": angular_body_rad_per_s[:, 1],
            "angular_z_radps": angular_body_rad_per_s[:, 2],
            "world_los_x": los_world[:, 0],
            "world_los_y": los_world[:, 1],
            "world_los_z": los_world[:, 2],
            "world_los": [row.copy() for row in los_world],
            "world_az_deg": world_az_deg,
            "world_el_deg": world_el_deg,
            "held_world_az_deg": held_world_az_deg,
            "held_world_el_deg": held_world_el_deg,
            "forward_world_x": forward_world[:, 0],
            "forward_world_y": forward_world[:, 1],
            "forward_world_z": forward_world[:, 2],
            "gimbal_offset_s": nearest_time_offsets(camera_time, gimbal_time),
            "attitude_offset_s": nearest_time_offsets(camera_time, nav_time),
            "ground_hit_valid": footprint["valid"].to_numpy(),
            "ground_hit_x_m": footprint["x_m"].to_numpy(),
            "ground_hit_y_m": footprint["y_m"].to_numpy(),
            "ground_hit_z_m": footprint["z_m"].to_numpy(),
        }
    )


def select_best_frame_convention(dataframe: pd.DataFrame) -> FrameConventionSelection:
    """Score plausible sign/frame conventions and choose the most coherent one."""

    candidate_rows: list[dict[str, Any]] = []
    for convention in iter_convention_candidates():
        synchronized = synchronize_q1_streams(dataframe, convention)
        az_rate = estimate_angle_rate(
            synchronized["time_s"].to_numpy(),
            synchronized["world_az_deg"].to_numpy(),
            method="local_polynomial",
            unwrap=True,
        )
        el_rate = estimate_angle_rate(
            synchronized["time_s"].to_numpy(),
            synchronized["world_el_deg"].to_numpy(),
            method="local_polynomial",
        )
        bundle_points = estimate_local_ray_bundle_points(
            time_s=synchronized["time_s"].to_numpy(),
            positions_world_m=synchronized[["position_x_m", "position_y_m", "position_z_m"]].to_numpy(),
            los_world=synchronized[["world_los_x", "world_los_y", "world_los_z"]].to_numpy(),
            window_size=9,
            stride=7,
        )
        pair_miss_median_m = float(bundle_points["residual_rms_m"].median()) if len(bundle_points) else np.inf
        pair_miss_p90_m = float(bundle_points["residual_rms_m"].quantile(0.9)) if len(bundle_points) else np.inf
        heading_error_deg = forward_velocity_heading_error_deg(synchronized)
        candidate_rows.append(
            {
                **asdict(convention),
                "pair_miss_median_m": pair_miss_median_m,
                "pair_miss_p90_m": pair_miss_p90_m,
                "bundle_positive_ratio": float(bundle_points["positive_range_ratio"].median()) if len(bundle_points) else 0.0,
                "bundle_count": int(len(bundle_points)),
                "combined_rate_std_deg_s": float(np.std(az_rate) + np.std(el_rate)),
                "heading_error_median_deg": heading_error_deg,
            }
        )

    candidate_table = pd.DataFrame(candidate_rows).sort_values(
        by=[
            "pair_miss_median_m",
            "pair_miss_p90_m",
            "combined_rate_std_deg_s",
            "heading_error_median_deg",
        ]
    ).reset_index(drop=True)
    best_row = candidate_table.iloc[0]
    return FrameConventionSelection(
        best_convention=FrameConvention(
            quaternion_is_body_to_world=bool(best_row["quaternion_is_body_to_world"]),
            camera_positive_azimuth_right=bool(best_row["camera_positive_azimuth_right"]),
            camera_positive_elevation_up=bool(best_row["camera_positive_elevation_up"]),
            gimbal_positive_pitch_raises=bool(best_row["gimbal_positive_pitch_raises"]),
        ),
        candidate_table=candidate_table,
    )


def summarize_streams(streams: GuidanceStreams) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, dataframe in {
        "camera": streams.camera,
        "gimbal": streams.gimbal,
        "nav": streams.nav,
        "empty_rows": streams.empty_rows,
        "unclassified_rows": streams.unclassified_rows,
    }.items():
        if len(dataframe) < 2:
            rows.append({"stream": name, "samples": int(len(dataframe)), "median_dt_s": np.nan, "nominal_rate_hz": np.nan})
            continue
        dt_s = np.diff(dataframe["time"].to_numpy())
        rows.append(
            {
                "stream": name,
                "samples": int(len(dataframe)),
                "median_dt_s": float(np.median(dt_s)),
                "nominal_rate_hz": float(1.0 / np.mean(dt_s)),
                "p95_dt_s": float(np.percentile(dt_s, 95.0)),
            }
        )
    return pd.DataFrame(rows)


def build_rate_metrics(
    time_s: np.ndarray,
    synchronized: pd.DataFrame,
    rate_candidates: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """Quantify rate-estimator behavior on the synchronized signal."""

    rows: list[dict[str, Any]] = []
    held_world_az = np.rad2deg(np.unwrap(np.deg2rad(synchronized["held_world_az_deg"].to_numpy())))
    held_world_el = synchronized["held_world_el_deg"].to_numpy()
    interpolated_world_az = np.rad2deg(np.unwrap(np.deg2rad(synchronized["world_az_deg"].to_numpy())))
    interpolated_world_el = synchronized["world_el_deg"].to_numpy()
    for channel, interpolated, held in (
        ("azimuth", interpolated_world_az, held_world_az),
        ("elevation", interpolated_world_el, held_world_el),
    ):
        for method_name, rates in rate_candidates.items():
            signal_rate = rates[channel]
            reconstructed = np.cumsum(signal_rate * np.gradient(time_s))
            reconstructed -= reconstructed[0]
            reconstructed += interpolated[0]
            rows.append(
                {
                    "channel": channel,
                    "method": method_name,
                    "noise_proxy_deg_per_s": float(np.std(np.diff(signal_rate))),
                    "edge_proxy_deg_per_s": float(max(np.max(np.abs(signal_rate[:10])), np.max(np.abs(signal_rate[-10:])))),
                    "lag_proxy_seconds": lag_proxy_seconds(signal_rate, np.gradient(interpolated, time_s), time_s),
                    "reconstruction_rmse_deg": float(np.sqrt(np.mean((reconstructed - interpolated) ** 2))),
                    "holdout_rmse_deg": float(np.sqrt(np.mean((interpolated - held) ** 2))),
                }
            )
    return pd.DataFrame(rows)


def select_rate_method(rate_metrics: pd.DataFrame) -> dict[str, Any]:
    """Rank rate estimators with a noise/latency-first policy."""

    ranking_table = (
        rate_metrics.groupby("method", as_index=False)
        .agg(
            lag_abs_mean_s=("lag_proxy_seconds", lambda values: float(np.mean(np.abs(values)))),
            reconstruction_rmse_mean_deg=("reconstruction_rmse_deg", "mean"),
            noise_proxy_mean_deg_per_s=("noise_proxy_deg_per_s", "mean"),
            edge_proxy_mean_deg_per_s=("edge_proxy_deg_per_s", "mean"),
            holdout_rmse_mean_deg=("holdout_rmse_deg", "mean"),
        )
    )
    for metric_name, weight in RATE_SELECTION_WEIGHTS.items():
        metric_values = ranking_table[metric_name]
        metric_span = float(metric_values.max() - metric_values.min())
        normalized_column = f"{metric_name}_normalized"
        contribution_column = f"{metric_name}_weighted_score"
        if np.isclose(metric_span, 0.0):
            ranking_table[normalized_column] = 0.0
        else:
            ranking_table[normalized_column] = (metric_values - float(metric_values.min())) / metric_span
        ranking_table[contribution_column] = ranking_table[normalized_column] * weight

    ranking_table["composite_score"] = sum(
        ranking_table[f"{metric_name}_weighted_score"] for metric_name in RATE_SELECTION_WEIGHTS
    )
    ranking_table = ranking_table.sort_values(
        by=[
            "composite_score",
            "lag_abs_mean_s",
            "reconstruction_rmse_mean_deg",
            "noise_proxy_mean_deg_per_s",
            "edge_proxy_mean_deg_per_s",
            "holdout_rmse_mean_deg",
        ]
    ).reset_index(drop=True)
    selected_row = ranking_table.iloc[0]
    return {
        "policy": (
            "Rank by weighted normalized composite score with lower-is-better metrics. "
            "Latency proxy and noise proxy carry the dominant weights so the selected "
            "method follows the prompt's noise-versus-latency requirement; "
            "reconstruction RMSE, edge proxy, and holdout RMSE remain as secondary "
            "stability checks."
        ),
        "weights": dict(RATE_SELECTION_WEIGHTS),
        "selected_method": str(selected_row["method"]),
        "selected_method_rationale": (
            f"{selected_row['method']} was selected because it gives the best overall "
            "trade between low latency and low noise on this log, while still staying "
            "well-behaved on reconstruction and edge checks."
        ),
        "ranking": ranking_table.to_dict(orient="records"),
    }


def run_q1_analysis(
    output_dir: Path,
    csv_path: Path,
    convention: FrameConvention | None = None,
    *,
    render_animation: bool = True,
    clean_outputs: bool = True,
    plot_config: PlotConfig | None = None,
) -> dict[str, Any]:
    """Run the complete Q1 pipeline and persist reviewer-facing artifacts."""

    from .q1_visualization import create_q1_visuals

    output_dir = Path(output_dir)
    if clean_outputs:
        clean_output_dir(output_dir)
    configure_matplotlib(plot_config)
    plot_config = plot_config or PlotConfig()

    dataframe = pd.read_csv(csv_path)
    streams = split_guidance_streams(dataframe)
    convention_selection = select_best_frame_convention(dataframe)
    selected_convention = convention or convention_selection.best_convention
    synchronized = synchronize_q1_streams(dataframe, selected_convention)

    time_s = synchronized["time_s"].to_numpy()
    world_az_deg = synchronized["world_az_deg"].to_numpy()
    world_el_deg = synchronized["world_el_deg"].to_numpy()
    rate_candidates = {
        "gradient": {
            "azimuth": estimate_angle_rate(time_s, world_az_deg, method="gradient", unwrap=True),
            "elevation": estimate_angle_rate(time_s, world_el_deg, method="gradient"),
        },
        "savgol": {
            "azimuth": estimate_angle_rate(time_s, world_az_deg, method="savgol", unwrap=True),
            "elevation": estimate_angle_rate(time_s, world_el_deg, method="savgol"),
        },
        "local_polynomial": {
            "azimuth": estimate_angle_rate(time_s, world_az_deg, method="local_polynomial", unwrap=True),
            "elevation": estimate_angle_rate(time_s, world_el_deg, method="local_polynomial"),
        },
        "spline": {
            "azimuth": estimate_angle_rate(time_s, world_az_deg, method="spline", unwrap=True),
            "elevation": estimate_angle_rate(time_s, world_el_deg, method="spline"),
        },
    }
    synchronized["world_az_unwrapped_deg"] = np.rad2deg(np.unwrap(np.deg2rad(world_az_deg)))
    synchronized["az_rate_gradient_deg_s"] = rate_candidates["gradient"]["azimuth"]
    synchronized["el_rate_gradient_deg_s"] = rate_candidates["gradient"]["elevation"]
    synchronized["az_rate_savgol_deg_s"] = rate_candidates["savgol"]["azimuth"]
    synchronized["el_rate_savgol_deg_s"] = rate_candidates["savgol"]["elevation"]
    synchronized["az_rate_local_polynomial_deg_s"] = rate_candidates["local_polynomial"]["azimuth"]
    synchronized["el_rate_local_polynomial_deg_s"] = rate_candidates["local_polynomial"]["elevation"]
    synchronized["az_rate_spline_deg_s"] = rate_candidates["spline"]["azimuth"]
    synchronized["el_rate_spline_deg_s"] = rate_candidates["spline"]["elevation"]

    bundle_points = estimate_local_ray_bundle_points(
        time_s=time_s,
        positions_world_m=synchronized[["position_x_m", "position_y_m", "position_z_m"]].to_numpy(),
        los_world=synchronized[["world_los_x", "world_los_y", "world_los_z"]].to_numpy(),
        window_size=9,
        stride=5,
    )
    ground_footprint = synchronized.loc[
        synchronized["ground_hit_valid"],
        ["time_s", "ground_hit_x_m", "ground_hit_y_m", "ground_hit_z_m"],
    ].rename(columns={"ground_hit_x_m": "x_m", "ground_hit_y_m": "y_m", "ground_hit_z_m": "z_m"})
    stream_summary = summarize_streams(streams)
    sync_offset_summary = pd.concat(
        [
            pd.DataFrame({"source": "gimbal", "offset_ms": synchronized["gimbal_offset_s"] * 1e3}),
            pd.DataFrame({"source": "attitude", "offset_ms": synchronized["attitude_offset_s"] * 1e3}),
        ],
        ignore_index=True,
    )
    rate_metrics = build_rate_metrics(time_s, synchronized, rate_candidates)
    rate_selection = select_rate_method(rate_metrics)
    selected_rate_method = rate_selection["selected_method"]

    data_csv = output_dir / "q1_world_angles.csv"
    candidates_csv = output_dir / "q1_convention_candidates.csv"
    bundle_csv = output_dir / "q1_bundle_points.csv"
    footprint_csv = output_dir / "q1_ground_footprint.csv"
    synchronized.to_csv(data_csv, index=False)
    convention_selection.candidate_table.to_csv(candidates_csv, index=False)
    bundle_points.to_csv(bundle_csv, index=False)
    ground_footprint.to_csv(footprint_csv, index=False)
    artifact_paths = [data_csv, candidates_csv, bundle_csv, footprint_csv]
    artifact_paths.extend(
        create_q1_visuals(
            output_dir=output_dir,
            streams=streams,
            stream_summary=stream_summary,
            sync_offset_summary=sync_offset_summary,
            synchronized=synchronized,
            convention_selection=convention_selection,
            selected_convention=selected_convention,
            rate_candidates=rate_candidates,
            rate_metrics=rate_metrics,
            selected_rate_method=selected_rate_method,
            bundle_points=bundle_points,
            ground_footprint=ground_footprint,
            render_animation=render_animation,
            plot_config=plot_config,
        )
    )

    selected_rates = rate_candidates[selected_rate_method]
    summary = {
        "source_csv": str(csv_path),
        "output_dir": str(output_dir),
        "selected_convention": asdict(selected_convention),
        "auto_selected_convention": asdict(convention_selection.best_convention),
        "convention_selection_note": (
            "Frame/sign conventions are inferred under the stated camera-body alignment model. "
            "The ranking is diagnostic evidence, not direct ground truth."
        ),
        "sensor_sync": {
            "target_samples": int(len(synchronized)),
            "duration_s": float(time_s[-1] - time_s[0]),
            "gimbal_offset_ms_median": float(np.median(synchronized["gimbal_offset_s"]) * 1e3),
            "gimbal_offset_ms_p95": float(np.percentile(synchronized["gimbal_offset_s"], 95.0) * 1e3),
            "attitude_offset_ms_median": float(np.median(synchronized["attitude_offset_s"]) * 1e3),
            "attitude_offset_ms_p95": float(np.percentile(synchronized["attitude_offset_s"], 95.0) * 1e3),
        },
        "stream_summary": stream_summary.to_dict(orient="records"),
        "world_angle_ranges_deg": {
            "azimuth_min": float(np.nanmin(world_az_deg)),
            "azimuth_max": float(np.nanmax(world_az_deg)),
            "elevation_min": float(np.nanmin(world_el_deg)),
            "elevation_max": float(np.nanmax(world_el_deg)),
        },
        "selected_rate_method": selected_rate_method,
        "selected_rate_rationale": rate_selection["selected_method_rationale"],
        "rate_selection": rate_selection,
        "rate_summary_deg_s": {
            "azimuth_std": float(np.std(selected_rates["azimuth"])),
            "elevation_std": float(np.std(selected_rates["elevation"])),
            "azimuth_peak_abs": float(np.max(np.abs(selected_rates["azimuth"]))),
            "elevation_peak_abs": float(np.max(np.abs(selected_rates["elevation"]))),
        },
        "rate_metrics": rate_metrics.to_dict(orient="records"),
        "best_candidate_by_composite_score": convention_selection.candidate_table.iloc[0].to_dict(),
        "bundle_geometry": {
            "bundle_count": int(len(bundle_points)),
            "bundle_residual_median_m": float(bundle_points["residual_rms_m"].median()) if len(bundle_points) else None,
            "bundle_residual_p90_m": float(bundle_points["residual_rms_m"].quantile(0.9)) if len(bundle_points) else None,
        },
        "artifact_paths": [str(path) for path in artifact_paths],
    }
    summary_path = output_dir / "q1_summary.json"
    summary_path.write_text(json.dumps(to_builtin(summary), indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def _sorted_unique(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)


def iter_convention_candidates() -> Iterable[FrameConvention]:
    for quaternion_is_body_to_world in (True, False):
        for camera_positive_azimuth_right in (True, False):
            for camera_positive_elevation_up in (True, False):
                for gimbal_positive_pitch_raises in (False, True):
                    yield FrameConvention(
                        quaternion_is_body_to_world=quaternion_is_body_to_world,
                        camera_positive_azimuth_right=camera_positive_azimuth_right,
                        camera_positive_elevation_up=camera_positive_elevation_up,
                        gimbal_positive_pitch_raises=gimbal_positive_pitch_raises,
                    )


def interpolate_scalar(query_time: np.ndarray, sample_time: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.interp(np.asarray(query_time, dtype=float), np.asarray(sample_time, dtype=float), np.asarray(values, dtype=float))


def interpolate_scalar_or_zero(query_time: np.ndarray, sample_time: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if not np.any(valid):
        return np.zeros_like(np.asarray(query_time, dtype=float))
    return interpolate_scalar(np.asarray(query_time, dtype=float), np.asarray(sample_time, dtype=float)[valid], values[valid])


def zero_order_hold(query_time: np.ndarray, sample_time: np.ndarray, values: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(sample_time, query_time, side="right") - 1
    indices = np.clip(indices, 0, len(sample_time) - 1)
    return np.asarray(values, dtype=float)[indices]


def navigation_rotations(nav_stream: pd.DataFrame) -> Rotation:
    return Rotation.from_quat(nav_stream[["orientation_x", "orientation_y", "orientation_z", "orientation_w"]].to_numpy())


def interpolate_rotations(query_time: np.ndarray, sample_time: np.ndarray, samples: Rotation) -> Rotation:
    clipped_time = np.clip(np.asarray(query_time, dtype=float), float(sample_time[0]), float(sample_time[-1]))
    return Slerp(np.asarray(sample_time, dtype=float), samples)(clipped_time)


def hold_rotations(query_time: np.ndarray, sample_time: np.ndarray, samples: Rotation) -> Rotation:
    indices = np.searchsorted(sample_time, query_time, side="right") - 1
    indices = np.clip(indices, 0, len(sample_time) - 1)
    return Rotation.from_quat(samples.as_quat()[indices])


def nearest_time_offsets(base_time: np.ndarray, reference_time: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(reference_time, base_time)
    left = np.clip(indices - 1, 0, len(reference_time) - 1)
    right = np.clip(indices, 0, len(reference_time) - 1)
    return np.minimum(np.abs(base_time - reference_time[left]), np.abs(base_time - reference_time[right]))


def world_los_from_body(los_body: np.ndarray, orientation: Rotation, convention: FrameConvention) -> np.ndarray:
    return orientation.apply(los_body) if convention.quaternion_is_body_to_world else orientation.inv().apply(los_body)


def body_axis_world(orientation: Rotation, *, axis_body: np.ndarray, convention: FrameConvention) -> np.ndarray:
    axes_body = np.repeat(np.asarray(axis_body, dtype=float)[None, :], len(orientation.as_quat()), axis=0)
    return world_los_from_body(axes_body, orientation, convention)


def estimate_ground_footprint(position_world_m: np.ndarray, los_world: np.ndarray, ground_z_m: float = 0.0) -> pd.DataFrame:
    dz = np.asarray(los_world, dtype=float)[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        ray_scale = (ground_z_m - np.asarray(position_world_m, dtype=float)[:, 2]) / dz
    valid = np.isfinite(ray_scale) & (ray_scale > 0.0)
    points = np.asarray(position_world_m, dtype=float) + ray_scale[:, None] * np.asarray(los_world, dtype=float)
    points[~valid] = np.nan
    return pd.DataFrame({"valid": valid, "x_m": points[:, 0], "y_m": points[:, 1], "z_m": points[:, 2]})


def forward_velocity_heading_error_deg(synchronized: pd.DataFrame) -> float:
    velocity = synchronized[["velocity_x_mps", "velocity_y_mps"]].to_numpy()
    forward = synchronized[["forward_world_x", "forward_world_y"]].to_numpy()
    speed = np.linalg.norm(velocity, axis=1)
    forward_norm = np.linalg.norm(forward, axis=1)
    valid = (speed > 0.2) & (forward_norm > 1e-6)
    if not np.any(valid):
        return np.inf
    velocity_hat = velocity[valid] / speed[valid, None]
    forward_hat = forward[valid] / forward_norm[valid, None]
    dot_product = np.clip(np.sum(velocity_hat * forward_hat, axis=1), -1.0, 1.0)
    return float(np.rad2deg(np.median(np.arccos(dot_product))))


def _estimate_savgol_rate(time_s: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    time_step = np.diff(time_s)
    median_step = float(np.median(time_step[time_step > 0.0]))
    uniform_time = np.arange(time_s[0], time_s[-1] + 0.5 * median_step, median_step)
    uniform_signal = np.interp(uniform_time, time_s, angle_deg)
    window_length = max(7, int(round(0.45 / median_step)) | 1)
    if window_length >= len(uniform_time):
        window_length = len(uniform_time) - (1 - len(uniform_time) % 2)
    window_length = max(window_length, 5)
    derivative = savgol_filter(uniform_signal, window_length=window_length, polyorder=min(3, window_length - 1), deriv=1, delta=median_step, mode="interp")
    return np.interp(time_s, uniform_time, derivative)


def _estimate_spline_rate(time_s: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    noise_scale = float(np.median(np.abs(np.diff(angle_deg))) / 0.6745) if len(angle_deg) > 2 else 1e-3
    spline = UnivariateSpline(time_s, angle_deg, k=min(3, len(time_s) - 1), s=len(time_s) * max(noise_scale, 1e-3) ** 2)
    return spline.derivative()(time_s)


def _estimate_local_polynomial_rate(time_s: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    derivative = np.zeros_like(angle_deg, dtype=float)
    half_width_s = max(0.18, 8.0 * np.median(np.diff(time_s)))
    minimum_points = 7
    for index, center_time in enumerate(time_s):
        centered_time = time_s - center_time
        mask = np.abs(centered_time) <= half_width_s
        if np.count_nonzero(mask) < minimum_points:
            mask = np.zeros_like(centered_time, dtype=bool)
            mask[np.argsort(np.abs(centered_time))[:minimum_points]] = True
        local_time = centered_time[mask]
        local_signal = angle_deg[mask]
        scale = max(np.max(np.abs(local_time)), 1e-9)
        weights = np.clip(1.0 - (np.abs(local_time) / scale) ** 3, 0.0, None) ** 3
        vandermonde = np.column_stack([np.ones_like(local_time), local_time, local_time**2, local_time**3])
        coefficients, *_ = np.linalg.lstsq(vandermonde * weights[:, None], local_signal * weights, rcond=None)
        derivative[index] = coefficients[1]
    return derivative


def lag_proxy_seconds(signal_a: np.ndarray, signal_b: np.ndarray, time_s: np.ndarray) -> float:
    signal_a = np.asarray(signal_a, dtype=float) - float(np.mean(signal_a))
    signal_b = np.asarray(signal_b, dtype=float) - float(np.mean(signal_b))
    correlation = np.correlate(signal_a, signal_b, mode="full")
    lags = np.arange(-len(signal_a) + 1, len(signal_a))
    return float(lags[int(np.argmax(correlation))] * np.median(np.diff(time_s)))
