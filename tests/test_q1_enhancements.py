import numpy as np
import pandas as pd

from helpers.paths import data_path
import helpers.q1_pipeline as q1_pipeline
from helpers.q1_pipeline import (
    FrameConvention,
    estimate_angle_rate,
    estimate_local_ray_bundle_points,
    select_rate_method,
    select_best_frame_convention,
    split_guidance_streams,
)


def test_split_guidance_streams_classifies_interleaved_rows():
    dataframe = pd.DataFrame(
        [
            {"time": 0.00, "target_x_deg": 1.0, "target_y_deg": -2.0},
            {"time": 0.01, "gimbal_pitch_rad": 0.1},
            {
                "time": 0.02,
                "orientation_w": 1.0,
                "orientation_x": 0.0,
                "orientation_y": 0.0,
                "orientation_z": 0.0,
                "position_x": 1.0,
                "position_y": 2.0,
                "position_z": 3.0,
                "angular_x": 0.1,
                "angular_y": 0.2,
                "angular_z": 0.3,
                "vel_x": 4.0,
                "vel_y": 5.0,
                "vel_z": 6.0,
            },
            {"time": 0.03},
        ]
    )

    streams = split_guidance_streams(dataframe)

    assert len(streams.camera) == 1
    assert len(streams.gimbal) == 1
    assert len(streams.nav) == 1
    assert len(streams.empty_rows) == 1
    assert len(streams.unclassified_rows) == 0


def test_local_polynomial_rate_beats_gradient_on_noisy_sine():
    rng = np.random.default_rng(7)
    time_s = np.linspace(0.0, 12.0, 721)
    clean_signal_deg = 25.0 * np.sin(0.75 * time_s)
    noisy_signal_deg = clean_signal_deg + rng.normal(scale=0.35, size=time_s.shape)
    truth_rate = 18.75 * np.cos(0.75 * time_s)

    gradient_rate = estimate_angle_rate(time_s, noisy_signal_deg, method="gradient")
    local_poly_rate = estimate_angle_rate(time_s, noisy_signal_deg, method="local_polynomial")

    gradient_rmse = float(np.sqrt(np.mean((gradient_rate - truth_rate) ** 2)))
    local_poly_rmse = float(np.sqrt(np.mean((local_poly_rate - truth_rate) ** 2)))
    assert local_poly_rmse < 0.65 * gradient_rmse


def test_ray_bundle_estimator_recovers_static_point():
    target_world = np.array([28.0, -6.0, 4.5], dtype=float)
    positions_world = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.2],
            [2.0, 0.2, 0.3],
            [3.0, 0.3, 0.4],
            [4.0, 0.5, 0.5],
            [5.0, 0.7, 0.6],
            [6.0, 0.9, 0.7],
        ],
        dtype=float,
    )
    los_world = target_world - positions_world
    los_world /= np.linalg.norm(los_world, axis=1, keepdims=True)
    time_s = np.linspace(0.0, 0.6, len(positions_world))

    bundle_points = estimate_local_ray_bundle_points(
        time_s=time_s,
        positions_world_m=positions_world,
        los_world=los_world,
        window_size=5,
        stride=1,
    )

    estimated_point = bundle_points[["x_m", "y_m", "z_m"]].median().to_numpy()
    assert np.linalg.norm(estimated_point - target_world) < 0.3
    assert float(bundle_points["residual_rms_m"].median()) < 1e-6


def test_select_best_frame_convention_matches_real_dataset_default():
    dataframe = pd.read_csv(data_path("examGuidance.csv"))

    selection = select_best_frame_convention(dataframe)

    assert selection.best_convention == FrameConvention(
        quaternion_is_body_to_world=False,
        camera_positive_azimuth_right=True,
        camera_positive_elevation_up=False,
        gimbal_positive_pitch_raises=False,
    )
    assert {"pair_miss_median_m", "combined_rate_std_deg_s", "camera_positive_azimuth_right"}.issubset(
        selection.candidate_table.columns
    )


def test_select_rate_method_reports_noise_latency_rationale():
    metrics = pd.DataFrame(
        [
            {
                "channel": channel,
                "method": method,
                "noise_proxy_deg_per_s": noise,
                "edge_proxy_deg_per_s": edge,
                "lag_proxy_seconds": lag,
                "reconstruction_rmse_deg": rmse,
                "holdout_rmse_deg": holdout,
            }
            for channel in ("azimuth", "elevation")
            for method, noise, edge, lag, rmse, holdout in (
                ("gradient", 11.0, 12.0, 0.0, 0.70, 0.12),
                ("savgol", 2.0, 4.0, 0.010, 0.38, 0.12),
                ("local_polynomial", 5.8, 6.0, 0.0, 0.25, 0.12),
                ("spline", 13.0, 14.0, 0.0, 0.26, 0.12),
            )
        ]
    )

    selection = select_rate_method(metrics)

    assert selection["selected_method"] == "local_polynomial"
    assert "noise" in selection["selected_method_rationale"].lower()
    assert "latency" in selection["selected_method_rationale"].lower()


def test_kalman_cv_returns_result_and_handles_irregular_dt():
    kalman_cv = getattr(q1_pipeline, "kalman_cv", None)
    assert callable(kalman_cv)

    time_s = np.array([0.0, 0.04, 0.11, 0.21, 0.36, 0.58, 0.91], dtype=float)
    truth_rate_deg_s = 4.25
    truth_angle_deg = -12.0 + truth_rate_deg_s * time_s
    measurements_deg = truth_angle_deg + np.array([0.18, -0.22, 0.14, -0.05, 0.09, -0.03, 0.11], dtype=float)

    result = kalman_cv(time_s, measurements_deg)

    assert hasattr(result, "filtered_angle_deg")
    assert hasattr(result, "estimated_rate_deg_s")
    assert hasattr(result, "sigma_z_deg")
    assert hasattr(result, "sigma_a_deg_s2")
    assert result.filtered_angle_deg.shape == measurements_deg.shape
    assert result.estimated_rate_deg_s.shape == measurements_deg.shape
    assert np.all(np.isfinite(result.filtered_angle_deg))
    assert np.all(np.isfinite(result.estimated_rate_deg_s))
    assert result.sigma_z_deg > 0.0
    assert result.sigma_a_deg_s2 > 0.0
    # The approved design starts with zero initial rate, so score after the
    # brief startup transient rather than demanding oracle knowledge at t0.
    assert float(np.sqrt(np.mean((result.estimated_rate_deg_s[2:] - truth_rate_deg_s) ** 2))) < 0.6


def test_kalman_cv_is_causal_for_earlier_samples():
    kalman_cv = getattr(q1_pipeline, "kalman_cv", None)
    assert callable(kalman_cv)

    time_s = np.array([0.0, 0.06, 0.17, 0.33, 0.56, 0.84], dtype=float)
    angle_deg = 7.5 + 2.0 * time_s + np.array([0.0, 0.07, -0.03, 0.05, -0.02, 0.01], dtype=float)

    baseline = kalman_cv(time_s, angle_deg)
    perturbed = angle_deg.copy()
    perturbed[-1] += 30.0
    changed = kalman_cv(time_s, perturbed)

    assert np.allclose(baseline.filtered_angle_deg[:-1], changed.filtered_angle_deg[:-1], atol=1e-12)
    assert np.allclose(baseline.estimated_rate_deg_s[:-1], changed.estimated_rate_deg_s[:-1], atol=1e-12)


def test_estimate_angle_rate_exposes_kalman_cv_and_short_sequence_falls_back():
    def run_kalman_rate(time_s: np.ndarray, angle_deg: np.ndarray) -> np.ndarray | None:
        try:
            return estimate_angle_rate(time_s, angle_deg, method="kalman_cv")
        except ValueError:
            return None

    short_time_s = np.array([0.0, 0.4], dtype=float)
    short_angle_deg = np.array([3.0, 3.8], dtype=float)
    short_rate = run_kalman_rate(short_time_s, short_angle_deg)
    assert short_rate is not None
    assert np.allclose(short_rate, np.array([2.0, 2.0], dtype=float), atol=1e-12)

    time_s = np.array([0.0, 0.05, 0.16, 0.31, 0.53], dtype=float)
    truth_rate_deg_s = -1.5
    angle_deg = 8.0 + truth_rate_deg_s * time_s + np.array([0.0, 0.14, -0.11, 0.08, -0.04], dtype=float)

    rate_from_estimator = run_kalman_rate(time_s, angle_deg)
    kalman_cv = getattr(q1_pipeline, "kalman_cv", None)
    assert callable(kalman_cv)
    direct_result = kalman_cv(time_s, angle_deg)

    assert rate_from_estimator is not None
    assert np.allclose(rate_from_estimator, direct_result.estimated_rate_deg_s, atol=1e-12)
    assert float(np.sqrt(np.mean((rate_from_estimator[2:] - truth_rate_deg_s) ** 2))) < 0.8


def test_kalman_cv_beats_raw_gradient_on_noisy_constant_rate_signal():
    kalman_cv = getattr(q1_pipeline, "kalman_cv", None)
    assert callable(kalman_cv)

    rng = np.random.default_rng(12)
    time_s = np.linspace(0.0, 15.0, 601, dtype=float)
    truth_rate_deg_s = 7.5
    truth_angle_deg = -18.0 + truth_rate_deg_s * time_s
    noisy_angle_deg = truth_angle_deg + rng.normal(scale=0.45, size=time_s.shape)

    kalman_result = kalman_cv(time_s, noisy_angle_deg)
    kalman_rate = estimate_angle_rate(time_s, noisy_angle_deg, method="kalman_cv")
    gradient_rate = np.gradient(noisy_angle_deg, time_s)

    kalman_rmse = float(np.sqrt(np.mean((kalman_rate - truth_rate_deg_s) ** 2)))
    gradient_rmse = float(np.sqrt(np.mean((gradient_rate - truth_rate_deg_s) ** 2)))
    assert kalman_rmse < 0.55 * gradient_rmse
    assert kalman_result.sigma_z_deg > 0.0
    assert kalman_result.sigma_a_deg_s2 > 0.0
