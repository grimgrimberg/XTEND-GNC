import numpy as np
import pandas as pd

from helpers.paths import data_path
from helpers.q1_pipeline import (
    FrameConvention,
    estimate_angle_rate,
    estimate_local_ray_bundle_points,
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
