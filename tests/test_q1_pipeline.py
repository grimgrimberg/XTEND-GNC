import numpy as np
import pandas as pd

from helpers.q1_pipeline import (
    FrameConvention,
    camera_angles_to_los,
    estimate_angle_rate,
    los_body_from_camera,
    los_to_world_angles_nwu,
    select_rate_method,
    synchronize_q1_streams,
)


def test_camera_angles_to_los_is_forward_at_zero_angles():
    los = camera_angles_to_los(np.array([0.0]), np.array([0.0]))
    assert np.allclose(los[0], np.array([1.0, 0.0, 0.0]))


def test_positive_gimbal_pitch_raises_body_los():
    convention = FrameConvention(gimbal_positive_pitch_raises=True)
    los = camera_angles_to_los(np.array([0.0]), np.array([0.0]))
    pitched = los_body_from_camera(los, np.array([np.deg2rad(10.0)]), convention)
    assert pitched[0, 2] > 0.0


def test_world_angles_follow_nwu_definition():
    az_deg, el_deg = los_to_world_angles_nwu(np.array([[1.0, 1.0, 1.0]]) / np.sqrt(3.0))
    assert np.isclose(az_deg[0], 45.0, atol=1e-6)
    assert np.isclose(el_deg[0], 35.26438968, atol=1e-6)


def test_camera_angles_round_trip_through_los_mapping():
    azimuth_deg = np.array([3.3, 20.0, 45.0, 60.0])
    elevation_deg = np.array([19.1, 20.0, 20.0, 30.0])

    los = camera_angles_to_los(azimuth_deg, elevation_deg)
    round_trip_az_deg, round_trip_el_deg = los_to_world_angles_nwu(los)

    assert np.allclose(round_trip_az_deg, azimuth_deg, atol=1e-6)
    assert np.allclose(round_trip_el_deg, elevation_deg, atol=1e-6)


def test_synchronize_q1_streams_interpolates_pitch_and_attitude():
    df = pd.DataFrame(
        [
            {"time": 0.0, "gimbal_pitch_rad": 0.0},
            {"time": 0.0, "orientation_w": 1.0, "orientation_x": 0.0, "orientation_y": 0.0, "orientation_z": 0.0},
            {"time": 0.5, "target_x_deg": 0.0, "target_y_deg": 0.0},
            {"time": 1.0, "gimbal_pitch_rad": 0.2},
            {
                "time": 1.0,
                "orientation_w": np.cos(np.deg2rad(45.0)),
                "orientation_x": 0.0,
                "orientation_y": 0.0,
                "orientation_z": np.sin(np.deg2rad(45.0)),
            },
        ]
    )
    synced = synchronize_q1_streams(df)
    assert np.isclose(synced["gimbal_pitch_rad"].iloc[0], 0.1, atol=1e-6)
    world_los = synced.loc[0, "world_los"]
    az_deg, _ = los_to_world_angles_nwu(np.array([world_los]))
    assert np.isclose(az_deg[0], 45.0, atol=1.0)


def test_estimate_angle_rate_beats_naive_gradient_on_clean_sine():
    time_s = np.linspace(0.0, 10.0, 401)
    signal_deg = 20.0 * np.sin(0.8 * time_s)
    truth = 16.0 * np.cos(0.8 * time_s)
    naive = np.gradient(signal_deg, time_s)
    estimate = estimate_angle_rate(time_s, signal_deg, method="savgol")
    assert np.sqrt(np.mean((estimate - truth) ** 2)) < np.sqrt(np.mean((naive - truth) ** 2))


def test_select_rate_method_uses_metric_ranking_not_fixed_method_name():
    rate_metrics = pd.DataFrame(
        [
            {"channel": "azimuth", "method": "gradient", "lag_proxy_seconds": 0.02, "reconstruction_rmse_deg": 0.4, "noise_proxy_deg_per_s": 4.0, "edge_proxy_deg_per_s": 6.0, "holdout_rmse_deg": 0.7},
            {"channel": "elevation", "method": "gradient", "lag_proxy_seconds": 0.02, "reconstruction_rmse_deg": 0.5, "noise_proxy_deg_per_s": 4.5, "edge_proxy_deg_per_s": 6.5, "holdout_rmse_deg": 0.8},
            {"channel": "azimuth", "method": "local_polynomial", "lag_proxy_seconds": 0.03, "reconstruction_rmse_deg": 0.6, "noise_proxy_deg_per_s": 3.5, "edge_proxy_deg_per_s": 4.0, "holdout_rmse_deg": 0.9},
            {"channel": "elevation", "method": "local_polynomial", "lag_proxy_seconds": 0.03, "reconstruction_rmse_deg": 0.7, "noise_proxy_deg_per_s": 3.8, "edge_proxy_deg_per_s": 4.2, "holdout_rmse_deg": 1.0},
            {"channel": "azimuth", "method": "savgol", "lag_proxy_seconds": 0.01, "reconstruction_rmse_deg": 0.2, "noise_proxy_deg_per_s": 1.8, "edge_proxy_deg_per_s": 2.2, "holdout_rmse_deg": 0.4},
            {"channel": "elevation", "method": "savgol", "lag_proxy_seconds": 0.01, "reconstruction_rmse_deg": 0.3, "noise_proxy_deg_per_s": 2.1, "edge_proxy_deg_per_s": 2.4, "holdout_rmse_deg": 0.5},
        ]
    )

    selection = select_rate_method(rate_metrics)

    assert selection["selected_method"] == "savgol"
    assert selection["ranking"][0]["method"] == "savgol"


def test_select_rate_method_does_not_let_tiny_lag_win_over_far_better_overall_fit():
    rate_metrics = pd.DataFrame(
        [
            {"channel": "azimuth", "method": "zero_lag_noisy", "lag_proxy_seconds": 0.0, "reconstruction_rmse_deg": 0.65, "noise_proxy_deg_per_s": 5.0, "edge_proxy_deg_per_s": 6.0, "holdout_rmse_deg": 0.7},
            {"channel": "elevation", "method": "zero_lag_noisy", "lag_proxy_seconds": 0.0, "reconstruction_rmse_deg": 0.60, "noise_proxy_deg_per_s": 5.5, "edge_proxy_deg_per_s": 6.2, "holdout_rmse_deg": 0.8},
            {"channel": "azimuth", "method": "small_lag_clean", "lag_proxy_seconds": 0.01, "reconstruction_rmse_deg": 0.20, "noise_proxy_deg_per_s": 1.5, "edge_proxy_deg_per_s": 2.0, "holdout_rmse_deg": 0.3},
            {"channel": "elevation", "method": "small_lag_clean", "lag_proxy_seconds": 0.01, "reconstruction_rmse_deg": 0.25, "noise_proxy_deg_per_s": 1.8, "edge_proxy_deg_per_s": 2.2, "holdout_rmse_deg": 0.35},
        ]
    )

    selection = select_rate_method(rate_metrics)

    assert selection["selected_method"] == "small_lag_clean"
    assert selection["ranking"][0]["method"] == "small_lag_clean"
