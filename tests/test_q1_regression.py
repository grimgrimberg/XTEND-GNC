import numpy as np
import pandas as pd

from helpers.q1_pipeline import build_rate_metrics, estimate_ground_footprint


def test_estimate_ground_footprint_only_keeps_forward_intersections():
    positions = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]])
    los_world = np.array([[1.0, 0.0, -1.0], [1.0, 0.0, 1.0]], dtype=float)
    los_world /= np.linalg.norm(los_world, axis=1, keepdims=True)

    footprint = estimate_ground_footprint(positions, los_world)

    assert bool(footprint.loc[0, "valid"])
    assert not bool(footprint.loc[1, "valid"])
    assert np.isclose(footprint.loc[0, "x_m"], 10.0, atol=1e-6)


def test_build_rate_metrics_reports_every_candidate_method():
    time_s = np.linspace(0.0, 1.0, 11)
    synchronized = pd.DataFrame(
        {
            "held_world_az_deg": np.linspace(0.0, 5.0, len(time_s)),
            "held_world_el_deg": np.linspace(-2.0, 1.0, len(time_s)),
            "world_az_deg": np.linspace(0.0, 5.0, len(time_s)) + 0.1,
            "world_el_deg": np.linspace(-2.0, 1.0, len(time_s)) - 0.1,
        }
    )
    rate_candidates = {
        "gradient": {"azimuth": np.ones_like(time_s), "elevation": np.ones_like(time_s) * 2.0},
        "local_polynomial": {"azimuth": np.ones_like(time_s) * 1.2, "elevation": np.ones_like(time_s) * 2.2},
        "savgol": {"azimuth": np.ones_like(time_s) * 1.1, "elevation": np.ones_like(time_s) * 2.1},
    }

    metrics = build_rate_metrics(time_s, synchronized, rate_candidates)

    assert set(metrics["method"]) == {"gradient", "local_polynomial", "savgol"}
    assert set(metrics["channel"]) == {"azimuth", "elevation"}
