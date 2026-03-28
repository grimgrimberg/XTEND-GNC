from __future__ import annotations

from itertools import product
from pathlib import Path

import helpers.plotting as plotting
from helpers.paths import data_path
from helpers.q1_pipeline import run_q1_analysis
from helpers.q2_simulation import run_q2_analysis


def test_matplotlib_uses_headless_backend():
    assert "agg" in plotting.plt.get_backend().lower()


def test_run_q1_analysis_completes_and_emits_full_static_artifact_set(tmp_path):
    summary = run_q1_analysis(tmp_path, data_path("examGuidance.csv"), render_animation=False)

    artifact_names = {Path(path).name for path in summary["artifact_paths"]}
    assert {
        "q1_world_angles.csv",
        "q1_convention_candidates.csv",
        "q1_bundle_points.csv",
        "q1_ground_footprint.csv",
        "q1_sync_diagnostics.png",
        "q1_frame_diagnostics.png",
        "q1_sync_quality.png",
        "q1_world_angles.png",
        "q1_rate_estimates.png",
        "q1_rate_raw_vs_clean.png",
        "q1_kalman_tracking.png",
        "q1_camera_fov.png",
        "q1_bundle_residuals.png",
        "q1_geometry_topdown.png",
        "q1_geometry_3d.png",
    }.issubset(artifact_names)
    assert Path(summary["summary_path"]).exists()
    assert summary["selected_rate_method"] == summary["rate_selection"]["selected_method"]


def test_run_q1_analysis_summary_includes_kalman_cv_ranking_and_tuning(tmp_path):
    summary = run_q1_analysis(tmp_path, data_path("examGuidance.csv"), render_animation=False)

    ranking_methods = {row["method"] for row in summary["rate_selection"]["ranking"]}
    assert "kalman_cv" in ranking_methods

    kalman_tuning = summary["kalman_tuning"]
    assert set(kalman_tuning) == {"azimuth", "elevation"}
    for channel_name in ("azimuth", "elevation"):
        channel_tuning = kalman_tuning[channel_name]
        assert channel_tuning["sigma_z_deg"] > 0.0
        assert channel_tuning["sigma_a_deg_s2"] > 0.0

    dataframe = __import__("pandas").read_csv(tmp_path / "q1_world_angles.csv")
    assert {
        "world_az_kalman_deg",
        "world_az_kalman_unwrapped_deg",
        "world_el_kalman_deg",
        "az_rate_kalman_cv_deg_s",
        "el_rate_kalman_cv_deg_s",
    }.issubset(dataframe.columns)


def test_run_q2_analysis_emits_12_bundle_directories_with_required_static_files(tmp_path):
    summary = run_q2_analysis(
        tmp_path,
        horizon_s=40.0,
        render_animation=False,
    )

    expected_pairs = set(product(("straight", "weave", "bounded_turn", "reactive"), ("predictive", "pure", "pn")))
    actual_pairs = {
        (bundle["target_mode"], bundle["guidance_mode"])
        for bundle in summary["scenario_bundles"]
    }
    assert actual_pairs == expected_pairs

    for target_mode, guidance_mode in expected_pairs:
        bundle_dir = tmp_path / target_mode / guidance_mode
        assert (bundle_dir / "summary.json").exists()
        assert (bundle_dir / "trajectory.png").exists()
        assert (bundle_dir / "constraints.png").exists()
        assert (bundle_dir / "engagement.png").exists()
        assert not (bundle_dir / "animation.gif").exists()

    assert (tmp_path / "q2_summary.json").exists()
    assert (tmp_path / "q2_overview_matrix.png").exists()


def test_run_q2_analysis_routes_each_bundle_to_the_matching_animation_result(tmp_path, monkeypatch):
    recorded_calls: list[tuple[str, str, str]] = []

    def fake_create_q2_animation(output_path, result, mode_name, plot_config):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("gif placeholder", encoding="utf-8")
        recorded_calls.append((result.target_mode, result.guidance_mode, output_path.name))

    monkeypatch.setattr("helpers.q2_visualization.create_q2_animation", fake_create_q2_animation)

    run_q2_analysis(
        tmp_path,
        horizon_s=25.0,
        render_animation=True,
    )

    expected_pairs = set(product(("straight", "weave", "bounded_turn", "reactive"), ("predictive", "pure", "pn")))
    observed_pairs = {(target_mode, guidance_mode) for target_mode, guidance_mode, _ in recorded_calls}

    assert observed_pairs == expected_pairs
    assert all(file_name == "animation.gif" for _, _, file_name in recorded_calls)
