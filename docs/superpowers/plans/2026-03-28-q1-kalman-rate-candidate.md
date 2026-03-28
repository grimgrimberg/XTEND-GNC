# Q1 Kalman Rate Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight causal constant-velocity Kalman filter as an additional Q1 azimuth/elevation rate-estimation candidate, persist its filtered-angle outputs, and generate reviewer-facing plots that show the Kalman estimate beside the raw measurements.

**Architecture:** Keep the current Q1 estimator-comparison architecture intact. Add one new Kalman helper in `helpers/q1_pipeline.py`, expose it through the existing rate-candidate flow as `kalman_cv`, and add one dedicated visualization that always shows the Kalman angle/rate behavior even if another estimator remains selected by the current noise/latency-first ranking policy.

**Tech Stack:** Python 3.10+, `numpy`, existing repo `pandas`/`matplotlib`/`pytest`

---

## File Map

- Modify: `helpers/q1_pipeline.py:38-713`
  - add Kalman result dataclass
  - add causal constant-velocity Kalman helper and tuning heuristic
  - expose `kalman_cv` through `estimate_angle_rate`
  - persist Kalman columns and summary metadata in `run_q1_analysis()`
- Modify: `helpers/q1_visualization.py:14-350`
  - add `kalman_cv` style to the estimator comparison plot
  - add a new `q1_kalman_tracking.png` artifact
- Modify: `tests/test_q1_enhancements.py:49-130`
  - add Kalman synthetic-behavior tests
- Modify: `tests/test_submission_hardening.py:16-38`
  - extend the artifact contract and summary assertions
- Modify: `README.md:72-93`
  - document the new Q1 artifact and Kalman candidate
- Modify: `submission_notes.md:9-27,114-123`
  - describe the Kalman candidate and the tuning/reporting policy

## BRAT / Ralphing Rules For Execution

- Before the Kalman tuning heuristic is coded, write down BRAT in the task notes:
  - Brainstorm: candidate `sigma_z` and `sigma_a` heuristics
  - Risks: over-smoothing, unstable early transients, wrap mistakes
  - Alternatives: fixed hand-tuned sigmas vs data-driven defaults
  - Tests: synthetic constant-rate test plus real Q1 artifact inspection
- Before and after each major task, do Ralphing:
  - review assumptions
  - separate proven vs uncertain
  - verify with tests and generated plots
  - only then move forward

## Execution Notes

- Execute this plan in an isolated worktree or feature branch, not directly on `master`.
- Keep the diff scoped to Q1 Kalman support and the minimum doc updates needed to explain the new output.
- Do not change the existing rate-selection weights unless the user explicitly asks for a selector-policy change.

### Task 1: Lock Down Kalman Core Behavior With Failing Tests

**Files:**
- Modify: `tests/test_q1_enhancements.py:49-130`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from helpers.q1_pipeline import (
    estimate_angle_rate,
    estimate_angle_rate_kalman_cv,
)


def test_kalman_cv_rate_beats_gradient_on_noisy_constant_rate_signal():
    rng = np.random.default_rng(12)
    time_s = np.linspace(0.0, 15.0, 601)
    truth_rate_deg_s = np.full_like(time_s, 7.5)
    truth_angle_deg = -18.0 + truth_rate_deg_s * time_s
    noisy_angle_deg = truth_angle_deg + rng.normal(scale=0.45, size=time_s.shape)

    gradient_rate = estimate_angle_rate(time_s, noisy_angle_deg, method="gradient")
    kalman = estimate_angle_rate_kalman_cv(time_s, noisy_angle_deg)

    gradient_rmse = float(np.sqrt(np.mean((gradient_rate - truth_rate_deg_s) ** 2)))
    kalman_rmse = float(np.sqrt(np.mean((kalman.estimated_rate_deg_s - truth_rate_deg_s) ** 2)))

    assert kalman_rmse < 0.55 * gradient_rmse
    assert kalman.sigma_z_deg > 0.0
    assert kalman.sigma_a_deg_s2 > 0.0


def test_kalman_cv_handles_irregular_sampling_with_finite_outputs():
    time_s = np.array([0.00, 0.04, 0.09, 0.14, 0.23, 0.31, 0.39, 0.53, 0.61], dtype=float)
    angle_deg = np.array([2.0, 2.4, 2.9, 3.6, 4.0, 4.8, 5.1, 6.0, 6.4], dtype=float)

    kalman = estimate_angle_rate_kalman_cv(time_s, angle_deg)

    assert kalman.filtered_angle_deg.shape == angle_deg.shape
    assert kalman.estimated_rate_deg_s.shape == angle_deg.shape
    assert np.isfinite(kalman.filtered_angle_deg).all()
    assert np.isfinite(kalman.estimated_rate_deg_s).all()


def test_estimate_angle_rate_exposes_kalman_cv_method():
    time_s = np.linspace(0.0, 2.0, 21)
    angle_deg = 5.0 * np.sin(time_s)

    rates = estimate_angle_rate(time_s, angle_deg, method="kalman_cv")

    assert rates.shape == angle_deg.shape
    assert np.isfinite(rates).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_q1_enhancements.py -k kalman -v
```

Expected:

```text
FAIL tests/test_q1_enhancements.py::test_kalman_cv_rate_beats_gradient_on_noisy_constant_rate_signal
FAIL tests/test_q1_enhancements.py::test_kalman_cv_handles_irregular_sampling_with_finite_outputs
FAIL tests/test_q1_enhancements.py::test_estimate_angle_rate_exposes_kalman_cv_method
```

- [ ] **Step 3: Write the minimal Kalman core implementation**

```python
@dataclass(frozen=True)
class KalmanAngleRateEstimate:
    filtered_angle_deg: np.ndarray
    estimated_rate_deg_s: np.ndarray
    sigma_z_deg: float
    sigma_a_deg_s2: float


def _robust_scale(values: np.ndarray, floor: float) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float(floor)
    centered = values - np.median(values)
    mad = float(np.median(np.abs(centered)))
    return max(float(1.4826 * mad), float(floor))


def _resolve_kalman_sigmas(
    time_s: np.ndarray,
    angle_deg: np.ndarray,
    *,
    sigma_z_deg: float | None,
    sigma_a_deg_s2: float | None,
) -> tuple[float, float]:
    if sigma_z_deg is None:
        angle_steps = np.diff(angle_deg)
        sigma_z_deg = _robust_scale(angle_steps, floor=0.05) / np.sqrt(2.0)
    if sigma_a_deg_s2 is None:
        raw_rate = np.gradient(angle_deg, time_s)
        dt_rate = np.diff(time_s)
        safe_dt = np.where(dt_rate > 1e-6, dt_rate, np.median(dt_rate[dt_rate > 0]))
        raw_accel = np.diff(raw_rate) / safe_dt
        sigma_a_deg_s2 = _robust_scale(raw_accel, floor=0.10)
    return float(sigma_z_deg), float(sigma_a_deg_s2)


def estimate_angle_rate_kalman_cv(
    time_s: np.ndarray,
    angle_deg: np.ndarray,
    *,
    unwrap: bool = False,
    sigma_z_deg: float | None = None,
    sigma_a_deg_s2: float | None = None,
) -> KalmanAngleRateEstimate:
    time_s = np.asarray(time_s, dtype=float)
    angle_deg = np.asarray(angle_deg, dtype=float)
    angle_work = np.rad2deg(np.unwrap(np.deg2rad(angle_deg))) if unwrap else angle_deg.copy()
    if len(time_s) < 3:
        return KalmanAngleRateEstimate(
            filtered_angle_deg=angle_work.copy(),
            estimated_rate_deg_s=np.gradient(angle_work, time_s),
            sigma_z_deg=float(0.05 if sigma_z_deg is None else sigma_z_deg),
            sigma_a_deg_s2=float(0.10 if sigma_a_deg_s2 is None else sigma_a_deg_s2),
        )

    sigma_z_deg, sigma_a_deg_s2 = _resolve_kalman_sigmas(
        time_s,
        angle_work,
        sigma_z_deg=sigma_z_deg,
        sigma_a_deg_s2=sigma_a_deg_s2,
    )
    x = np.array([angle_work[0], 0.0], dtype=float)
    p = np.diag([max(sigma_z_deg**2, 1.0), max(sigma_a_deg_s2**2, 1.0)])
    h = np.array([[1.0, 0.0]], dtype=float)
    r = np.array([[sigma_z_deg**2]], dtype=float)
    filtered = np.zeros_like(angle_work)
    rates = np.zeros_like(angle_work)
    filtered[0] = x[0]
    rates[0] = x[1]

    nominal_dt = float(np.median(np.diff(time_s)[np.diff(time_s) > 0]))
    for index in range(1, len(time_s)):
        dt = float(time_s[index] - time_s[index - 1])
        dt = dt if dt > 1e-6 else nominal_dt
        f = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
        q = (sigma_a_deg_s2**2) * np.array([[0.25 * dt**4, 0.5 * dt**3], [0.5 * dt**3, dt**2]], dtype=float)
        x = f @ x
        p = f @ p @ f.T + q
        innovation = angle_work[index] - float(h @ x)
        s = h @ p @ h.T + r
        k = (p @ h.T) / s[0, 0]
        x = x + (k[:, 0] * innovation)
        p = (np.eye(2) - k @ h) @ p
        filtered[index] = x[0]
        rates[index] = x[1]

    return KalmanAngleRateEstimate(
        filtered_angle_deg=filtered,
        estimated_rate_deg_s=rates,
        sigma_z_deg=sigma_z_deg,
        sigma_a_deg_s2=sigma_a_deg_s2,
    )
```

- [ ] **Step 4: Wire `estimate_angle_rate()` to the new method**

```python
def estimate_angle_rate(
    time_s: np.ndarray,
    angle_deg: np.ndarray,
    *,
    method: str = "local_polynomial",
    unwrap: bool = False,
) -> np.ndarray:
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
    if method == "kalman_cv":
        return estimate_angle_rate_kalman_cv(time_s, angle_deg, unwrap=unwrap).estimated_rate_deg_s
    raise ValueError(f"Unsupported method: {method}")
```

- [ ] **Step 5: Run the Kalman tests to verify they pass**

Run:

```bash
python -m pytest tests/test_q1_enhancements.py -k kalman -v
```

Expected:

```text
PASSED tests/test_q1_enhancements.py::test_kalman_cv_rate_beats_gradient_on_noisy_constant_rate_signal
PASSED tests/test_q1_enhancements.py::test_kalman_cv_handles_irregular_sampling_with_finite_outputs
PASSED tests/test_q1_enhancements.py::test_estimate_angle_rate_exposes_kalman_cv_method
```

- [ ] **Step 6: Commit the core test-first change**

```bash
git add tests/test_q1_enhancements.py helpers/q1_pipeline.py
git commit -m "feat: add Kalman CV rate estimator for Q1"
```

### Task 2: Integrate Kalman Outputs Into The Q1 Pipeline And Summary

**Files:**
- Modify: `helpers/q1_pipeline.py:563-713`

- [ ] **Step 1: Write the failing pipeline contract assertions**

```python
def test_run_q1_analysis_reports_kalman_tuning_and_candidate(tmp_path):
    summary = run_q1_analysis(tmp_path, data_path("examGuidance.csv"), render_animation=False)

    ranking_methods = {row["method"] for row in summary["rate_selection"]["ranking"]}

    assert "kalman_cv" in ranking_methods
    assert "kalman_tuning" in summary
    assert {"azimuth", "elevation"} == set(summary["kalman_tuning"])
    assert summary["kalman_tuning"]["azimuth"]["sigma_z_deg"] > 0.0
    assert summary["kalman_tuning"]["elevation"]["sigma_a_deg_s2"] > 0.0
```

- [ ] **Step 2: Run the pipeline contract test to verify it fails**

Run:

```bash
python -m pytest tests/test_submission_hardening.py -k kalman_tuning -v
```

Expected:

```text
FAIL tests/test_submission_hardening.py::test_run_q1_analysis_reports_kalman_tuning_and_candidate
```

- [ ] **Step 3: Add Kalman candidate generation and persistence in `run_q1_analysis()`**

```python
kalman_az = estimate_angle_rate_kalman_cv(time_s, world_az_deg, unwrap=True)
kalman_el = estimate_angle_rate_kalman_cv(time_s, world_el_deg)

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
    "kalman_cv": {
        "azimuth": kalman_az.estimated_rate_deg_s,
        "elevation": kalman_el.estimated_rate_deg_s,
    },
}

synchronized["world_az_kalman_deg"] = ((kalman_az.filtered_angle_deg + 180.0) % 360.0) - 180.0
synchronized["world_az_kalman_unwrapped_deg"] = kalman_az.filtered_angle_deg
synchronized["world_el_kalman_deg"] = kalman_el.filtered_angle_deg
synchronized["az_rate_kalman_cv_deg_s"] = kalman_az.estimated_rate_deg_s
synchronized["el_rate_kalman_cv_deg_s"] = kalman_el.estimated_rate_deg_s
```

- [ ] **Step 4: Add summary metadata for resolved Kalman tuning**

```python
summary["kalman_tuning"] = {
    "azimuth": {
        "sigma_z_deg": float(kalman_az.sigma_z_deg),
        "sigma_a_deg_s2": float(kalman_az.sigma_a_deg_s2),
    },
    "elevation": {
        "sigma_z_deg": float(kalman_el.sigma_z_deg),
        "sigma_a_deg_s2": float(kalman_el.sigma_a_deg_s2),
    },
}
```

- [ ] **Step 5: Do a BRAT / Ralphing tuning checkpoint on the real Q1 dataset**

Run:

```bash
python Q1.py --skip-animation --output-dir outputs/q1_kalman_tuning_check
@'
import json
from pathlib import Path
summary = json.loads(Path("outputs/q1_kalman_tuning_check/q1_summary.json").read_text())
print("selected_rate_method =", summary["selected_rate_method"])
print("kalman_tuning =", summary["kalman_tuning"])
for row in summary["rate_selection"]["ranking"]:
    if row["method"] == "kalman_cv":
        print("kalman_ranking_row =", row)
'@ | python -
```

Expected:

- one line starting with `selected_rate_method = `
- one line starting with `kalman_tuning = {'azimuth': {'sigma_z_deg':`
- one line starting with `kalman_ranking_row = {'method': 'kalman_cv'`

Review before moving on:

- Proven:
  - the filter runs end-to-end on the real dataset
  - the tuning values are finite and reported
  - `kalman_cv` appears in the selector ranking
- Uncertain:
  - whether the visual smoothness/lag trade is presentation-quality yet
  - whether the default `sigma_a` floor is too aggressive or too sluggish

If the ranking row shows obviously extreme lag or noise, adjust only the heuristic floors and rerun this checkpoint before proceeding.

- [ ] **Step 6: Run the pipeline contract test to verify it passes**

Run:

```bash
python -m pytest tests/test_submission_hardening.py -k kalman_tuning -v
```

Expected:

```text
PASSED tests/test_submission_hardening.py::test_run_q1_analysis_reports_kalman_tuning_and_candidate
```

- [ ] **Step 7: Commit the pipeline integration**

```bash
git add helpers/q1_pipeline.py tests/test_submission_hardening.py
git commit -m "feat: persist Q1 Kalman tracking outputs"
```

### Task 3: Add Kalman Visualization And Artifact Coverage

**Files:**
- Modify: `helpers/q1_visualization.py:14-350`
- Modify: `tests/test_submission_hardening.py:16-38`

- [ ] **Step 1: Extend the failing artifact contract**

```python
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
```

- [ ] **Step 2: Run the artifact contract to verify it fails**

Run:

```bash
python -m pytest tests/test_submission_hardening.py::test_run_q1_analysis_completes_and_emits_full_static_artifact_set -v
```

Expected:

```text
FAIL tests/test_submission_hardening.py::test_run_q1_analysis_completes_and_emits_full_static_artifact_set
```

- [ ] **Step 3: Add `kalman_cv` styling to the estimator comparison plot**

```python
styles = {
    "gradient": {"color": PALETTE["soft"], "linewidth": 1.1},
    "savgol": {"color": PALETTE["accent"], "linewidth": 1.4},
    "local_polynomial": {"color": PALETTE["success"], "linewidth": 2.0},
    "spline": {"color": PALETTE["danger"], "linewidth": 1.2, "alpha": 0.75},
    "kalman_cv": {"color": PALETTE["secondary"], "linewidth": 1.8, "linestyle": "--"},
}
```

- [ ] **Step 4: Add the dedicated Kalman tracking plot**

```python
def plot_kalman_tracking(output_path: Path, synchronized: pd.DataFrame, rate_candidates: dict[str, dict[str, np.ndarray]], plot_config: PlotConfig) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(16.0, 9.5), sharex="col", constrained_layout=True)
    time_s = synchronized["time_s"].to_numpy()

    axes[0, 0].plot(time_s, synchronized["world_az_deg"], color=PALETTE["soft"], linewidth=0.9, label="raw azimuth")
    axes[0, 0].plot(time_s, synchronized["world_az_kalman_deg"], color=PALETTE["secondary"], linewidth=1.8, label="Kalman azimuth")
    axes[0, 0].set_ylabel("Angle [deg]")
    axes[0, 0].set_title("Azimuth Tracking")
    axes[0, 0].legend(loc="upper right")

    axes[1, 0].plot(time_s, rate_candidates["gradient"]["azimuth"], color=PALETTE["soft"], linewidth=0.9, label="gradient")
    axes[1, 0].plot(time_s, rate_candidates["kalman_cv"]["azimuth"], color=PALETTE["secondary"], linewidth=1.8, label="Kalman rate")
    axes[1, 0].set_ylabel("Rate [deg/s]")
    axes[1, 0].set_xlabel("Time Since First Camera Sample [s]")
    axes[1, 0].legend(loc="upper right")

    axes[0, 1].plot(time_s, synchronized["world_el_deg"], color=PALETTE["soft"], linewidth=0.9, label="raw elevation")
    axes[0, 1].plot(time_s, synchronized["world_el_kalman_deg"], color=PALETTE["secondary"], linewidth=1.8, label="Kalman elevation")
    axes[0, 1].set_ylabel("Angle [deg]")
    axes[0, 1].set_title("Elevation Tracking")
    axes[0, 1].legend(loc="upper right")

    axes[1, 1].plot(time_s, rate_candidates["gradient"]["elevation"], color=PALETTE["soft"], linewidth=0.9, label="gradient")
    axes[1, 1].plot(time_s, rate_candidates["kalman_cv"]["elevation"], color=PALETTE["secondary"], linewidth=1.8, label="Kalman rate")
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
```

- [ ] **Step 5: Register the new plot artifact in `create_q1_visuals()`**

```python
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
]
```

- [ ] **Step 6: Run the artifact contract to verify it passes**

Run:

```bash
python -m pytest tests/test_submission_hardening.py::test_run_q1_analysis_completes_and_emits_full_static_artifact_set -v
```

Expected:

```text
PASSED tests/test_submission_hardening.py::test_run_q1_analysis_completes_and_emits_full_static_artifact_set
```

- [ ] **Step 7: Do the Ralphing visual check on the real Q1 artifacts**

Run:

```bash
python Q1.py --skip-animation --output-dir outputs/q1_kalman_visual_check
```

Review before moving on:

- Proven:
  - `q1_kalman_tracking.png` exists
  - `q1_rate_estimates.png` now includes `kalman_cv`
- Uncertain:
  - whether the Kalman angle trace is visibly cleaner without looking sluggish
  - whether the Kalman rate panel is meaningfully better than raw gradient for a reviewer

Open these artifacts and inspect them:

- `outputs/q1_kalman_visual_check/q1_kalman_tracking.png`
- `outputs/q1_kalman_visual_check/q1_rate_estimates.png`
- `outputs/q1_kalman_visual_check/q1_rate_raw_vs_clean.png`

If the Kalman trace is visibly over-damped, return to Task 2 Step 5, adjust the heuristic floors, and rerun the visual check.

- [ ] **Step 8: Commit the plotting changes**

```bash
git add helpers/q1_visualization.py tests/test_submission_hardening.py
git commit -m "feat: add Q1 Kalman tracking plot"
```

### Task 4: Update Reviewer Docs And Run Full Verification

**Files:**
- Modify: `README.md:72-93`
- Modify: `submission_notes.md:9-27,114-123`

- [ ] **Step 1: Update the README output layout and Q1 method description**

```markdown
- The Q1 rate comparison now includes a causal `kalman_cv` candidate built from a 1D constant-velocity Kalman filter.
- `outputs/q1/` also contains `q1_kalman_tracking.png`, which shows raw vs Kalman-filtered world angles and raw-gradient vs Kalman-estimated rates.
```

- [ ] **Step 2: Update `submission_notes.md` with the Kalman tuning explanation**

```markdown
- Added a causal constant-velocity Kalman candidate for Q1 rate estimation.
- The filter tracks `[theta, theta_dot]^T` independently for azimuth and elevation.
- Default `sigma_z` and `sigma_a` are data-driven heuristics and are written into `q1_summary.json` so the tuning remains reviewable.
- The selector still ranks all candidates using the same noise/latency-first policy; Kalman is compared, not silently forced.
```

- [ ] **Step 3: Run focused regression tests**

Run:

```bash
python -m pytest tests/test_q1_enhancements.py tests/test_submission_hardening.py -q
```

Expected:

- `tests/test_q1_enhancements.py` passes
- `tests/test_submission_hardening.py` passes

- [ ] **Step 4: Run full repository verification**

Run:

```bash
python -m pytest -q
python Q1.py --skip-animation --output-dir outputs/q1_verification_kalman
```

Expected:

- `python -m pytest -q` exits with code `0`
- `python Q1.py --skip-animation --output-dir outputs/q1_verification_kalman` exits with code `0`
- `outputs/q1_verification_kalman/q1_summary.json` exists

- [ ] **Step 5: Final Ralphing pass**

Checklist:

- Proven:
  - Kalman helper is causal and `numpy`-only
  - `kalman_cv` is ranked by the existing selector
  - filtered angles and Kalman rates are persisted and plotted
  - tests and fresh Q1 verification passed
- Uncertain:
  - whether Kalman should become the preferred submission answer on this dataset

Do not change selector policy here. Report the observed ranking and leave the decision explicit.

- [ ] **Step 6: Commit the docs and verification-backed finish**

```bash
git add README.md submission_notes.md outputs/q1_verification_kalman
git commit -m "docs: document Q1 Kalman rate candidate"
```
