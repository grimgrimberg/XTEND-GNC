# Submission Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve submission clarity without drifting from the assignment by refining the Q2 baseline geometry slightly, polishing Q2 visuals, clarifying Q1 rate-selection rationale, and adding targeted reviewer-facing comments.

**Architecture:** Keep the existing `Q1.py` / `Q2.py` entry points and `helpers/` module split intact. Constrain changes to the Q2 baseline configuration and plotting layer, the Q1 rate-selection policy/narrative, and submission-facing markdown so the behavior remains easy to verify against the current tests and artifacts.

**Tech Stack:** Python 3.13, NumPy, Pandas, Matplotlib, SciPy, Pytest

---

### Task 1: Lock Down The New Visual Contracts

**Files:**
- Modify: `tests/test_plot_annotations.py`
- Modify: `tests/test_q2_regression.py`

- [ ] **Step 1: Write failing tests for the overview-matrix miss label and the split-unit constraint plot**

```python
def test_q2_overview_matrix_marks_misses_explicitly(monkeypatch, tmp_path):
    import helpers.q2_visualization as q2_viz

    captured = _capture_figure(monkeypatch, q2_viz)
    scenario_grid = pd.DataFrame(
        [
            {"target_mode": "straight", "guidance_mode": "predictive", "intercept_time_s": 10.0, "minimum_distance_m": 1.0},
            {"target_mode": "straight", "guidance_mode": "pure", "intercept_time_s": 12.0, "minimum_distance_m": 1.5},
            {"target_mode": "reactive", "guidance_mode": "predictive", "intercept_time_s": np.nan, "minimum_distance_m": 4.5},
            {"target_mode": "reactive", "guidance_mode": "pure", "intercept_time_s": 18.0, "minimum_distance_m": 1.8},
        ]
    )

    q2_viz.plot_overview_matrix(tmp_path / "overview.png", scenario_grid, PlotConfig())
    figure = captured["figure"]
    text_strings = {text.get_text() for axis in figure.axes for text in axis.texts}
    assert "MISS" in text_strings
    assert "nan" not in {value.lower() for value in text_strings}
    plt.close(figure)


def test_q2_constraint_plot_uses_explicit_units(monkeypatch, tmp_path):
    import helpers.q2_visualization as q2_viz

    captured = _capture_figure(monkeypatch, q2_viz)
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
    result = SimulationResult(
        time_s=time_s,
        target_position_m=np.column_stack([100.0 - 2.0 * time_s, np.full_like(time_s, 20.0), np.full_like(time_s, 30.0)]),
        target_velocity_mps=np.tile(np.array([6.0, 0.0, 0.0]), (len(time_s), 1)),
        interceptor_position_m=np.column_stack([10.0 * time_s, np.zeros_like(time_s), np.zeros_like(time_s)]),
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

    q2_viz.plot_constraint_traces(tmp_path / "constraints.png", config, result, PlotConfig())
    figure = captured["figure"]
    ylabels = [axis.get_ylabel() for axis in figure.axes if axis.get_title() or axis.get_ylabel()]
    assert "Speed / Rate" not in ylabels
    assert "LOS Rate [rad/s]" in ylabels
    assert "Closing Speed [m/s]" in ylabels
    plt.close(figure)
```

- [ ] **Step 2: Run the targeted tests to verify they fail for the right reason**

Run:

```bash
python -m pytest tests/test_plot_annotations.py -q
```

Expected: failure because the overview matrix still emits `nan` text and the constraint plot still uses a mixed-unit bottom panel.

- [ ] **Step 3: Add a regression test for the Q2 baseline heading assumption**

```python
def test_run_q2_analysis_uses_a_fixed_documented_baseline_heading(tmp_path):
    summary = run_q2_analysis(tmp_path, horizon_s=40.0, render_animation=False)

    heading_note = summary["scenario"]["heading_assumption"]
    assert "35" in heading_note
    assert summary["scenario"]["target_velocity_mps"][0] > 0.0
```

- [ ] **Step 4: Run the specific regression test and verify it fails**

Run:

```bash
python -m pytest tests/test_q2_regression.py::test_run_q2_analysis_uses_a_fixed_documented_baseline_heading -q
```

Expected: failure because the baseline heading is still described as the initial line-of-sight default.

### Task 2: Implement The Q2 Baseline And Plot Polish

**Files:**
- Modify: `helpers/q2_simulation.py`
- Modify: `helpers/q2_visualization.py`

- [ ] **Step 1: Implement the smallest baseline-heading refinement**

Update the Q2 baseline so it uses a fixed horizontal heading of `35 deg` with the existing vertical-speed assumption preserved.

- [ ] **Step 2: Update the overview matrix text rendering**

Render miss cells as `MISS` for intercept-time heatmaps while leaving minimum-distance cells numeric.

- [ ] **Step 3: Split the bottom constraint panel into explicit closing-speed and LOS-rate panels**

Keep the existing closure, speed, and acceleration panels unchanged; replace the mixed-unit panel with separate unit-clean plots.

- [ ] **Step 4: Run the Task 1 targeted tests and verify they pass**

Run:

```bash
python -m pytest tests/test_plot_annotations.py tests/test_q2_regression.py -q
```

Expected: PASS.

### Task 3: Clarify Q1 Rate Selection And Reviewer Narrative

**Files:**
- Modify: `helpers/q1_pipeline.py`
- Modify: `README.md`
- Modify: `submission_notes.md`

- [ ] **Step 1: Add a Q1 rate-selection regression test**

Create a focused test that checks the selector prefers a low-latency, lower-noise method when lag becomes materially nonzero.

- [ ] **Step 2: Run the new Q1 selector test to verify it fails first**

Run:

```bash
python -m pytest tests/test_q1_enhancements.py -q
```

Expected: failure because the current selector is still framed as a generic weighted composite instead of an explicit noise/latency-first rule.

- [ ] **Step 3: Update the selector and comments**

Keep the current metrics, but make the policy explicitly prioritize latency and noise, with reconstruction / edge / holdout as secondary guards. Add short comments where the policy would otherwise require reverse engineering.

- [ ] **Step 4: Rewrite the submission-facing narrative**

Make `README.md` and `submission_notes.md` state:
- why the Q2 heading was adjusted slightly,
- what `response_time_s = 0.8` means,
- why the selected Q1 rate method is the best trade between noise and latency on this log,
- how the quaternion/frame choice is evidence-driven rather than magic.

- [ ] **Step 5: Re-run the Q1-focused tests**

Run:

```bash
python -m pytest tests/test_q1_enhancements.py tests/test_submission_hardening.py -q
```

Expected: PASS.

### Task 4: Full Verification

**Files:**
- Verify only

- [ ] **Step 1: Run the full test suite**

Run:

```bash
python -m pytest -q
```

Expected: PASS with no failures.

- [ ] **Step 2: Regenerate Q1 outputs**

Run:

```bash
python Q1.py --skip-animation
```

Expected: clean completion and a fresh `outputs/q1/q1_summary.json`.

- [ ] **Step 3: Regenerate Q2 outputs to a fresh directory**

Run:

```bash
python Q2.py --skip-animation --output-dir outputs/q2_verification
```

Expected: clean completion and a fresh `outputs/q2_verification/q2_summary.json`.

- [ ] **Step 4: Review the key artifacts**

Check:
- `outputs/q2_verification/q2_nominal_trajectory.png`
- `outputs/q2_verification/q2_constraint_traces.png`
- `outputs/q2_verification/q2_overview_matrix.png`
- `outputs/q1/q1_rate_estimates.png`

- [ ] **Step 5: Commit**

```bash
git add helpers/q1_pipeline.py helpers/q2_simulation.py helpers/q2_visualization.py README.md submission_notes.md tests/test_plot_annotations.py tests/test_q1_enhancements.py tests/test_q2_regression.py docs/superpowers/plans/2026-03-28-submission-polish.md
git commit -m "polish submission clarity and reviewer-facing visuals"
```
