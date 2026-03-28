# XTEND Guidance Engineer Test Submission

This repository contains a submission-ready answer to the two tasks in `1000191377.pdf`.

- `Q1.py` reconstructs world-frame target azimuth/elevation from the asynchronous guidance log, estimates azimuth/elevation rates, and emits reviewer-facing diagnostics.
- `Q2.py` simulates the constrained 3D intercept baseline from the prompt and also generates a labeled 12-bundle extension matrix (`4 target modes x 3 guidance laws`) for comparison.

## Quick Start

Use Python `3.10+`.

```bash
python3 -m pip install -r requirements.txt
python3 Q1.py
python3 Q2.py
python3 -m pytest -q -s
```

Fast paths:

```bash
python3 Q1.py --skip-animation
python3 Q2.py --skip-animation
```

## Assignment Scope

### Q1

- Inputs used directly from the dataset:
  - camera target angles: `target_x_deg`, `target_y_deg`
  - drone attitude quaternion
  - gimbal pitch angle
- Required outputs:
  - world-frame target azimuth/elevation in `NWU`
  - azimuth rate / elevation rate

### Q2

- Assignment baseline:
  - target starts at `[800, 200, 300]` in `NWU`
  - target speed is fixed at `13 m/s`
  - interceptor starts at `[0, 0, 0]`
  - interceptor constraints:
    - max speed `20.0 m/s`
    - max world-`X` acceleration `2.5 m/s^2`
    - max world-`Y` acceleration `1.0 m/s^2`
- Required outputs:
  - whether interception occurs
  - intercept time
  - clear explanatory plots

## Key Assumptions

### Q1

- The CSV is treated as an interleaved event log, not a row-synchronized table.
- Camera timestamps are the master timeline because LOS exists only on camera samples.
- The camera/body alignment model is explicit, but the quaternion direction and sign conventions are still inferred from data-driven diagnostics rather than observed directly.
- The reported rate method is chosen with a noise/latency-first policy; reconstruction and edge metrics are secondary guards against a numerically pretty but untrustworthy derivative.
- Q1 geometry plots are plausibility diagnostics only. The dataset does not contain target range truth.

### Q2

- The prompt does not specify target heading. The baseline therefore fixes a mild `35 deg` horizontal heading so the straight-target plot remains readable without turning the assignment into an aggressive crossing-only case.
- The baseline preserves the climb component implied by the original radial-away default, so the only intentional geometry change is the horizontal heading.
- Only `X/Y` acceleration limits are enforced numerically because the prompt does not specify a `Z` limit.
- The guidance command uses a first-order velocity-shaping time constant `tau = 0.8 s`. This is a tuning constant, not an identified vehicle model.
- The `straight + predictive` case is the assignment-aligned baseline.
- All other target modes and guidance-law comparisons are labeled extensions, not silent changes to the assignment scenario.

## Output Layout

### Q1

`outputs/q1/` contains:

- `q1_world_angles.csv`
- `q1_convention_candidates.csv`
- `q1_bundle_points.csv`
- `q1_ground_footprint.csv`
- `q1_sync_diagnostics.png`
- `q1_frame_diagnostics.png`
- `q1_sync_quality.png`
- `q1_world_angles.png`
- `q1_rate_estimates.png`
- `q1_rate_raw_vs_clean.png`
- `q1_camera_fov.png`
- `q1_bundle_residuals.png`
- `q1_geometry_topdown.png`
- `q1_geometry_3d.png`
- `q1_geometry_animation.gif`
- `q1_summary.json`

### Q2

Top-level `outputs/q2/` contains:

- `q2_summary.json`
- `q2_overview_matrix.png`
- baseline comparison artifacts and CSV summaries

Per-scenario bundles live under:

```text
outputs/q2/<target_mode>/<guidance_mode>/
```

Each of the 12 bundles contains:

- `summary.json`
- `trajectory.png`
- `constraints.png`
- `engagement.png`
- `animation.gif` unless `--skip-animation` is used

## Review Order

If reviewing quickly, open these first:

1. `outputs/q1/q1_sync_diagnostics.png`
2. `outputs/q1/q1_frame_diagnostics.png`
3. `outputs/q1/q1_geometry_topdown.png`
4. `outputs/q1/q1_geometry_3d.png`
5. `outputs/q2/q2_overview_matrix.png`
6. `outputs/q2/straight/predictive/trajectory.png`
7. `outputs/q2/reactive/predictive/trajectory.png`
8. `outputs/q2/reactive/pure/engagement.png`
9. `submission_notes.md`

## Repo Layout

- `Q1.py`, `Q2.py`: thin CLI entry points
- `helpers/`: implementation modules
- `tests/`: lightweight regression and contract tests
- `outputs/`: generated submission artifacts
- `submission_notes.md`: concise technical memo

## Validation

Canonical verification commands:

```bash
python3 -m pytest -q -s
python3 Q1.py
python3 Q2.py
```

Fresh verification from the repo root on `2026-03-28`:

- `python -m pytest -q` -> `48 passed in 443.11s`
- `python Q1.py --skip-animation --output-dir outputs/q1_verification` -> regenerated a clean static Q1 artifact set and `outputs/q1_verification/q1_summary.json`
- `python Q2.py --skip-animation --output-dir outputs/q2_verification_final` -> regenerated `outputs/q2_verification_final/q2_summary.json`, `outputs/q2_verification_final/q2_overview_matrix.png`, and all `12` scenario bundles
