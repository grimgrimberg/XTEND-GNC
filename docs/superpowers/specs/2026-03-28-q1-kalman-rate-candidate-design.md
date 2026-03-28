# Q1 Kalman Rate Candidate Design

## Goal

Add a lightweight causal constant-velocity Kalman filter as an additional Q1 world-angle rate-estimation candidate, while preserving the current submission workflow that compares multiple estimators and selects one using the existing noise/latency-first policy.

## Proven Constraints

- The assignment requires world-frame azimuth/elevation rates and explicitly says the method should aim to minimize both noise and latency.
- The current Q1 pipeline already computes synchronized `world_az_deg` and `world_el_deg` signals on the camera timeline.
- The current repo already compares multiple derivative estimators (`gradient`, `savgol`, `local_polynomial`, `spline`) and selects one using score-based ranking.
- The requested Kalman filter must be lightweight, causal, and implemented with visible math using `numpy` only for the filtering step.
- The filter must estimate state `[theta, theta_dot]^T` independently for azimuth and elevation.

## Scope

In scope:

- Add a causal 1D linear Kalman filter with a constant-velocity kinematic model.
- Integrate it as a new `kalman_cv` candidate in the existing rate-comparison pipeline.
- Expose both filtered angle and estimated rate for azimuth and elevation.
- Add reviewer-facing plots that show raw measurements, filtered angles, and Kalman rates.
- Add focused regression tests around Kalman behavior and artifact generation.

Out of scope:

- Replacing the existing rate-selection policy.
- Refactoring every estimator into a new common result object.
- Changing the Q1 synchronization or frame-convention logic.

## Alternatives Considered

### 1. Add `kalman_cv` as one more candidate in the current pipeline

This keeps the current structure intact: rate candidates remain comparable, the selector remains free to choose, and the Kalman filter can be inspected without forcing it to become the submission answer.

This is the recommended option because it is the smallest correct change and fits the repo's current design.

### 2. Refactor all estimators to return `{filtered_angle, rate}`

This is architecturally cleaner but would expand the diff across selection, persistence, and plotting paths that are already working. It adds risk without improving the assignment outcome.

### 3. Pre-filter then differentiate

This would be smaller, but it is not the requested constant-velocity state estimator and hides the model structure the user wants to explain.

## Selected Design

### Filter model

Add a small Kalman helper in `helpers/q1_pipeline.py` with:

- State: `x = [theta_deg, theta_rate_deg_s]^T`
- Transition: `F(dt) = [[1, dt], [0, 1]]`
- Observation: `H = [1, 0]`
- Process covariance:
  - `Q(dt) = sigma_a^2 * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]`
- Measurement covariance:
  - `R = [[sigma_z^2]]`

The implementation will run one forward predict/update pass only. No smoothing pass and no future samples will be used.

### Signal handling

- Azimuth will be unwrapped before filtering so the estimator does not see false jumps at `+-180 deg`.
- Elevation will be filtered directly.
- The Kalman result will return:
  - `filtered_angle_deg`
  - `estimated_rate_deg_s`
  - resolved tuning values used for that channel
- For display/storage, filtered azimuth will also be wrapped back into the conventional angle range while keeping the unwrapped version available if needed internally.

### Tuning approach

The filter will accept explicit `sigma_z` and `sigma_a` inputs, but the default Q1 path will derive them from the observed signal so the feature works without manual tuning.

The defaults will be simple and explainable:

- `sigma_z` from a robust measurement-noise proxy based on adjacent angle increments.
- `sigma_a` from a conservative rate-change scale derived from the raw gradient.

The exact resolved values will be recorded in `q1_summary.json` so the tuning remains inspectable and not magical.

### Pipeline integration

`run_q1_analysis()` will:

- compute Kalman azimuth and elevation results once
- add `kalman_cv` to `rate_candidates`
- persist Kalman rate columns to `q1_world_angles.csv`
- persist filtered-angle columns so the user can inspect the state estimate directly
- add Kalman tuning metadata to the Q1 summary

The existing rate-selection logic will remain unchanged apart from ranking one extra candidate.

### Visualization

Existing plots:

- `q1_rate_estimates.png` will include `kalman_cv` in the same comparison view as the other methods.
- `q1_rate_raw_vs_clean.png` will remain selection-driven so it still shows the chosen estimator versus the raw gradient.

New plot:

- `q1_kalman_tracking.png` will show, for azimuth and elevation:
  - raw world angle versus filtered world angle
  - raw gradient versus Kalman estimated rate

This satisfies the explicit request to show filtered angles and estimated rates alongside the raw measurements even if the Kalman estimator is not ultimately selected by the ranking policy.

## Error Handling And Edge Cases

- Very short input sequences will fall back to a simple gradient path instead of pretending the Kalman state is meaningful.
- Nonpositive or repeated `dt` values will be guarded so covariance propagation stays finite.
- The filter will initialize from the first measurement with zero initial rate and a broad covariance so early samples do not overconstrain the estimate.
- Azimuth unwrap/wrap behavior will be isolated to the Kalman helper so the rest of Q1 keeps its current conventions.

## Testing Strategy

Follow TDD:

1. Add a synthetic noisy constant-rate test showing Kalman rate RMSE beats the raw gradient.
2. Add a shape/causality-oriented test on irregular sample spacing to verify same-length outputs and finite estimates.
3. Update the Q1 artifact regression test so the static artifact set includes the new Kalman plot.
4. After implementation, run targeted tests, then full `pytest`, then regenerate Q1 artifacts with `python Q1.py --skip-animation`.

## Risks And Mitigations

- Risk: poor default tuning could make Kalman look artificially good or unnecessarily sluggish.
  - Mitigation: keep tuning heuristic simple, expose the resolved values, and let the selector compare Kalman against the existing methods rather than forcing it as the answer.

- Risk: adding filtered-angle persistence could bloat the Q1 CSV with unclear columns.
  - Mitigation: use explicit column names that make the source and units obvious.

- Risk: azimuth wrap handling could introduce visual discontinuities.
  - Mitigation: filter on the unwrapped signal and only wrap the display column after estimation.

## Success Criteria

The change is successful if:

- `kalman_cv` appears as a valid Q1 rate-estimation candidate.
- The filter uses a visible causal constant-velocity Kalman formulation implemented with `numpy`.
- Q1 outputs include filtered-angle and Kalman-rate visibility for both channels.
- Existing behavior remains intact unless the selector legitimately ranks `kalman_cv` first.
- Tests and regenerated Q1 artifacts pass cleanly.
