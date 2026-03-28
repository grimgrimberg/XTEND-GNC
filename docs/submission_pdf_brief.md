# XTEND Guidance Engineer Test

Concise submission brief for the Q1 and Q2 deliverables. Full detail lives in `README.md`, `submission_notes.md`, and `outputs/`.

## Q1

- Treat the guidance log as an asynchronous event record, not a row-synchronized table.
- Use the camera stream as the master timeline, then interpolate gimbal and body attitude onto each camera sample.
- Reconstruct world-frame target azimuth/elevation under an explicit camera/body convention model.
- Compare the rate candidates with the existing noise/latency-first selector; the causal `kalman_cv` candidate is visible, not forced.
- `outputs/q1/q1_kalman_tracking.png` shows raw vs Kalman-filtered world angles and raw-gradient vs Kalman-estimated rates.
- Other key Q1 artifacts: `q1_sync_diagnostics.png`, `q1_frame_diagnostics.png`, `q1_rate_estimates.png`, `q1_rate_raw_vs_clean.png`, `q1_geometry_topdown.png`, `q1_geometry_3d.png`.

## Q2

- Simulate the constrained 3D intercept baseline from the prompt.
- Keep the target heading explicit for readability and reproduce a 12-bundle comparison matrix: 4 target modes x 3 guidance laws.
- Report interception success, intercept time, minimum miss distance, and constraint compliance.

## Key Assumptions

- Q1 convention selection is diagnostic evidence under the stated model, not direct ground truth.
- The Q1 `kalman_cv` candidate uses a causal `[theta, theta_dot]` state and is compared against the other estimators.
- Q2 uses the prompt's speed and acceleration limits; non-specified behavior is documented as a comparison assumption.

## Recommended Review Order

1. `outputs/q1/q1_sync_diagnostics.png`
2. `outputs/q1/q1_frame_diagnostics.png`
3. `outputs/q1/q1_kalman_tracking.png`
4. `outputs/q1/q1_rate_estimates.png`
5. `outputs/q1/q1_geometry_topdown.png`
6. `outputs/q1/q1_geometry_3d.png`
7. `outputs/q2/q2_overview_matrix.png`
8. `outputs/q2/straight/predictive/trajectory.png`
9. `outputs/q2/reactive/predictive/trajectory.png`
10. `outputs/q2/reactive/pure/engagement.png`
