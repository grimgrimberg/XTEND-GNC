# Submission Notes

## Short Summary

- Q1 is implemented as asynchronous sensor fusion rather than row-wise CSV arithmetic.
- Q2 preserves the assignment baseline and adds a labeled 12-bundle comparison matrix for reviewer visibility.
- The markdown surface is intentionally small: `README.md` and this file are the only submission-facing narratives.

## Q1 Method

- Split the dataset into camera, gimbal, nav, and empty-row streams.
- Use the camera stream as the master timeline.
- Interpolate gimbal pitch linearly and attitude with SLERP.
- Convert camera azimuth/elevation to LOS with a standard spherical unit-vector mapping.
- Rotate LOS through gimbal pitch and attitude into the `NWU` world frame.
- Estimate rates with four candidate methods:
  - `gradient`
  - `savgol`
  - `local_polynomial`
  - `spline`
- Select the reported rate method with a weighted normalized score that is explicitly dominated by the two quantities named in the prompt:
  - mean absolute lag, weight `0.40`
  - mean noise proxy, weight `0.40`
  - mean reconstruction RMSE, weight `0.10`
  - mean edge proxy, weight `0.05`
  - mean holdout RMSE, weight `0.05`
- This keeps the decision anchored to low latency and low noise, while still rejecting methods that look smooth only because they distort the underlying angle history.

## Q1 Bugs Fixed During Review

- The previous LOS conversion used a tangent-based approximation that was only locally correct near small angles. This distorted general azimuth/elevation pairs and could bias the downstream world-angle solution.
- The previous top-down plot normalized a read-only NumPy view in place, which crashed Q1 mid-run and left only a partial artifact set in `outputs/q1/`.
- The previous summary claimed the selected rate method came from quantitative comparison, but the code hardcoded `local_polynomial`.
- The previous rate-selector narrative was more generic than the assignment requires. The current selector is framed directly around the prompt's noise-and-latency trade, with reconstruction / edge / holdout metrics acting only as secondary guardrails.
- The previous write-up spoke too confidently about the “true” convention selection. The current repo treats it as diagnostic evidence under an explicit camera/body model, not as direct truth.

## Q2 Method

- The prompt leaves target heading unspecified, so the baseline fixes a mild `35 deg` horizontal heading.
- The baseline keeps the climb component implied by the original radial-away 3D velocity. This preserves the earlier vertical assumption while avoiding a nearly collinear tail chase in the reviewer-facing 3D plot.
- The interceptor is simulated with numerical clipping of:
  - max speed
  - max `|ax|`
  - max `|ay|`
- The guidance command is a first-order velocity-shaping law:
  - `a_cmd = (v_des - v) / tau`
  - with `tau = 0.8 s`
- `tau` is a tuning constant, not a vehicle-ID result. In practice it sets how aggressively the interceptor tries to close the velocity error before the axis limits clip the command.
- Guidance laws compared:
  - `predictive`
  - `pure`
  - `pn`
- Guidance-law intent:
  - `pure`: aim at the target's current position; simplest and most curved pursuit path.
  - `predictive`: solve a lead-intercept problem against a constant-velocity target and aim at the predicted intercept point.
  - `pn`: keep the predictive baseline and add a proportional-navigation lateral correction based on LOS rate.
- Target modes compared:
  - `straight`
  - `weave`
  - `bounded_turn`
  - `reactive`
- Output contract:
  - top-level manifest and overview figure in `outputs/q2/`
  - 12 scenario bundles under `outputs/q2/<target_mode>/<guidance_mode>/`

## Q2 Bugs / Cleanup During Review

- The previous Q2 output layout was flat and did not match the requested 12-package submission contract.
- The previous GIF logic could use nominal straight-flight results even when the selected target mode was not `straight`.
- The extra top-level selected-scenario GIF was removed so the artifact surface stays organized around the bundle matrix.
- The previous baseline geometry was technically valid but visually degenerate because it was too close to a pure tail chase. The current baseline uses a small heading adjustment so the 3D plot reads clearly without turning the assignment into an extreme crossing case.
- The overview matrix now labels non-intercepts explicitly instead of leaking `NaN` into the plot text.
- The constraint plot no longer mixes closing speed and LOS rate on the same axis.

## Rejected Alternatives

- Q1 row-wise synchronization:
  - rejected because the CSV is observably interleaved and asynchronous.
- Q1 tangent LOS mapping:
  - rejected because it is not a correct general azimuth/elevation mapping.
- Q2 one-off baseline-only artifact set:
  - rejected because it hides the guidance-law comparison work the repo now supports.
- Keeping the old `docs/` narrative set:
  - rejected because it duplicated and contradicted the main submission story.

## Known Limits

- Q1 does not have target range truth, so bundle points and ground intersections are diagnostic only.
- Q1 convention ranking still depends on an explicit camera/body alignment model; it cannot prove the underlying hardware convention from the log alone.
- Q2 leaves `Z` acceleration unconstrained by default because the prompt does not provide a limit.
- Q2 evasive target modes are bounded illustrative scenarios, not claims about a validated adversarial autopilot.

## Validation

Canonical commands:

```bash
python3 -m pytest -q -s
python3 Q1.py
python3 Q2.py
```

Observed on `2026-03-28`:

- `python -m pytest -q` -> `48 passed in 443.11s`
- `python Q1.py --skip-animation --output-dir outputs/q1_verification` -> completed cleanly and regenerated the full static Q1 artifact set
- `python Q2.py --skip-animation --output-dir outputs/q2_verification_final` -> completed cleanly and regenerated:
  - `outputs/q2_verification_final/q2_summary.json`
  - `outputs/q2_verification_final/q2_overview_matrix.png`
  - `12` scenario bundles under `outputs/q2_verification_final/<target_mode>/<guidance_mode>/`

Current headline runtime results:

- Q1 selected convention:
  - quaternion interpreted as `world-to-body`
  - camera azimuth positive right
  - camera elevation positive down
  - gimbal positive pitch down
- Q1 selected rate method:
  - `local_polynomial`
- Q1 note on `savgol`:
  - visually smoother, but it carries slightly higher lag on this log; `local_polynomial` keeps the lag proxy at zero while remaining materially cleaner than raw gradient differentiation
- Q2 baseline predictive result:
  - intercepted = `True`
  - intercept time = `129.5 s`
  - minimum distance = `1.973 m`
