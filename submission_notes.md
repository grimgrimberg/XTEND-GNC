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
- Select the reported rate method by a weighted normalized composite score:
  - mean absolute lag, weight `0.35`
  - mean reconstruction RMSE, weight `0.30`
  - mean noise proxy, weight `0.20`
  - mean edge proxy, weight `0.10`
  - mean holdout RMSE, weight `0.05`
- This keeps temporal fidelity as the top concern for a derivative estimate, but a tiny lag advantage is not allowed to dominate a much cleaner overall rate trace.

## Q1 Bugs Fixed During Review

- The previous LOS conversion used a tangent-based approximation that was only locally correct near small angles. This distorted general azimuth/elevation pairs and could bias the downstream world-angle solution.
- The previous top-down plot normalized a read-only NumPy view in place, which crashed Q1 mid-run and left only a partial artifact set in `outputs/q1/`.
- The previous summary claimed the selected rate method came from quantitative comparison, but the code hardcoded `local_polynomial`.
- The previous rate selector was lexicographic and over-weighted lag. The current selector still prioritizes lag for rate estimation, but balances it against reconstruction quality and smoothness.
- The previous write-up spoke too confidently about the “true” convention selection. The current repo treats it as diagnostic evidence under an explicit camera/body model, not as direct truth.

## Q2 Method

- Baseline target motion follows the initial line-of-sight direction at constant speed `13 m/s`.
- The interceptor is simulated with numerical clipping of:
  - max speed
  - max `|ax|`
  - max `|ay|`
- Guidance laws compared:
  - `predictive`
  - `pure`
  - `pn`
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

Observed on `2026-03-26`:

- `python3 -m pytest -q -s` -> `45 passed in 590.91s`
- `python3 Q1.py` -> completed cleanly and regenerated the full Q1 artifact set
- `python3 Q2.py` -> completed cleanly and regenerated:
  - `outputs/q2/q2_summary.json`
  - `outputs/q2/q2_overview_matrix.png`
  - `12` scenario bundles under `outputs/q2/<target_mode>/<guidance_mode>/`

Current headline runtime results:

- Q1 selected convention:
  - quaternion interpreted as `world-to-body`
  - camera azimuth positive right
  - camera elevation positive down
  - gimbal positive pitch down
- Q1 selected rate method:
  - `local_polynomial`
- Q1 note on `savgol`:
  - visually smoother, but it carries slightly higher lag and worse reconstruction RMSE than `local_polynomial` on this log, so it is not the selected analysis filter
- Q2 baseline predictive result:
  - intercepted = `True`
  - intercept time = `134.4 s`
  - minimum distance = `1.997 m`
