from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .plotting import PlotConfig, configure_matplotlib
from .q2_guidance import GuidanceConfig, analytic_lead_intercept_time, build_guidance_command, proportional_navigation_command
from .q2_targets import TargetBehaviorConfig, target_velocity_for_behavior
from .utils import to_builtin

BASELINE_Q2_HORIZONTAL_HEADING_DEG = 35.0


@dataclass(frozen=True)
class InterceptorConstraints:
    max_speed: float
    max_accel_x: float
    max_accel_y: float
    max_accel_z: float | None = None


@dataclass(frozen=True)
class SimulationConfig:
    """Full scenario description for a single Q2 interception simulation.

    The ``guidance.response_time_s`` field is the single source of truth for
    the first-order velocity-shaping time constant.  There is no separate
    ``response_time_s`` on this dataclass.
    """

    target_initial_position: np.ndarray
    interceptor_initial_position: np.ndarray
    interceptor_initial_velocity: np.ndarray
    constraints: InterceptorConstraints
    target_velocity: np.ndarray | None = None
    target_speed_mps: float = 13.0
    target_heading_rad: float | None = None
    target_vertical_speed_mps: float | None = None
    target_behavior: TargetBehaviorConfig = field(default_factory=TargetBehaviorConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    dt: float = 0.05
    horizon_s: float = 180.0
    intercept_radius_m: float = 2.0


@dataclass
class SimulationResult:
    time_s: np.ndarray
    target_position_m: np.ndarray
    target_velocity_mps: np.ndarray
    interceptor_position_m: np.ndarray
    interceptor_velocity_mps: np.ndarray
    commanded_accel_mps2: np.ndarray
    applied_accel_mps2: np.ndarray
    distance_m: np.ndarray
    closing_speed_mps: np.ndarray
    los_rate_norm_radps: np.ndarray
    guidance_mode: str
    target_mode: str
    intercepted: bool
    intercept_time_s: float | None


def make_nominal_target_velocity(initial_position: np.ndarray, speed: float) -> np.ndarray:
    initial_position = np.asarray(initial_position, dtype=float)
    norm = np.linalg.norm(initial_position)
    if norm == 0.0:
        raise ValueError("Initial position must be non-zero.")
    return speed * initial_position / norm


def make_heading_based_target_velocity(
    *,
    speed: float,
    horizontal_heading_rad: float,
    vertical_speed_mps: float,
) -> np.ndarray:
    """Build a constant-speed 3D velocity from a horizontal heading and fixed climb rate."""

    horizontal_speed = float(np.sqrt(max(speed ** 2 - vertical_speed_mps ** 2, 0.0)))
    return np.array(
        [
            horizontal_speed * np.cos(horizontal_heading_rad),
            horizontal_speed * np.sin(horizontal_heading_rad),
            vertical_speed_mps,
        ],
        dtype=float,
    )


def apply_constraints(
    *,
    current_velocity: np.ndarray,
    commanded_accel: np.ndarray,
    dt: float,
    constraints: InterceptorConstraints,
) -> tuple[np.ndarray, np.ndarray]:
    current_velocity = np.asarray(current_velocity, dtype=float)
    applied_accel = np.asarray(commanded_accel, dtype=float).copy()
    applied_accel[0] = np.clip(applied_accel[0], -constraints.max_accel_x, constraints.max_accel_x)
    applied_accel[1] = np.clip(applied_accel[1], -constraints.max_accel_y, constraints.max_accel_y)
    if constraints.max_accel_z is not None:
        applied_accel[2] = np.clip(applied_accel[2], -constraints.max_accel_z, constraints.max_accel_z)
    delta_velocity = applied_accel * dt
    next_velocity = current_velocity + delta_velocity
    speed = float(np.linalg.norm(next_velocity))
    if speed > constraints.max_speed:
        scale = _speed_feasible_delta_scale(current_velocity, delta_velocity, constraints.max_speed)
        delta_velocity *= scale
        next_velocity = current_velocity + delta_velocity
        applied_accel = delta_velocity / dt
    return next_velocity, applied_accel


def simulate_interception(config: SimulationConfig, *, guidance: str | None = None) -> SimulationResult:
    guidance_config = _resolve_guidance_config(config, override_mode=guidance)
    base_speed_mps, base_heading_rad, base_vertical_speed_mps = _resolve_target_motion(config)

    step_count = int(np.ceil(config.horizon_s / config.dt))
    target_position = np.asarray(config.target_initial_position, dtype=float).copy()
    interceptor_position = np.asarray(config.interceptor_initial_position, dtype=float).copy()
    interceptor_velocity = np.asarray(config.interceptor_initial_velocity, dtype=float).copy()
    target_velocity = target_velocity_for_behavior(
        time_s=0.0,
        current_position_m=target_position,
        interceptor_position_m=interceptor_position,
        base_speed_mps=base_speed_mps,
        base_heading_rad=base_heading_rad,
        base_vertical_speed_mps=base_vertical_speed_mps,
        config=config.target_behavior,
    )

    time_history: list[float] = []
    target_position_history: list[np.ndarray] = []
    target_velocity_history: list[np.ndarray] = []
    interceptor_position_history: list[np.ndarray] = []
    interceptor_velocity_history: list[np.ndarray] = []
    commanded_accel_history: list[np.ndarray] = []
    applied_accel_history: list[np.ndarray] = []
    distance_history: list[float] = []
    closing_speed_history: list[float] = []
    los_rate_history: list[float] = []

    applied_accel = np.zeros(3, dtype=float)
    intercepted = False
    intercept_time_s: float | None = None

    # ---- Integration scheme (documented) --------------------------------
    # Both vehicles use a semi-implicit (symplectic Euler) update:
    #   1. Compute guidance command from state at time t.
    #   2. Apply constraints → get new interceptor velocity v_i(t+1).
    #   3. Advance interceptor position: p_i(t+1) = p_i(t) + v_i(t+1)*dt.
    #   4. Advance target position FIRST: p_t(t+1) = p_t(t) + v_t(t)*dt.
    #   5. Compute target velocity for the NEXT step using p_t(t+1) and
    #      p_i(t+1) so the reactive mode sees updated geometry.
    #
    # This avoids the 1-step information-leak that would occur if the
    # target velocity were computed with the old target position but the
    # new interceptor position.
    # -------------------------------------------------------------------
    for step in range(step_count + 1):
        time_s = step * config.dt
        relative_position = target_position - interceptor_position
        relative_velocity = target_velocity - interceptor_velocity
        distance = float(np.linalg.norm(relative_position))
        los_rate_norm = float(np.linalg.norm(np.cross(relative_position, relative_velocity)) / max(distance**2, 1e-9))
        closing_speed = max(-float(np.dot(relative_position, relative_velocity)) / max(distance, 1e-9), 0.0)

        time_history.append(time_s)
        target_position_history.append(target_position.copy())
        target_velocity_history.append(target_velocity.copy())
        interceptor_position_history.append(interceptor_position.copy())
        interceptor_velocity_history.append(interceptor_velocity.copy())
        distance_history.append(distance)
        commanded_accel_history.append(np.zeros(3) if step == 0 else commanded_accel.copy())
        applied_accel_history.append(applied_accel.copy())
        closing_speed_history.append(closing_speed)
        los_rate_history.append(los_rate_norm)

        if distance <= config.intercept_radius_m:
            intercepted = True
            intercept_time_s = time_s
            break
        if step == step_count:
            break

        # --- interceptor update (symplectic: velocity first, then position)
        commanded_accel, _ = build_guidance_command(
            guidance=guidance_config,
            target_position_m=target_position,
            target_velocity_mps=target_velocity,
            interceptor_position_m=interceptor_position,
            interceptor_velocity_mps=interceptor_velocity,
            max_speed_mps=config.constraints.max_speed,
            dt_s=config.dt,
        )
        interceptor_velocity, applied_accel = apply_constraints(
            current_velocity=interceptor_velocity,
            commanded_accel=commanded_accel,
            dt=config.dt,
            constraints=config.constraints,
        )
        interceptor_position = interceptor_position + interceptor_velocity * config.dt

        # --- target update (position first with CURRENT velocity, then
        #     compute NEXT velocity from updated geometry)
        target_position = target_position + target_velocity * config.dt
        target_velocity = target_velocity_for_behavior(
            time_s=time_s + config.dt,
            current_position_m=target_position,
            interceptor_position_m=interceptor_position,
            base_speed_mps=base_speed_mps,
            base_heading_rad=base_heading_rad,
            base_vertical_speed_mps=base_vertical_speed_mps,
            config=config.target_behavior,
        )

    return SimulationResult(
        time_s=np.asarray(time_history),
        target_position_m=np.asarray(target_position_history),
        target_velocity_mps=np.asarray(target_velocity_history),
        interceptor_position_m=np.asarray(interceptor_position_history),
        interceptor_velocity_mps=np.asarray(interceptor_velocity_history),
        commanded_accel_mps2=np.asarray(commanded_accel_history),
        applied_accel_mps2=np.asarray(applied_accel_history),
        distance_m=np.asarray(distance_history),
        closing_speed_mps=np.asarray(closing_speed_history),
        los_rate_norm_radps=np.asarray(los_rate_history),
        guidance_mode=guidance_config.mode,
        target_mode=config.target_behavior.mode,
        intercepted=intercepted,
        intercept_time_s=intercept_time_s,
    )


def result_to_dataframe(result: SimulationResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": result.time_s,
            "target_x_m": result.target_position_m[:, 0],
            "target_y_m": result.target_position_m[:, 1],
            "target_z_m": result.target_position_m[:, 2],
            "target_vx_mps": result.target_velocity_mps[:, 0],
            "target_vy_mps": result.target_velocity_mps[:, 1],
            "target_vz_mps": result.target_velocity_mps[:, 2],
            "interceptor_x_m": result.interceptor_position_m[:, 0],
            "interceptor_y_m": result.interceptor_position_m[:, 1],
            "interceptor_z_m": result.interceptor_position_m[:, 2],
            "interceptor_vx_mps": result.interceptor_velocity_mps[:, 0],
            "interceptor_vy_mps": result.interceptor_velocity_mps[:, 1],
            "interceptor_vz_mps": result.interceptor_velocity_mps[:, 2],
            "commanded_ax_mps2": result.commanded_accel_mps2[:, 0],
            "commanded_ay_mps2": result.commanded_accel_mps2[:, 1],
            "commanded_az_mps2": result.commanded_accel_mps2[:, 2],
            "applied_ax_mps2": result.applied_accel_mps2[:, 0],
            "applied_ay_mps2": result.applied_accel_mps2[:, 1],
            "applied_az_mps2": result.applied_accel_mps2[:, 2],
            "distance_m": result.distance_m,
            "closing_speed_mps": result.closing_speed_mps,
            "los_rate_norm_radps": result.los_rate_norm_radps,
        }
    )


def sweep_horizontal_headings(
    base_config: SimulationConfig,
    *,
    heading_deg_step: float = 15.0,
    guidance_modes: tuple[str, ...] = ("pure", "predictive", "pn"),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for heading_deg in np.arange(0.0, 360.0, heading_deg_step):
        heading_rad = float(np.deg2rad(heading_deg))
        config = replace(base_config, target_heading_rad=heading_rad, target_velocity=None)
        row: dict[str, Any] = {"heading_deg": float(heading_deg)}
        for mode in guidance_modes:
            result = simulate_interception(config, guidance=mode)
            row[f"{mode}_intercept_time_s"] = result.intercept_time_s
            row[f"{mode}_min_distance_m"] = float(np.min(result.distance_m))
        rows.append(row)
    return pd.DataFrame(rows)


def run_scenario_grid(
    base_config: SimulationConfig,
    *,
    guidance_modes: tuple[str, ...] = ("predictive", "pure", "pn"),
    target_modes: tuple[str, ...] = ("straight", "weave", "bounded_turn", "reactive"),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_mode in target_modes:
        scenario_config = replace(base_config, target_behavior=replace(base_config.target_behavior, mode=target_mode))
        for guidance_mode in guidance_modes:
            result = simulate_interception(scenario_config, guidance=guidance_mode)
            rows.append(
                {
                    "target_mode": target_mode,
                    "guidance_mode": guidance_mode,
                    "intercepted": bool(result.intercepted),
                    "intercept_time_s": result.intercept_time_s,
                    "minimum_distance_m": float(np.min(result.distance_m)),
                    "max_speed_mps": float(np.max(np.linalg.norm(result.interceptor_velocity_mps, axis=1))),
                    "max_abs_accel_x_mps2": float(np.max(np.abs(result.applied_accel_mps2[:, 0]))),
                    "max_abs_accel_y_mps2": float(np.max(np.abs(result.applied_accel_mps2[:, 1]))),
                }
            )
    return pd.DataFrame(rows)


def run_q2_analysis(
    output_dir: Path,
    *,
    intercept_radius_m: float = 2.0,
    horizon_s: float = 180.0,
    guidance_mode: str = "predictive",
    target_mode: str = "straight",
    render_animation: bool = True,
    plot_config: PlotConfig | None = None,
) -> dict[str, Any]:
    """Run Q2 simulations, verify constraints, and generate comparison artifacts."""
    from .q2_visualization import create_q2_visuals

    configure_matplotlib(plot_config)
    plot_config = plot_config or PlotConfig()
    target_initial_position = np.array([800.0, 200.0, 300.0], dtype=float)
    baseline_vertical_speed_mps = float(make_nominal_target_velocity(target_initial_position, speed=13.0)[2])
    baseline_heading_rad = float(np.deg2rad(BASELINE_Q2_HORIZONTAL_HEADING_DEG))
    # Keep the original climb component, but rotate the horizontal heading slightly
    # away from the radial tail-chase so the baseline geometry is still conservative
    # and materially easier to read in the reviewer-facing plots.
    nominal_velocity = make_heading_based_target_velocity(
        speed=13.0,
        horizontal_heading_rad=baseline_heading_rad,
        vertical_speed_mps=baseline_vertical_speed_mps,
    )
    constraints = InterceptorConstraints(max_speed=20.0, max_accel_x=2.5, max_accel_y=1.0)
    base_config = SimulationConfig(
        target_initial_position=target_initial_position,
        target_velocity=None,
        interceptor_initial_position=np.zeros(3, dtype=float),
        interceptor_initial_velocity=np.zeros(3, dtype=float),
        constraints=constraints,
        target_speed_mps=13.0,
        target_heading_rad=baseline_heading_rad,
        target_vertical_speed_mps=baseline_vertical_speed_mps,
        target_behavior=TargetBehaviorConfig(mode=target_mode, seed=7),
        guidance=GuidanceConfig(mode=guidance_mode, response_time_s=0.8, navigation_constant=3.5),
        horizon_s=horizon_s,
        intercept_radius_m=intercept_radius_m,
    )

    guidance_modes = ("predictive", "pure", "pn")
    target_modes = ("straight", "weave", "bounded_turn", "reactive")
    scenario_results = {
        target_behavior_mode: {
            mode: simulate_interception(
                replace(base_config, target_behavior=replace(base_config.target_behavior, mode=target_behavior_mode)),
                guidance=mode,
            )
            for mode in guidance_modes
        }
        for target_behavior_mode in target_modes
    }
    baseline_results = scenario_results["straight"]
    selected_target_results = scenario_results[target_mode]
    selected_result = selected_target_results[guidance_mode]
    selected_guidance_results = {
        target_behavior_mode: guidance_results[guidance_mode]
        for target_behavior_mode, guidance_results in scenario_results.items()
    }
    evasive_scenario_results = {
        target_behavior_mode: guidance_results
        for target_behavior_mode, guidance_results in scenario_results.items()
        if target_behavior_mode != "straight"
    }
    comparison_config = replace(
        base_config,
        dt=max(base_config.dt, 0.08),
        horizon_s=min(base_config.horizon_s, 160.0),
    )
    heading_sweep = sweep_horizontal_headings(
        replace(comparison_config, target_behavior=replace(comparison_config.target_behavior, mode="straight")),
        heading_deg_step=30.0,
    )
    scenario_grid = run_scenario_grid(base_config, guidance_modes=guidance_modes, target_modes=target_modes)

    csv_paths = []
    for mode, result in baseline_results.items():
        csv_path = Path(output_dir) / f"q2_nominal_{mode}.csv"
        result_to_dataframe(result).to_csv(csv_path, index=False)
        csv_paths.append(csv_path)
    heading_csv = Path(output_dir) / "q2_heading_sweep.csv"
    grid_csv = Path(output_dir) / "q2_mode_matrix.csv"
    heading_sweep.to_csv(heading_csv, index=False)
    scenario_grid.to_csv(grid_csv, index=False)
    csv_paths.extend([heading_csv, grid_csv])

    visual_report = create_q2_visuals(
        output_dir,
        base_config,
        scenario_results,
        heading_sweep,
        scenario_grid,
        selected_guidance=guidance_mode,
        selected_target=target_mode,
        render_animation=render_animation,
        plot_config=plot_config,
    )
    artifact_paths = csv_paths + [Path(path) for path in visual_report["artifact_paths"]]
    predictive = baseline_results["predictive"]
    summary = {
        "output_dir": str(output_dir),
        "scenario": {
            "target_initial_position_m": target_initial_position,
            "target_velocity_mps": nominal_velocity,
            "target_heading_deg": float(BASELINE_Q2_HORIZONTAL_HEADING_DEG),
            "target_vertical_speed_mps": baseline_vertical_speed_mps,
            "interceptor_initial_position_m": base_config.interceptor_initial_position,
            "intercept_radius_m": intercept_radius_m,
            "heading_assumption": (
                "Baseline target heading is fixed at 35 deg in the horizontal plane. "
                "The prompt leaves heading unspecified; 35 deg is a small shift away "
                "from the radial tail-chase so the baseline 3D geometry stays readable."
            ),
            "target_vertical_speed_assumption": (
                "The baseline preserves the climb component implied by the original "
                "radial-away default, so the only intentional geometry change is the "
                "horizontal heading."
            ),
            "z_acceleration_assumption": "No Z-axis acceleration limit was provided, so only X/Y limits are enforced numerically by default.",
        },
        "constraints": asdict(constraints),
        "selected_guidance": guidance_mode,
        "selected_target_mode": target_mode,
        "selected_result": _result_summary(selected_result),
        "predictive_result": _result_summary(predictive),
        "pure_pursuit_reference": _result_summary(baseline_results["pure"]),
        "pn_result": _result_summary(baseline_results["pn"]),
        "heading_sweep_summary": {
            f"{mode}_success_count": int(heading_sweep[f"{mode}_intercept_time_s"].notna().sum()) for mode in ("predictive", "pure", "pn")
        } | {
            f"{mode}_mean_intercept_time_s": float(heading_sweep[f"{mode}_intercept_time_s"].mean()) for mode in ("predictive", "pure", "pn")
        },
        "scenario_grid": scenario_grid.to_dict(orient="records"),
        "selected_guidance_extension_results": {
            mode: _result_summary(result) for mode, result in selected_guidance_results.items()
        },
        "evasive_scenario_guidance_results": {
            target_behavior_mode: {mode: _result_summary(result) for mode, result in guidance_results.items()}
            for target_behavior_mode, guidance_results in evasive_scenario_results.items()
        },
        "scenario_bundles": visual_report["scenario_bundles"],
        "overview_matrix_path": visual_report["overview_matrix_path"],
        "artifact_paths": [str(path) for path in artifact_paths],
    }
    summary_path = Path(output_dir) / "q2_summary.json"
    summary_path.write_text(json.dumps(to_builtin(summary), indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def _resolve_guidance_config(config: SimulationConfig, *, override_mode: str | None) -> GuidanceConfig:
    """Return the effective guidance config, optionally overriding the mode."""
    guidance = config.guidance
    if override_mode is not None:
        guidance = replace(guidance, mode=override_mode)
    return guidance


def _resolve_target_motion(config: SimulationConfig) -> tuple[float, float, float]:
    if config.target_velocity is not None:
        target_velocity = np.asarray(config.target_velocity, dtype=float)
        target_speed_mps = float(np.linalg.norm(target_velocity))
        target_heading_rad = float(np.arctan2(target_velocity[1], target_velocity[0]))
        target_vertical_speed_mps = float(target_velocity[2])
        return target_speed_mps, target_heading_rad, target_vertical_speed_mps
    target_speed_mps = float(config.target_speed_mps)
    target_heading_rad = (
        float(config.target_heading_rad)
        if config.target_heading_rad is not None
        else float(np.arctan2(config.target_initial_position[1], config.target_initial_position[0]))
    )
    target_vertical_speed_mps = (
        float(config.target_vertical_speed_mps)
        if config.target_vertical_speed_mps is not None
        else float(make_nominal_target_velocity(config.target_initial_position, target_speed_mps)[2])
    )
    return target_speed_mps, target_heading_rad, target_vertical_speed_mps


def _result_summary(result: SimulationResult) -> dict[str, float | bool | None]:
    speed = np.linalg.norm(result.interceptor_velocity_mps, axis=1)
    return {
        "intercepted": bool(result.intercepted),
        "intercept_time_s": result.intercept_time_s,
        "minimum_distance_m": float(np.min(result.distance_m)),
        "max_speed_mps": float(np.max(speed)),
        "max_abs_accel_x_mps2": float(np.max(np.abs(result.applied_accel_mps2[:, 0]))),
        "max_abs_accel_y_mps2": float(np.max(np.abs(result.applied_accel_mps2[:, 1]))),
        "peak_los_rate_radps": float(np.max(result.los_rate_norm_radps)),
    }


def _speed_feasible_delta_scale(current_velocity: np.ndarray, delta_velocity: np.ndarray, max_speed: float) -> float:
    """Find the largest α in [0, 1] such that ‖v_curr + α·Δv‖ ≤ max_speed.

    Solves the quadratic  ‖v + α·d‖² = S²  in closed form:
        (d·d) α² + 2(v·d) α + (v·v − S²) = 0
    """

    v = np.asarray(current_velocity, dtype=float)
    d = np.asarray(delta_velocity, dtype=float)
    if np.linalg.norm(v + d) <= max_speed:
        return 1.0
    a = float(np.dot(d, d))
    b = float(2.0 * np.dot(v, d))
    c = float(np.dot(v, v) - max_speed ** 2)
    if a < 1e-15:
        return 0.0
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return 0.0
    sqrt_disc = np.sqrt(discriminant)
    alpha1 = (-b - sqrt_disc) / (2.0 * a)
    alpha2 = (-b + sqrt_disc) / (2.0 * a)
    # We want the largest α in [0, 1] that puts us ON the sphere.
    candidates = [x for x in (alpha1, alpha2) if 0.0 <= x <= 1.0]
    return max(candidates) if candidates else 0.0
