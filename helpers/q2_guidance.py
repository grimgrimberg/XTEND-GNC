from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GuidanceConfig:
    """High-level guidance-law configuration."""

    mode: str = "predictive"
    response_time_s: float = 0.8
    navigation_constant: float = 3.0


def analytic_lead_intercept_time(
    *,
    relative_position_m: np.ndarray,
    target_velocity_mps: np.ndarray,
    pursuer_speed_mps: float,
) -> float | None:
    """Closed-form intercept time for a constant-speed pursuer against constant-velocity target.

    Solves  ‖r + v_t · t‖ = V_p · t  for the smallest positive *t*:

        (v_t·v_t − V_p²) t² + 2(r·v_t) t + r·r = 0
    """

    relative_position_m = np.asarray(relative_position_m, dtype=float)
    target_velocity_mps = np.asarray(target_velocity_mps, dtype=float)
    a = float(np.dot(target_velocity_mps, target_velocity_mps) - pursuer_speed_mps**2)
    b = float(2.0 * np.dot(relative_position_m, target_velocity_mps))
    c = float(np.dot(relative_position_m, relative_position_m))
    if abs(a) < 1e-12:
        # Linear case: pursuer speed ≈ target speed
        if abs(b) < 1e-12:
            return None
        solution = -c / b
        return solution if solution > 0.0 else None
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None
    sqrt_disc = float(np.sqrt(discriminant))
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    # Return the smallest positive root.
    positive = sorted(t for t in (t1, t2) if t > 0.0)
    return positive[0] if positive else None


def desired_velocity_toward(
    target_point_m: np.ndarray,
    interceptor_position_m: np.ndarray,
    max_speed_mps: float,
) -> np.ndarray:
    delta = np.asarray(target_point_m, dtype=float) - np.asarray(interceptor_position_m, dtype=float)
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-9:
        return np.zeros(3, dtype=float)
    return max_speed_mps * delta / distance


def proportional_navigation_command(
    *,
    relative_position_m: np.ndarray,
    relative_velocity_mps: np.ndarray,
    interceptor_velocity_mps: np.ndarray,
    navigation_constant: float,
) -> np.ndarray:
    """Classical lateral PN command for a given LOS geometry."""

    relative_position_m = np.asarray(relative_position_m, dtype=float)
    relative_velocity_mps = np.asarray(relative_velocity_mps, dtype=float)
    interceptor_velocity_mps = np.asarray(interceptor_velocity_mps, dtype=float)
    range_m = float(np.linalg.norm(relative_position_m))
    interceptor_speed = float(np.linalg.norm(interceptor_velocity_mps))
    if range_m <= 1e-9 or interceptor_speed <= 1e-9:
        return np.zeros(3, dtype=float)
    closing_speed = max(-float(np.dot(relative_position_m, relative_velocity_mps)) / range_m, 0.0)
    if closing_speed <= 0.0:
        return np.zeros(3, dtype=float)
    los_rate_vector = np.cross(relative_position_m, relative_velocity_mps) / max(range_m**2, 1e-9)
    velocity_hat = interceptor_velocity_mps / interceptor_speed
    return navigation_constant * closing_speed * np.cross(los_rate_vector, velocity_hat)


def build_guidance_command(
    *,
    guidance: GuidanceConfig,
    target_position_m: np.ndarray,
    target_velocity_mps: np.ndarray,
    interceptor_position_m: np.ndarray,
    interceptor_velocity_mps: np.ndarray,
    max_speed_mps: float,
    dt_s: float,
) -> tuple[np.ndarray, dict[str, float | None]]:
    """Generate a commanded interceptor acceleration and diagnostic scalars."""

    mode = guidance.mode.lower()
    relative_position = np.asarray(target_position_m, dtype=float) - np.asarray(interceptor_position_m, dtype=float)
    relative_velocity = np.asarray(target_velocity_mps, dtype=float) - np.asarray(interceptor_velocity_mps, dtype=float)
    response_time_s = max(guidance.response_time_s, dt_s)
    lead_time_s: float | None = None
    los_rate_norm = float(np.linalg.norm(np.cross(relative_position, relative_velocity)) / max(np.linalg.norm(relative_position) ** 2, 1e-9))
    closing_speed = max(-float(np.dot(relative_position, relative_velocity)) / max(np.linalg.norm(relative_position), 1e-9), 0.0)

    if mode == "pure":
        desired_velocity = desired_velocity_toward(target_position_m, interceptor_position_m, max_speed_mps)
        command = (desired_velocity - interceptor_velocity_mps) / response_time_s
    elif mode == "predictive":
        lead_time_s = analytic_lead_intercept_time(
            relative_position_m=relative_position,
            target_velocity_mps=target_velocity_mps,
            pursuer_speed_mps=max_speed_mps,
        )
        lead_point = target_position_m if lead_time_s is None else target_position_m + lead_time_s * target_velocity_mps
        desired_velocity = desired_velocity_toward(lead_point, interceptor_position_m, max_speed_mps)
        command = (desired_velocity - interceptor_velocity_mps) / response_time_s
    elif mode == "pn":
        lead_time_s = analytic_lead_intercept_time(
            relative_position_m=relative_position,
            target_velocity_mps=target_velocity_mps,
            pursuer_speed_mps=max_speed_mps,
        )
        lead_point = target_position_m if lead_time_s is None else target_position_m + lead_time_s * target_velocity_mps
        desired_velocity = desired_velocity_toward(lead_point, interceptor_position_m, max_speed_mps)
        predictive_term = (desired_velocity - interceptor_velocity_mps) / response_time_s
        pn_term = proportional_navigation_command(
            relative_position_m=relative_position,
            relative_velocity_mps=relative_velocity,
            interceptor_velocity_mps=interceptor_velocity_mps,
            navigation_constant=guidance.navigation_constant,
        )
        command = predictive_term + pn_term
    else:
        raise ValueError(f"Unsupported guidance mode: {guidance.mode}")

    return command, {
        "lead_time_s": lead_time_s,
        "los_rate_norm_radps": los_rate_norm,
        "closing_speed_mps": closing_speed,
    }
