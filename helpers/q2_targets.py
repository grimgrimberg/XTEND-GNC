from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TargetBehaviorConfig:
    """Target-behavior extension options for Q2."""

    mode: str = "straight"
    seed: int = 0
    weave_amplitude_deg: float = 18.0
    weave_period_s: float = 6.0
    turn_amplitude_deg: float = 28.0
    turn_period_s: float = 10.0
    bounded_turn_rate_deg_s: float = 12.0
    reactive_trigger_distance_m: float = 220.0
    reactive_turn_amplitude_deg: float = 35.0


def target_velocity_for_behavior(
    *,
    time_s: float,
    current_position_m: np.ndarray,
    interceptor_position_m: np.ndarray,
    base_speed_mps: float,
    base_heading_rad: float,
    base_vertical_speed_mps: float,
    config: TargetBehaviorConfig,
) -> np.ndarray:
    """Generate a deterministic target velocity for the selected behavior mode."""

    horizontal_speed = float(np.sqrt(max(base_speed_mps**2 - base_vertical_speed_mps**2, 0.0)))
    mode = config.mode.lower()
    if mode == "straight":
        heading_offset = 0.0
    elif mode == "weave":
        heading_offset = np.deg2rad(config.weave_amplitude_deg) * np.sin(2.0 * np.pi * time_s / max(config.weave_period_s, 1e-6))
    elif mode == "bounded_turn":
        amplitude = np.deg2rad(config.turn_amplitude_deg)
        min_period = 4.0 * amplitude / max(np.deg2rad(config.bounded_turn_rate_deg_s), 1e-6)
        period = max(config.turn_period_s, min_period)
        phase = (time_s / period) % 1.0
        triangle = 2.0 * np.abs(2.0 * phase - 1.0) - 1.0
        heading_offset = amplitude * triangle
    elif mode == "reactive":
        horizontal_delta = np.asarray(current_position_m, dtype=float)[:2] - np.asarray(interceptor_position_m, dtype=float)[:2]
        separation = float(np.linalg.norm(horizontal_delta))
        if separation > config.reactive_trigger_distance_m:
            heading_offset = 0.0
        else:
            away_heading = np.arctan2(current_position_m[1] - interceptor_position_m[1], current_position_m[0] - interceptor_position_m[0])
            pseudo_random = (
                np.sin(0.73 * time_s + 0.41 * config.seed)
                + 0.6 * np.sin(1.91 * time_s + 0.13 * config.seed)
            ) / 1.6
            blended_heading = wrap_angle_rad(0.35 * base_heading_rad + 0.65 * away_heading)
            heading_offset = wrap_angle_rad(blended_heading - base_heading_rad)
            heading_offset += np.deg2rad(config.reactive_turn_amplitude_deg) * 0.35 * pseudo_random
    else:
        raise ValueError(f"Unsupported target mode: {config.mode}")

    heading = wrap_angle_rad(base_heading_rad + heading_offset)
    horizontal_velocity = horizontal_speed * np.array([np.cos(heading), np.sin(heading)], dtype=float)
    return np.array([horizontal_velocity[0], horizontal_velocity[1], base_vertical_speed_mps], dtype=float)


def wrap_angle_rad(angle_rad: float) -> float:
    """Wrap an angle to [-pi, pi]."""

    return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))
