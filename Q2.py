from __future__ import annotations

import argparse

from helpers.paths import clean_output_dir, resolve_output_dir
from helpers.q2_simulation import run_q2_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XTEND Q2 interception simulation.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--intercept-radius-m", type=float, default=2.0)
    parser.add_argument("--horizon-s", type=float, default=180.0)
    parser.add_argument(
        "--guidance-mode",
        choices=["predictive", "pure", "pn"],
        default="predictive",
        help=(
            "Primary guidance mode: 'predictive' (analytic lead), 'pure' (pursuit), "
            "'pn' (augmented: predictive velocity-shaping + lateral PN correction). "
            "All three are always compared in the generated artifacts."
        ),
    )
    parser.add_argument(
        "--target-mode",
        choices=["straight", "weave", "bounded_turn", "reactive"],
        default="straight",
        help="Target behavior mode. `straight` is the assignment-aligned baseline; the others are explicit extensions.",
    )
    parser.add_argument(
        "--skip-animation",
        action="store_true",
        help="Skip the generated 3D interception GIF if you only want static metrics and plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir("q2", args.output_dir)
    clean_output_dir(output_dir)
    summary = run_q2_analysis(
        output_dir,
        intercept_radius_m=args.intercept_radius_m,
        horizon_s=args.horizon_s,
        guidance_mode=args.guidance_mode,
        target_mode=args.target_mode,
        render_animation=not args.skip_animation,
    )
    selected_guidance = summary["selected_guidance"]
    selected_result = summary["selected_result"]
    print(f"Q2 simulation complete. Outputs: {output_dir}")
    print(f"Selected extension modes: guidance={selected_guidance} | target={summary['selected_target_mode']}")
    print(
        f"{selected_guidance.capitalize()} guidance: "
        f"intercepted={selected_result['intercepted']} | "
        f"intercept_time_s={selected_result['intercept_time_s']} | "
        f"min_distance_m={selected_result['minimum_distance_m']:.3f}"
    )
    print(
        "Constraint peaks: "
        f"speed={selected_result['max_speed_mps']:.2f} m/s, "
        f"|ax|={selected_result['max_abs_accel_x_mps2']:.2f} m/s^2, "
        f"|ay|={selected_result['max_abs_accel_y_mps2']:.2f} m/s^2"
    )
    print(f"Summary JSON: {summary['summary_path']}")


if __name__ == "__main__":
    main()
