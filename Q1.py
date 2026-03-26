from __future__ import annotations

import argparse
from pathlib import Path

from helpers.paths import data_path, resolve_output_dir
from helpers.q1_pipeline import FrameConvention, run_q1_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XTEND Q1 world-angle analysis.")
    parser.add_argument("--csv-path", type=Path, default=data_path("examGuidance.csv"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--inverse-quaternion",
        action="store_true",
        help="Interpret the provided quaternion as world-to-body instead of body-to-world.",
    )
    parser.add_argument(
        "--gimbal-positive-pitch-raises",
        action="store_true",
        help="Override the default convention and treat positive gimbal pitch as upward.",
    )
    parser.add_argument(
        "--camera-positive-elevation-down",
        action="store_true",
        help="Flip the camera elevation sign if positive image elevation should mean downward.",
    )
    parser.add_argument(
        "--camera-positive-azimuth-left",
        action="store_true",
        help="Flip the camera azimuth sign if positive image azimuth should mean left.",
    )
    parser.add_argument(
        "--manual-convention",
        action="store_true",
        help="Use the CLI convention flags directly instead of the data-driven auto-selection path.",
    )
    parser.add_argument(
        "--skip-animation",
        action="store_true",
        help="Skip the saved Q1 geometry GIF if you only want static artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir("q1", args.output_dir)
    manual_requested = (
        args.manual_convention
        or args.inverse_quaternion
        or args.gimbal_positive_pitch_raises
        or args.camera_positive_elevation_down
        or args.camera_positive_azimuth_left
    )
    convention = None
    if manual_requested:
        convention = FrameConvention(
            quaternion_is_body_to_world=not args.inverse_quaternion,
            camera_positive_azimuth_right=not args.camera_positive_azimuth_left,
            camera_positive_elevation_up=not args.camera_positive_elevation_down,
            gimbal_positive_pitch_raises=args.gimbal_positive_pitch_raises,
        )
    summary = run_q1_analysis(output_dir, args.csv_path, convention=convention, render_animation=not args.skip_animation)
    print(f"Q1 analysis complete. Outputs: {output_dir}")
    print(f"Selected convention: {summary['selected_convention']}")
    print(
        "Sync offsets [ms]: "
        f"gimbal median={summary['sensor_sync']['gimbal_offset_ms_median']:.2f}, "
        f"attitude median={summary['sensor_sync']['attitude_offset_ms_median']:.2f}"
    )
    print(
        "Selected rate method: "
        f"{summary['selected_rate_method']} | "
        f"azimuth std={summary['rate_summary_deg_s']['azimuth_std']:.2f} deg/s | "
        f"elevation std={summary['rate_summary_deg_s']['elevation_std']:.2f} deg/s"
    )
    print(
        "Bundle geometry: "
        f"windows={summary['bundle_geometry']['bundle_count']} | "
        f"median residual={summary['bundle_geometry']['bundle_residual_median_m']}"
    )
    print(f"Summary JSON: {summary['summary_path']}")


if __name__ == "__main__":
    main()
