# Copyright 2026 petrify contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-License-Identifier: Apache-2.0

"""Argparse CLI entry point for petrify."""

from __future__ import annotations

import argparse
import logging
import sys

from petrify import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="petrify",
        description=(
            "Cinemagraph-style motion freeze with live timestamp overlay.\n\n"
            "petrify detects motion frame-by-frame using OpenCV MOG2.  When motion "
            "exceeds the threshold the background is frozen (last clean still frame). "
            "A live timestamp ROI is composited on top of the frozen background, then "
            "the result is re-encoded with FFmpeg while the original audio is muxed "
            "back untouched."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Basic freeze, default settings\n"
            "  petrify input.mp4 output.mp4\n\n"
            "  # With a timestamp ROI at top-left corner (320x80 px)\n"
            "  petrify input.mp4 output.mp4 --roi 10 10 320 80\n\n"
            "  # More sensitive detection, slow-preset, high quality\n"
            "  petrify input.mp4 output.mp4 --threshold 200 --crf 18 --preset slow\n\n"
            "  # GPU encode on NVIDIA via CUDA + NVENC\n"
            "  petrify input.mp4 output.mp4 --hwaccel cuda --codec h264_nvenc\n\n"
            "  # Apple Silicon GPU encode\n"
            "  petrify input.mp4 output.mp4 --hwaccel videotoolbox --codec h264_videotoolbox\n"
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # -- Positional arguments -----------------------------------------------
    parser.add_argument(
        "input",
        metavar="INPUT",
        help="Path to the input video file.",
    )
    parser.add_argument(
        "output",
        metavar="OUTPUT",
        help="Path for the output video file.",
    )

    # -- ROI / Compositor ---------------------------------------------------
    roi_group = parser.add_argument_group("ROI / Compositor")
    roi_group.add_argument(
        "--roi",
        metavar=("X", "Y", "W", "H"),
        nargs=4,
        type=int,
        default=None,
        help=(
            "Bounding box of the live timestamp region to composite over the frozen "
            "background.  Format: X Y W H in pixels (top-left origin).  "
            "When not set, petrify auto-detects the timestamp region by analysing "
            "which pixels change consistently every second.  The detected value is "
            "printed so you can hard-code it with this flag on future runs to skip "
            "the scan entirely."
        ),
    )
    roi_group.add_argument(
        "--no-roi",
        action="store_true",
        default=False,
        dest="no_roi",
        help=(
            "Disable timestamp ROI compositing entirely (and skip auto-detection). "
            "Output will be a pure motion-freeze with no live clock overlay."
        ),
    )
    roi_group.add_argument(
        "--roi-scan-samples",
        type=int,
        default=60,
        metavar="N",
        dest="roi_scan_samples",
        help=(
            "Number of frame pairs to diff during auto ROI detection.  "
            "More samples = more robust at the cost of a longer scan.  Default: 60."
        ),
    )
    roi_group.add_argument(
        "--roi-scan-min-freq",
        type=float,
        default=0.6,
        metavar="FLOAT",
        dest="roi_scan_min_freq",
        help=(
            "Fraction of sample pairs in which a pixel must change to be considered "
            "part of the timestamp.  Lower = more permissive; raise to 0.8+ when "
            "the subject occupies the scene for more than ~60%% of the clip.  "
            "Default: 0.6."
        ),
    )
    roi_group.add_argument(
        "--roi-scan-margin",
        type=int,
        default=8,
        metavar="PX",
        dest="roi_scan_margin",
        help="Pixel padding added around the auto-detected ROI bounding box.  Default: 8.",
    )
    roi_group.add_argument(
        "--roi-scan-max-area-frac",
        type=float,
        default=0.10,
        metavar="FLOAT",
        dest="roi_scan_max_area_frac",
        help=(
            "Maximum candidate area as a fraction of total frame area.  "
            "Blobs larger than this are assumed to be motion regions, not a timestamp.  "
            "Default: 0.10 (10%%)."
        ),
    )
    roi_group.add_argument(
        "--roi-scan-edge-threshold",
        type=float,
        default=80.0,
        metavar="FLOAT",
        dest="roi_scan_edge_threshold",
        help=(
            "Laplacian edge energy threshold used in Pass 2 (edge-based X extent).  "
            "Text glyphs produce sharp spatial edges (>90) while diffuse background "
            "texture is typically below 70.  Default: 80."
        ),
    )
    roi_group.add_argument(
        "--roi-scan-edge-gap-tolerance",
        type=int,
        default=30,
        metavar="PX",
        dest="roi_scan_edge_gap_tolerance",
        help=(
            "Maximum inter-character gap (px) before the Pass 2 scan stops.  "
            "30 bridges typical letter spacing without jumping past the blank "
            "region after the last timestamp character.  Default: 30."
        ),
    )
    roi_group.add_argument(
        "--roi-scan-edge-samples",
        type=int,
        default=3,
        metavar="N",
        dest="roi_scan_edge_samples",
        help=(
            "Number of frames combined (per-pixel max) for the Pass 2 edge map.  "
            "Default: 3."
        ),
    )

    # -- Detector mode ------------------------------------------------------
    det_group = parser.add_argument_group("Detector Mode")
    det_group.add_argument(
        "--detector",
        choices=["motion", "person", "both"],
        default="motion",
        help=(
            "Freeze trigger strategy. "
            "'motion': MOG2 background subtraction (fast, default). "
            "'person': OpenCV HOG person detector (more accurate, slower). "
            "'both': freeze if EITHER motion OR a person is detected. "
            "Default: motion."
        ),
    )
    det_group.add_argument(
        "--hog-downsample",
        type=float,
        default=0.5,
        metavar="FACTOR",
        dest="hog_downsample",
        help="Downsample factor for HOG detection (default: 0.5).",
    )
    det_group.add_argument(
        "--hog-scale",
        type=float,
        default=1.05,
        metavar="FLOAT",
        dest="hog_scale",
        help="HOG image pyramid scale factor. Larger = faster, less accurate (default: 1.05).",
    )
    det_group.add_argument(
        "--hog-min-confidence",
        type=float,
        default=0.3,
        metavar="FLOAT",
        dest="hog_min_confidence",
        help="Minimum SVM confidence to accept a person detection (default: 0.3).",
    )
    det_group.add_argument(
        "--hog-min-box-area",
        type=int,
        default=1500,
        metavar="INT",
        dest="hog_min_box_area",
        help="Minimum bounding-box area in downsampled pixels to accept a detection (default: 1500).",
    )

    # -- Gap fill -----------------------------------------------------------
    gap_group = parser.add_argument_group("Temporal Gap Fill")
    gap_group.add_argument(
        "--gap-fill",
        type=int,
        default=10,
        metavar="FRAMES",
        dest="gap_fill",
        help=(
            "Expand every detected region by this many frames on both sides. "
            "Covers brief missed detections (person bending out of car, "
            "partial occlusion, between detector pulses). "
            "Default: 10 (0.4s at 25fps)."
        ),
    )

    # -- Background reference -----------------------------------------------
    bg_group = parser.add_argument_group("Background Reference")
    bg_group.add_argument(
        "--background-frame",
        type=int,
        default=None,
        metavar="N",
        dest="background_frame",
        help=(
            "Frame index (0-based) to use as the locked background reference. "
            "When not set, the video is scanned automatically to find the latest "
            "clean frame (no person detected) — typically from the end of the clip "
            "after the subject leaves. Override this when auto-detection picks the "
            "wrong frame."
        ),
    )
    bg_group.add_argument(
        "--no-bg-diff",
        action="store_true",
        default=False,
        dest="no_bg_diff",
        help=(
            "Disable the static background-difference detector.  By default petrify "
            "compares every frame against the locked clean background to catch subjects "
            "that MOG2 has absorbed (standing still) or that HOG misses (tiny, partial, "
            "or at the frame edge).  Use this flag to turn that check off."
        ),
    )
    bg_group.add_argument(
        "--bg-diff-pixel-threshold",
        type=int,
        default=25,
        metavar="INT",
        dest="bg_diff_pixel_threshold",
        help=(
            "Per-pixel grayscale difference (0–255) above which a pixel is considered "
            "changed by the static background differ.  25 suppresses sensor noise while "
            "detecting even dark clothing against a light background.  Default: 25."
        ),
    )
    bg_group.add_argument(
        "--bg-diff-area-threshold",
        type=int,
        default=200,
        metavar="PIXELS",
        dest="bg_diff_area_threshold",
        help=(
            "Total changed-pixel contour area (px² in downsampled space) above which "
            "a frame is classified as containing a foreground subject by the static "
            "background differ.  200 catches a person 30+ m down a corridor without "
            "tripping on sensor noise.  Default: 200."
        ),
    )

    # -- Motion detection ---------------------------------------------------
    motion_group = parser.add_argument_group("Motion Detection (MOG2)")
    motion_group.add_argument(
        "--threshold",
        type=int,
        default=500,
        metavar="PIXELS",
        help=(
            "Total foreground-contour area (px²) above which a frame is classified "
            "as containing motion.  Measured in downsampled-frame space.  "
            "Lower = more sensitive.  Default: 500."
        ),
    )
    motion_group.add_argument(
        "--history",
        type=int,
        default=200,
        metavar="FRAMES",
        help=(
            "Number of frames used to build the MOG2 background model.  "
            "Larger values produce a more stable background estimate but adapt "
            "more slowly to scene changes.  Default: 200."
        ),
    )
    motion_group.add_argument(
        "--var-threshold",
        type=float,
        default=16.0,
        metavar="FLOAT",
        dest="var_threshold",
        help=(
            "MOG2 varThreshold: Mahalanobis distance threshold for foreground "
            "classification.  Higher = less sensitive.  Default: 16."
        ),
    )
    motion_group.add_argument(
        "--detect-shadows",
        action="store_true",
        default=False,
        dest="detect_shadows",
        help=(
            "Enable MOG2 shadow detection.  Shadows are marked but excluded from "
            "the motion area calculation.  Adds a small CPU overhead.  Default: off."
        ),
    )
    motion_group.add_argument(
        "--downsample",
        type=float,
        default=0.5,
        metavar="FACTOR",
        help=(
            "Scale factor (0 < x <= 1) applied to frames before motion detection.  "
            "0.5 = half resolution, which gives ~4x speedup with minimal accuracy "
            "loss.  Remember: --threshold is evaluated in downsampled space.  "
            "Default: 0.5."
        ),
    )

    # -- Encoding -----------------------------------------------------------
    encode_group = parser.add_argument_group("Encoding (FFmpeg)")
    encode_group.add_argument(
        "--codec",
        default="libx264",
        metavar="CODEC",
        help=(
            "FFmpeg video codec to use for the output.  "
            "Examples: libx264, libx265, h264_nvenc, h264_videotoolbox.  "
            "Default: libx264."
        ),
    )
    encode_group.add_argument(
        "--crf",
        type=int,
        default=23,
        metavar="INT",
        help=(
            "Constant Rate Factor quality setting (lower = better quality / larger "
            "file).  Valid range: 0–51 for libx264/libx265.  Default: 23."
        ),
    )
    encode_group.add_argument(
        "--preset",
        default="medium",
        metavar="PRESET",
        help=(
            "FFmpeg encoding preset.  Slower presets produce smaller files at the "
            "same quality.  Options: ultrafast, superfast, veryfast, faster, fast, "
            "medium, slow, slower, veryslow.  Default: medium."
        ),
    )
    encode_group.add_argument(
        "--hwaccel",
        default=None,
        metavar="ACCEL",
        help=(
            "FFmpeg -hwaccel value for hardware-accelerated *decoding* of the "
            "intermediate file.  Examples: cuda, videotoolbox, dxva2.  "
            "Pair with a matching hardware encoder via --codec.  Default: None."
        ),
    )

    # -- Logging ------------------------------------------------------------
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.  Default: INFO level.",
    )

    return parser


_BANNER = r"""
██████╗ ███████╗████████╗██████╗ ██╗███████╗██╗   ██╗
██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██║██╔════╝╚██╗ ██╔╝
██████╔╝█████╗     ██║   ██████╔╝██║█████╗   ╚████╔╝
██╔═══╝ ██╔══╝     ██║   ██╔══██╗██║██╔══╝    ╚██╔╝
██║     ███████╗   ██║   ██║  ██║██║██║        ██║
╚═╝     ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝        ╚═╝

  Freeze the world. Keep the clock ticking.
  Made with ❤️  by starside.io  ·  https://starside.io
"""


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns an exit code (0 = success)."""
    # Print banner to stderr before parse_args so it shows on every real run.
    # Suppress it for --help / --version (pure informational flags).
    _argv = sys.argv[1:] if argv is None else argv
    if not any(f in _argv for f in ("-h", "--help", "--version")):
        print(_BANNER, file=sys.stderr)

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Configure logging before importing pipeline (which logs at module level).
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    # Validate --downsample range here (argparse doesn't support float ranges).
    if not (0.0 < args.downsample <= 1.0):
        parser.error(f"--downsample must be in (0, 1], got {args.downsample}")

    # Import here to keep startup fast when --version / --help is used.
    from petrify.core.config import PetrifyConfig  # noqa: PLC0415
    from petrify.pipeline import run_file  # noqa: PLC0415

    try:
        # A report with failed-closed frames still exits 0: those frames are
        # written black, and run_file logs a warning for them.
        run_file(args.input, args.output, PetrifyConfig.from_namespace(args))
    except FileNotFoundError as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 2
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Interrupted by user.")
        return 130
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).error("Fatal error: %s", exc, exc_info=args.verbose)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
