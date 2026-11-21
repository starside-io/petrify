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
import dataclasses
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from petrify import __version__

if TYPE_CHECKING:
    from petrify.core.config import PetrifyConfig
    from petrify.pipeline import OutputConfig


def _add_roi_group(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add the ROI / Compositor options shared by both command forms."""
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
    return roi_group


def _add_detector_group(parser: argparse.ArgumentParser) -> None:
    """Add the detector mode and HOG options shared by both command forms."""
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


def _add_gap_fill_group(parser: argparse.ArgumentParser) -> None:
    """Add --gap-fill (file form only; ``petrify run`` uses --lookahead)."""
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


def _add_bg_diff_arguments(bg_group: argparse._ArgumentGroup) -> None:
    """Add the background differ options shared by both command forms."""
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


def _add_motion_group(parser: argparse.ArgumentParser) -> None:
    """Add the MOG2 motion detection options shared by both command forms."""
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


def _add_encode_group(parser: argparse.ArgumentParser, hwaccel_help: str | None = None) -> None:
    """Add the FFmpeg encoding options shared by both command forms.

    ``hwaccel_help`` replaces the file form's --hwaccel help text.
    """
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
        help=hwaccel_help
        or (
            "FFmpeg -hwaccel value for hardware-accelerated *decoding* of the "
            "intermediate file.  Examples: cuda, videotoolbox, dxva2.  "
            "Pair with a matching hardware encoder via --codec.  Default: None."
        ),
    )


def _add_verbose(parser: argparse.ArgumentParser) -> None:
    """Add --verbose, shared by both command forms."""
    # -- Logging ------------------------------------------------------------
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.  Default: INFO level.",
    )


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
            "  petrify input.mp4 output.mp4 --hwaccel videotoolbox --codec h264_videotoolbox\n\n"
            "Live sources and other outputs (webcam, RTSP or HTTP stream, image folder;\n"
            "virtual camera, preview window) use the run form, see: petrify run --help\n"
            "  petrify run --source camera:0 --virtualcam\n\n"
            "The run form is used when the first argument that is not an option is 'run'\n"
            "and --source is given; otherwise the arguments are INPUT OUTPUT, so a file\n"
            "named run still works: petrify run out.mp4\n"
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

    _add_roi_group(parser)
    _add_detector_group(parser)
    _add_gap_fill_group(parser)

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
    _add_bg_diff_arguments(bg_group)

    _add_motion_group(parser)
    _add_encode_group(parser)
    _add_verbose(parser)

    return parser


# Modes ``petrify run --mode`` accepts. Only the built-in erase mode exists so far.
RUN_MODES = ("erase",)

# Default --lookahead: the file form's --gap-fill default for a file or image folder
# and for a stream URL (CCTV, where a subject may enter slowly), a short delay for a
# webcam. Frames are buffered uncompressed, so the value is capped.
_FILE_LOOKAHEAD = 10
_STREAM_LOOKAHEAD = 10
_CAMERA_LOOKAHEAD = 3
_MAX_LOOKAHEAD = 600

_RUN_HWACCEL_HELP = (
    "FFmpeg -hwaccel value, passed to the FFmpeg process that encodes --output. In this "
    "form petrify decodes the source itself and sends FFmpeg raw frames, so there is "
    "nothing for FFmpeg to decode and this does not speed anything up. Choose a hardware "
    "encoder with --codec instead (for example h264_videotoolbox or h264_nvenc). "
    "Default: None."
)

_LIVE_KINDS = ("camera", "stream")


def _build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="petrify run",
        description=(
            "Run petrify on any source: a video file, an image folder or glob, a webcam,\n"
            "or an RTSP or HTTP stream. Output goes to a video file, a virtual camera,\n"
            "a preview window, or several of them at once.\n\n"
            "Frames are processed in one streaming pass. On a live source the run ends\n"
            "with Ctrl-C (or q in the preview window) and the output file is still\n"
            "finalized. Frames that cannot be processed safely (a detector error, the\n"
            "background not ready yet, a frame over budget) are written black."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Same result as: petrify input.mp4 output.mp4\n"
            "  petrify run --source input.mp4 --output output.mp4\n\n"
            "  # Webcam to a virtual camera for video calls, with a preview window\n"
            "  petrify run --source camera:0 --virtualcam --preview\n\n"
            "  # IP camera to a file; step out of view for the first 5 seconds\n"
            "  petrify run --source rtsp://camera.local/stream --output out.mp4 \\\n"
            "      --background calibrate --calibrate-seconds 5\n\n"
            "  # Image folder at 10 fps\n"
            "  petrify run --source 'frames/*.png' --fps 10 --output out.mp4\n\n"
            "The file form's --gap-fill is not accepted here: --lookahead has the same\n"
            "meaning. --background-frame is accepted with --background locked.\n"
        ),
    )

    # -- Source and mode ----------------------------------------------------
    src_group = parser.add_argument_group("Source and Mode")
    src_group.add_argument(
        "--source",
        required=True,
        metavar="SPEC",
        help=(
            "What to read: a video file path, an image folder or glob (for example "
            "'frames/*.png'), camera:N for webcam number N, or an rtsp://, rtsps://, "
            "http:// or https:// stream URL. A bare number is a file name, not a camera."
        ),
    )
    src_group.add_argument(
        "--mode",
        choices=RUN_MODES,
        default="erase",
        help="Processing mode. 'erase': freeze the subject out of the scene. Default: erase.",
    )
    src_group.add_argument(
        "--fps",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Frame rate of an image folder source (image folders only). Default: 25.",
    )

    # -- Outputs ------------------------------------------------------------
    out_group = parser.add_argument_group("Outputs (at least one is required)")
    out_group.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "Write the result to this video file, encoded by FFmpeg in one pass. For a "
            "video file source its audio is copied."
        ),
    )
    out_group.add_argument(
        "--virtualcam",
        action="store_true",
        default=False,
        help=(
            "Publish the result as a virtual camera (needs pyvirtualcam: pip install "
            "petrify[virtualcam]; on macOS also OBS 30 or newer)."
        ),
    )
    out_group.add_argument(
        "--virtualcam-device",
        default=None,
        metavar="NAME",
        dest="virtualcam_device",
        help="Virtual camera device to use (with --virtualcam). Default: the first one.",
    )
    out_group.add_argument(
        "--preview",
        action="store_true",
        default=False,
        help=(
            "Show the result in a local window; q or Escape ends the run (needs an OpenCV "
            "build with GUI support)."
        ),
    )

    # -- Streaming engine ---------------------------------------------------
    engine_group = parser.add_argument_group("Streaming Engine")
    engine_group.add_argument(
        "--lookahead",
        type=int,
        default=None,
        metavar="FRAMES",
        help=(
            "Frames of delay before a frame is written. Every detection is expanded by "
            "this many frames on both sides (the file form's --gap-fill). A smaller "
            "lookahead lowers the latency but lets more of a slowly entering subject "
            "through before it is detected. At most "
            f"{_MAX_LOOKAHEAD} (frames are buffered uncompressed). Default: "
            f"{_FILE_LOOKAHEAD} for a video file or image folder, {_STREAM_LOOKAHEAD} for "
            f"a stream URL, {_CAMERA_LOOKAHEAD} for a webcam (camera:N)."
        ),
    )
    engine_group.add_argument(
        "--frame-budget-ms",
        type=float,
        default=None,
        metavar="MS",
        dest="frame_budget_ms",
        help=(
            "Per-frame time budget for detection plus rendering, in milliseconds. A "
            "frame over budget is written black. Default: no budget."
        ),
    )
    # Rejected with a pointer to --lookahead; hidden from the help.
    engine_group.add_argument(
        "--gap-fill", type=int, default=None, dest="run_gap_fill", help=argparse.SUPPRESS
    )

    roi_group = _add_roi_group(parser)
    roi_group.add_argument(
        "--roi-auto",
        action="store_true",
        default=False,
        dest="roi_auto",
        help=(
            "Auto-detect the timestamp ROI on a webcam (camera:N). A webcam has no "
            "burned-in timestamp and detection there can lock onto a person, so for "
            "camera:N the ROI is disabled unless --roi or --roi-auto is given. Other "
            "sources auto-detect by default. Cannot be combined with --roi or --no-roi."
        ),
    )
    roi_group.add_argument(
        "--roi-warmup",
        type=float,
        default=None,
        metavar="SECONDS",
        dest="roi_warmup",
        help=(
            "On a source that cannot be scanned in advance (live sources, image "
            "folders), auto-detect the ROI on the first SECONDS of input, then lock it. "
            "Nothing is pasted before it locks. Only with ROI auto-detection. Default: 10."
        ),
    )
    roi_group.add_argument(
        "--roi-max-area",
        type=float,
        default=None,
        metavar="FRAC",
        dest="roi_max_area",
        help=(
            "Largest ROI the warmup detection accepts, as a fraction of the frame area "
            "(0 < FRAC <= 1); a larger box is discarded as too large to be a timestamp, "
            "so nothing is pasted. Only with ROI auto-detection. Default: 0.15."
        ),
    )
    _add_detector_group(parser)

    # -- Background reference -----------------------------------------------
    bg_group = parser.add_argument_group("Background Reference")
    bg_group.add_argument(
        "--background",
        choices=("locked", "running", "calibrate"),
        default=None,
        help=(
            "How the clean background is found. 'locked': pre-scan the source for one "
            "clean frame (video files and image folders only). 'running': rolling median "
            "of recent frames with no subject. 'calibrate': median of the first "
            "--calibrate-seconds, while the scene is empty. Frames are written black "
            "until the background is ready. Default: locked for a video file or image "
            "folder, running for a live source."
        ),
    )
    bg_group.add_argument(
        "--calibrate-seconds",
        type=float,
        default=None,
        metavar="SECONDS",
        dest="calibrate_seconds",
        help="Length of the calibration (with --background calibrate). Default: 3.",
    )
    bg_group.add_argument(
        "--no-background-check",
        action="store_false",
        default=True,
        dest="background_check",
        help=(
            "UNSAFE. Turn off the person check on the frames the running or calibrate "
            "background is learned from. By default a frame in which the HOG person "
            "detector finds a person is not used. Without the check a person in view "
            "while the background is built is absorbed into it and shown after they "
            "leave. The check reduces but does not eliminate this: start with the scene "
            "empty either way."
        ),
    )
    bg_group.add_argument(
        "--background-frame",
        type=int,
        default=None,
        metavar="N",
        dest="background_frame",
        help=(
            "Frame index (0-based) of the locked background (with --background locked). "
            "When not set, the source is scanned for the latest frame with no person."
        ),
    )
    _add_bg_diff_arguments(bg_group)

    _add_motion_group(parser)
    _add_encode_group(parser, hwaccel_help=_RUN_HWACCEL_HELP)
    _add_verbose(parser)
    return parser


@dataclasses.dataclass(frozen=True)
class _RunSettings:
    """What ``petrify run`` resolved from its arguments."""

    kind: str
    config: PetrifyConfig
    outputs: OutputConfig


def _resolve_run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> _RunSettings:
    """Validate the run arguments, apply the per-source defaults, and build the config.

    Every problem is reported with ``parser.error`` (exit code 2). No message holds
    the source spec, which may be a URL with credentials.
    """
    from petrify.core.config import PetrifyConfig  # noqa: PLC0415
    from petrify.pipeline import OutputConfig, _same_file  # noqa: PLC0415
    from petrify.sources import source_kind  # noqa: PLC0415

    if args.run_gap_fill is not None:
        parser.error("--gap-fill is not a petrify run option; use --lookahead (same meaning)")
    try:
        kind = source_kind(args.source)
    except ValueError as exc:
        parser.error(str(exc))  # Holds only the redacted URL.
    live = kind in _LIVE_KINDS

    if not (args.output or args.virtualcam or args.preview):
        parser.error("at least one output is required: --output, --virtualcam or --preview")
    if args.virtualcam_device is not None and not args.virtualcam:
        parser.error("--virtualcam-device requires --virtualcam")

    if not (0.0 < args.downsample <= 1.0):
        parser.error(f"--downsample must be in (0, 1], got {args.downsample}")
    if args.lookahead is not None and args.lookahead < 0:
        parser.error(f"--lookahead must be >= 0, got {args.lookahead}")
    if args.lookahead is not None and args.lookahead > _MAX_LOOKAHEAD:
        parser.error(
            f"--lookahead must be <= {_MAX_LOOKAHEAD} (frames are buffered uncompressed), "
            f"got {args.lookahead}"
        )
    if args.output is not None and kind == "file" and _same_file(
        Path(args.source), Path(args.output)
    ):
        parser.error("--output is the input file; choose another output path")
    for flag, value in (
        ("--fps", args.fps),
        ("--frame-budget-ms", args.frame_budget_ms),
        ("--calibrate-seconds", args.calibrate_seconds),
        ("--roi-warmup", args.roi_warmup),
    ):
        if value is not None and not value > 0:
            parser.error(f"{flag} must be > 0, got {value}")
    if args.background_frame is not None and args.background_frame < 0:
        parser.error(f"--background-frame must be >= 0, got {args.background_frame}")
    if args.fps is not None and kind != "images":
        parser.error("--fps applies only to an image folder or glob source")

    if sum((args.roi is not None, args.no_roi, args.roi_auto)) > 1:
        parser.error("--roi, --no-roi and --roi-auto cannot be combined")
    if args.roi is not None:
        roi_mode = "manual"
    elif args.no_roi or (kind == "camera" and not args.roi_auto):
        roi_mode = "disabled"
    else:
        roi_mode = "auto"
    for flag, value in (("--roi-warmup", args.roi_warmup), ("--roi-max-area", args.roi_max_area)):
        if value is None or roi_mode == "auto":
            continue
        if args.roi is None and not args.no_roi:
            parser.error(
                f"{flag} needs ROI auto-detection, which is off by default for camera:N; "
                "add --roi-auto"
            )
        parser.error(f"{flag} needs ROI auto-detection; it does not apply with --roi or --no-roi")
    if args.roi_max_area is not None and not 0 < args.roi_max_area <= 1:
        parser.error(f"--roi-max-area must be in (0, 1], got {args.roi_max_area}")

    background = args.background or ("running" if live else "locked")
    if background == "locked" and live:
        parser.error(
            "--background locked needs a video file or image folder to pre-scan; "
            "for a live source use --background running or --background calibrate"
        )
    if args.background_frame is not None and background != "locked":
        parser.error("--background-frame requires --background locked")
    if args.calibrate_seconds is not None and background != "calibrate":
        parser.error("--calibrate-seconds requires --background calibrate")

    base = PetrifyConfig.from_namespace(args)
    bg_changes: dict[str, object] = {
        "strategy": background,
        "sample_check": args.background_check,
    }
    if args.calibrate_seconds is not None:
        bg_changes["calibrate_seconds"] = args.calibrate_seconds
    roi_changes: dict[str, object] = {}
    if roi_mode != base.roi.mode:
        roi_changes["mode"] = roi_mode
    if args.roi_warmup is not None:
        roi_changes["warmup_seconds"] = args.roi_warmup
    if args.roi_max_area is not None:
        roi_changes["max_area_cap"] = args.roi_max_area
    lookahead = args.lookahead
    if lookahead is None:
        lookahead = {"camera": _CAMERA_LOOKAHEAD, "stream": _STREAM_LOOKAHEAD}.get(
            kind, _FILE_LOOKAHEAD
        )
    config = dataclasses.replace(
        base,
        background=dataclasses.replace(base.background, **bg_changes),
        roi=dataclasses.replace(base.roi, **roi_changes),
        engine=dataclasses.replace(
            base.engine, lookahead=lookahead, frame_budget_ms=args.frame_budget_ms
        ),
    )
    outputs = OutputConfig(
        file=None if args.output is None else Path(args.output),
        virtualcam=args.virtualcam,
        virtualcam_device=args.virtualcam_device,
        preview=args.preview,
    )
    return _RunSettings(kind=kind, config=config, outputs=outputs)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _run_main(argv: list[str]) -> int:
    """``petrify run``: open the source, run the streaming engine, return an exit code."""
    parser = _build_run_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    settings = _resolve_run(parser, args)

    from petrify.pipeline import output_overwrites_input, run_stream  # noqa: PLC0415
    from petrify.sources import display_spec, open_source  # noqa: PLC0415

    log = logging.getLogger(__name__)
    try:
        source = open_source(args.source, fps=args.fps)
        if settings.outputs.file is not None and output_overwrites_input(
            source, settings.outputs.file
        ):
            source.close()
            parser.error("--output is one of the input files; choose another output path")
        log.info(
            "Source: %s (%s), mode: %s",
            display_spec(source.info.uri or ""),
            settings.kind,
            args.mode,
        )
        # Failed-closed frames still exit 0 (they are written black and logged), and
        # so does a live run ended with Ctrl-C: its outputs were finalized. A missing
        # FFmpeg is a fatal error (exit 1), not a missing input (exit 2).
        run_stream(source, settings.config, settings.outputs)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        log.info("Interrupted by user; the run was aborted.")
        return 130
    except Exception as exc:  # noqa: BLE001
        log.error("Fatal error: %s", exc, exc_info=args.verbose)
        return 1
    return 0


def _run_form_argv(argv: list[str]) -> list[str] | None:
    """Return the run form's arguments when ``argv`` selects it, else None.

    The run form is selected when the first non-option argument is ``run`` and the
    other arguments contain ``--source`` (or ``-h`` / ``--help``); the ``run`` word
    is removed and the options around it are kept, so ``petrify --verbose run
    --source ...`` works. Option values are skipped when looking for the first
    non-option argument. Anything else is the file form, so a file named ``run``
    still works there: ``petrify run out.mp4``.
    """
    actions = _build_run_parser()._option_string_actions
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--":
            return None
        if token.startswith("-") and token != "-":
            action = actions.get(token)
            nargs = action.nargs if action is not None else 0
            if nargs is None:
                i += 1
            elif isinstance(nargs, int):
                i += nargs
            i += 1
            continue
        if token != "run":
            return None
        rest = argv[:i] + argv[i + 1 :]
        wants_run = any(
            a in ("--source", "-h", "--help") or a.startswith("--source=") for a in rest
        )
        return rest if wants_run else None
    return None


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

    run_argv = _run_form_argv(_argv)
    if run_argv is not None:
        return _run_main(run_argv)

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
    if args.gap_fill < 0:
        parser.error(f"--gap-fill must be >= 0, got {args.gap_fill}")

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
