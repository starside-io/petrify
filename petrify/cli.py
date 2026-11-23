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
import math
import secrets
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from petrify import __version__

if TYPE_CHECKING:
    from collections.abc import Iterator

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
            "the detection.  Exception: on a webcam (petrify run --source camera:N) "
            "nothing is detected and no ROI is used unless this flag or --roi-auto "
            "is given."
        ),
    )
    roi_group.add_argument(
        "--no-roi",
        action="store_true",
        default=False,
        dest="no_roi",
        help=(
            "Disable timestamp ROI compositing entirely (and skip auto-detection). "
            "Nothing from the live frame is pasted over the mode's output, so no live "
            "clock is shown."
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
            "Per-pixel grayscale difference (0 to 255) above which a pixel is considered "
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
            "file).  Valid range: 0 to 51 for libx264/libx265.  Default: 23."
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
            "clean frame (no person detected), typically from the end of the clip "
            "after the subject leaves. Override this when auto-detection picks the "
            "wrong frame."
        ),
    )
    _add_bg_diff_arguments(bg_group)

    _add_motion_group(parser)
    _add_encode_group(parser)
    _add_verbose(parser)

    return parser


# Built-in modes ``petrify run --mode`` accepts; installed plugins may add more (see
# _ModeCatalog).
RUN_MODES = ("erase",)

# The mode used when --mode is not given.
_DEFAULT_MODE = "erase"

# TrackerConfig.hold_frames default, for the --track-hold help text (importing the
# config here would slow down every invocation).
_TRACK_HOLD_DEFAULT = 15

# Defaults of PersonDnnConfig.score_threshold, FaceYuNetConfig.score_threshold and
# SegmentConfig.threshold, for the help texts, for the same reason.
_DNN_SCORE_DEFAULT = 0.4
_FACE_SCORE_DEFAULT = 0.6
_SEGMENT_THRESHOLD_DEFAULT = 0.5

# Default --lookahead: the file form's --gap-fill default for a file or image folder
# and for a stream URL (CCTV, where a subject may enter slowly), a short delay for a
# webcam. Frames are buffered uncompressed, so the value is capped.
_FILE_LOOKAHEAD = 10
_STREAM_LOOKAHEAD = 10
_CAMERA_LOOKAHEAD = 3
_MAX_LOOKAHEAD = 600

# Bounds of --threads (OpenCV worker threads per process).
_MAX_THREADS = 64

_RUN_HWACCEL_HELP = (
    "FFmpeg -hwaccel value, passed to the FFmpeg process that encodes --output. In this "
    "form petrify decodes the source itself and sends FFmpeg raw frames, so there is "
    "nothing for FFmpeg to decode and this does not speed anything up. Choose a hardware "
    "encoder with --codec instead (for example h264_videotoolbox or h264_nvenc). "
    "Default: None."
)

_LIVE_KINDS = ("camera", "stream")

# The presets --profile accepts. A webcam profile was left out on purpose: its
# settings are not measured yet.
_PROFILES = ("cctv",)
# Values --profile cctv sets (see _profile_before_resolve and _profile_after_mode).
_CCTV_LOOKAHEAD = 10
_CCTV_SOURCE_TIMEOUT = 10.0
_CCTV_BACKGROUND_CHECK = "hog,dnn"

# Default port of the --web page (on 127.0.0.1 only).
_WEB_PORT = 8765


def _one_line(value: object) -> str:
    """Return ``value`` as text on one line, with control characters replaced by spaces."""
    text = "".join(ch if ch.isprintable() else " " for ch in str(value))
    return " ".join(text.split())


class _ModeCatalog:
    """The modes ``--mode`` accepts, discovered on first use.

    It serves as the ``choices`` of ``--mode``: membership and iteration cover the
    selectable modes only, so an unknown name gets argparse's usual "invalid
    choice" error listing them. Discovery (:func:`petrify.core.registry.list_modes`)
    runs only when the run form needs it, never while the parser is built, so the
    file form and ``--help`` do not pay for it. A built-in name never needs it.
    Discovery reads entry point metadata alone and imports no plugin.
    """

    def __init__(self) -> None:
        self._entries: tuple[Any, ...] | None = None

    def entries(self) -> tuple[Any, ...]:
        """Every listed mode (``ModeEntry``), selectable or not."""
        if self._entries is None:
            from petrify.core.registry import list_modes  # noqa: PLC0415

            self._entries = list_modes()
        return self._entries

    def selectable(self) -> list[str]:
        """The names ``--mode`` accepts, in listing order."""
        names: list[str] = []
        for entry in self.entries():
            if entry.problem is None and entry.name not in names:
                names.append(entry.name)
        return names

    def selected(self, name: str) -> Any:
        """The selectable entry named ``name``, or None."""
        for entry in self.entries():
            if entry.name == name and entry.problem is None:
                return entry
        return None

    def __contains__(self, name: object) -> bool:
        return name in RUN_MODES or name in self.selectable()

    def __iter__(self) -> Iterator[str]:
        return iter(self.selectable())

    def check_name(self, text: str) -> str:
        """The ``--mode`` type: reject a listed but unselectable name with its reason.

        Any other name is returned unchanged; argparse then checks it against the
        selectable names.
        """
        if text in RUN_MODES:
            return text
        matching = [entry for entry in self.entries() if entry.name == text]
        if matching and all(entry.problem is not None for entry in matching):
            reasons = "; ".join(
                f"{_one_line(entry.origin)}: {_one_line(entry.problem)}" for entry in matching
            )
            raise argparse.ArgumentTypeError(
                f"the mode {_one_line(text)!r} cannot be selected: {reasons}"
            )
        return text


def _profile_name(text: str) -> str:
    """The ``--profile`` type: accept a known profile, otherwise name the ones available."""
    if text in _PROFILES:
        return text
    raise argparse.ArgumentTypeError(
        f"unknown profile {_one_line(text)!r}: only {', '.join(_PROFILES)} is available"
    )


def _build_run_parser(modes: _ModeCatalog | None = None) -> argparse.ArgumentParser:
    """Build the run form's parser; ``modes`` is the catalog ``--mode`` selects from."""
    catalog = modes if modes is not None else _ModeCatalog()
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
    # Required, except with --list-modes; _resolve_run reports it missing.
    src_group.add_argument(
        "--source",
        default=None,
        metavar="SPEC",
        help=(
            "What to read: a video file path, an image folder or glob (for example "
            "'frames/*.png'), camera:N for webcam number N, or an rtsp://, rtsps://, "
            "http:// or https:// stream URL. A bare number is a file name, not a camera. "
            "Required, except with --list-modes."
        ),
    )
    src_group.add_argument(
        "--mode",
        type=catalog.check_name,
        choices=catalog,
        default=None,
        metavar="NAME",
        help=(
            "Processing mode. 'erase' (built in): freeze the subject out of the scene. "
            "Installed plugins can add modes; a plugin mode is ordinary Python code that "
            "runs in the petrify process and receives the unprocessed frames, so install "
            "only plugins you trust. --list-modes shows the available modes. "
            "Default: erase."
        ),
    )
    src_group.add_argument(
        "--mode-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="mode_option",
        help=(
            "Set an option of the selected mode; repeat it for several options. "
            "--list-modes --mode NAME shows a mode's options. The erase mode has none."
        ),
    )
    src_group.add_argument(
        "--list-modes",
        action="store_true",
        default=False,
        dest="list_modes",
        help=(
            "Print the available modes and exit; needs no --source. With --mode NAME, "
            "also print that mode's options. No plugin is imported except the one "
            "named by --mode."
        ),
    )
    src_group.add_argument(
        "--fps",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Frame rate of an image folder source (image folders only). Default: 25.",
    )
    src_group.add_argument(
        "--source-timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        dest="source_timeout",
        help=(
            "End the run when a live source (camera:N or a stream URL) delivers no frame "
            "for SECONDS, the first frame included, instead of waiting for it forever. "
            "The recording made so far is kept and petrify exits with code 1, so a "
            "service manager can restart it. Live sources only. Default: no limit."
        ),
    )
    src_group.add_argument(
        "--profile",
        type=_profile_name,
        default=None,
        metavar="NAME",
        help=(
            "Preset of defaults. 'cctv', for a fixed security camera, sets --detector "
            "both, --person-detector dnn (needs its model file), --lookahead "
            f"{_CCTV_LOOKAHEAD}, --faces when the mode uses faces or face landmarks, "
            f"--background-check {_CCTV_BACKGROUND_CHECK} when the mode uses a running or "
            "calibrate background, and --source-timeout "
            f"{_CCTV_SOURCE_TIMEOUT:g} on a live source. A flag given on the command line "
            "always wins. It never turns on --track, --segment or a frame budget. Only "
            "cctv is available. Default: no profile."
        ),
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
    out_group.add_argument(
        "--web",
        action="store_true",
        default=False,
        help=(
            "Serve a local web page on 127.0.0.1 to watch the processed feed and change "
            "the run's settings, the selected mode's options included, while the run is "
            "in progress (needs FastAPI and uvicorn: "
            "pip install petrify[web]). The link is printed at start; it holds a "
            "per-run token and gives control of the run to whoever has it."
        ),
    )
    out_group.add_argument(
        "--web-port",
        type=int,
        default=_WEB_PORT,
        metavar="PORT",
        dest="web_port",
        help=f"Port of the web page on 127.0.0.1 (with --web); 0 picks a free port. Default: {_WEB_PORT}.",
    )
    out_group.add_argument(
        "--web-raw-preview",
        action="store_true",
        default=False,
        dest="web_raw_preview",
        help=(
            "Also show the UNPROCESSED input feed next to the processed one on the web "
            "page (with --web). This exposes the original frames, people included, to "
            "the local page. Off by default."
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
    engine_group.add_argument(
        "--realtime",
        action="store_true",
        default=False,
        help=(
            "Process a video file or image folder no faster than its frame rate, so it "
            "can be watched on the web page. It changes no output frame. Ignored for "
            "live sources (camera:N and stream URLs), which are already real time."
        ),
    )
    engine_group.add_argument(
        "--threads",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Limit OpenCV's worker threads in this process to N (1 to "
            f"{_MAX_THREADS}). When several petrify processes share a machine (one per "
            "camera), 2 is enough for one camera and keeps the processes from competing "
            "for the cores. Some OpenCV builds (the macOS ones) honor only 1 and keep "
            "their own count for other values; the count in use is logged. FFmpeg's "
            "encoder threads are not limited by this flag. Default: OpenCV's own "
            "thread count."
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

    # -- Tracking and models ------------------------------------------------
    track_group = parser.add_argument_group("Tracking and Models")
    track_group.add_argument(
        "--track",
        action="store_true",
        default=False,
        help=(
            "Enable the box tracker: a person or face box is held for a few frames after "
            "the detector stops finding it. Only for modes that use boxes; the erase "
            "mode does not accept it."
        ),
    )
    track_group.add_argument(
        "--track-hold",
        type=int,
        default=None,
        metavar="N",
        dest="track_hold",
        help=(
            "Frames a tracked box is held after its last detection (implies --track). "
            f"Default: {_TRACK_HOLD_DEFAULT}."
        ),
    )
    track_group.add_argument(
        "--model-dir",
        default=None,
        metavar="DIR",
        dest="model_dir",
        help=(
            "Directory searched first for detector model files, before "
            "$PETRIFY_MODEL_DIR and the per-user model directory. Only the detectors "
            "below and their background checks use model files; petrify never "
            "downloads them."
        ),
    )

    # -- Detectors with model files ------------------------------------------
    model_group = parser.add_argument_group("Detectors With Model Files")
    model_group.add_argument(
        "--person-detector",
        choices=("hog", "dnn"),
        default="hog",
        dest="person_detector",
        help=(
            "Person box detector used by --detector person and both. 'hog': OpenCV HOG "
            "(default). 'dnn': the NanoDet-Plus model, which needs its model file. The "
            "DNN misses people smaller than roughly 110 px tall on a 1080p frame (the "
            "frame is scaled down to 416 px for it); the background differ covers that "
            "gap only while it is enabled (not with --no-bg-diff). The locked background "
            "pre-scan always uses HOG. Default: hog."
        ),
    )
    model_group.add_argument(
        "--dnn-score-threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        dest="dnn_score_threshold",
        help=(
            "Minimum score in (0, 1) to accept a DNN person box (with --person-detector "
            "dnn). It also applies when dnn checks the background samples "
            f"(--background-check). Default: {_DNN_SCORE_DEFAULT}."
        ),
    )
    model_group.add_argument(
        "--faces",
        action="store_true",
        default=False,
        help=(
            "Enable the YuNet face detector (needs its model file) for modes that use "
            "faces or face landmarks. The erase mode does not use it."
        ),
    )
    model_group.add_argument(
        "--face-score-threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        dest="face_score_threshold",
        help=(
            "Minimum score in (0, 1) to accept a face (with --faces). It also applies "
            "when yunet checks the background samples (--background-check). Default: "
            f"{_FACE_SCORE_DEFAULT}."
        ),
    )
    model_group.add_argument(
        "--segment",
        action="store_true",
        default=False,
        help=(
            "Enable the PP-HumanSeg person segmenter (needs its model file) for modes "
            "that use a person mask. The erase mode does not use it."
        ),
    )
    model_group.add_argument(
        "--segment-threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        dest="segment_threshold",
        help=(
            "Person probability in (0, 1) from which a pixel belongs to the mask (with "
            f"--segment). Default: {_SEGMENT_THRESHOLD_DEFAULT}."
        ),
    )

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
            "detector finds a person is not used (--background-check chooses the "
            "detectors). Without the check a person in view while the background is "
            "built is absorbed into it and shown after they leave. The check reduces "
            "but does not eliminate this: start with the scene empty either way."
        ),
    )
    bg_group.add_argument(
        "--background-check",
        default=None,
        metavar="NAMES",
        dest="background_checkers",
        help=(
            "Comma-separated detectors that check the frames the running or calibrate "
            "background is learned from: hog, yunet (faces), dnn (person boxes). A "
            "frame is not used when any of them finds a person. The list replaces the "
            "default, so include hog to keep it (for example hog,dnn). yunet and dnn "
            "need their model files. Default: hog."
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
    source_timeout: float | None = None


def _resolve_run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> _RunSettings:
    """Validate the run arguments, apply the per-source defaults, and build the config.

    Every problem is reported with ``parser.error`` (exit code 2). No message holds
    the source spec, which may be a URL with credentials.
    """
    from petrify.core.config import (  # noqa: PLC0415
        BackgroundConfig,
        FaceYuNetConfig,
        ModelConfig,
        PersonDnnConfig,
        PetrifyConfig,
        SegmentConfig,
        TrackerConfig,
    )
    from petrify.pipeline import OutputConfig, _same_file  # noqa: PLC0415
    from petrify.sources import source_kind  # noqa: PLC0415

    if args.source is None:
        parser.error("the following arguments are required: --source")
    if args.run_gap_fill is not None:
        parser.error("--gap-fill is not a petrify run option; use --lookahead (same meaning)")
    try:
        kind = source_kind(args.source)
    except ValueError as exc:
        parser.error(str(exc))  # Holds only the redacted URL.
    live = kind in _LIVE_KINDS

    if not (args.output or args.virtualcam or args.preview or args.web):
        parser.error(
            "at least one output is required: --output, --virtualcam, --preview or --web"
        )
    if args.virtualcam_device is not None and not args.virtualcam:
        parser.error("--virtualcam-device requires --virtualcam")
    if args.web_raw_preview and not args.web:
        parser.error("--web-raw-preview requires --web")
    if not 0 <= args.web_port <= 65535:
        parser.error(f"--web-port must be in 0..65535, got {args.web_port}")

    if not (0.0 < args.downsample <= 1.0):
        parser.error(f"--downsample must be in (0, 1], got {args.downsample}")
    if args.lookahead is not None and args.lookahead < 0:
        parser.error(f"--lookahead must be >= 0, got {args.lookahead}")
    if args.threads is not None and not 1 <= args.threads <= _MAX_THREADS:
        parser.error(f"--threads must be in 1..{_MAX_THREADS}, got {args.threads}")
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
    if args.source_timeout is not None:
        if not (math.isfinite(args.source_timeout) and args.source_timeout > 0):
            parser.error(f"--source-timeout must be > 0, got {args.source_timeout}")
        if not live:
            parser.error(
                "--source-timeout applies only to a live source (camera:N or a stream URL)"
            )

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

    tracker: TrackerConfig | None = None
    if args.track_hold is not None:
        try:
            tracker = TrackerConfig(hold_frames=args.track_hold)
        except ValueError as exc:
            parser.error(f"--track-hold: {exc}")
    elif args.track:
        tracker = TrackerConfig()
    try:
        models = ModelConfig(dir=args.model_dir)
    except ValueError as exc:
        parser.error(f"--model-dir: {exc}")

    for flag, value, needs, enabled in (
        ("--dnn-score-threshold", args.dnn_score_threshold, "--person-detector dnn",
         args.person_detector == "dnn"),
        ("--face-score-threshold", args.face_score_threshold, "--faces", args.faces),
        ("--segment-threshold", args.segment_threshold, "--segment", args.segment),
    ):
        if value is not None and not enabled:
            parser.error(f"{flag} requires {needs}")
    person_dnn: PersonDnnConfig | None = None
    faces: FaceYuNetConfig | None = None
    segment: SegmentConfig | None = None
    try:
        if args.person_detector == "dnn":
            person_dnn = PersonDnnConfig(
                **_given(score_threshold=args.dnn_score_threshold)  # type: ignore[arg-type]
            )
        if args.faces:
            faces = FaceYuNetConfig(
                **_given(score_threshold=args.face_score_threshold)  # type: ignore[arg-type]
            )
        if args.segment:
            segment = SegmentConfig(
                **_given(threshold=args.segment_threshold)  # type: ignore[arg-type]
            )
    except ValueError as exc:
        parser.error(str(exc))

    sample_checkers: tuple[str, ...] | None = None
    if args.background_checkers is not None:
        if not args.background_check:
            parser.error("--background-check cannot be combined with --no-background-check")
        if background == "locked":
            parser.error(
                "--background-check applies to --background running or calibrate; the "
                "locked background pre-scan uses HOG"
            )
        sample_checkers = tuple(name.strip() for name in args.background_checkers.split(","))
        try:
            BackgroundConfig(sample_checkers=sample_checkers)
        except ValueError as exc:
            parser.error(f"--background-check: {exc}")

    base = PetrifyConfig.from_namespace(args)
    bg_changes: dict[str, object] = {
        "strategy": background,
        "sample_check": args.background_check,
    }
    if sample_checkers is not None:
        bg_changes["sample_checkers"] = sample_checkers
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
        mode=args.mode or _DEFAULT_MODE,
        tracker=tracker,
        models=models,
        person_dnn=person_dnn,
        faces=faces,
        segment=segment,
    )
    outputs = OutputConfig(
        file=None if args.output is None else Path(args.output),
        virtualcam=args.virtualcam,
        virtualcam_device=args.virtualcam_device,
        preview=args.preview,
    )
    return _RunSettings(
        kind=kind, config=config, outputs=outputs, source_timeout=args.source_timeout
    )


def _explicit_dests(catalog: _ModeCatalog, argv: list[str]) -> frozenset[str]:
    """The ``dest`` of every run option given on the command line.

    ``argv`` is parsed again with every default replaced by ``argparse.SUPPRESS``,
    so only the options argparse actually consumed appear in the namespace. This
    covers abbreviations (``--detect``), the ``--flag=value`` form, and a value
    equal to the default, which matching the argument strings would get wrong.
    ``argv`` already parsed once, so this parse cannot fail.
    """
    parser = _build_run_parser(catalog)
    for action in parser._actions:
        action.default = argparse.SUPPRESS
    return frozenset(vars(parser.parse_args(argv)))


def _profile_before_resolve(args: argparse.Namespace, explicit: frozenset[str]) -> list[str]:
    """Apply the cctv values that do not depend on the mode; return them as flags.

    They are set on ``args`` before ``_resolve_run``, so its checks see them (for
    example ``--dnn-score-threshold`` then needs no ``--person-detector dnn``).
    """
    from petrify.sources import source_kind  # noqa: PLC0415

    applied: list[str] = []
    for dest, flag, value, shown in (
        ("detector", "--detector", "both", "both"),
        ("person_detector", "--person-detector", "dnn", "dnn"),
        ("lookahead", "--lookahead", _CCTV_LOOKAHEAD, str(_CCTV_LOOKAHEAD)),
    ):
        if dest not in explicit:
            setattr(args, dest, value)
            applied.append(f"{flag} {shown}")
    if "source_timeout" not in explicit and args.source is not None:
        try:
            live = source_kind(args.source) in _LIVE_KINDS
        except ValueError:
            live = False  # _resolve_run reports the spec as a usage error.
        if live:
            args.source_timeout = _CCTV_SOURCE_TIMEOUT
            applied.append(f"--source-timeout {_CCTV_SOURCE_TIMEOUT:g}")
    return applied


def _profile_after_mode(
    args: argparse.Namespace, explicit: frozenset[str], factory: Any, config: PetrifyConfig
) -> list[str]:
    """Apply the cctv values that depend on the loaded mode; return them as flags.

    ``--faces`` is set only for a mode that uses faces or landmarks, and
    ``--background-check`` only for a mode that uses a running or calibrate
    background, so the profile never causes a "has no effect" warning for them.
    """
    from petrify.core.detections import DetectionKind  # noqa: PLC0415
    from petrify.pipeline import _mode_options  # noqa: PLC0415

    applied: list[str] = []
    if "faces" not in explicit:
        try:
            kinds = factory.requires(_mode_options(factory, config))
        except Exception:  # noqa: BLE001
            # The run reports a failing requires at startup with its own error; the
            # profile only has to avoid turning on faces on a guess.
            kinds = frozenset()
        if DetectionKind.FACES in kinds or DetectionKind.LANDMARKS in kinds:
            args.faces = True
            applied.append("--faces")
    if (
        not explicit & {"background_checkers", "background_check"}
        and factory.needs_background
        and config.background.strategy in ("running", "calibrate")
    ):
        args.background_checkers = _CCTV_BACKGROUND_CHECK
        applied.append(f"--background-check {_CCTV_BACKGROUND_CHECK}")
    return applied


def _given(**values: object) -> dict[str, object]:
    """The keyword arguments whose value is not None (flags left at their default)."""
    return {name: value for name, value in values.items() if value is not None}


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


_TRUST_WARNING = (
    "The mode {name!r} comes from the installed plugin {origin}. It runs in the petrify "
    "process with petrify's permissions and receives the unprocessed frames, people "
    "included."
)


def _load_mode(catalog: _ModeCatalog, name: str) -> tuple[Any, Any]:
    """Load the mode ``name``; return its factory and its entry (None for a built-in).

    Raises ``ModeLoadError``. Only the selected mode's code is imported.
    """
    from petrify.core.registry import load_mode  # noqa: PLC0415

    if name in RUN_MODES:
        return load_mode(name), None
    return load_mode(name, catalog.entries()), catalog.selected(name)


def _mode_failure(exc: BaseException, name: str, entry: Any) -> str:
    """The one-line error for a mode that cannot be loaded or built, naming its origin."""
    text = _one_line(exc)
    origin = "built in" if entry is None else _one_line(entry.origin)
    if origin not in text:
        text = f"{text} (mode {_one_line(name)!r}, {origin})"
    return text


def _option_summary(specs: tuple[Any, ...]) -> str:
    return ", ".join(f"{spec.name} ({spec.kind})" for spec in specs)


def _mode_options_config(
    parser: argparse.ArgumentParser, args: argparse.Namespace, factory: Any, config: PetrifyConfig
) -> PetrifyConfig:
    """Return ``config`` with the ``--mode-option`` values of the loaded mode.

    Every declared option is stored, defaults filled in, each in its declared type.
    Every problem is a usage error (``parser.error``, exit code 2).
    """
    from petrify.core.config import mode_options_from  # noqa: PLC0415
    from petrify.core.plugin import resolve_option_text  # noqa: PLC0415

    specs = factory.options
    if args.mode_option and not specs:
        parser.error(f"--mode-option: the mode {config.mode!r} has no options")
    try:
        options = resolve_option_text(specs, args.mode_option)
    except ValueError as exc:
        parser.error(
            f"--mode-option: {_one_line(exc)}. Options of the mode {config.mode!r}: "
            f"{_option_summary(specs)}"
        )
    return dataclasses.replace(config, mode_options=mode_options_from(options))


def _format_value(value: object) -> str:
    """An option value as it would be written after ``--mode-option KEY=``."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, tuple):
        return ",".join(value) if value else "(empty)"
    return _one_line(value)


def _option_lines(spec: Any) -> list[str]:
    lines = [f"  {spec.name}", f"    kind: {spec.kind}", f"    default: {_format_value(spec.default)}"]
    if spec.kind in ("choice", "choices"):
        lines.append(f"    choices: {', '.join(spec.choices)}")
    elif spec.kind in ("int", "float"):
        if spec.minimum is not None and spec.maximum is not None:
            bounds = f"{spec.minimum} to {spec.maximum}"
        elif spec.minimum is not None:
            bounds = f"at least {spec.minimum}"
        elif spec.maximum is not None:
            bounds = f"at most {spec.maximum}"
        else:
            bounds = "any"
        lines.append(f"    range: {bounds}")
    lines.append(f"    changeable while running: {'yes' if spec.runtime else 'no'}")
    lines.append(f"    help: {_one_line(spec.help)}")
    return lines


def _list_modes(catalog: _ModeCatalog, name: str | None) -> int:
    """Print the modes, and the options of the mode ``name`` when given; return the exit code.

    Only built-in modes and the mode ``name`` are imported: a plugin's summary is
    shown only for the mode ``name``, since reading it needs the plugin's code.
    """
    from petrify.core.registry import ModeLoadError, load_mode  # noqa: PLC0415

    lines = ["Modes (select one with --mode NAME):"]
    for entry in catalog.entries():
        shown = _one_line(entry.name)
        if entry.builtin:
            try:
                summary = _one_line(load_mode(entry.name).summary)
            except ModeLoadError as exc:
                summary = f"cannot be loaded: {_one_line(exc)}"
            lines.append(f"  {shown} (built in): {summary}")
            continue
        line = f"  {shown} (plugin {_one_line(entry.origin)})"
        if entry.problem is not None:
            line += f": not selectable: {_one_line(entry.problem)}"
        lines.append(line)
    print("\n".join(lines), flush=True)
    if name is None:
        return 0
    try:
        factory, entry = _load_mode(catalog, name)
    except ModeLoadError as exc:
        logging.getLogger(__name__).error(
            "Fatal error: %s", _mode_failure(exc, name, catalog.selected(name))
        )
        return 1
    origin = "built in" if entry is None else f"plugin {_one_line(entry.origin)}"
    lines = ["", f"Mode {name} ({origin}): {_one_line(factory.summary)}"]
    if factory.options:
        lines.append("Options (set with --mode-option NAME=VALUE):")
        for spec in factory.options:
            lines.extend(_option_lines(spec))
    else:
        lines.append("Options: none")
    print("\n".join(lines), flush=True)
    return 0


def _set_opencv_threads(count: int) -> None:
    """Set OpenCV's worker thread count for this process (``--threads``).

    Called before the mode is loaded and before the source, any detector or any
    model is built, so all OpenCV work of the run, model loading included, uses
    the requested count from the start.
    """
    import cv2  # noqa: PLC0415

    cv2.setNumThreads(count)
    # Some OpenCV builds ignore the request (the macOS wheels honor only 1), so the
    # log shows the count OpenCV reports, not the one asked for.
    actual = cv2.getNumThreads()
    log = logging.getLogger(__name__)
    log.info("OpenCV worker threads: %d", actual)
    if actual != count:
        log.warning(
            "--threads %d was requested, but this OpenCV build uses %d worker threads; "
            "some builds honor only 1",
            count,
            actual,
        )


def _run_main(argv: list[str]) -> int:
    """``petrify run``: open the source, run the streaming engine, return an exit code."""
    catalog = _ModeCatalog()
    parser = _build_run_parser(catalog)
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    if args.list_modes:
        return _list_modes(catalog, args.mode)
    explicit = _explicit_dests(catalog, argv) if args.profile is not None else frozenset()
    profile_applied: list[str] = []
    if args.profile is not None:
        profile_applied = _profile_before_resolve(args, explicit)
    settings = _resolve_run(parser, args)
    if args.threads is not None:
        _set_opencv_threads(args.threads)

    from petrify.core.registry import ModeLoadError  # noqa: PLC0415
    from petrify.models import ModelError  # noqa: PLC0415
    from petrify.pipeline import (  # noqa: PLC0415
        OutputsKeptInterrupt,
        output_overwrites_input,
        run_stream,
    )
    from petrify.sources import display_spec, open_source  # noqa: PLC0415

    log = logging.getLogger(__name__)
    finished = False  # The run returned: its outputs are final.
    mode_name = settings.config.mode
    # The mode is loaded before the source is opened, to validate its options.
    try:
        factory, mode_entry = _load_mode(catalog, mode_name)
    except ModeLoadError as exc:
        log.error("Fatal error: %s", _mode_failure(exc, mode_name, catalog.selected(mode_name)),
                  exc_info=args.verbose)
        return 1
    settings = dataclasses.replace(
        settings, config=_mode_options_config(parser, args, factory, settings.config)
    )
    if args.profile is not None:
        late = _profile_after_mode(args, explicit, factory, settings.config)
        if late:
            # Resolved again so the new values pass the same checks as typed flags.
            settings = _resolve_run(parser, args)
            settings = dataclasses.replace(
                settings, config=_mode_options_config(parser, args, factory, settings.config)
            )
        profile_applied += late
        log.info(
            "Profile %s: set %s; explicit flags win",
            args.profile,
            ", ".join(profile_applied) if profile_applied else "nothing",
        )
    if mode_entry is not None:
        log.warning(_TRUST_WARNING.format(name=mode_name, origin=_one_line(mode_entry.origin)))
    if args.web:
        from petrify.server.app import require_web  # noqa: PLC0415

        try:
            require_web()
        except RuntimeError as exc:
            log.error("%s", exc)
            return 1
    try:
        # read_timeout is passed only when set, so a run without --source-timeout
        # opens the source exactly as before.
        timeout = (
            {} if settings.source_timeout is None else {"read_timeout": settings.source_timeout}
        )
        source = open_source(args.source, fps=args.fps, **timeout)
        if settings.outputs.file is not None and output_overwrites_input(
            source, settings.outputs.file
        ):
            source.close()
            parser.error("--output is one of the input files; choose another output path")
        log.info(
            "Source: %s (%s), mode: %s",
            display_spec(source.info.uri or ""),
            settings.kind,
            mode_name,
        )
        # Failed-closed frames still exit 0 (they are written black and logged), and
        # so does a live run ended with Ctrl-C: its outputs were finalized. A missing
        # FFmpeg is a fatal error (exit 1), not a missing input (exit 2).
        if args.web:
            _run_with_web(args, settings, source, factory=factory, modes=catalog.entries())
        elif args.realtime:
            run_stream(
                source, settings.config, settings.outputs, realtime=True, factory=factory
            )
        else:
            run_stream(source, settings.config, settings.outputs, factory=factory)
        finished = True
    except ModelError as exc:
        # Before FileNotFoundError: a missing model is a ModelNotFoundError, which is
        # also a FileNotFoundError, but it is a fatal error (exit 1), not a missing input.
        log.error("Fatal error: %s", _one_line(exc), exc_info=args.verbose)
        return 1
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2
    except KeyboardInterrupt as exc:
        if finished or isinstance(exc, OutputsKeptInterrupt):
            # The stop came after the outputs were finalized: nothing was aborted.
            log.info("Stopped by user; the outputs were kept.")
            return 0
        log.info("Interrupted by user; the run was aborted.")
        return 130
    except ModeLoadError as exc:
        # The same one-line report as a mode that fails to load above.
        log.error("Fatal error: %s", _mode_failure(exc, mode_name, mode_entry),
                  exc_info=args.verbose)
        return 1
    except Exception as exc:  # noqa: BLE001
        log.error("Fatal error: %s", exc, exc_info=args.verbose)
        return 1
    return 0


def _run_with_web(
    args: argparse.Namespace,
    settings: _RunSettings,
    source: Any,
    *,
    factory: Any,
    modes: tuple[Any, ...],
) -> None:
    """Run the stream with the local web page: serve it on a thread, process on this one.

    The processing loop stays on the calling (main) thread; the server runs on a
    daemon thread and is stopped when the run ends for any reason. The process
    does not wait after the run: the page then reports that the run has ended.
    The link printed here is the only place the token is shown. ``factory`` is
    the loaded mode and ``modes`` the listed ones (metadata only), for the page.
    """
    from petrify.pipeline import OutputsKeptInterrupt, run_stream  # noqa: PLC0415
    from petrify.server.app import create_app, serve_in_thread  # noqa: PLC0415
    from petrify.server.control import RunControl  # noqa: PLC0415
    from petrify.sinks.mjpeg import MjpegSink  # noqa: PLC0415

    try:
        token = secrets.token_urlsafe(32)
        control = RunControl(
            settings.config,
            (source.info.width, source.info.height),
            modes=modes,
            factory=factory,
        )
        processed = MjpegSink()
        raw = MjpegSink() if args.web_raw_preview else None
        app = create_app(control, processed, token=token, raw=raw, port=args.web_port)
        handle = serve_in_thread(app, args.web_port)
    except BaseException:
        source.close()
        raise
    try:
        print(
            f"Web page (this link gives control of this run to whoever has it): "
            f"http://127.0.0.1:{handle.port}/#token={token}",
            file=sys.stderr,
            flush=True,
        )
        run_stream(
            source,
            settings.config,
            settings.outputs,
            web_sink=processed,
            raw_preview=raw,
            control=control,
            realtime=args.realtime,
            factory=factory,
        )
    except BaseException:
        handle.stop()
        raise
    try:
        handle.stop()
    except KeyboardInterrupt as exc:
        # The run already returned, so its outputs are final.
        raise OutputsKeptInterrupt from exc


def _run_form_argv(argv: list[str]) -> list[str] | None:
    """Return the run form's arguments when ``argv`` selects it, else None.

    The run form is selected when the first non-option argument is ``run`` and the
    other arguments contain ``--source`` (or ``--list-modes``, ``-h`` / ``--help``);
    the ``run`` word is removed and the options around it are kept, so ``petrify --verbose run
    --source ...`` works. Option values are skipped when looking for the first
    non-option argument. Anything else is the file form, so a file named ``run``
    still works there: ``petrify run out.mp4``.
    """
    rest = _after_run_word(argv)
    if rest is None:
        return None
    wants_run = any(
        a in ("--source", "--list-modes", "-h", "--help") or a.startswith("--source=")
        for a in rest
    )
    return rest if wants_run else None


def _run_form_without_source(argv: list[str]) -> bool:
    """True when ``argv`` starts with ``run`` and uses an option only the run form has.

    Such a command is meant for ``petrify run`` but has no ``--source``, so it
    would otherwise be read as the file form (an input file named ``run``) and
    fail with an "unrecognized arguments" message that does not say what is
    missing. A valid file form command never uses these options.
    """
    rest = _after_run_word(argv)
    if rest is None:
        return False
    file_options = set(_build_parser()._option_string_actions)
    run_only = set(_build_run_parser()._option_string_actions) - file_options
    return any(a.split("=", 1)[0] in run_only for a in rest if a.startswith("--"))


def _after_run_word(argv: list[str]) -> list[str] | None:
    """Return ``argv`` without its first non-option argument when that is ``run``, else None.

    Option values are skipped when looking for the first non-option argument.
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
        return argv[:i] + argv[i + 1 :]
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
    # Suppress it for --help / --version / --list-modes (pure informational flags).
    _argv = sys.argv[1:] if argv is None else argv
    if not any(f in _argv for f in ("-h", "--help", "--version", "--list-modes")):
        print(_BANNER, file=sys.stderr)

    run_argv = _run_form_argv(_argv)
    if run_argv is not None:
        return _run_main(run_argv)
    if _run_form_without_source(_argv):
        # parser.error exits with code 2, as for every other usage error.
        _build_run_parser().error(
            "petrify run requires --source (for example: petrify run --source INPUT "
            "--output OUTPUT.mp4)"
        )

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
