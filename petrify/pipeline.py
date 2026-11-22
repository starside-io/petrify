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

"""Main pipeline orchestrator: two-pass motion/person detection and compositing."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import signal
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FrameType, TracebackType
from typing import TYPE_CHECKING, Any, TypeVar

import cv2
import numpy as np
from tqdm import tqdm

from petrify.background.calibrate import CalibratedBackground, frames_for
from petrify.background.locked import locked_background
from petrify.background.running import RunningBackground
from petrify.background.vetting import PersonCheck, any_person, detector_check, person_free
from petrify.core.config import (
    BackgroundConfig,
    FaceYuNetConfig,
    OptionValue,
    PersonDnnConfig,
    PetrifyConfig,
    mode_options_from,
)
from petrify.core.detections import DetectionKind
from petrify.core.engine import Engine, EngineStats, FrameTrace
from petrify.core.frame import Frame, SourceInfo
from petrify.core.plugin import (
    BackgroundModel,
    Detector,
    Mode,
    ModeContext,
    ModeFactory,
    Refiner,
    Sink,
    Source,
    Stage,
    _describe_error,
    resolve_options,
)
from petrify.core.registry import ModeLoadError, build_mode, load_mode
from petrify.detectors.bgdiff import BackgroundDiffer
from petrify.detectors.face_yunet import FaceYuNetDetector
from petrify.detectors.motion import MotionDetector
from petrify.detectors.person_dnn import PersonDnnDetector
from petrify.detectors.person_hog import PersonDetector
from petrify.detectors.segment import PersonSegmenter
from petrify.detectors.tracker import TrackerRefiner
from petrify.encoder import ffmpeg_encode
from petrify.models import (
    FACE_YUNET_MODEL,
    PERSON_DNN_MODEL,
    SEGMENT_MODEL,
    ModelError,
    ModelSpec,
    model_dirs,
    pop_loaded_path,
)
from petrify.modes.erase import EraseMode
from petrify.roi_detect import WarmupRoiDetector, auto_detect_roi
from petrify.sinks import file as file_sinks
from petrify.sinks.file import FfmpegPipeSink, FileSink
from petrify.sinks.preview import PreviewSink
from petrify.sinks.virtualcam import VirtualCamSink
from petrify.sources.file import FileSource
from petrify.sources.images import ImageFolderSource
from petrify.stages.roi_overlay import Compositor, RoiOverlay, WarmupRoiOverlay

if TYPE_CHECKING:
    from petrify.server.control import RunControl
    from petrify.sinks.mjpeg import MjpegSink

logger = logging.getLogger(__name__)

_ModelDetectorT = TypeVar("_ModelDetectorT")


# ---------------------------------------------------------------------------
# Pre-pass helpers
# ---------------------------------------------------------------------------

def _find_clean_background(
    cap: cv2.VideoCapture,
    total_frames: int,
    person_detector: PersonDetector,
    scan_step: int = 5,
    fps: float = 25.0,
) -> tuple[np.ndarray, int]:
    """Scan the entire video and return the *latest* frame with no person detected.

    Scanning every ``scan_step`` frames keeps the scan fast while being
    thorough enough to catch a clean background near the end of the clip.
    Falls back to frame 0 if every sampled frame contains a person.
    """
    logger.info("Auto-background scan: checking every %d frames for a clean frame...", scan_step)

    best_frame: np.ndarray | None = None
    best_idx: int = -1

    for idx in tqdm(
        range(0, total_frames, scan_step),
        desc="Background scan",
        unit="fr",
        dynamic_ncols=True,
        leave=False,
    ):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        if not person_detector.is_person(frame):
            # Keep updating — we want the LATEST clean frame.
            best_idx = idx
            best_frame = frame.copy()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if best_frame is None:
        logger.warning(
            "Background scan found no person-free frame — falling back to frame 0. "
            "Consider specifying --background-frame manually."
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _, best_frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        best_idx = 0
    else:
        logger.info(
            "Auto background: frame %d  (%.1fs into video)",
            best_idx,
            best_idx / fps,
        )

    return best_frame, best_idx


def _classify_frames(
    cap: cv2.VideoCapture,
    total_frames: int,
    motion_detector: MotionDetector | None,
    person_detector: PersonDetector | None,
    bg_differ: BackgroundDiffer | None = None,
) -> list[bool]:
    """First pass — classify every frame as *has subject* (True) or *clean* (False).

    A frame is marked as containing a subject when at least one of the enabled
    detectors fires.  Detectors are run in order and short-circuit once one fires:

    * ``motion`` (MOG2): fast, catches moving subjects.
    * ``person`` (HOG): catches stopped subjects that MOG2 adapted away.
    * ``bg_differ`` (static diff): compares against the locked clean background —
      catches subjects that are both stopped AND have been absorbed by MOG2, as well
      as tiny/partial subjects that HOG misses (e.g. person far down corridor, or
      body partially cut off at frame edge).

    Returns a ``list[bool]`` indexed by frame number (0 … total_frames-1).
    """
    logger.info("Pass 1 — classifying %d frames...", total_frames)

    if motion_detector is not None:
        motion_detector.reset()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    has_subject: list[bool] = []

    for _ in tqdm(range(total_frames), desc="Pass 1 classify", unit="fr", dynamic_ncols=True):
        ret, frame = cap.read()
        if not ret:
            break

        detected = False
        if motion_detector is not None:
            detected = motion_detector.is_motion(frame)
        if person_detector is not None and not detected:
            detected = person_detector.is_person(frame)
        if bg_differ is not None and not detected:
            detected = bg_differ.is_different(frame)

        has_subject.append(detected)

    # Pad to total_frames in case read stopped early.
    while len(has_subject) < total_frames:
        has_subject.append(False)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if motion_detector is not None:
        motion_detector.reset()

    n_subject = sum(has_subject)
    logger.info(
        "Pass 1 done — subject detected in %d / %d frames (%.1f%%)",
        n_subject,
        total_frames,
        100.0 * n_subject / total_frames if total_frames else 0,
    )
    return has_subject


def _fill_gaps(has_subject: list[bool], gap: int) -> list[bool]:
    """Expand every detected region by *gap* frames on both sides.

    This covers short missed detections (person bent, partially occluded,
    between HOG/MOG2 detections) by assuming they are still present for up
    to ``gap`` frames around a confirmed detection.
    """
    if gap <= 0:
        return list(has_subject)

    filled = list(has_subject)
    n = len(has_subject)

    for i in range(n):
        if has_subject[i]:
            start = max(0, i - gap)
            end = min(n, i + gap + 1)
            for j in range(start, end):
                filled[j] = True

    added = sum(filled) - sum(has_subject)
    logger.info("Gap fill (%d frames): expanded by %d frames", gap, added)
    return filled


def _freeze_cold_start(has_subject: list[bool], warmup_frames: int = 50) -> list[bool]:
    """Unconditionally freeze the first *warmup_frames* frames.

    MOG2 has two cold-start failure modes:

    1. **Frame 0 init spike** — the very first frame produces a huge spurious
       foreground mask (MOG2 initialises with no prior history), which is
       then immediately discarded, leaving frames 1–N with a nearly-empty
       background model and artificially low motion scores.

    2. **Warmup lag** — for the first ``history`` frames MOG2 hasn't seen
       enough samples to reliably distinguish foreground from background, so
       slow motion (person getting out of a car) is missed entirely.

    Freezing the first *warmup_frames* frames unconditionally is the safest
    fix: if the background is truly empty at the start those frames look
    identical to the locked background anyway, so there is no visual cost.
    """
    if warmup_frames < 0:
        raise ValueError(f"warmup_frames must be >= 0, got {warmup_frames}")
    filled = list(has_subject)
    n = min(warmup_frames, len(filled))
    for i in range(n):
        filled[i] = True
    logger.info("Cold-start freeze: frames 0–%d forced (MOG2 warmup window)", n - 1)
    return filled


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    """Execute the full two-pass petrify pipeline.

    Pass 1 — Classification
    -----------------------
    Every frame is classified as *has subject* or *clean* using the
    configured detector(s).  The resulting boolean list is then expanded by
    ``--gap-fill`` frames on each side of every detected region to cover
    brief missed detections (e.g. person bending out of a car).

    Background selection
    --------------------
    If ``--background-frame N`` is given that frame is used directly.
    Otherwise the video is scanned to find the *latest* clean frame
    (no person detected) which is locked as the frozen background for the
    entire output.  For security-camera footage the scene is typically clear
    at the end of the clip — this auto-detection handles that correctly.

    Pass 2 — Compositing
    --------------------
    Each frame is either frozen (background + live timestamp ROI) or passed
    through unchanged based on the classification from Pass 1.

    FFmpeg re-encodes the intermediate file and muxes the original audio.
    """
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # ------------------------------------------------------------------
    # Open input video
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video file: {input_path}")

    fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        "Input: %s  |  %dx%d @ %.2f fps  |  ~%d frames",
        input_path, width, height, fps, total_frames,
    )

    # ------------------------------------------------------------------
    # Build detectors
    # ------------------------------------------------------------------
    detector_mode: str = getattr(args, "detector", "motion")

    motion_detector: MotionDetector | None = None
    person_detector: PersonDetector | None = None

    if detector_mode in ("motion", "both"):
        motion_detector = MotionDetector(
            history=args.history,
            var_threshold=args.var_threshold,
            detect_shadows=args.detect_shadows,
            threshold=args.threshold,
            downsample=args.downsample,
        )

    if detector_mode in ("person", "both"):
        person_detector = PersonDetector(
            downsample=getattr(args, "hog_downsample", 0.5),
            win_stride=tuple(getattr(args, "hog_win_stride", [8, 8])),  # type: ignore[arg-type]
            scale=getattr(args, "hog_scale", 1.05),
            min_confidence=getattr(args, "hog_min_confidence", 0.3),
            min_box_area=getattr(args, "hog_min_box_area", 1500),
        )

    logger.info("Detector mode: %s", detector_mode)

    # ------------------------------------------------------------------
    # Resolve ROI (manual > auto-detect > disabled)
    # ------------------------------------------------------------------
    if args.roi:
        roi: tuple[int, int, int, int] | None = tuple(args.roi)  # type: ignore[assignment]
        logger.info("ROI: manual  x=%d y=%d w=%d h=%d", *roi)
    elif getattr(args, "no_roi", False):
        roi = None
        logger.info("ROI: disabled (--no-roi)")
    else:
        roi = auto_detect_roi(
            cap,
            total_frames,
            fps,
            n_samples=getattr(args, "roi_scan_samples", 60),
            min_freq=getattr(args, "roi_scan_min_freq", 0.6),
            margin=getattr(args, "roi_scan_margin", 8),
            max_area_frac=getattr(args, "roi_scan_max_area_frac", 0.10),
            edge_threshold=getattr(args, "roi_scan_edge_threshold", 80.0),
            edge_gap_tolerance=getattr(args, "roi_scan_edge_gap_tolerance", 30),
            n_edge_samples=getattr(args, "roi_scan_edge_samples", 3),
        )
        if roi is None:
            logger.warning(
                "ROI auto-detection found nothing — running without timestamp overlay. "
                "Use --roi X Y W H to set it manually, or --no-roi to silence this warning."
            )

    compositor = Compositor(roi)

    # ------------------------------------------------------------------
    # Find / load background reference frame
    # ------------------------------------------------------------------
    if args.background_frame is not None:
        bg_idx: int = args.background_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, bg_idx)
        ret_bg, locked_background = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if not ret_bg:
            cap.release()
            raise ValueError(
                f"Cannot read background frame at index {bg_idx} "
                f"(video has ~{total_frames} frames)."
            )
        logger.info("Background reference: frame %d (manual)", bg_idx)
    else:
        # Always need a PersonDetector for the background scan.
        scan_detector = person_detector or PersonDetector(
            downsample=getattr(args, "hog_downsample", 0.5),
            scale=getattr(args, "hog_scale", 1.05),
            min_confidence=getattr(args, "hog_min_confidence", 1.2),
        )
        locked_background, bg_idx = _find_clean_background(
            cap, total_frames, scan_detector, scan_step=5, fps=fps
        )

    # ------------------------------------------------------------------
    # Pass 1 — classify every frame
    # ------------------------------------------------------------------
    # Build static background differ (always on unless --no-bg-diff is set).
    # This catches subjects that MOG2 has adapted away (standing still) and
    # tiny/partial subjects that HOG misses.
    bg_differ: BackgroundDiffer | None = None
    if not getattr(args, "no_bg_diff", False):
        bg_differ = BackgroundDiffer(
            reference=locked_background,
            pixel_threshold=getattr(args, "bg_diff_pixel_threshold", 25),
            area_threshold=getattr(args, "bg_diff_area_threshold", 200),
            downsample=args.downsample,
        )
        logger.info(
            "BackgroundDiffer enabled: pixel_threshold=%d area_threshold=%d",
            bg_differ.pixel_threshold,
            bg_differ.area_threshold,
        )

    has_subject = _classify_frames(cap, total_frames, motion_detector, person_detector, bg_differ)
    gap_fill: int = getattr(args, "gap_fill", 10)
    has_subject = _fill_gaps(has_subject, gap_fill)
    has_subject = _freeze_cold_start(has_subject)

    # ------------------------------------------------------------------
    # Pass 2 — compositing
    # ------------------------------------------------------------------
    fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    logger.info("Pass 2 — compositing frames -> %s", temp_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        os.unlink(temp_path)
        raise OSError("cv2.VideoWriter failed to open the intermediate file.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames_frozen = 0
    frames_pass = 0

    try:
        for frame_idx in tqdm(range(total_frames), desc="Pass 2 composite", unit="fr", dynamic_ncols=True):
            ret, frame = cap.read()
            if not ret:
                break

            # Track B: live ROI (timestamp) — always from real frame
            live_roi = compositor.extract_roi(frame)

            # Track A: background selection
            if has_subject[frame_idx]:
                frames_frozen += 1
                bg = locked_background.copy()
            else:
                frames_pass += 1
                bg = frame

            output_frame = compositor.compose(bg, live_roi)
            writer.write(output_frame)

    finally:
        cap.release()
        writer.release()

    total_processed = frames_frozen + frames_pass
    if total_processed > 0:
        logger.info(
            "Pass 2 done — frozen=%d (%.1f%%)  pass-through=%d (%.1f%%)",
            frames_frozen, 100.0 * frames_frozen / total_processed,
            frames_pass,  100.0 * frames_pass  / total_processed,
        )

    # ------------------------------------------------------------------
    # Re-encode + mux audio
    # ------------------------------------------------------------------
    logger.info("Re-encoding with FFmpeg -> %s", output_path)
    try:
        ffmpeg_encode(
            video_path=temp_path,
            original_audio_source=input_path,
            output_path=output_path,
            codec=args.codec,
            crf=args.crf,
            preset=args.preset,
            hwaccel=args.hwaccel,
        )
    finally:
        try:
            os.unlink(temp_path)
        except OSError as exc:
            logger.warning("Could not remove temp file %s: %s", temp_path, exc)

    logger.info("Done. Output written to: %s", output_path)


# ---------------------------------------------------------------------------
# Streaming engine file runner
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunReport:
    """Outcome of :func:`run_file` or :func:`run_stream`.

    ``background_index`` is the locked background's frame index, or None for an
    online background strategy. ``roi`` is the ROI in use at the end of the run
    (None when disabled or not found). ``dropped_frames`` counts frames a live
    source captured but never delivered. ``interrupted`` is True when a live run
    was ended with Ctrl-C and finished cleanly.
    """

    stats: EngineStats
    background_index: int | None
    roi: tuple[int, int, int, int] | None
    dropped_frames: int = 0
    interrupted: bool = False


def _resolve_roi(source: FileSource, config: PetrifyConfig) -> tuple[int, int, int, int] | None:
    """Return the manual ROI, None when disabled, or the auto-detected one."""
    roi_cfg = config.roi
    if roi_cfg.mode == "manual":
        assert roi_cfg.box is not None
        logger.info("ROI: manual  x=%d y=%d w=%d h=%d", *roi_cfg.box)
        return roi_cfg.box
    if roi_cfg.mode == "disabled":
        logger.info("ROI: disabled (--no-roi)")
        return None
    info = source.info
    roi = auto_detect_roi(
        source.capture,
        info.frame_count or 0,
        info.fps,
        n_samples=roi_cfg.scan_samples,
        min_freq=roi_cfg.scan_min_freq,
        margin=roi_cfg.scan_margin,
        max_area_frac=roi_cfg.scan_max_area_frac,
        edge_threshold=roi_cfg.scan_edge_threshold,
        edge_gap_tolerance=roi_cfg.scan_edge_gap_tolerance,
        n_edge_samples=roi_cfg.scan_edge_samples,
    )
    if roi is None:
        logger.warning(
            "ROI auto-detection found nothing, running without timestamp overlay. "
            "Use --roi X Y W H to set it manually, or --no-roi to silence this warning."
        )
    return roi


def run_file(
    input_path: str | Path,
    output_path: str | Path,
    config: PetrifyConfig,
    observer: Callable[[FrameTrace], None] | None = None,
) -> RunReport:
    """Run the streaming engine on a video file: one pass, then FFmpeg re-encode.

    Builds the same components as :func:`run` in the same order (source, detectors,
    ROI, locked background, background differ) and drives them with :class:`Engine`.
    """
    source = FileSource(input_path)
    try:
        detectors: list[Detector] = []
        if config.detector in ("motion", "both"):
            detectors.append(MotionDetector.from_config(config.motion))
        if config.detector in ("person", "both"):
            detectors.append(PersonDetector.from_config(config.person))
        logger.info("Detector mode: %s", config.detector)

        roi = _resolve_roi(source, config)
        background, bg_idx = locked_background(
            source, config.background, config.background_scan_person_config()
        )
        if config.background_diff is not None:
            bg_differ = BackgroundDiffer.from_config(background, config.background_diff)
            detectors.append(bg_differ)
            logger.info(
                "BackgroundDiffer enabled: pixel_threshold=%d area_threshold=%d",
                bg_differ.pixel_threshold,
                bg_differ.area_threshold,
            )

        with tqdm(
            total=source.info.frame_count or None,
            desc="Petrify",
            unit="fr",
            dynamic_ncols=True,
        ) as progress:

            def on_frame(trace: FrameTrace) -> None:
                progress.update(1)
                if observer is not None:
                    observer(trace)

            engine = Engine(
                detectors,
                EraseMode(background),
                [RoiOverlay(roi)],
                [FileSink(output_path, input_path, config.encode)],
                config.engine,
                on_frame,
            )
            stats = engine.run(source)
    finally:
        source.close()

    if stats.frames > 0:
        logger.info(
            "Engine done: frozen=%d (%.1f%%)  pass-through=%d (%.1f%%)",
            stats.frozen, 100.0 * stats.frozen / stats.frames,
            stats.passed_through, 100.0 * stats.passed_through / stats.frames,
        )
    if stats.failed_closed > 0:
        logger.warning(
            "%d frame(s) failed closed and were written as black frames",
            stats.failed_closed,
        )
    return RunReport(stats=stats, background_index=bg_idx, roi=roi)


# ---------------------------------------------------------------------------
# Streaming engine runner for any source
# ---------------------------------------------------------------------------

ONLINE_BACKGROUND_WARNING = (
    "Background limitation: the clean background is learned from the first seconds "
    "of the source. A person in view during that time can be absorbed into it and is "
    "then shown on frozen frames after they leave. The person check reduces this but "
    "does not eliminate it (it can miss a seated person framed from the chest up). "
    "Start with the scene empty."
)
NO_BACKGROUND_CHECK_WARNING = (
    "Background person check disabled (--no-background-check): this is unsafe. "
    "Nothing checks the frames the background is learned from, so a person in view "
    "while it is built is absorbed into it and shown after they leave."
)
NO_BG_DIFF_LIVE_WARNING = (
    "Background differ disabled (--no-bg-diff) on a live source: a subject who stops "
    "moving is no longer detected and can be shown. The differ is the only protection "
    "against that."
)
DNN_WITHOUT_DIFFER_WARNING = (
    "The DNN person detector runs without the background differ (--no-bg-diff, or a "
    "mode that does not use it): the DNN misses people smaller than about 110 px tall "
    "on a 1080p frame, and nothing else covers that gap. Small or distant people can "
    "be shown."
)
DNN_NOT_USED_WARNING = (
    "--person-detector dnn has no effect: --detector motion runs no person detector. "
    "It takes effect when the detector is switched to person or both."
)
FACES_NOT_USED_WARNING = (
    "--faces has no effect: the mode {mode!r} does not use faces or face landmarks, so "
    "the face detector does not run on the frames."
)
SEGMENT_NOT_USED_WARNING = (
    "--segment has no effect: the mode {mode!r} does not use a person mask, so the "
    "segmenter does not run on the frames."
)
BACKGROUND_CHECK_NOT_USED_WARNING = (
    "--background-check has no effect: the mode {mode!r} uses no background, so no "
    "background sample is checked."
)
SAMPLE_REJECTION_STREAK = 50
"""Consecutive rejected samples, with no background yet, after which the run warns."""
SAMPLE_CHECK_STUCK_WARNING = (
    "The background sample check ({checkers}) rejected {count} samples in a row and "
    "the background is still not ready: it cannot be built while these checkers see a "
    "person in every sample, and until then every frame is written black."
)


class NoFramesError(RuntimeError):
    """The source delivered no frames, so no output could be written."""


class FfmpegNotFoundError(RuntimeError):
    """FFmpeg is needed for the file output and was not found."""


def _same_file(a: Path, b: Path) -> bool:
    """True when ``b`` exists and is the same file as ``a``."""
    try:
        return a.exists() and b.exists() and os.path.samefile(a, b)
    except OSError:
        return False


def output_overwrites_input(source: Source, output: str | Path) -> bool:
    """True when ``output`` is the file source itself or one of an image folder's images."""
    out = Path(output)
    if isinstance(source, FileSource) and source.info.uri:
        return _same_file(Path(source.info.uri), out)
    if isinstance(source, ImageFolderSource):
        return any(_same_file(p, out) for p in source.paths)
    return False


@dataclass(frozen=True)
class OutputConfig:
    """Where :func:`run_stream` writes: a video file, a virtual camera, a preview window."""

    file: Path | None = None
    virtualcam: bool = False
    virtualcam_device: str | None = None
    preview: bool = False

    def __post_init__(self) -> None:
        if self.virtualcam_device is not None and not self.virtualcam:
            raise ValueError("OutputConfig.virtualcam_device requires virtualcam=True")

    @property
    def selected(self) -> bool:
        """True when at least one output is selected."""
        return self.file is not None or self.virtualcam or self.preview


def _build_sinks(source: Source, config: PetrifyConfig, outputs: OutputConfig) -> list[Sink]:
    """Build the sinks ``outputs`` selects.

    Every sink runs on the calling thread, for every source kind. A live source
    already decouples capture from processing (its capture thread keeps only the
    latest frame), so the file and virtual camera sinks are not wrapped in a
    ``ThreadedSink``: the wrapper delivered frames to FFmpeg in bursts and its write
    timeout could end a run. The preview must stay on the calling thread anyway, as
    GUI calls must run there.
    """
    info = source.info
    live = info.live
    sinks: list[Sink] = []
    if outputs.file is not None:
        audio = Path(info.uri) if isinstance(source, FileSource) and info.uri else None
        sinks.append(
            FfmpegPipeSink(
                outputs.file,
                config.encode,
                audio_source=audio,
                wallclock_timestamps=live,
            )
        )
    if outputs.virtualcam:
        sinks.append(VirtualCamSink(device=outputs.virtualcam_device))
    if outputs.preview:
        sinks.append(PreviewSink())
    return sinks


def _background_for(
    source: Source,
    config: PetrifyConfig,
    sample_check: PersonCheck | None = None,
) -> tuple[np.ndarray | BackgroundModel, int | None]:
    """Return the locked image and its index, or an online model and None.

    The online models get a person check on their samples unless
    ``config.background.sample_check`` is False: ``sample_check`` when given (built
    by :func:`_sample_check` at startup), else the one :func:`_sample_check` builds
    here. With the default checkers that is the HOG detector with the settings of
    the locked pre-scan. Every run with an online model logs
    :data:`ONLINE_BACKGROUND_WARNING`.
    """
    bg_cfg = config.background
    info = source.info
    if bg_cfg.strategy == "locked":
        if not info.seekable or not hasattr(source, "read_frame"):
            raise ValueError(
                "The locked background needs a seekable source (a video file or an image "
                "folder) to pre-scan; live sources need the running or calibrate background"
            )
        return locked_background(
            source,  # type: ignore[arg-type]
            bg_cfg,
            config.background_scan_person_config(),
        )
    logger.warning(ONLINE_BACKGROUND_WARNING)
    sample_filter: Callable[[np.ndarray], bool] | None = None
    watch: _RejectionWatch | None = None
    if bg_cfg.sample_check:
        if sample_check is None:
            sample_check = _sample_check(config, _ModelDetectors(config))
        watch = _RejectionWatch(person_free(sample_check), bg_cfg.sample_checkers)
        sample_filter = watch
    else:
        logger.warning(NO_BACKGROUND_CHECK_WARNING)
    if bg_cfg.strategy == "running":
        model: BackgroundModel = RunningBackground(
            bg_cfg.running_window,
            bg_cfg.running_interval,
            bg_cfg.running_min_samples,
            recompute_in_background=info.live,
            sample_filter=sample_filter,
        )
        if watch is not None:
            watch.model = model
        logger.info(
            "Background: running median (window=%d interval=%d min_samples=%d)",
            bg_cfg.running_window,
            bg_cfg.running_interval,
            bg_cfg.running_min_samples,
        )
        return model, None
    frames = frames_for(bg_cfg.calibrate_seconds, info.fps)
    logger.info(
        "Background: calibrate on the first %d frames (%.1f s); keep the scene empty",
        frames,
        bg_cfg.calibrate_seconds,
    )
    calibrated = CalibratedBackground(frames, sample_filter=sample_filter)
    if watch is not None:
        watch.model = calibrated
    return calibrated, None


class _RejectionWatch:
    """A sample filter that passes on ``sample_filter``'s answer and watches for a stuck check.

    When :data:`SAMPLE_REJECTION_STREAK` samples in a row are rejected (a filter
    error counts as a rejection) and ``model`` still has no background, it logs
    :data:`SAMPLE_CHECK_STUCK_WARNING` once per run. It changes no answer:
    rejected samples stay rejected and errors propagate.
    """

    def __init__(
        self, sample_filter: Callable[[np.ndarray], bool], checkers: Sequence[str]
    ) -> None:
        self._filter = sample_filter
        self._checkers = tuple(checkers)
        self._streak = 0
        self._warned = False
        self.model: BackgroundModel | None = None

    def __call__(self, image: np.ndarray) -> bool:
        try:
            accepted = bool(self._filter(image))
        except Exception:
            self._rejected()
            raise
        if accepted:
            self._streak = 0
        else:
            self._rejected()
        return accepted

    def _rejected(self) -> None:
        self._streak += 1
        if self._warned or self._streak < SAMPLE_REJECTION_STREAK:
            return
        if self.model is not None and self.model.current() is not None:
            return
        self._warned = True
        logger.warning(
            SAMPLE_CHECK_STUCK_WARNING.format(
                checkers=", ".join(self._checkers), count=self._streak
            )
        )


def _warn_large_manual_roi(box: tuple[int, int, int, int], cap: float, source: Source) -> None:
    """Warn when a manual ROI box covers more than ``cap`` of the source's frame."""
    frame_area = source.info.width * source.info.height
    if frame_area <= 0:
        return
    share = box[2] * box[3] / frame_area
    if share > cap:
        logger.warning(
            "The manual ROI %d %d %d %d covers %.1f percent of the frame, more than the "
            "%g percent cap for auto-detection: that region is shown live (not anonymized) "
            "on every output frame.",
            *box,
            share * 100,
            cap * 100,
        )


def _roi_stage(source: Source, config: PetrifyConfig) -> RoiOverlay | WarmupRoiOverlay:
    """Return the ROI stage: manual box, disabled, whole-clip scan, or warmup detection.

    A manual box over ``roi.max_area_cap`` of the frame is allowed (an explicit
    choice at startup) but logged as a warning: its region is shown live.
    """
    roi_cfg = config.roi
    if roi_cfg.mode == "manual" and roi_cfg.box is not None:
        _warn_large_manual_roi(roi_cfg.box, roi_cfg.max_area_cap, source)
    if roi_cfg.mode != "auto" or (isinstance(source, FileSource) and source.info.seekable):
        return RoiOverlay(_resolve_roi(source, config))  # type: ignore[arg-type]
    logger.info("ROI: auto-detect on the first %.1f s, then locked", roi_cfg.warmup_seconds)
    detector = WarmupRoiDetector(
        source.info.fps,
        roi_cfg.warmup_seconds,
        min_freq=roi_cfg.scan_min_freq,
        margin=roi_cfg.scan_margin,
        max_area_frac=roi_cfg.scan_max_area_frac,
        edge_threshold=roi_cfg.scan_edge_threshold,
        edge_gap_tolerance=roi_cfg.scan_edge_gap_tolerance,
        n_edge_samples=roi_cfg.scan_edge_samples,
        max_roi_area_frac=roi_cfg.max_area_cap,
    )
    return WarmupRoiOverlay(detector)


class _InterruptGuard:
    """Defer a Ctrl-C that arrives inside ``Engine.step`` until the step returns.

    An interrupt inside a step would abort the engine and discard the recording, so
    the first one received there only sets :attr:`pending`; the loop then stops and
    closes cleanly. An interrupt anywhere else (for example while waiting for the
    next frame), or a second one, raises ``KeyboardInterrupt`` as usual. With
    ``enabled=False`` (non-live runs), on a thread other than the main thread, or
    when a handler other than Python's default is installed, the guard does nothing.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.in_step = False
        self.pending = False
        self._previous: Any = None
        self._installed = False

    def __enter__(self) -> _InterruptGuard:
        if (
            self.enabled
            and threading.current_thread() is threading.main_thread()
            and signal.getsignal(signal.SIGINT) is signal.default_int_handler
        ):
            self._previous = signal.signal(signal.SIGINT, self._handle)
            self._installed = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._installed:
            signal.signal(signal.SIGINT, self._previous)
            self._installed = False

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        if self.in_step and not self.pending:
            self.pending = True
            logger.info("Interrupt received: finishing the current frame, then stopping")
            return
        raise KeyboardInterrupt


def _drive(
    engine: Engine,
    source: Source,
    should_stop: Callable[[], bool],
    before_step: Callable[[Frame], None] | None = None,
    after_step: Callable[[], None] | None = None,
) -> tuple[EngineStats, bool]:
    """Run the engine over the source with the step API; return the stats and ``interrupted``.

    ``before_step`` is called with each frame before :meth:`Engine.step` and
    ``after_step`` right after it (both on this thread; ``after_step`` counts as
    part of the step for the Ctrl-C guard, so it may reconfigure the engine).

    On a live source a ``KeyboardInterrupt`` ends the run: reading stops and the
    engine is closed normally, so the outputs are finalized. If the interrupt
    landed inside a step that already aborted the engine, or on a non-live source,
    it propagates. The interrupt guard is removed before the close, so a second
    Ctrl-C there aborts.

    Any other exception (not ``SystemExit``) that ends a live run while the engine
    is still open finalizes the outputs too: everything written so far is processed
    footage, so it is kept, an error is logged, and the exception propagates. If
    that close fails as well, the engine is aborted and the original exception
    propagates. An error inside a step has already aborted the engine (nothing is
    left to finalize); on a non-live source any error aborts the engine. When the
    source delivered no frame and closing fails, :class:`NoFramesError` is raised.
    """
    live = source.info.live
    interrupted = False
    failure: Exception | None = None
    delivered = 0
    try:
        engine.open(source.info)
        with _InterruptGuard(enabled=live) as guard:
            try:
                for frame in source:
                    delivered += 1
                    if before_step is not None:
                        before_step(frame)
                    guard.in_step = True
                    engine.step(frame)
                    if after_step is not None:
                        after_step()
                    guard.in_step = False
                    if guard.pending:
                        interrupted = True
                        break
                    if should_stop():
                        logger.info("Stop requested; finishing the run")
                        break
            except KeyboardInterrupt:
                if not live or not engine.is_open:
                    raise
                interrupted = True
            except Exception as exc:
                if not live or not engine.is_open:
                    raise
                failure = exc
        if failure is not None:
            _finalize_after_error(engine, failure)
        if interrupted:
            logger.info("Interrupted by user; finishing the outputs (Ctrl-C again to abort)")
        try:
            stats = engine.close()
        except Exception as exc:
            if delivered == 0:
                raise NoFramesError(
                    "The source delivered no frames; no output was written"
                ) from exc
            raise
        return stats, interrupted
    except BaseException:
        engine.abort()
        raise


def _finalize_after_error(engine: Engine, failure: Exception) -> None:
    """Close the engine after a live run failed, keeping the outputs; always raise ``failure``."""
    try:
        engine.close()
    except Exception:
        logger.exception("Finalizing the outputs after the error failed too; nothing was kept")
        engine.abort()
        raise failure from None
    logger.error(
        "The run ended early because of an error (%s: %s); the recording up to that "
        "point was kept",
        type(failure).__name__,
        failure,
    )
    raise failure


# ---------------------------------------------------------------------------
# Components, and rebuilding them for a reconfiguration at a frame boundary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Components:
    """What :func:`run_stream` built from a config, kept so a later change can reuse it.

    ``background`` (the locked image or the online model) is built once and never
    rebuilt: rebuilding it would discard the learned background. It is None when
    the mode does not use a background. ``roi_stage`` is None when the mode does
    not allow the ROI overlay. ``factory`` is the loaded mode factory, ``options``
    its resolved options and ``requires`` what the mode requires for them.
    ``person`` is the HOG or the DNN person detector, as the config selects.
    ``models`` holds the detectors built from model files, so that a later change
    reuses them instead of reading their files again.
    """

    config: PetrifyConfig
    motion: MotionDetector | None
    person: PersonDetector | PersonDnnDetector | None
    faces: FaceYuNetDetector | None
    segment: PersonSegmenter | None
    differ: BackgroundDiffer | None
    background: np.ndarray | BackgroundModel | None
    mode: Mode
    roi_stage: Stage | None
    factory: ModeFactory
    options: Mapping[str, OptionValue]
    requires: frozenset[DetectionKind]
    refiners: tuple[Refiner, ...]
    source: SourceInfo
    models: _ModelDetectors

    @property
    def detectors(self) -> list[Detector]:
        """The detector list in the engine's order: motion, person, faces, mask, differ."""
        found = (self.motion, self.person, self.faces, self.segment, self.differ)
        return [d for d in found if d is not None]

    @property
    def stages(self) -> list[Stage]:
        """The stage list: the ROI stage when the mode allows it, else nothing."""
        return [self.roi_stage] if self.roi_stage is not None else []


def _mode_options(factory: ModeFactory, config: PetrifyConfig) -> dict[str, OptionValue]:
    """Resolve ``config.mode_options`` against the factory's options, defaults filled in.

    Raises ``ValueError`` for an unknown option, an invalid value, or a value that
    resolving would change (for example ``5.0`` for an int option): every value in
    the config must already be in its declared type.
    """
    given = config.mode_options_dict()
    options = resolve_options(factory.options, given)
    for name, value in given.items():
        resolved = options[name]
        if type(resolved) is not type(value) or resolved != value:
            raise ValueError(
                f"mode option {name!r} is {value!r} in the config, which is not its "
                f"declared form {resolved!r}"
            )
    return options


# Kinds that only a detector the user turns on provides: the config section that
# turns it on, and how the error names that detector and its flag.
_OPT_IN_PROVIDERS: tuple[tuple[DetectionKind, str, str], ...] = (
    (DetectionKind.FACES, "faces", "the YuNet face detector with --faces"),
    (DetectionKind.LANDMARKS, "faces", "the YuNet face detector with --faces"),
    (DetectionKind.MASK, "segment", "the person segmenter with --segment"),
)

MISSING_PROVIDER_ERROR = (
    "The mode {mode!r} requires {kind} but no enabled detector provides it; enable {provider}"
)


def _mode_requires(
    factory: ModeFactory, options: Mapping[str, OptionValue], config: PetrifyConfig
) -> frozenset[DetectionKind]:
    """Return ``factory.requires(options)``, checked against the detectors ``config`` enables.

    A ``ValueError`` from the factory propagates (the settings are not acceptable);
    any other error, or a result that is not a frozenset of DetectionKind, raises
    :class:`ModeLoadError` (the plugin is faulty), as in :func:`build_mode`.

    Raises ``ValueError`` (:data:`MISSING_PROVIDER_ERROR`) when the mode requires
    faces, landmarks or a mask and the detector that provides it is not enabled:
    without it the mode would receive None in that field on every frame. Motion
    and person boxes keep the engine's rule (at least one required kind provided),
    which is what lets erase run with ``--detector motion``.

    An enabled provider is not enough on its own: under the ``first_hit``
    policy the engine skips it on frames where an earlier detector fired. A mode
    that needs these kinds together with another kind must therefore use
    ``detect_policy = "all"``, which :func:`build_mode` enforces.
    """
    try:
        kinds = factory.requires(options)
    except ValueError:
        raise
    except Exception as exc:
        raise ModeLoadError(
            f"mode {factory.name!r}: requires failed: {_describe_error(exc)}"
        ) from exc
    if not isinstance(kinds, frozenset) or not all(isinstance(k, DetectionKind) for k in kinds):
        raise ModeLoadError(
            f"mode {factory.name!r}: requires must return a frozenset of DetectionKind"
        )
    for kind, section, provider in _OPT_IN_PROVIDERS:
        if kind in kinds and getattr(config, section) is None:
            raise ValueError(
                MISSING_PROVIDER_ERROR.format(mode=factory.name, kind=kind.value, provider=provider)
            )
    return kinds


def _mode_context(
    source: SourceInfo,
    config: PetrifyConfig,
    factory: ModeFactory,
    options: Mapping[str, OptionValue],
    background: np.ndarray | BackgroundModel | None,
) -> ModeContext:
    """Return the context :func:`build_mode` gets: options filled in, background if needed."""
    filled = mode_options_from(options)
    ctx_config = (
        config if filled == config.mode_options else dataclasses.replace(config, mode_options=filled)
    )
    return ModeContext(
        source, ctx_config, options, background if factory.needs_background else None
    )


def _wants(detector: type, requires: frozenset[DetectionKind]) -> bool:
    """True when a built-in detector class provides a kind the mode requires."""
    return bool(detector.provides & requires)  # type: ignore[attr-defined]


DIFFER_WITHOUT_BACKGROUND_ERROR = (
    "The mode {mode!r} requires the background differ but does not use a background, "
    "and the differ needs one to compare against; turn the background differ off "
    "(--no-bg-diff) for this mode"
)


class _ModelDetectors:
    """The detectors of one run that are built from model files, one per class and settings.

    A model file is read and verified only when a detector with settings not seen
    before in the run is built, so a reconfiguration that keeps a detector, or
    brings back one used earlier, reads nothing. The background sample check and
    frame detection share an instance: both run on the engine's thread (the sample
    filter runs inside the mode's background update), and these detectors keep no
    state between frames.
    """

    def __init__(self, config: PetrifyConfig) -> None:
        self.models = config.models
        self.dirs = model_dirs(config.models.dir)
        self._built: dict[tuple[type, object], Any] = {}

    def get(self, cls: type[_ModelDetectorT], settings: object) -> _ModelDetectorT:
        """Return the run's ``cls`` detector for ``settings``, building it on first use.

        Raises :class:`~petrify.models.ModelError` when the model file is missing,
        unreadable, or fails its size or SHA-256 check.
        """
        key = (cls, settings)
        found = self._built.get(key)
        if found is None:
            spec = _MODEL_OF[cls]
            pop_loaded_path(spec)  # Drop a path left by an earlier load.
            found = cls.from_config(settings, self.dirs)  # type: ignore[attr-defined]
            self._built[key] = found
            path = pop_loaded_path(spec)
            logger.info(
                "Detector %s ready: model %s loaded and verified",
                cls.__name__,
                path if path is not None else spec.filename,
            )
        return found  # type: ignore[no-any-return]


# The model file each detector class loads, to report where it was found.
_MODEL_OF: dict[type, ModelSpec] = {
    FaceYuNetDetector: FACE_YUNET_MODEL,
    PersonDnnDetector: PERSON_DNN_MODEL,
    PersonSegmenter: SEGMENT_MODEL,
}


def _person_detector(
    config: PetrifyConfig, models: _ModelDetectors
) -> PersonDetector | PersonDnnDetector:
    """Build the person box detector the config selects: DNN with a ``person_dnn`` section."""
    if config.person_dnn is not None:
        return models.get(PersonDnnDetector, config.person_dnn)
    return PersonDetector.from_config(config.person)


def _faces_detector(
    config: PetrifyConfig, requires: frozenset[DetectionKind], models: _ModelDetectors
) -> FaceYuNetDetector | None:
    """The face detector when the config enables it and the mode requires faces or landmarks."""
    if config.faces is None or not _wants(FaceYuNetDetector, requires):
        return None
    return models.get(FaceYuNetDetector, config.faces)


def _segmenter(
    config: PetrifyConfig, requires: frozenset[DetectionKind], models: _ModelDetectors
) -> PersonSegmenter | None:
    """The person segmenter when the config enables it and the mode requires a mask."""
    if config.segment is None or not _wants(PersonSegmenter, requires):
        return None
    return models.get(PersonSegmenter, config.segment)


def _frame_detectors(
    config: PetrifyConfig, requires: frozenset[DetectionKind], models: _ModelDetectors
) -> tuple[
    MotionDetector | None,
    PersonDetector | PersonDnnDetector | None,
    FaceYuNetDetector | None,
    PersonSegmenter | None,
]:
    """Build the frame detectors the config enables and the mode requires.

    Motion and person boxes follow ``config.detector``; the person boxes come from
    the DNN detector instead of HOG when ``config.person_dnn`` is set. Faces and
    the mask follow their own sections. Raises
    :class:`~petrify.models.ModelError` when a model file cannot be used.
    """
    motion = None
    person = None
    if config.detector in ("motion", "both") and _wants(MotionDetector, requires):
        motion = MotionDetector.from_config(config.motion)
    if config.detector in ("person", "both") and _wants(PersonDetector, requires):
        person = _person_detector(config, models)
    return motion, person, _faces_detector(config, requires, models), _segmenter(
        config, requires, models
    )


def _sample_checker(name: str, config: PetrifyConfig, models: _ModelDetectors) -> PersonCheck:
    """Build the background sample check named ``name`` (one of ``SAMPLE_CHECKERS``).

    A model-based checker uses the settings of its detector section when the
    config enables that detector, and the section's defaults otherwise.
    """
    if name == "hog":
        return PersonDetector.from_config(config.background_scan_person_config())
    if name == "yunet":
        return detector_check(models.get(FaceYuNetDetector, config.faces or FaceYuNetConfig()))
    if name == "dnn":
        return detector_check(
            models.get(PersonDnnDetector, config.person_dnn or PersonDnnConfig())
        )
    # BackgroundConfig accepts only the names above; this guards a name added there
    # without a checker here, which must not silently vet nothing.
    raise ValueError(f"The background sample checker {name!r} is not available")


def _sample_check(config: PetrifyConfig, models: _ModelDetectors) -> PersonCheck:
    """Build the background sample check from ``config.background.sample_checkers``.

    A single checker is used as is, so the default (HOG alone) is the same check
    as before the other checkers existed. Several are combined with
    :func:`~petrify.background.vetting.any_person`: a sample is rejected as soon as
    any of them sees a person.
    """
    checks = [_sample_checker(name, config, models) for name in config.background.sample_checkers]
    return checks[0] if len(checks) == 1 else any_person(checks)


def _uses_sample_check(factory: ModeFactory, config: PetrifyConfig) -> bool:
    """True when the run builds an online background whose samples are checked."""
    bg_cfg = config.background
    return factory.needs_background and bg_cfg.strategy != "locked" and bg_cfg.sample_check


def _refiners(config: PetrifyConfig, requires: frozenset[DetectionKind]) -> tuple[Refiner, ...]:
    """Build the tracker refiners for the kinds the mode requires, or none without a tracker."""
    tracker = config.tracker
    if tracker is None:
        return ()
    refiners: list[Refiner] = []
    if DetectionKind.PERSONS in requires:
        refiners.append(TrackerRefiner.from_config(DetectionKind.PERSONS, tracker))
    if DetectionKind.FACES in requires:
        refiners.append(
            TrackerRefiner.from_config(
                DetectionKind.FACES, tracker, landmarks=DetectionKind.LANDMARKS in requires
            )
        )
    return tuple(refiners)


def _tracked_kinds(requires: frozenset[DetectionKind]) -> tuple[bool, bool, bool]:
    """What :func:`_refiners` builds for ``requires``: persons, faces, landmarks."""
    return (
        DetectionKind.PERSONS in requires,
        DetectionKind.FACES in requires,
        DetectionKind.LANDMARKS in requires,
    )


def _background_differ(
    background: np.ndarray | BackgroundModel, config: PetrifyConfig, live: bool
) -> BackgroundDiffer | None:
    """Build the background differ against the locked image or the online model, or None.

    Logs the differ's thresholds, or :data:`NO_BG_DIFF_LIVE_WARNING` when it is
    off on a live source.
    """
    if config.background_diff is None:
        if live:
            logger.warning(NO_BG_DIFF_LIVE_WARNING)
        return None
    if isinstance(background, np.ndarray):
        bg_differ = BackgroundDiffer.from_config(background, config.background_diff)
    else:
        bg_differ = BackgroundDiffer.from_model(background, config.background_diff)
    logger.info(
        "BackgroundDiffer enabled: pixel_threshold=%d area_threshold=%d",
        bg_differ.pixel_threshold,
        bg_differ.area_threshold,
    )
    return bg_differ


def _requires_differ(
    background: np.ndarray | BackgroundModel, differ: BackgroundDiffer | None
) -> bool:
    """``require_background_diff`` for the erase mode: True exactly with a model-backed differ."""
    return differ is not None and not isinstance(background, np.ndarray)


def _warn_unused(
    config: PetrifyConfig, factory: ModeFactory, requires: frozenset[DetectionKind]
) -> None:
    """Log a warning for each detector setting that has no effect in this run."""
    if config.person_dnn is not None and config.detector == "motion":
        logger.warning(DNN_NOT_USED_WARNING)
    if config.faces is not None and not _wants(FaceYuNetDetector, requires):
        logger.warning(FACES_NOT_USED_WARNING.format(mode=config.mode))
    if config.segment is not None and not _wants(PersonSegmenter, requires):
        logger.warning(SEGMENT_NOT_USED_WARNING.format(mode=config.mode))
    default_checkers = BackgroundConfig().sample_checkers
    if config.background.sample_checkers != default_checkers and not factory.needs_background:
        logger.warning(BACKGROUND_CHECK_NOT_USED_WARNING.format(mode=config.mode))


def _build_components(source: Source, config: PetrifyConfig) -> tuple[_Components, int | None]:
    """Build the mode factory, detectors, ROI stage, background, mode and refiners at startup.

    The factory for ``config.mode`` is loaded from the registry first; a
    :class:`ModeLoadError` propagates unchanged. Then, in the same order as
    :func:`run_file`: detectors, ROI, background, the background differ, and the
    mode built with :func:`build_mode`. A built-in detector is built only when
    the config enables it and it provides a kind the mode requires. The
    background is built only when the factory needs one, and the ROI stage only
    when the factory allows the overlay. Raises ``ValueError`` when the settings
    cannot work with the mode. Returns the components and the locked
    background's index.

    Every model file the run needs (frame detectors and background sample
    checkers) is loaded and verified before the ROI scan and the background are
    built, which are the first steps that read frames, so a missing or corrupt
    model fails the run with :class:`~petrify.models.ModelError` within seconds.
    """
    factory = load_mode(config.mode)
    options = _mode_options(factory, config)
    requires = _mode_requires(factory, options, config)
    wants_differ = DetectionKind.BACKGROUND_DIFF in requires
    if wants_differ and not factory.needs_background and config.background_diff is not None:
        raise ValueError(DIFFER_WITHOUT_BACKGROUND_ERROR.format(mode=config.mode))
    models = _ModelDetectors(config)
    motion, person, faces, segment = _frame_detectors(config, requires, models)
    sample_check = _sample_check(config, models) if _uses_sample_check(factory, config) else None
    logger.info("Detector mode: %s", config.detector)
    if person is not None and config.person_dnn is not None:
        logger.info("Person detector: DNN (NanoDet-Plus) in place of HOG")
    _warn_unused(config, factory, requires)
    roi_stage: Stage | None = None
    if factory.roi_overlay:
        roi_stage = _roi_stage(source, config)
    else:
        logger.info("ROI: not used, the mode %r does not allow the overlay", config.mode)
    background: np.ndarray | BackgroundModel | None = None
    bg_idx: int | None = None
    differ: BackgroundDiffer | None = None
    if factory.needs_background:
        background, bg_idx = _background_for(source, config, sample_check)
        if wants_differ:
            differ = _background_differ(background, config, source.info.live)
    if person is not None and config.person_dnn is not None and differ is None:
        logger.warning(DNN_WITHOUT_DIFFER_WARNING)
    ctx = _mode_context(source.info, config, factory, options, background)
    mode = build_mode(factory, ctx)
    components = _Components(
        config=config,
        motion=motion,
        person=person,
        faces=faces,
        segment=segment,
        differ=differ,
        background=background,
        mode=mode,
        roi_stage=roi_stage,
        factory=factory,
        options=options,
        requires=requires,
        refiners=_refiners(config, requires),
        source=source.info,
        models=models,
    )
    return components, bg_idx


def _rebuild(old: _Components, new: PetrifyConfig) -> tuple[_Components, dict[str, Any]]:
    """Return the components for ``new`` and the :meth:`Engine.reconfigure` arguments.

    Only what changed is rebuilt; an unchanged detector is reused (the engine
    still resets every detector when the detector list changes). The background
    model or image is always kept, and a model-backed differ is rebuilt from that
    same model. The mode is rebuilt with :func:`build_mode` and the current mode
    as ``previous``, so the factory keeps the same object when nothing it depends
    on changed (for erase: the same background and ``require_background_diff``).
    Whenever the detectors change, the refiners are handed over too, so the
    engine resets them. At runtime the ROI is either a manual box or disabled, and
    stays absent for a mode without the overlay. The mode itself cannot change.

    Raises ``ValueError`` for an invalid combination, including a faulty plugin
    (its :class:`ModeLoadError` becomes a ``ValueError``) and a detector whose
    model file cannot be used (its :class:`~petrify.models.ModelError`); nothing
    is installed then.
    """
    try:
        return _rebuild_components(old, new)
    except (ModeLoadError, ModelError) as exc:
        raise ValueError(str(exc)) from exc


def _rebuild_components(
    old: _Components, new: PetrifyConfig
) -> tuple[_Components, dict[str, Any]]:
    cfg = old.config
    if new.mode != cfg.mode:
        raise ValueError(
            f"The mode cannot be changed during a run (from {cfg.mode!r} to {new.mode!r}); "
            "start a new run with the other mode"
        )
    factory = old.factory
    options = _mode_options(factory, new)
    requires = _mode_requires(factory, options, new)
    models = old.models if new.models == old.models.models else _ModelDetectors(new)
    motion: MotionDetector | None = None
    if new.detector in ("motion", "both") and _wants(MotionDetector, requires):
        motion = old.motion
        if motion is None or new.motion != cfg.motion:
            motion = MotionDetector.from_config(new.motion)
    person: PersonDetector | PersonDnnDetector | None = None
    if new.detector in ("person", "both") and _wants(PersonDetector, requires):
        if new.person_dnn is not None:
            person = models.get(PersonDnnDetector, new.person_dnn)
        else:
            person = old.person
            if person is None or isinstance(person, PersonDnnDetector) or new.person != cfg.person:
                person = PersonDetector.from_config(new.person)
    faces = _faces_detector(new, requires, models)
    segment = _segmenter(new, requires, models)
    differ: BackgroundDiffer | None = None
    if new.background_diff is not None and DetectionKind.BACKGROUND_DIFF in requires:
        differ = old.differ
        if differ is None or new.background_diff != cfg.background_diff:
            if old.background is None:
                raise ValueError(DIFFER_WITHOUT_BACKGROUND_ERROR.format(mode=new.mode))
            if isinstance(old.background, np.ndarray):
                differ = BackgroundDiffer.from_config(old.background, new.background_diff)
            else:
                differ = BackgroundDiffer.from_model(old.background, new.background_diff)
    ctx = _mode_context(old.source, new, factory, options, old.background)
    mode = build_mode(factory, ctx, old.mode)
    roi_stage = old.roi_stage
    if roi_stage is not None and new.roi != cfg.roi:
        roi_stage = RoiOverlay(new.roi.box if new.roi.mode == "manual" else None)
    refiners = old.refiners
    if new.tracker != cfg.tracker or _tracked_kinds(requires) != _tracked_kinds(old.requires):
        refiners = _refiners(new, requires)

    built = _Components(
        config=new,
        motion=motion,
        person=person,
        faces=faces,
        segment=segment,
        differ=differ,
        background=old.background,
        mode=mode,
        roi_stage=roi_stage,
        factory=factory,
        options=options,
        requires=requires,
        refiners=refiners,
        source=old.source,
        models=models,
    )
    kwargs: dict[str, Any] = {}
    detectors_changed = [id(d) for d in built.detectors] != [id(d) for d in old.detectors]
    if detectors_changed:
        kwargs["detectors"] = built.detectors
    refiners_changed = [id(r) for r in refiners] != [id(r) for r in old.refiners]
    if refiners_changed or (detectors_changed and refiners):
        kwargs["refiners"] = list(refiners)
    if mode is not old.mode:
        kwargs["mode"] = mode
    if roi_stage is not old.roi_stage:
        kwargs["stages"] = built.stages
    if new.engine != cfg.engine:
        kwargs["config"] = new.engine
    return built, kwargs


def _apply_pending(
    engine: Engine, components: _Components, control: RunControl, live: bool
) -> _Components:
    """Apply the latest config the control accepted, if any; return the components in use.

    A ``ValueError`` from rebuilding or from :meth:`Engine.reconfigure` (an
    invalid combination) keeps the old configuration running: the message goes
    to the control and a warning is logged. Any other error propagates.
    """
    new = control.take_pending()
    if new is None:
        return components
    try:
        built, kwargs = _rebuild(components, new)
        if kwargs:
            engine.reconfigure(**kwargs)
    except ValueError as exc:
        if not engine.is_open:
            raise  # Writing the tail failed and aborted the engine: not a bad setting.
        from petrify.server.control import PROBLEM_TEXT_LIMIT, untrusted_text  # noqa: PLC0415

        # The message may come from plugin code: plain text, capped, before it is logged.
        message = untrusted_text(exc, PROBLEM_TEXT_LIMIT)
        logger.warning("The new settings were not applied, the previous ones stay: %s", message)
        control.report_rejected(message)
        return components
    if live and new.background_diff is None and components.config.background_diff is not None:
        logger.warning(NO_BG_DIFF_LIVE_WARNING)
    if kwargs:
        logger.info("Settings changed at a frame boundary: %s", ", ".join(sorted(kwargs)))
    control.report_applied(new)
    return built


# ---------------------------------------------------------------------------
# Run state for the web page
# ---------------------------------------------------------------------------


def _engine_stats(engine: Engine) -> EngineStats:
    """Return a copy of the engine's running counters.

    ``Engine`` returns its stats only from :meth:`Engine.close`; the page needs
    them after every frame, so this reads the private counters (read only).
    """
    stats = getattr(engine, "_stats", None)
    return dataclasses.replace(stats) if isinstance(stats, EngineStats) else EngineStats()


class _Publisher:
    """Builds :class:`~petrify.server.control.RunSnapshot` values and publishes them."""

    def __init__(self, control: RunControl, source: Source) -> None:
        from petrify.sources.file import display_spec  # noqa: PLC0415

        self._control = control
        self._source = source
        uri = source.info.uri
        self._uri = display_spec(uri) if uri else None
        self._config: PetrifyConfig | None = None
        self._config_dict: dict[str, Any] = {}
        self._components: _Components | None = None
        self._engine: Engine | None = None
        self._stats = EngineStats()
        self._ended = False

    def attach(self, components: _Components, engine: Engine | None = None) -> None:
        """Use ``components`` (and ``engine``, when given) for the next snapshots."""
        self._components = components
        if engine is not None:
            self._engine = engine

    def publish(self, *, running: bool = True, stats: EngineStats | None = None) -> None:
        """Publish the current state; ``running=False`` marks the run as ended (once)."""
        from petrify.server.control import (  # noqa: PLC0415
            BackgroundState,
            RunSnapshot,
            RunStats,
            RunTimings,
            SourceFacts,
            config_to_dict,
        )

        if self._ended:
            return
        components = self._components
        config = components.config if components is not None else self._control.applied
        if config is not self._config:
            self._config = config
            self._config_dict = config_to_dict(config)
        engine = self._engine
        if stats is not None:
            self._stats = stats
        elif engine is not None:
            self._stats = _engine_stats(engine)
        timings = RunTimings()
        if engine is not None:
            t = engine.timings
            timings = RunTimings(
                detect_mean_ms=t.detect.mean_ms,
                detect_last_ms=t.detect.last_ms,
                render_mean_ms=t.render.mean_ms,
                render_last_ms=t.render.last_ms,
                write_mean_ms=t.write.mean_ms,
                write_last_ms=t.write.last_ms,
                fps=t.frames_per_second,
            )
        background = BackgroundState(strategy=config.background.strategy)
        roi = None
        if components is not None:
            # A plugin mode need not have a ``model``; fall back to the built background.
            model = getattr(components.mode, "model", None)
            if model is None:
                model = components.background
            if isinstance(model, np.ndarray):
                background = BackgroundState(strategy=config.background.strategy, ready=True)
            elif model is not None:
                try:
                    ready = model.current() is not None
                except Exception:  # noqa: BLE001 (a failing model is simply not ready)
                    ready = False
                try:
                    rejected = int(getattr(model, "rejected", 0) or 0)
                except Exception:  # noqa: BLE001 (a plugin attribute that is not a count)
                    rejected = 0
                background = BackgroundState(
                    strategy=config.background.strategy,
                    ready=ready,
                    rejected=rejected,
                )
            else:
                # The mode uses no background: neither ready nor still learning.
                background = BackgroundState(strategy=config.background.strategy, in_use=False)
            roi = getattr(components.roi_stage, "roi", None)
        info = self._source.info
        self._control.publish(
            RunSnapshot(
                config=self._config_dict,
                stats=RunStats(**dataclasses.asdict(self._stats)),
                timings=timings,
                source=SourceFacts(
                    uri=self._uri,
                    width=info.width,
                    height=info.height,
                    fps=float(info.fps),
                    live=info.live,
                    dropped=int(getattr(self._source, "dropped", 0) or 0),
                ),
                background=background,
                roi=roi,
                running=running,
                ended=not running,
            )
        )
        if not running:
            self._ended = True


# ---------------------------------------------------------------------------
# Raw preview tap (opt-in) and realtime pacing
# ---------------------------------------------------------------------------

RAW_PREVIEW_WARNING = (
    "Raw preview enabled (--web-raw-preview): the UNPROCESSED feed is exposed to the "
    "local web page. Anyone with the page link sees the original frames."
)


def _expose_unprocessed_frame(raw_preview: MjpegSink, frame: Frame) -> None:
    """Hand an UNPROCESSED input frame to the opt-in raw preview sink.

    This is the only place in petrify where an unprocessed frame is given to an
    output. It is reached only when the caller of :func:`run_stream` passed a
    ``raw_preview`` sink explicitly, and that sink is never in the engine's sink
    list.
    """
    raw_preview.write(frame.image, frame.meta)


class _RawTap:
    """Opens, feeds and closes the raw preview sink; a failure only turns the tap off."""

    def __init__(self, sink: MjpegSink) -> None:
        self._sink = sink
        self._open = False

    def open(self, info: Any) -> None:
        try:
            self._sink.open(info)
            self._open = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("The raw preview could not start and is off: %r", exc)

    def __call__(self, frame: Frame) -> None:
        if not self._open:
            return
        try:
            _expose_unprocessed_frame(self._sink, frame)
        except Exception as exc:  # noqa: BLE001
            logger.warning("The raw preview failed and is off for the rest of the run: %r", exc)
            self.close()

    def close(self) -> None:
        self._open = False
        try:
            self._sink.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("The raw preview failed to close: %r", exc)


class _Pacer:
    """Wait until ``start + frame.timestamp`` so a file is processed at its own speed.

    The first frame sets the start. Each wait is cut into short sleeps so that a
    stop request ends it at once; a Ctrl-C raises from the sleep as usual.
    """

    _SLICE = 0.05

    def __init__(
        self,
        clock: Callable[[], float],
        sleep: Callable[[float], None],
        should_stop: Callable[[], bool],
    ) -> None:
        self._clock = clock
        self._sleep = sleep
        self._should_stop = should_stop
        self._origin: float | None = None

    def __call__(self, frame: Frame) -> None:
        if self._origin is None:
            self._origin = self._clock() - frame.timestamp
            return
        target = self._origin + frame.timestamp
        while not self._should_stop():
            remaining = target - self._clock()
            if remaining <= 0:
                return
            self._sleep(min(remaining, self._SLICE))


def run_stream(
    source: Source,
    config: PetrifyConfig,
    outputs: OutputConfig,
    *,
    observer: Callable[[FrameTrace], None] | None = None,
    sinks: Sequence[Sink] | None = None,
    stop: Callable[[], bool] | None = None,
    web_sink: MjpegSink | None = None,
    raw_preview: MjpegSink | None = None,
    control: RunControl | None = None,
    realtime: bool = False,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> RunReport:
    """Run the streaming engine on any source and write to the selected outputs.

    ``sinks`` replaces the sinks built from ``outputs``. Components are built in
    the same order as :func:`run_file`: detectors, ROI, background, then the
    background differ. On a file source with the locked background and whole-clip
    ROI detection the written frames are exactly the ones :func:`run_file` writes.

    The run stops at the end of the source, when ``stop()`` returns True,
    ``control.stop_requested`` is set, or a preview window asks to stop (all
    checked after each frame), or on Ctrl-C for a live source; the lookahead tail
    is still written and the outputs are finalized. A ``KeyboardInterrupt`` on a
    non-live source aborts the outputs and propagates. Any other error aborts the
    outputs and propagates, except on a live source where the outputs written so
    far are finalized first (see :func:`_drive`). The source is always closed.

    Web page support (all optional):

    - ``web_sink`` is the processed feed for the local web page. It is added to
      the sinks and counts as an output.
    - ``control`` links the run to the page: after each frame the latest config
      it accepted is applied at that frame boundary (see :func:`_apply_pending`),
      and a snapshot is published after every frame, plus a final one with
      ``running=False`` on every exit path.
    - ``raw_preview`` receives every INPUT frame before it is processed (see
      :func:`_expose_unprocessed_frame`). Pass it only when the user asked for it.
    - ``realtime=True`` on a non-live source processes frames no faster than
      their timestamps (``clock`` and ``sleep`` do the pacing). It changes no
      output frame. It is ignored on a live source.

    Raises ``ValueError`` when ``outputs.file`` is the input file, or one of the
    images of an image folder source, and :class:`FfmpegNotFoundError` before any
    processing when the file output is built here and FFmpeg is missing.
    """
    publisher = _Publisher(control, source) if control is not None else None
    raw_tap = _RawTap(raw_preview) if raw_preview is not None else None
    final_stats: EngineStats | None = None
    try:
        if outputs.file is not None and output_overwrites_input(source, outputs.file):
            raise ValueError("The output file is the input; choose another output path")
        if sinks is None and outputs.file is not None:
            try:
                file_sinks.find_ffmpeg()
            except FileNotFoundError as exc:
                raise FfmpegNotFoundError(str(exc)) from exc
        if sinks is not None:
            sink_list = list(sinks)
        else:
            if not outputs.selected and web_sink is None:
                raise ValueError(
                    "At least one output is required: a file, the virtual camera, the "
                    "preview, or the web page"
                )
            sink_list = _build_sinks(source, config, outputs)
        if web_sink is not None:
            sink_list.append(web_sink)
        if not sink_list:
            raise ValueError("At least one sink is required")
        previews = [s for s in sink_list if isinstance(s, PreviewSink)]
        if raw_tap is not None:
            logger.warning(RAW_PREVIEW_WARNING)

        components, bg_idx = _build_components(source, config)
        live = source.info.live

        def should_stop() -> bool:
            if stop is not None and stop():
                return True
            if control is not None and control.stop_requested:
                return True
            return any(p.stop_requested for p in previews)

        hooks: list[Callable[[Frame], None]] = []
        if realtime and not live:
            hooks.append(_Pacer(clock, sleep, should_stop))
        elif realtime:
            logger.info("Realtime pacing is ignored on a live source")
        if raw_tap is not None:
            hooks.append(raw_tap)

        def before_step(frame: Frame) -> None:
            for hook in hooks:
                hook(frame)

        with tqdm(
            total=source.info.frame_count or None,
            desc="Petrify",
            unit="fr",
            dynamic_ncols=True,
        ) as progress:

            def on_frame(trace: FrameTrace) -> None:
                progress.update(1)
                if observer is not None:
                    observer(trace)

            try:
                engine = Engine(
                    components.detectors,
                    components.mode,
                    components.stages,
                    sink_list,
                    config.engine,
                    on_frame,
                    refiners=components.refiners,
                )
            except ValueError as exc:
                raise ValueError(
                    f"The settings cannot run with the mode {config.mode!r}: {exc}"
                ) from exc
            after_step: Callable[[], None] | None = None
            if control is not None and publisher is not None:
                publisher.attach(components, engine)
                publisher.publish(running=True)
                ctl, pub = control, publisher

                def after_step() -> None:
                    nonlocal components
                    components = _apply_pending(engine, components, ctl, live)
                    pub.attach(components)
                    pub.publish(running=True)

            if raw_tap is not None:
                raw_tap.open(source.info)
            stats, interrupted = _drive(
                engine,
                source,
                should_stop,
                before_step if hooks else None,
                after_step,
            )
            final_stats = stats
        dropped = int(getattr(source, "dropped", 0) or 0)
    finally:
        try:
            source.close()
        finally:
            if raw_tap is not None:
                raw_tap.close()
            if publisher is not None:
                publisher.publish(running=False, stats=final_stats)

    _log_summary(stats, dropped)
    return RunReport(
        stats=stats,
        background_index=bg_idx,
        roi=getattr(components.roi_stage, "roi", None),
        dropped_frames=dropped,
        interrupted=interrupted,
    )


def _log_summary(stats: EngineStats, dropped: int) -> None:
    """Log one summary line, and a warning when frames failed closed."""
    logger.info(
        "Run done: frames=%d frozen=%d pass-through=%d failed-closed=%d (over budget %d) "
        "source-dropped=%d sink-failures=%d",
        stats.frames,
        stats.frozen,
        stats.passed_through,
        stats.failed_closed,
        stats.over_budget,
        dropped,
        stats.sink_failures,
    )
    if stats.failed_closed > 0:
        logger.warning(
            "%d frame(s) failed closed and were written as black frames (%d over budget)",
            stats.failed_closed,
            stats.over_budget,
        )
