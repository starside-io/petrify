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
import logging
import os
import signal
import tempfile
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FrameType, TracebackType
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from petrify.background.calibrate import CalibratedBackground, frames_for
from petrify.background.locked import locked_background
from petrify.background.running import RunningBackground
from petrify.core.config import PetrifyConfig
from petrify.core.engine import Engine, EngineStats, FrameTrace
from petrify.core.plugin import BackgroundModel, Detector, Sink, Source, Stage
from petrify.detectors.bgdiff import BackgroundDiffer
from petrify.detectors.motion import MotionDetector
from petrify.detectors.person_hog import PersonDetector
from petrify.encoder import ffmpeg_encode
from petrify.modes.erase import EraseMode
from petrify.roi_detect import WarmupRoiDetector, auto_detect_roi
from petrify.sinks.file import FfmpegPipeSink, FileSink
from petrify.sinks.preview import PreviewSink
from petrify.sinks.threaded import ThreadedSink
from petrify.sinks.virtualcam import VirtualCamSink
from petrify.sources.file import FileSource
from petrify.stages.roi_overlay import Compositor, RoiOverlay, WarmupRoiOverlay

logger = logging.getLogger(__name__)


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

    On a live source the file and virtual camera sinks run on their own thread
    (``ThreadedSink``, which blocks rather than drops for these critical sinks) so a
    slow encoder does not stall capture. The preview always stays on the calling
    thread: GUI calls must run there.
    """
    info = source.info
    live = info.live
    sinks: list[Sink] = []
    if outputs.file is not None:
        audio = Path(info.uri) if isinstance(source, FileSource) and info.uri else None
        pipe = FfmpegPipeSink(
            outputs.file,
            config.encode,
            audio_source=audio,
            wallclock_timestamps=live,
        )
        sinks.append(ThreadedSink(pipe) if live else pipe)
    if outputs.virtualcam:
        cam = VirtualCamSink(device=outputs.virtualcam_device)
        sinks.append(ThreadedSink(cam) if live else cam)
    if outputs.preview:
        sinks.append(PreviewSink())
    return sinks


def _background_for(
    source: Source, config: PetrifyConfig
) -> tuple[np.ndarray | BackgroundModel, int | None]:
    """Return the locked image and its index, or an online model and None."""
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
    if bg_cfg.strategy == "running":
        model: BackgroundModel = RunningBackground(
            bg_cfg.running_window,
            bg_cfg.running_interval,
            bg_cfg.running_min_samples,
            recompute_in_background=info.live,
        )
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
    return CalibratedBackground(frames), None


def _roi_stage(source: Source, config: PetrifyConfig) -> RoiOverlay | WarmupRoiOverlay:
    """Return the ROI stage: manual box, disabled, whole-clip scan, or warmup detection."""
    roi_cfg = config.roi
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
) -> tuple[EngineStats, bool]:
    """Run the engine over the source with the step API; return the stats and ``interrupted``.

    On a live source a ``KeyboardInterrupt`` ends the run: reading stops and the
    engine is closed normally, so the outputs are finalized. If the interrupt
    landed inside a step that already aborted the engine, or on a non-live source,
    it propagates. Any other exception aborts the engine and propagates. The
    interrupt guard is removed before the close, so a second Ctrl-C there aborts.
    """
    live = source.info.live
    interrupted = False
    try:
        engine.open(source.info)
        with _InterruptGuard(enabled=live) as guard:
            try:
                for frame in source:
                    guard.in_step = True
                    engine.step(frame)
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
        if interrupted:
            logger.info("Interrupted by user; finishing the outputs (Ctrl-C again to abort)")
        return engine.close(), interrupted
    except BaseException:
        engine.abort()
        raise


def run_stream(
    source: Source,
    config: PetrifyConfig,
    outputs: OutputConfig,
    *,
    observer: Callable[[FrameTrace], None] | None = None,
    sinks: Sequence[Sink] | None = None,
    stop: Callable[[], bool] | None = None,
) -> RunReport:
    """Run the streaming engine on any source and write to the selected outputs.

    ``sinks`` replaces the sinks built from ``outputs``. Components are built in
    the same order as :func:`run_file`: detectors, ROI, background, then the
    background differ. On a file source with the locked background and whole-clip
    ROI detection the written frames are exactly the ones :func:`run_file` writes.

    The run stops at the end of the source, when ``stop()`` returns True or a
    preview window asks to stop (both checked after each frame), or on Ctrl-C for a
    live source; the lookahead tail is still written and the outputs are
    finalized. A ``KeyboardInterrupt`` on a non-live source, or any other error,
    aborts the outputs and propagates. The source is always closed.
    """
    try:
        if sinks is not None:
            sink_list = list(sinks)
        else:
            if not outputs.selected:
                raise ValueError(
                    "At least one output is required: a file, the virtual camera, or the preview"
                )
            sink_list = _build_sinks(source, config, outputs)
        if not sink_list:
            raise ValueError("At least one sink is required")
        previews = [s for s in sink_list if isinstance(s, PreviewSink)]

        detectors: list[Detector] = []
        if config.detector in ("motion", "both"):
            detectors.append(MotionDetector.from_config(config.motion))
        if config.detector in ("person", "both"):
            detectors.append(PersonDetector.from_config(config.person))
        logger.info("Detector mode: %s", config.detector)

        roi_stage = _roi_stage(source, config)
        background, bg_idx = _background_for(source, config)
        model_backed = False
        if config.background_diff is not None:
            if isinstance(background, np.ndarray):
                bg_differ = BackgroundDiffer.from_config(background, config.background_diff)
            else:
                bg_differ = BackgroundDiffer.from_model(background, config.background_diff)
                model_backed = True
            detectors.append(bg_differ)
            logger.info(
                "BackgroundDiffer enabled: pixel_threshold=%d area_threshold=%d",
                bg_differ.pixel_threshold,
                bg_differ.area_threshold,
            )
        mode = EraseMode(background, require_background_diff=model_backed)

        def should_stop() -> bool:
            if stop is not None and stop():
                return True
            return any(p.stop_requested for p in previews)

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

            stages: list[Stage] = [roi_stage]
            engine = Engine(detectors, mode, stages, sink_list, config.engine, on_frame)
            stats, interrupted = _drive(engine, source, should_stop)
        dropped = int(getattr(source, "dropped", 0) or 0)
    finally:
        source.close()

    _log_summary(stats, dropped)
    return RunReport(
        stats=stats,
        background_index=bg_idx,
        roi=roi_stage.roi,
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
