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

"""Typed, frozen configuration for the streaming engine.

Configs are immutable plain values, which is what makes a later swap at a frame
boundary safe.
"""

from __future__ import annotations

import argparse
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

# A plugin mode option value. Tuples, not lists, keep the config hashable.
OptionValue = bool | int | float | str | tuple[str, ...]

# Mode names and mode option names: lowercase, starting with a letter, at most 32 characters.
_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,31}")

# Detectors that may vet a background sample, in BackgroundConfig.sample_checkers.
SAMPLE_CHECKERS: tuple[str, ...] = ("hog", "yunet", "dnn", "segment")


def _check_model_path(owner: str, value: str | None) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{owner} must be a non-empty string or None, got {value!r}")


def _check_option_value(name: str, value: object) -> None:
    if isinstance(value, tuple):
        if not all(isinstance(item, str) for item in value):
            raise ValueError(
                f"mode option {name!r}: a tuple value must hold only strings, got {value!r}"
            )
        return
    if not isinstance(value, (bool, int, float, str)):
        raise ValueError(
            f"mode option {name!r} must be a bool, int, float, str or tuple of str, "
            f"got {type(value).__name__}"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"mode option {name!r} must be a finite number, got {value!r}")


def _check_mode_options(options: object) -> None:
    if not isinstance(options, tuple):
        raise ValueError(
            f"PetrifyConfig.mode_options must be a tuple of (name, value) pairs, "
            f"got {type(options).__name__}"
        )
    previous: str | None = None
    for entry in options:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise ValueError(
                f"PetrifyConfig.mode_options entries must be (name, value) pairs, got {entry!r}"
            )
        name, value = entry
        if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name):
            raise ValueError(
                f"mode option name must match [a-z][a-z0-9_]{{0,31}}, got {name!r}"
            )
        if previous is not None and name <= previous:
            raise ValueError(
                "PetrifyConfig.mode_options must be sorted by name with unique names, "
                f"got {name!r} after {previous!r}"
            )
        _check_option_value(name, value)
        previous = name


def mode_options_from(options: Mapping[str, OptionValue]) -> tuple[tuple[str, OptionValue], ...]:
    """Return ``options`` as the sorted, validated pairs ``PetrifyConfig.mode_options`` holds."""
    for name in options:
        if not isinstance(name, str):
            raise ValueError(f"mode option names must be strings, got {name!r}")
    pairs = tuple(sorted(options.items(), key=lambda pair: pair[0]))
    _check_mode_options(pairs)
    return pairs


@dataclass(frozen=True)
class MotionConfig:
    """MOG2 motion detector settings."""

    history: int = 200
    var_threshold: float = 16.0
    detect_shadows: bool = False
    threshold: int = 500
    downsample: float = 0.5


@dataclass(frozen=True)
class PersonHogConfig:
    """HOG person detector settings."""

    downsample: float = 0.5
    win_stride: tuple[int, int] = (8, 8)
    scale: float = 1.05
    min_confidence: float = 0.3
    min_box_area: int = 1500


@dataclass(frozen=True)
class BackgroundDiffConfig:
    """Static background-difference detector settings."""

    pixel_threshold: int = 25
    area_threshold: int = 200
    downsample: float = 0.5


@dataclass(frozen=True)
class RoiConfig:
    """Live timestamp ROI: a manual box, auto-detection, or disabled."""

    mode: Literal["manual", "auto", "disabled"] = "auto"
    box: tuple[int, int, int, int] | None = None
    scan_samples: int = 60
    scan_min_freq: float = 0.6
    scan_margin: int = 8
    scan_max_area_frac: float = 0.10
    scan_edge_threshold: float = 80.0
    scan_edge_gap_tolerance: int = 30
    scan_edge_samples: int = 3
    # Auto mode on a source that cannot be scanned in advance: seconds of warmup.
    warmup_seconds: float = 10.0
    # Warmup detection only: a final box larger than this fraction of the frame is
    # discarded as too large to be a timestamp.
    max_area_cap: float = 0.15

    def __post_init__(self) -> None:
        if self.mode not in ("manual", "auto", "disabled"):
            raise ValueError(f"RoiConfig.mode must be manual, auto or disabled, got {self.mode!r}")
        if self.mode == "manual" and self.box is None:
            raise ValueError("RoiConfig.mode 'manual' requires a box")
        if self.mode != "manual" and self.box is not None:
            raise ValueError(f"RoiConfig.mode {self.mode!r} requires box to be None")
        if not self.warmup_seconds > 0:
            raise ValueError(
                f"RoiConfig.warmup_seconds must be > 0, got {self.warmup_seconds}"
            )
        if not 0 < self.max_area_cap <= 1:
            raise ValueError(
                f"RoiConfig.max_area_cap must be in (0, 1], got {self.max_area_cap}"
            )


@dataclass(frozen=True)
class BackgroundConfig:
    """Background reference strategy and its settings.

    ``locked`` pre-scans a file source for one clean frame. ``running`` keeps a
    rolling median of recent frames classified as clean. ``calibrate`` builds the
    background from the first ``calibrate_seconds`` while the user is out of frame.
    ``sample_check`` (running and calibrate only) rejects background samples in
    which the HOG person detector finds a person; turning it off is unsafe.
    ``sample_checkers`` names the detectors that vet a sample, each one of
    :data:`SAMPLE_CHECKERS`; the default is the HOG detector alone, as today.
    """

    strategy: Literal["locked", "running", "calibrate"] = "locked"
    frame_index: int | None = None  # None means pre-scan for the latest clean frame.
    scan_step: int = 5
    running_window: int = 30
    running_interval: int = 5
    running_min_samples: int = 5
    calibrate_seconds: float = 3.0
    sample_check: bool = True
    sample_checkers: tuple[str, ...] = ("hog",)

    def __post_init__(self) -> None:
        if not isinstance(self.sample_checkers, tuple) or not self.sample_checkers:
            raise ValueError(
                "BackgroundConfig.sample_checkers must be a non-empty tuple, "
                f"got {self.sample_checkers!r}"
            )
        for checker in self.sample_checkers:
            if checker not in SAMPLE_CHECKERS:
                raise ValueError(
                    "BackgroundConfig.sample_checkers entries must be one of "
                    f"{', '.join(SAMPLE_CHECKERS)}, got {checker!r}"
                )
        if len(set(self.sample_checkers)) != len(self.sample_checkers):
            raise ValueError(
                "BackgroundConfig.sample_checkers must not repeat a checker, "
                f"got {self.sample_checkers!r}"
            )
        if self.strategy not in ("locked", "running", "calibrate"):
            raise ValueError(
                "BackgroundConfig.strategy must be locked, running or calibrate, "
                f"got {self.strategy!r}"
            )
        if self.running_window < 1:
            raise ValueError(
                f"BackgroundConfig.running_window must be >= 1, got {self.running_window}"
            )
        if self.running_interval < 1:
            raise ValueError(
                f"BackgroundConfig.running_interval must be >= 1, got {self.running_interval}"
            )
        if not 1 <= self.running_min_samples <= self.running_window:
            raise ValueError(
                "BackgroundConfig.running_min_samples must be in [1, running_window], "
                f"got {self.running_min_samples} with running_window {self.running_window}"
            )
        if not self.calibrate_seconds > 0:
            raise ValueError(
                f"BackgroundConfig.calibrate_seconds must be > 0, got {self.calibrate_seconds}"
            )


@dataclass(frozen=True)
class EncodeConfig:
    """FFmpeg output encoding settings."""

    codec: str = "libx264"
    crf: int = 23
    preset: str = "medium"
    hwaccel: str | None = None


@dataclass(frozen=True)
class EngineConfig:
    """Streaming engine settings."""

    lookahead: int = 10  # Frames of delay; file mode sets it to the gap fill.
    cold_start_frames: int = 50
    fail_closed: Literal["black"] = "black"
    # Per-frame time budget for detection plus rendering; None disables it.
    frame_budget_ms: float | None = None

    def __post_init__(self) -> None:
        if self.lookahead < 0:
            raise ValueError(f"EngineConfig.lookahead must be >= 0, got {self.lookahead}")
        if self.cold_start_frames < 0:
            raise ValueError(
                f"EngineConfig.cold_start_frames must be >= 0, got {self.cold_start_frames}"
            )
        if self.frame_budget_ms is not None and not self.frame_budget_ms > 0:
            raise ValueError(
                f"EngineConfig.frame_budget_ms must be > 0 or None, got {self.frame_budget_ms}"
            )


@dataclass(frozen=True)
class FaceYuNetConfig:
    """YuNet face detector settings."""

    score_threshold: float = 0.6
    nms_threshold: float = 0.3
    top_k: int = 5000
    model_path: str | None = None  # Explicit model file; None searches the model directories.

    def __post_init__(self) -> None:
        if not 0 < self.score_threshold < 1:
            raise ValueError(
                f"FaceYuNetConfig.score_threshold must be in (0, 1), got {self.score_threshold}"
            )
        if not 0 < self.nms_threshold <= 1:
            raise ValueError(
                f"FaceYuNetConfig.nms_threshold must be in (0, 1], got {self.nms_threshold}"
            )
        if self.top_k <= 0:
            raise ValueError(f"FaceYuNetConfig.top_k must be > 0, got {self.top_k}")
        _check_model_path("FaceYuNetConfig.model_path", self.model_path)


@dataclass(frozen=True)
class PersonDnnConfig:
    """DNN person box detector settings."""

    score_threshold: float = 0.4
    nms_threshold: float = 0.5
    model_path: str | None = None  # Explicit model file; None searches the model directories.

    def __post_init__(self) -> None:
        if not 0 < self.score_threshold < 1:
            raise ValueError(
                f"PersonDnnConfig.score_threshold must be in (0, 1), got {self.score_threshold}"
            )
        if not 0 < self.nms_threshold <= 1:
            raise ValueError(
                f"PersonDnnConfig.nms_threshold must be in (0, 1], got {self.nms_threshold}"
            )
        _check_model_path("PersonDnnConfig.model_path", self.model_path)


@dataclass(frozen=True)
class SegmentConfig:
    """Person segmentation settings."""

    threshold: float = 0.5
    dilate_px: int = 0  # Safety margin grown around the mask, in pixels.
    # Mask share of the frame from which the frame counts as containing a person.
    min_area_share: float = 0.002
    model_path: str | None = None  # Explicit model file; None searches the model directories.

    def __post_init__(self) -> None:
        if not 0 < self.threshold < 1:
            raise ValueError(f"SegmentConfig.threshold must be in (0, 1), got {self.threshold}")
        if self.dilate_px < 0:
            raise ValueError(f"SegmentConfig.dilate_px must be >= 0, got {self.dilate_px}")
        if not 0 < self.min_area_share <= 1:
            raise ValueError(
                f"SegmentConfig.min_area_share must be in (0, 1], got {self.min_area_share}"
            )
        _check_model_path("SegmentConfig.model_path", self.model_path)


@dataclass(frozen=True)
class TrackerConfig:
    """IoU box tracker settings, as in :class:`petrify.detectors.tracker.IouTracker`."""

    iou_threshold: float = 0.3
    hold_frames: int = 15
    max_tracks: int = 256

    def __post_init__(self) -> None:
        if not 0 < self.iou_threshold <= 1:
            raise ValueError(
                f"TrackerConfig.iou_threshold must be in (0, 1], got {self.iou_threshold}"
            )
        if self.hold_frames < 0:
            raise ValueError(f"TrackerConfig.hold_frames must be >= 0, got {self.hold_frames}")
        if self.max_tracks <= 0:
            raise ValueError(f"TrackerConfig.max_tracks must be > 0, got {self.max_tracks}")


@dataclass(frozen=True)
class ModelConfig:
    """Where detector model files are looked up."""

    dir: str | None = None  # Extra model directory, searched before the default ones.

    def __post_init__(self) -> None:
        _check_model_path("ModelConfig.dir", self.dir)


@dataclass(frozen=True)
class PetrifyConfig:
    """Top-level petrify configuration.

    ``mode`` names the plugin mode and ``mode_options`` holds its options as
    ``(name, value)`` pairs sorted by name, so the config stays hashable; build
    them from a mapping with :func:`mode_options_from` and read them back with
    :meth:`mode_options_dict`. Whether a mode of that name exists is checked by
    the mode registry, not here. ``faces``, ``person_dnn``, ``segment`` and
    ``tracker`` are off when None.
    """

    detector: Literal["motion", "person", "both"] = "motion"
    motion: MotionConfig = field(default_factory=MotionConfig)
    person: PersonHogConfig = field(default_factory=PersonHogConfig)
    background_diff: BackgroundDiffConfig | None = field(default_factory=BackgroundDiffConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    encode: EncodeConfig = field(default_factory=EncodeConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    mode: str = "erase"
    mode_options: tuple[tuple[str, OptionValue], ...] = ()
    faces: FaceYuNetConfig | None = None
    person_dnn: PersonDnnConfig | None = None
    segment: SegmentConfig | None = None
    tracker: TrackerConfig | None = None
    models: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or not _NAME_PATTERN.fullmatch(self.mode):
            raise ValueError(
                f"PetrifyConfig.mode must match [a-z][a-z0-9_]{{0,31}}, got {self.mode!r}"
            )
        _check_mode_options(self.mode_options)

    def mode_options_dict(self) -> dict[str, OptionValue]:
        """Return ``mode_options`` as a new dict from option name to value."""
        return dict(self.mode_options)

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> PetrifyConfig:
        """Build a config from the namespace of ``petrify.cli._build_parser()``.

        Mirrors the values ``petrify.pipeline.run`` derives, including its
        ``getattr`` fallbacks for the attributes it reads that way.
        """
        motion = MotionConfig(
            history=args.history,
            var_threshold=args.var_threshold,
            detect_shadows=args.detect_shadows,
            threshold=args.threshold,
            downsample=args.downsample,
        )
        person = PersonHogConfig(
            downsample=getattr(args, "hog_downsample", 0.5),
            win_stride=tuple(getattr(args, "hog_win_stride", [8, 8])),  # type: ignore[arg-type]
            scale=getattr(args, "hog_scale", 1.05),
            min_confidence=getattr(args, "hog_min_confidence", 0.3),
            min_box_area=getattr(args, "hog_min_box_area", 1500),
        )

        background_diff: BackgroundDiffConfig | None = None
        if not getattr(args, "no_bg_diff", False):
            background_diff = BackgroundDiffConfig(
                pixel_threshold=getattr(args, "bg_diff_pixel_threshold", 25),
                area_threshold=getattr(args, "bg_diff_area_threshold", 200),
                downsample=args.downsample,
            )

        scan = {
            "scan_samples": getattr(args, "roi_scan_samples", 60),
            "scan_min_freq": getattr(args, "roi_scan_min_freq", 0.6),
            "scan_margin": getattr(args, "roi_scan_margin", 8),
            "scan_max_area_frac": getattr(args, "roi_scan_max_area_frac", 0.10),
            "scan_edge_threshold": getattr(args, "roi_scan_edge_threshold", 80.0),
            "scan_edge_gap_tolerance": getattr(args, "roi_scan_edge_gap_tolerance", 30),
            "scan_edge_samples": getattr(args, "roi_scan_edge_samples", 3),
        }
        if args.roi:
            roi = RoiConfig(mode="manual", box=tuple(args.roi), **scan)  # type: ignore[arg-type]
        elif getattr(args, "no_roi", False):
            roi = RoiConfig(mode="disabled", **scan)
        else:
            roi = RoiConfig(mode="auto", **scan)

        return cls(
            detector=getattr(args, "detector", "motion"),
            motion=motion,
            person=person,
            background_diff=background_diff,
            roi=roi,
            background=BackgroundConfig(frame_index=args.background_frame),
            encode=EncodeConfig(
                codec=args.codec,
                crf=args.crf,
                preset=args.preset,
                hwaccel=args.hwaccel,
            ),
            engine=EngineConfig(lookahead=getattr(args, "gap_fill", 10)),
        )

    def background_scan_person_config(self) -> PersonHogConfig:
        """Return the HOG config the locked-background pre-scan must use.

        When the detector mode includes the person detector, the scan reuses it.
        """
        if self.detector in ("person", "both"):
            return self.person
        # Deliberate parity quirk: in "motion" mode pipeline.run() builds a separate
        # PersonDetector from only downsample, scale and min_confidence, leaving
        # win_stride and min_box_area at the class defaults. run() reads
        # min_confidence with a getattr fallback of 1.2, but the CLI always sets
        # hog_min_confidence, so the fallback never triggers and self.person's value
        # (from args, default 0.3) is the one that applies.
        return PersonHogConfig(
            downsample=self.person.downsample,
            scale=self.person.scale,
            min_confidence=self.person.min_confidence,
        )
