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

"""Tests for the core engine data types, configs, and protocols."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import numpy as np
import pytest

from petrify.cli import _build_parser
from petrify.core import (
    BackgroundConfig,
    BackgroundDiffConfig,
    Box,
    DetectionKind,
    Detections,
    Detector,
    EngineConfig,
    Face,
    Frame,
    FrameMeta,
    Mode,
    PersonHogConfig,
    PetrifyConfig,
    RoiConfig,
    Sink,
    Source,
    SourceInfo,
    Stage,
)


def _config(*extra: str) -> PetrifyConfig:
    return PetrifyConfig.from_namespace(_build_parser().parse_args(["in.mp4", "out.mp4", *extra]))


def _info() -> SourceInfo:
    return SourceInfo(fps=25.0, width=4, height=2, frame_count=10, seekable=True, live=False)


def _frame() -> Frame:
    return Frame(image=np.zeros((2, 4, 3), dtype=np.uint8), index=0, timestamp=0.0, source=_info())


# ---------------------------------------------------------------------------
# Frame / SourceInfo
# ---------------------------------------------------------------------------


def test_source_info_constructs_and_is_frozen() -> None:
    info = _info()
    assert info.uri is None
    assert info.frame_count == 10
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.fps = 30.0  # type: ignore[misc]


def test_source_info_unknown_frame_count() -> None:
    info = SourceInfo(fps=30.0, width=640, height=480, frame_count=None, seekable=False,
                      live=True, uri="rtsp://cam")
    assert info.frame_count is None
    assert info.live


def test_frame_constructs_and_is_frozen() -> None:
    frame = _frame()
    assert frame.image.shape == (2, 4, 3)
    assert frame.source.fps == 25.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        frame.index = 1  # type: ignore[misc]


def test_frame_meta_has_no_pixels_and_is_frozen() -> None:
    frame = _frame()
    meta = frame.meta
    assert meta == FrameMeta(index=0, timestamp=0.0, source=_info())
    assert not hasattr(meta, "image")
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.index = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------


def test_detection_kind_values() -> None:
    assert {k.value for k in DetectionKind} == {
        "motion", "persons", "background_diff", "faces", "landmarks", "mask",
    }
    assert DetectionKind.MOTION == "motion"


def test_detections_default_all_none() -> None:
    d = Detections()
    for f in dataclasses.fields(d):
        assert getattr(d, f.name) is None


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, False),
        ({"motion": True}, True),
        ({"motion": False}, False),
        ({"persons": ()}, False),
        ({"persons": (Box(0, 0, 1, 1),)}, True),
        ({"background_diff": True}, True),
        ({"background_diff": False}, False),
        ({"motion": False, "persons": (), "background_diff": False}, False),
        ({"motion": None, "persons": None, "background_diff": True}, True),
        ({"faces": (Face(Box(0, 0, 1, 1)),)}, False),
        ({"mask": np.ones((2, 2), dtype=np.uint8)}, False),
    ],
)
def test_subject_present(kwargs: dict, expected: bool) -> None:
    assert Detections(**kwargs).subject_present is expected


def test_merged_keeps_and_overrides() -> None:
    boxes = (Box(1, 2, 3, 4, score=0.9),)
    base = Detections(motion=False, persons=boxes)
    other = Detections(motion=True, background_diff=False)
    out = base.merged(other)
    assert out.motion is True
    assert out.persons == boxes
    assert out.background_diff is False
    assert out.faces is None
    assert out.mask is None
    # Originals are untouched.
    assert base.motion is False
    assert base.background_diff is None


def test_merged_with_empty_other_keeps_everything() -> None:
    mask = np.ones((2, 2), dtype=np.uint8)
    base = Detections(motion=True, mask=mask)
    out = base.merged(Detections())
    assert out.motion is True
    assert out.mask is mask


def test_detections_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        Detections().motion = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_cli_defaults_match_config_defaults() -> None:
    from_cli = _config()
    default = PetrifyConfig()
    assert from_cli.detector == default.detector == "motion"
    assert from_cli.motion == default.motion
    assert from_cli.person == default.person
    assert from_cli.background_diff == default.background_diff
    assert from_cli.background_diff is not None
    assert from_cli.roi == default.roi
    assert from_cli.roi.mode == "auto"
    assert from_cli.roi.box is None
    assert from_cli.background == default.background
    assert from_cli.background.frame_index is None
    assert from_cli.background.scan_step == 5
    assert from_cli.encode == default.encode
    assert from_cli.engine == default.engine
    assert from_cli.engine.lookahead == 10
    assert from_cli.engine.cold_start_frames == 50
    assert from_cli == default


def test_detector_flag() -> None:
    assert _config("--detector", "both").detector == "both"
    assert _config("--detector", "person").detector == "person"


def test_no_bg_diff() -> None:
    assert _config("--no-bg-diff").background_diff is None


def test_bg_diff_flags() -> None:
    cfg = _config("--bg-diff-pixel-threshold", "40", "--bg-diff-area-threshold", "300")
    assert cfg.background_diff == BackgroundDiffConfig(pixel_threshold=40, area_threshold=300)


def test_gap_fill_maps_to_lookahead() -> None:
    assert _config("--gap-fill", "3").engine.lookahead == 3


def test_manual_roi() -> None:
    roi = _config("--roi", "1", "2", "3", "4").roi
    assert roi.mode == "manual"
    assert roi.box == (1, 2, 3, 4)


def test_no_roi() -> None:
    roi = _config("--no-roi").roi
    assert roi.mode == "disabled"
    assert roi.box is None


def test_manual_roi_wins_over_no_roi() -> None:
    assert _config("--roi", "1", "2", "3", "4", "--no-roi").roi.mode == "manual"


def test_roi_scan_flags() -> None:
    roi = _config(
        "--roi-scan-samples", "30",
        "--roi-scan-min-freq", "0.8",
        "--roi-scan-margin", "4",
        "--roi-scan-max-area-frac", "0.2",
        "--roi-scan-edge-threshold", "90",
        "--roi-scan-edge-gap-tolerance", "20",
        "--roi-scan-edge-samples", "5",
    ).roi
    assert roi.mode == "auto"
    assert roi.scan_samples == 30
    assert roi.scan_min_freq == 0.8
    assert roi.scan_margin == 4
    assert roi.scan_max_area_frac == 0.2
    assert roi.scan_edge_threshold == 90.0
    assert roi.scan_edge_gap_tolerance == 20
    assert roi.scan_edge_samples == 5


def test_background_frame() -> None:
    assert _config("--background-frame", "7").background.frame_index == 7


def test_downsample_feeds_motion_and_bg_diff_only() -> None:
    cfg = _config("--downsample", "0.25")
    assert cfg.motion.downsample == 0.25
    assert cfg.background_diff is not None
    assert cfg.background_diff.downsample == 0.25
    assert cfg.person.downsample == 0.5


def test_hog_downsample_feeds_person_only() -> None:
    cfg = _config("--hog-downsample", "0.75")
    assert cfg.person.downsample == 0.75
    assert cfg.motion.downsample == 0.5
    assert cfg.background_diff is not None
    assert cfg.background_diff.downsample == 0.5


def test_motion_flags() -> None:
    cfg = _config("--threshold", "200", "--history", "100", "--var-threshold", "20",
                  "--detect-shadows")
    assert cfg.motion.threshold == 200
    assert cfg.motion.history == 100
    assert cfg.motion.var_threshold == 20.0
    assert cfg.motion.detect_shadows is True


def test_hog_flags() -> None:
    cfg = _config("--hog-scale", "1.2", "--hog-min-confidence", "0.5",
                  "--hog-min-box-area", "900")
    assert cfg.person == PersonHogConfig(scale=1.2, min_confidence=0.5, min_box_area=900)


def test_encode_flags() -> None:
    enc = _config("--crf", "30", "--codec", "libx265", "--preset", "fast",
                  "--hwaccel", "cuda").encode
    assert enc.crf == 30
    assert enc.codec == "libx265"
    assert enc.preset == "fast"
    assert enc.hwaccel == "cuda"


def test_background_scan_person_config_motion_uses_class_defaults() -> None:
    cfg = _config("--hog-downsample", "0.75", "--hog-scale", "1.2",
                  "--hog-min-confidence", "0.5", "--hog-min-box-area", "900")
    scan = cfg.background_scan_person_config()
    assert scan.downsample == 0.75
    assert scan.scale == 1.2
    assert scan.min_confidence == 0.5
    assert scan.win_stride == PersonHogConfig().win_stride
    assert scan.min_box_area == PersonHogConfig().min_box_area == 1500


@pytest.mark.parametrize("mode", ["both", "person"])
def test_background_scan_person_config_reuses_person(mode: str) -> None:
    cfg = _config("--detector", mode, "--hog-min-box-area", "900")
    assert cfg.background_scan_person_config() is cfg.person
    assert cfg.background_scan_person_config().min_box_area == 900


def test_roi_manual_without_box_rejected() -> None:
    with pytest.raises(ValueError):
        RoiConfig(mode="manual")


@pytest.mark.parametrize("mode", ["auto", "disabled"])
def test_roi_box_rejected_outside_manual(mode: str) -> None:
    with pytest.raises(ValueError):
        RoiConfig(mode=mode, box=(1, 2, 3, 4))  # type: ignore[arg-type]


def test_roi_warmup_seconds_default_and_validation() -> None:
    assert RoiConfig().warmup_seconds == 10.0
    assert RoiConfig(warmup_seconds=2.5).warmup_seconds == 2.5
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="warmup_seconds"):
            RoiConfig(warmup_seconds=bad)


def test_negative_lookahead_rejected() -> None:
    with pytest.raises(ValueError):
        EngineConfig(lookahead=-1)


def test_negative_cold_start_rejected() -> None:
    with pytest.raises(ValueError):
        EngineConfig(cold_start_frames=-1)


def test_frame_budget_defaults_to_disabled() -> None:
    assert EngineConfig().frame_budget_ms is None
    assert _config().engine.frame_budget_ms is None
    assert EngineConfig(frame_budget_ms=33.3).frame_budget_ms == 33.3


@pytest.mark.parametrize("budget", [0.0, -5.0, float("nan")])
def test_non_positive_frame_budget_rejected(budget: float) -> None:
    with pytest.raises(ValueError, match="frame_budget_ms"):
        EngineConfig(frame_budget_ms=budget)


def test_background_config_strategy_defaults() -> None:
    cfg = BackgroundConfig()
    assert cfg.strategy == "locked"
    assert (cfg.running_window, cfg.running_interval, cfg.calibrate_seconds) == (30, 5, 3.0)
    assert cfg.running_min_samples == 5
    small = BackgroundConfig(running_window=1, running_min_samples=1)
    assert (small.running_window, small.running_min_samples) == (1, 1)
    assert _config().background.strategy == "locked"
    for strategy in ("locked", "running", "calibrate"):
        assert BackgroundConfig(strategy=strategy).strategy == strategy  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strategy": "rolling"},
        {"running_window": 0},
        {"running_interval": 0},
        {"running_min_samples": 0},
        {"running_window": 4},
        {"running_window": 10, "running_min_samples": 11},
        {"calibrate_seconds": 0.0},
        {"calibrate_seconds": -1.0},
        {"calibrate_seconds": float("nan")},
    ],
)
def test_background_config_rejects_bad_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="BackgroundConfig"):
        BackgroundConfig(**kwargs)  # type: ignore[arg-type]


def test_config_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        PetrifyConfig().detector = "both"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class _DummyDetector:
    provides = frozenset({DetectionKind.MOTION})
    stateful = True

    def detect(self, frame: Frame) -> Detections:
        return Detections(motion=False)

    def reset(self) -> None:
        pass


class _DummyMode:
    name = "dummy"
    requires = frozenset({DetectionKind.MOTION})

    def render(self, frame: Frame, detections: Detections, subject_present: bool) -> np.ndarray:
        return frame.image


class _DummyStage:
    def apply(self, frame: Frame, image: np.ndarray) -> np.ndarray:
        return image


class _DummySink:
    critical = True

    def open(self, info: SourceInfo) -> None:
        pass

    def write(self, image: np.ndarray, meta: FrameMeta) -> None:
        pass

    def close(self) -> None:
        pass

    def abort(self) -> None:
        pass


class _DummySource:
    def __init__(self) -> None:
        self.info = _info()

    def __iter__(self) -> Iterator[Frame]:
        yield _frame()

    def close(self) -> None:
        pass


@pytest.mark.parametrize(
    ("obj", "proto"),
    [
        (_DummyDetector(), Detector),
        (_DummyMode(), Mode),
        (_DummyStage(), Stage),
        (_DummySink(), Sink),
        (_DummySource(), Source),
    ],
)
def test_dummy_satisfies_protocol(obj: object, proto: type) -> None:
    assert isinstance(obj, proto)


def test_object_missing_members_fails_protocol() -> None:
    assert not isinstance(_DummyStage(), Sink)
    assert not isinstance(_DummySink(), Detector)
