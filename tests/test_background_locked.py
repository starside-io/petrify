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

"""Tests for petrify.background.locked."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

import petrify.background.locked as locked_mod
from petrify.background.locked import find_clean_background, locked_background
from petrify.core import BackgroundConfig, PersonHogConfig
from petrify.sources.file import FileSource

WIDTH, HEIGHT, FPS, N_FRAMES = 160, 256, 10.0, 20
LEVEL0, LEVEL_STEP = 10, 12


def _level(i: int) -> int:
    return LEVEL0 + LEVEL_STEP * i


def _index_of(frame: np.ndarray) -> int:
    """Recover the frame index from the flat gray level drawn into it."""
    return round((float(frame.mean()) - LEVEL0) / LEVEL_STEP)


@pytest.fixture
def source(tmp_path: Path) -> Iterator[FileSource]:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened()
    for i in range(N_FRAMES):
        writer.write(np.full((HEIGHT, WIDTH, 3), _level(i), dtype=np.uint8))
    writer.release()
    src = FileSource(path)
    yield src
    src.close()


class _FakeDetector:
    def __init__(self, dirty: set[int]) -> None:
        self.dirty = dirty
        self.seen: list[int] = []

    def is_person(self, frame: np.ndarray) -> bool:
        idx = _index_of(frame)
        self.seen.append(idx)
        return idx in self.dirty


def test_markers_decode(source: FileSource) -> None:
    for i in range(N_FRAMES):
        frame = source.read_frame(i)
        assert frame is not None
        assert _index_of(frame) == i


def test_latest_clean_scanned_index(source: FileSource) -> None:
    det = _FakeDetector(dirty={15})
    frame, idx = find_clean_background(source, det, scan_step=5)
    assert det.seen == [0, 5, 10, 15]
    assert idx == 10
    assert _index_of(frame) == 10


def test_respects_scan_step(source: FileSource) -> None:
    det = _FakeDetector(dirty={18})
    _, idx = find_clean_background(source, det, scan_step=3)
    assert det.seen == list(range(0, N_FRAMES, 3))
    assert idx == 15


def test_all_dirty_falls_back_to_zero(source: FileSource, caplog: pytest.LogCaptureFixture) -> None:
    det = _FakeDetector(dirty=set(range(N_FRAMES)))
    with caplog.at_level("WARNING"):
        frame, idx = find_clean_background(source, det, scan_step=4)
    assert idx == 0
    assert _index_of(frame) == 0
    assert "no person-free frame" in caplog.text


def test_result_is_a_copy(source: FileSource) -> None:
    frame, _ = find_clean_background(source, _FakeDetector(dirty=set()), scan_step=5)
    assert frame.flags.owndata


def test_manual_frame_index_skips_detector(
    source: FileSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("PersonDetector must not be built")

    monkeypatch.setattr(locked_mod, "PersonDetector", _boom)
    frame, idx = locked_background(source, BackgroundConfig(frame_index=7), PersonHogConfig())
    assert idx == 7
    assert _index_of(frame) == 7


def test_manual_frame_index_unreadable(source: FileSource) -> None:
    with pytest.raises(
        ValueError,
        match=rf"Cannot read background frame at index 500 \(video has ~{N_FRAMES} frames\)\.",
    ):
        locked_background(source, BackgroundConfig(frame_index=500), PersonHogConfig())


def test_scan_builds_detector_from_config(
    source: FileSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: dict[str, Any] = {}

    class _Recorder(_FakeDetector):
        def __init__(self, **kwargs: Any) -> None:
            built.update(kwargs)
            super().__init__(dirty={10})

    monkeypatch.setattr(locked_mod, "PersonDetector", _Recorder)
    cfg = PersonHogConfig(
        downsample=0.75, win_stride=(4, 4), scale=1.1, min_confidence=0.9, min_box_area=42
    )
    _, idx = locked_background(source, BackgroundConfig(scan_step=10), cfg)
    assert built == {
        "downsample": 0.75,
        "win_stride": (4, 4),
        "scale": 1.1,
        "min_confidence": 0.9,
        "min_box_area": 42,
    }
    assert idx == 0


def test_detector_error_propagates(source: FileSource, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Raising:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def is_person(self, frame: np.ndarray) -> bool:
            raise ValueError("frame too small for person detection")

    with pytest.raises(ValueError, match="too small"):
        find_clean_background(source, _Raising(), scan_step=5)

    monkeypatch.setattr(locked_mod, "PersonDetector", _Raising)
    with pytest.raises(ValueError, match="too small"):
        locked_background(source, BackgroundConfig(), PersonHogConfig())


def test_real_hog_scan_on_empty_clip(source: FileSource) -> None:
    # 160x256 at downsample 1.0 is above the 64x128 HOG window: safe for HOG.
    cfg = PersonHogConfig(downsample=1.0)
    _, idx = locked_background(source, BackgroundConfig(scan_step=5), cfg)
    assert idx == 15
