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

"""Tests for the person-free sample filter."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from petrify.background import CalibratedBackground, RunningBackground, person_free
from petrify.core import Frame, SourceInfo
from petrify.detectors import PersonDetector

W, H = 160, 120
INFO = SourceInfo(fps=25.0, width=W, height=H, frame_count=None, seekable=False, live=True)


def _frame(image: np.ndarray, index: int = 0) -> Frame:
    return Frame(image=image, index=index, timestamp=index / INFO.fps, source=INFO)


def _level(level: int) -> np.ndarray:
    return np.full((H, W, 3), level, dtype=np.uint8)


class _FakeDetector:
    """Duck-typed detector: a person is present when the first pixel is 255."""

    def __init__(self) -> None:
        self.calls = 0

    def is_person(self, frame: np.ndarray) -> bool:
        self.calls += 1
        return bool(frame[0, 0, 0] == 255)


class _TooSmall:
    """Behaves like ``PersonDetector`` on a frame under the HOG window."""

    def is_person(self, frame: np.ndarray) -> bool:
        raise ValueError("frame too small for person detection")


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "sample filter raised" in r.getMessage()
    ]


def test_person_free_is_the_negation_of_is_person() -> None:
    detector = _FakeDetector()
    check = person_free(detector)
    assert check(_level(10)) is True
    assert check(_level(255)) is False
    assert detector.calls == 2


def test_person_free_lets_detector_errors_propagate() -> None:
    check = person_free(_TooSmall())
    with pytest.raises(ValueError, match="too small"):
        check(_level(10))


def test_person_free_keeps_people_out_of_the_running_model() -> None:
    model = RunningBackground(
        window=5, interval=1, min_samples=1, sample_filter=person_free(_FakeDetector())
    )
    model.update(_frame(_level(255), 0), subject_present=False)
    assert model.current() is None
    model.update(_frame(_level(10), 1), subject_present=False)
    current = model.current()
    assert current is not None and np.all(current == 10)
    assert model.rejected == 1


@pytest.mark.parametrize("kind", ["running", "calibrate"])
def test_undetectable_frames_are_rejected_with_one_warning(
    kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    check = person_free(_TooSmall())
    model: RunningBackground | CalibratedBackground
    if kind == "running":
        model = RunningBackground(window=5, interval=1, min_samples=1, sample_filter=check)
    else:
        model = CalibratedBackground(frames=5, sample_filter=check)
    with caplog.at_level(logging.WARNING, logger="petrify.background.running"):
        for i in range(10):
            model.update(_frame(_level(10), i), subject_present=False)  # Does not raise.
    assert model.current() is None
    assert model.rejected == 10
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "ValueError" in warnings[0].getMessage()
    assert "too small" in warnings[0].getMessage()


def test_real_person_detector_plumbing() -> None:
    """Only checks that the real detector plugs in; no detections are asserted."""
    check = person_free(PersonDetector())
    rng = np.random.default_rng(0)
    for w, h in ((128, 256), (320, 256)):
        image = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        result = check(image)
        assert isinstance(result, bool)
