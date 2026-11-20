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

"""Tests for EraseMode."""

from __future__ import annotations

import numpy as np
import pytest

from petrify.core import DetectionKind, Detections, Frame, Mode, SourceInfo
from petrify.modes import EraseMode

W, H = 160, 120


def _frame(image: np.ndarray) -> Frame:
    h, w = image.shape[:2]
    info = SourceInfo(fps=25.0, width=w, height=h, frame_count=None, seekable=False, live=False)
    return Frame(image=image, index=0, timestamp=0.0, source=info)


def _bg() -> np.ndarray:
    return np.full((H, W, 3), 50, dtype=np.uint8)


def _live() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (H, W, 3), dtype=np.uint8)


def test_satisfies_mode() -> None:
    mode = EraseMode(_bg())
    assert isinstance(mode, Mode)
    assert mode.name == "erase"
    assert mode.requires == frozenset(
        {DetectionKind.MOTION, DetectionKind.PERSONS, DetectionKind.BACKGROUND_DIFF}
    )


def test_present_returns_background_copy() -> None:
    bg = _bg()
    mode = EraseMode(bg)
    out = mode.render(_frame(_live()), Detections(), subject_present=True)
    assert np.array_equal(out, bg)
    assert out is not bg
    out[:] = 0
    assert np.array_equal(mode.background, _bg())


def test_absent_returns_frame_image_itself() -> None:
    live = _live()
    out = EraseMode(_bg()).render(_frame(live), Detections(motion=True), subject_present=False)
    assert out is live


def test_shape_mismatch_raises() -> None:
    mode = EraseMode(_bg())
    wrong = np.zeros((H, W + 1, 3), dtype=np.uint8)
    for present in (True, False):
        with pytest.raises(ValueError, match="does not match background shape"):
            mode.render(_frame(wrong), Detections(), subject_present=present)


@pytest.mark.parametrize("shape", [(H, W), (H, W, 1), (H, W, 4)])
def test_background_must_be_three_channel(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="3-channel"):
        EraseMode(np.zeros(shape, dtype=np.uint8))
