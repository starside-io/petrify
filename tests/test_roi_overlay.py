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

"""Tests for the RoiOverlay stage."""

from __future__ import annotations

import numpy as np

from petrify.core import Frame, SourceInfo, Stage
from petrify.stages import Compositor, RoiOverlay

W, H = 160, 120


def _frame(image: np.ndarray) -> Frame:
    info = SourceInfo(fps=25.0, width=W, height=H, frame_count=None, seekable=False, live=False)
    return Frame(image=image, index=0, timestamp=0.0, source=info)


def _images() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    live = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    mode_image = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    return live, mode_image


def test_satisfies_stage() -> None:
    assert isinstance(RoiOverlay((0, 0, 10, 10)), Stage)
    assert isinstance(RoiOverlay(None), Stage)


def test_roi_region_is_live_rest_is_mode_image() -> None:
    live, mode_image = _images()
    x, y, w, h = 20, 30, 50, 40
    out = RoiOverlay((x, y, w, h)).apply(_frame(live), mode_image)

    mask = np.zeros((H, W), dtype=bool)
    mask[y : y + h, x : x + w] = True
    assert np.array_equal(out[mask], live[mask])
    assert np.array_equal(out[~mask], mode_image[~mask])


def test_matches_compositor_pass_two() -> None:
    live, mode_image = _images()
    roi = (100, 90, 100, 100)  # Clamped at the frame edge.
    comp = Compositor(roi)
    expected = comp.compose(mode_image, comp.extract_roi(live))
    assert np.array_equal(RoiOverlay(roi).apply(_frame(live), mode_image), expected)


def test_none_roi_returns_image_unchanged() -> None:
    live, mode_image = _images()
    out = RoiOverlay(None).apply(_frame(live), mode_image)
    assert out is mode_image


def test_inputs_not_mutated() -> None:
    live, mode_image = _images()
    live_before, mode_before = live.copy(), mode_image.copy()
    RoiOverlay((10, 10, 30, 30)).apply(_frame(live), mode_image)
    assert np.array_equal(live, live_before)
    assert np.array_equal(mode_image, mode_before)
