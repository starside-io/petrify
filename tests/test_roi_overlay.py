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


# ---------------------------------------------------------------------------
# WarmupRoiOverlay
# ---------------------------------------------------------------------------


def _warmup_stage(warmup_seconds: float = 3.0, **scene_kwargs):
    """A WarmupRoiOverlay on a real detector plus the synthetic scene it will see."""
    from petrify.roi_detect import WarmupRoiDetector
    from petrify.stages import WarmupRoiOverlay
    from tests.test_roi_warmup import FPS, scene

    detector = WarmupRoiDetector(FPS, warmup_seconds)
    frames = [scene(i, **scene_kwargs) for i in range(detector.warmup_frames + 5)]
    return WarmupRoiOverlay(detector), detector, frames


def _scene_frame(image: np.ndarray, index: int) -> Frame:
    h, w = image.shape[:2]
    info = SourceInfo(fps=4.0, width=w, height=h, frame_count=None, seekable=False, live=True)
    return Frame(image=image, index=index, timestamp=index / 4.0, source=info)


def test_warmup_overlay_satisfies_stage() -> None:
    stage, _, _ = _warmup_stage()
    assert isinstance(stage, Stage)


def test_warmup_overlay_is_identity_during_warmup() -> None:
    stage, detector, frames = _warmup_stage()
    for i in range(detector.warmup_frames - 1):
        mode_image = np.full_like(frames[i], 7)
        before = mode_image.copy()
        out = stage.apply(_scene_frame(frames[i], i), mode_image)
        assert out is mode_image
        assert np.array_equal(out, before)  # Nothing from frame.image was pasted in.
        assert not stage.locked
        assert stage.roi is None


def test_warmup_overlay_matches_roi_overlay_after_lock() -> None:
    stage, detector, frames = _warmup_stage()
    for i, image in enumerate(frames):
        mode_image = np.full_like(image, 7)
        out = stage.apply(_scene_frame(image, i), mode_image)
        if i >= detector.warmup_frames - 1:
            assert stage.locked
            assert stage.roi == detector.roi is not None
            expected = RoiOverlay(stage.roi).apply(_scene_frame(image, i), mode_image)
            assert np.array_equal(out, expected)
            assert not np.array_equal(out, mode_image)


def test_warmup_overlay_none_roi_is_identity_forever() -> None:
    stage, detector, frames = _warmup_stage(clock=None)
    for i, image in enumerate(frames):
        mode_image = np.full_like(image, 7)
        assert stage.apply(_scene_frame(image, i), mode_image) is mode_image
    assert detector.done
    assert stage.locked
    assert stage.roi is None


def test_warmup_overlay_locks_once(caplog) -> None:
    stage, detector, frames = _warmup_stage()
    with caplog.at_level("INFO", logger="petrify.stages.roi_overlay"):
        for i, image in enumerate(frames):
            stage.apply(_scene_frame(image, i), np.full_like(image, 7))
    locks = [r for r in caplog.records if r.getMessage().startswith("Auto ROI detected:")]
    assert len(locks) == 1
    x, y, w, h = detector.roi
    assert locks[0].getMessage() == f"Auto ROI detected: x={x} y={y} w={w} h={h}"
    # Later frames no longer reach the detector.
    assert detector.frames_seen == detector.warmup_frames
