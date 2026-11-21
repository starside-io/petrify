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

"""Tests for WarmupRoiDetector and the warmup ROI stage inside the engine."""

from __future__ import annotations

import logging
import math
from typing import ClassVar

import cv2
import numpy as np
import pytest

from petrify.core import (
    DetectionKind,
    Detections,
    Engine,
    EngineConfig,
    Frame,
    FrameMeta,
    SourceInfo,
)
from petrify.roi_detect import WarmupRoiDetector, auto_detect_roi, band_edge_magnitude
from petrify.stages import RoiOverlay, WarmupRoiOverlay

W, H = 320, 240
FPS = 4.0  # pair_gap = 4
PAIR_GAP = 4

# Fake clock: a 4 x 4 checkerboard that inverts every PAIR_GAP frames, so the two
# frames of every pair differ at every clock pixel.
CLOCK = (8, 4, 24, 16)  # x, y, w, h
# Static "text" right of the clock: 4 px glyph columns every 8 px, ending at x=119.
TEXT_X0, TEXT_X1 = 36, 120


def _background() -> np.ndarray:
    """Static low-contrast texture: Laplacian magnitude stays below the edge threshold."""
    rng = np.random.default_rng(1)
    return (100 + rng.integers(0, 10, (H, W))).astype(np.uint8)


_BACKGROUND = _background()


def scene(
    i: int,
    *,
    clock: tuple[int, int, int, int] | None = CLOCK,
    text: bool = True,
    pair_gap: int = PAIR_GAP,
) -> np.ndarray:
    """Frame ``i`` of the synthetic scene as BGR uint8."""
    gray = _BACKGROUND.copy()
    if clock is not None:
        x, y, w, h = clock
        yy, xx = np.mgrid[0:h, 0:w]
        checker = (((yy // 4) + (xx // 4)) % 2).astype(bool)
        if (i // pair_gap) % 2:
            checker = ~checker
        gray[y : y + h, x : x + w] = np.where(checker, 255, 0)
        if text:
            for gx in range(TEXT_X0, TEXT_X1, 8):
                gray[y : y + h, gx : gx + 4] = 255
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _feed(detector: WarmupRoiDetector, n: int, **scene_kwargs) -> None:
    for i in range(n):
        detector.update(scene(i, **scene_kwargs))


# ---------------------------------------------------------------------------
# WarmupRoiDetector on synthetic frames
# ---------------------------------------------------------------------------


def test_locks_after_exactly_ceil_warmup_frames() -> None:
    det = WarmupRoiDetector(FPS, 2.6)
    n = math.ceil(2.6 * FPS)
    assert det.warmup_frames == n == 11
    assert det.pair_gap == PAIR_GAP
    for i in range(n - 1):
        det.update(scene(i))
        assert not det.done
        assert det.roi is None
    det.update(scene(n - 1))
    assert det.done
    assert det.frames_seen == n


def test_roi_contains_clock_and_extends_over_static_text() -> None:
    det = WarmupRoiDetector(FPS, 3.0)
    _feed(det, det.warmup_frames)
    assert det.roi is not None
    x, y, w, h = det.roi
    cx, cy, cw, ch = CLOCK
    # Top-left corner snap.
    assert (x, y) == (0, 0)
    # Contains the changing block.
    assert x <= cx and y <= cy and x + w >= cx + cw and y + h >= cy + ch
    # Extends over the static text (last glyph column is TEXT_X1 - 1) ...
    assert x + w >= TEXT_X1
    # ... but stops at the gap after it instead of running to the frame edge.
    assert x + w <= TEXT_X1 + 8 + 2
    # Exact value for the record: far_x = 120 (column next to the last glyph) + margin 8.
    assert det.roi == (0, 0, 128, 40)


def test_ignores_frames_after_done() -> None:
    det = WarmupRoiDetector(FPS, 3.0)
    _feed(det, det.warmup_frames)
    roi = det.roi
    # A different shape would raise before done; after done it is ignored.
    det.update(np.zeros((10, 10, 3), dtype=np.uint8))
    det.update(scene(0, clock=None))
    assert det.roi == roi
    assert det.frames_seen == det.warmup_frames


def test_frees_buffers_when_done() -> None:
    det = WarmupRoiDetector(FPS, 3.0)
    _feed(det, det.warmup_frames - 1)
    assert det.buffered_frames == PAIR_GAP
    det.update(scene(det.warmup_frames - 1))
    assert det.done
    assert det.buffered_frames == 0
    assert det._accumulator is None
    assert det._edge_grays == []


def test_buffer_never_exceeds_pair_gap() -> None:
    det = WarmupRoiDetector(FPS, 20.0)
    held = []
    for i in range(det.warmup_frames - 1):
        det.update(scene(i))
        held.append(det.buffered_frames)
        assert len(det._edge_grays) <= 3
    assert max(held) == PAIR_GAP
    assert held[:PAIR_GAP] == [1, 2, 3, 4]


def test_rejects_shape_change() -> None:
    det = WarmupRoiDetector(FPS, 3.0)
    det.update(scene(0))
    with pytest.raises(ValueError, match="shape"):
        det.update(np.zeros((H // 2, W, 3), dtype=np.uint8))
    assert not det.done


def test_static_scene_gives_none_with_one_warning(caplog: pytest.LogCaptureFixture) -> None:
    det = WarmupRoiDetector(FPS, 3.0)
    with caplog.at_level(logging.WARNING, logger="petrify.roi_detect"):
        _feed(det, det.warmup_frames, clock=None)
    assert det.done
    assert det.roi is None
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "no consistently-changing region found" in warnings[0].getMessage()


def test_too_short_window_gives_none_with_one_warning(caplog: pytest.LogCaptureFixture) -> None:
    det = WarmupRoiDetector(FPS, 1.0)  # 4 frames, pair_gap 4: no pair can form.
    with caplog.at_level(logging.WARNING, logger="petrify.roi_detect"):
        _feed(det, 4)
    assert det.done
    assert det.roi is None
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "too short" in warnings[0].getMessage()


@pytest.mark.parametrize(
    "kwargs",
    [{"fps": 0.0}, {"fps": 25.0, "warmup_seconds": 0.0}, {"fps": 25.0, "n_edge_samples": 0}],
)
def test_rejects_invalid_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        WarmupRoiDetector(**kwargs)


@pytest.mark.parametrize("frac", [0.0, -0.1, 1.01, 2.0])
def test_rejects_invalid_max_roi_area_frac(frac: float) -> None:
    with pytest.raises(ValueError, match="max_roi_area_frac"):
        WarmupRoiDetector(FPS, 3.0, max_roi_area_frac=frac)


# ---------------------------------------------------------------------------
# Final-ROI area cap
# ---------------------------------------------------------------------------

# A textured "head" that bobs up and down during warmup, in a scene with no
# timestamp (a webcam). It qualifies as the changing anchor, and snapping and edge
# extension then grow it into a large box.
_HEAD = np.random.default_rng(5).integers(0, 256, (80, 64)).astype(np.uint8)


def _head_scene(i: int) -> np.ndarray:
    gray = _BACKGROUND.copy()
    y = 10 + (i % 3) * 6
    gray[y : y + 80, 20:84] = _HEAD
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _run_head(det: WarmupRoiDetector) -> None:
    for i in range(det.warmup_frames):
        det.update(_head_scene(i))
    assert det.done


def test_large_roi_is_rejected_by_the_default_cap(caplog: pytest.LogCaptureFixture) -> None:
    uncapped = WarmupRoiDetector(FPS, 3.0, max_roi_area_frac=1.0)
    _run_head(uncapped)
    assert uncapped.roi is not None
    x, y, w, h = uncapped.roi
    share = w * h / (W * H)
    assert share > 0.15  # Without the cap the box is accepted.

    det = WarmupRoiDetector(FPS, 3.0)
    with caplog.at_level(logging.WARNING, logger="petrify.roi_detect"):
        _run_head(det)
    assert det.roi is None
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert str(uncapped.roi) in message
    assert f"{share * 100:.1f}%" in message
    assert "too large to be a timestamp" in message
    assert "disabled" in message


def test_small_corner_timestamp_is_unaffected_by_the_default_cap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    det = WarmupRoiDetector(FPS, 3.0)
    with caplog.at_level(logging.WARNING, logger="petrify.roi_detect"):
        _feed(det, det.warmup_frames)
    assert det.roi == (0, 0, 128, 40)
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_cap_compares_against_the_final_box() -> None:
    """A cap just under the corner timestamp's share rejects it, one just over keeps it."""
    share = 128 * 40 / (W * H)
    below = WarmupRoiDetector(FPS, 3.0, max_roi_area_frac=share - 0.001)
    _feed(below, below.warmup_frames)
    assert below.roi is None
    at = WarmupRoiDetector(FPS, 3.0, max_roi_area_frac=share)
    _feed(at, at.warmup_frames)
    assert at.roi == (0, 0, 128, 40)


# ---------------------------------------------------------------------------
# Equivalence with the file version on identical frames
# ---------------------------------------------------------------------------


class _SeekableCap:
    """Minimal seekable cv2.VideoCapture stand-in over in-memory frames."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames
        self._pos = 0

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._frames[0].shape[0])
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._frames[0].shape[1])
        raise AssertionError(f"unexpected property {prop}")

    def set(self, prop: int, value: float) -> bool:
        assert prop == cv2.CAP_PROP_POS_FRAMES
        self._pos = int(value)
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._pos >= len(self._frames):
            return False, None
        frame = self._frames[self._pos]
        self._pos += 1
        return True, frame


@pytest.mark.parametrize(
    "clock",
    [
        CLOCK,  # top-left: band starts at row 0
        (8, 110, 24, 16),  # middle-left: both band borders are interior rows
        (W - 40, H - 24, 24, 16),  # bottom-right: right-to-left scan
    ],
    ids=["top-left", "middle-left", "bottom-right"],
)
def test_matches_auto_detect_roi_on_identical_frames(clock: tuple[int, int, int, int]) -> None:
    # With n - pair_gap <= n_samples the file version uses every pair, and with
    # total_frames == n it picks the same edge frames, so both see the same input.
    n = 11
    frames = [scene(i, clock=clock, text=clock is CLOCK) for i in range(n)]
    expected = auto_detect_roi(_SeekableCap(frames), n, FPS)

    det = WarmupRoiDetector(FPS, n / FPS)
    assert det.warmup_frames == n
    for frame in frames:
        det.update(frame)
    assert expected is not None
    assert det.roi == expected


def test_band_laplacian_differs_from_full_frame_slice_only_at_border_rows() -> None:
    """Why the detector keeps grayscale frames instead of a full-frame Laplacian max.

    ``cv2.Laplacian`` on a band reflects the band's own first and last rows at its
    border, while on the full frame those rows see their real neighbours. Interior
    rows agree exactly; the two border rows do not.
    """
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 256, (H, W), dtype=np.uint8)
    y0, y1 = 50, 80
    band = band_edge_magnitude(gray, y0, y1)
    full = np.abs(cv2.Laplacian(gray, cv2.CV_64F))[y0:y1]
    assert np.array_equal(band[1:-1], full[1:-1])
    assert not np.array_equal(band[0], full[0])
    assert not np.array_equal(band[-1], full[-1])
    # At the top of the frame both reflect the same way, so row 0 agrees.
    assert np.array_equal(band_edge_magnitude(gray, 0, 30)[0], np.abs(cv2.Laplacian(gray, cv2.CV_64F))[0])


# ---------------------------------------------------------------------------
# Warmup window versus full-clip scan on the demo clips
# ---------------------------------------------------------------------------

# Full-clip ROIs pinned by the golden fixtures, and what a warmup window of the
# first N seconds gives (measured; see the assertions below for what is checked).
#   camera.mp4  full (0, 0, 642, 129)  10 s (0, 0, 642, 127)  20 s (0, 0, 642, 127)
#   camera2.mp4 full (0, 0, 440, 29)   10 s (0, 0, 440, 29)   20 s (0, 0, 440, 28)
_FULL_CLIP_ROI = {"camera.mp4": (0, 0, 642, 129), "camera2.mp4": (0, 0, 440, 29)}


def _clip_path(name: str):
    from tests.golden.cases import clip_path

    return clip_path(name)


@pytest.mark.golden
@pytest.mark.parametrize("clip", sorted(_FULL_CLIP_ROI))
def test_file_version_roi_on_demo_clip(clip: str) -> None:
    path = _clip_path(clip)
    if not path.exists():
        pytest.skip(f"demo clip not available: {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        roi = auto_detect_roi(cap, n, cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    assert roi == _FULL_CLIP_ROI[clip]


@pytest.mark.golden
@pytest.mark.parametrize("seconds", [10, 20])
@pytest.mark.parametrize("clip", sorted(_FULL_CLIP_ROI))
def test_warmup_window_on_demo_clip(clip: str, seconds: int) -> None:
    """A warmup window finds the same corner and width; height may differ by a few px.

    Height is ``ay + ah + 2 * max(margin, ah // 2)`` for a top-snapped anchor, so a
    one-pixel change in the anchor's y or height (the seconds digits' blob, which a
    shorter window samples differently) moves it by up to 3 px. Width comes from
    the edge scan of static text and does not depend on the window on these clips.
    """
    path = _clip_path(clip)
    if not path.exists():
        pytest.skip(f"demo clip not available: {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        det = WarmupRoiDetector(cap.get(cv2.CAP_PROP_FPS), seconds)
        while not det.done:
            ok, frame = cap.read()
            assert ok, "clip ended before the warmup window"
            det.update(frame)
    finally:
        cap.release()

    full = _FULL_CLIP_ROI[clip]
    roi = det.roi
    assert roi is not None
    x, y, w, h = roi
    fx, fy, fw, fh = full
    assert (x, y) == (fx, fy)  # same corner
    assert abs(w - fw) <= 2
    assert abs(h - fh) <= 3
    # The live region stays within a few pixels of the full-clip one: never more
    # than 3 rows of the full ROI are left out.
    assert x <= fx and x + w >= fx + fw - 2 and y + h >= fy + fh - 3


# ---------------------------------------------------------------------------
# End to end with the real Engine
# ---------------------------------------------------------------------------

CONSTANT = 7  # Never appears in the synthetic scene (background >= 100, glyphs 0/255).


class ConstantMode:
    name = "constant"
    requires: ClassVar[frozenset[DetectionKind]] = frozenset()

    def render(self, frame: Frame, detections: Detections, subject_present: bool) -> np.ndarray:
        return np.full_like(frame.image, CONSTANT)


class CollectSink:
    critical = True

    def __init__(self) -> None:
        self.images: list[np.ndarray] = []
        self.metas: list[FrameMeta] = []

    def open(self, info: SourceInfo) -> None:
        pass

    def write(self, image: np.ndarray, meta: FrameMeta) -> None:
        self.images.append(image.copy())
        self.metas.append(meta)

    def close(self) -> None:
        pass

    def abort(self) -> None:
        pass


def test_engine_hides_live_pixels_until_lock_then_shows_only_the_roi() -> None:
    info = SourceInfo(fps=FPS, width=W, height=H, frame_count=None, seekable=False, live=True)
    detector = WarmupRoiDetector(FPS, 3.0)
    stage = WarmupRoiOverlay(detector)
    sink = CollectSink()
    engine = Engine([], ConstantMode(), [stage], [sink], EngineConfig(lookahead=2, cold_start_frames=0))

    n = detector.warmup_frames + 6
    inputs = [scene(i) for i in range(n)]
    engine.open(info)
    for i, image in enumerate(inputs):
        engine.step(Frame(image=image, index=i, timestamp=i / FPS, source=info))
    stats = engine.close()

    assert stats.frames == n and stats.failed_closed == 0
    assert [m.index for m in sink.metas] == list(range(n))
    assert stage.locked
    roi = stage.roi
    assert roi == (0, 0, 128, 40)
    x, y, w, h = roi
    mask = np.zeros((H, W), dtype=bool)
    mask[y : y + h, x : x + w] = True

    lock_index = detector.warmup_frames - 1  # The frame that completes the warmup.
    for i, out in enumerate(sink.images):
        if i < lock_index:
            # No pixel from any input frame: the output is the mode's constant image.
            assert np.all(out == CONSTANT), f"frame {i} leaked live pixels during warmup"
        else:
            assert np.array_equal(out[mask], inputs[i][mask]), f"frame {i} ROI is not live"
            assert np.all(out[~mask] == CONSTANT), f"frame {i} leaked pixels outside the ROI"
            expected = RoiOverlay(roi).apply(
                Frame(image=inputs[i], index=i, timestamp=i / FPS, source=info),
                np.full_like(inputs[i], CONSTANT),
            )
            assert np.array_equal(out, expected)
