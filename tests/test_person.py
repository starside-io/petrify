"""Regression tests for PersonDetector frame-size handling.

Downsampling uses one scale for both axes and never goes under the HOG
window when the source is large enough. Every frame below the window must
raise ValueError before reaching detectMultiScale, so none of these tests
can crash the interpreter.
"""

from __future__ import annotations

import numpy as np
import pytest

from petrify.detectors.person_hog import PersonDetector

WIN_W, WIN_H = 64, 128


def _frame(w: int, h: int) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _gray(w: int, h: int) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (h, w), dtype=np.uint8)


@pytest.mark.parametrize(
    ("w", "h", "downsample"),
    [(160, 120, 0.5), (63, 128, 0.5), (64, 127, 0.5), (80, 60, 1.0), (63, 128, 1.0)],
)
def test_too_small_frame_raises(w: int, h: int, downsample: float) -> None:
    det = PersonDetector(downsample=downsample)
    with pytest.raises(ValueError, match="minimum is 64x128"):
        det.is_person(_frame(w, h))


def test_narrow_tall_source_raises() -> None:
    with pytest.raises(ValueError, match="minimum is 64x128"):
        PersonDetector(downsample=0.5).is_person(_frame(50, 400))


@pytest.mark.parametrize("downsample", [0.5, 1.0])
def test_frame_at_minimum_does_not_raise(downsample: float) -> None:
    det = PersonDetector(downsample=downsample)
    assert isinstance(det.is_person(_frame(64, 128)), bool)


def test_non_default_padding_and_stride_at_minimum() -> None:
    det = PersonDetector(padding=(16, 16), win_stride=(4, 4))
    assert isinstance(det.is_person(_frame(64, 128)), bool)


def test_downsample_clamped_to_window() -> None:
    # 160x240 at 0.5 would be 80x120, under the 128 px window height.
    det = PersonDetector(downsample=0.5)
    assert det._maybe_downsample(_frame(160, 240)).shape[:2] == (128, 85)
    assert isinstance(det.is_person(_frame(160, 240)), bool)


def test_width_limited_clamp() -> None:
    # 100x300 at 0.5 would be 50x150, under the 64 px window width.
    det = PersonDetector(downsample=0.5)
    assert det._maybe_downsample(_frame(100, 300)).shape[:2] == (192, 64)


@pytest.mark.parametrize(("w", "h"), [(320, 240), (1920, 140), (100, 300)])
def test_clamp_preserves_aspect_ratio(w: int, h: int) -> None:
    small = PersonDetector(downsample=0.5)._maybe_downsample(_frame(w, h))
    new_h, new_w = small.shape[:2]
    assert new_w >= WIN_W and new_h >= WIN_H
    assert abs(new_w - new_h * w / h) <= 1
    assert abs(new_h - new_w * h / w) <= 1


def test_large_frame_resize_unchanged() -> None:
    det = PersonDetector(downsample=0.5)
    assert det._maybe_downsample(_frame(640, 480)).shape[:2] == (240, 320)
    assert det._maybe_downsample(_frame(1920, 1080)).shape[:2] == (540, 960)


def test_grayscale_frames() -> None:
    det = PersonDetector(downsample=0.5)
    assert isinstance(det.is_person(_gray(320, 240)), bool)
    with pytest.raises(ValueError, match="minimum is 64x128"):
        det.is_person(_gray(160, 120))


@pytest.mark.parametrize("downsample", [0.333, 0.999999, 0.1])
@pytest.mark.parametrize(("w", "h"), [(64, 128), (65, 129), (97, 193), (1920, 1080)])
def test_odd_downsample_never_under_window(downsample: float, w: int, h: int) -> None:
    det = PersonDetector(downsample=downsample)
    small = det._maybe_downsample(_frame(w, h))
    assert small.shape[1] >= WIN_W and small.shape[0] >= WIN_H
    if w * h <= 97 * 193:
        assert isinstance(det.is_person(_frame(w, h)), bool)


@pytest.mark.parametrize("downsample", [0.0, -0.5, 1.5])
def test_invalid_downsample_raises(downsample: float) -> None:
    with pytest.raises(ValueError, match="downsample must be in"):
        PersonDetector(downsample=downsample)
