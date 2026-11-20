"""Regression tests for PersonDetector frame-size handling (FIX-2).

Every frame below the HOG window must raise ValueError before reaching
detectMultiScale, so none of these tests can crash the interpreter.
"""

from __future__ import annotations

import numpy as np
import pytest

from petrify.person import PersonDetector


def _frame(w: int, h: int) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    ("w", "h", "downsample"),
    [(160, 120, 0.5), (63, 128, 0.5), (64, 127, 0.5), (80, 60, 1.0), (63, 128, 1.0)],
)
def test_too_small_frame_raises(w: int, h: int, downsample: float) -> None:
    det = PersonDetector(downsample=downsample)
    with pytest.raises(ValueError, match="minimum is 64x128"):
        det.is_person(_frame(w, h))


@pytest.mark.parametrize("downsample", [0.5, 1.0])
def test_frame_at_minimum_does_not_raise(downsample: float) -> None:
    det = PersonDetector(downsample=downsample)
    assert isinstance(det.is_person(_frame(64, 128)), bool)


def test_downsample_clamped_to_window() -> None:
    # 160x240 at 0.5 would be 80x120, under the 128 px window height.
    det = PersonDetector(downsample=0.5)
    assert det._maybe_downsample(_frame(160, 240)).shape[:2] == (128, 80)
    assert isinstance(det.is_person(_frame(160, 240)), bool)


def test_large_frame_resize_unchanged() -> None:
    det = PersonDetector(downsample=0.5)
    assert det._maybe_downsample(_frame(640, 480)).shape[:2] == (240, 320)


@pytest.mark.parametrize("downsample", [0.0, -0.5, 1.5])
def test_invalid_downsample_raises(downsample: float) -> None:
    with pytest.raises(ValueError, match="downsample must be in"):
        PersonDetector(downsample=downsample)
