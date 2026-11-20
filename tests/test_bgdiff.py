"""Tests for BackgroundDiffer."""

from __future__ import annotations

import numpy as np
import pytest

from petrify.detectors.bgdiff import BackgroundDiffer


def _solid(h: int, w: int, value: int = 128) -> np.ndarray:
    """Create a solid BGR frame filled with *value*."""
    return np.full((h, w, 3), value, dtype=np.uint8)


class TestBackgroundDiffer:
    def test_identical_frames_not_different(self) -> None:
        bg = _solid(240, 320, 100)
        differ = BackgroundDiffer(bg, area_threshold=200)
        assert not differ.is_different(bg)

    def test_slightly_noisy_frame_not_different(self) -> None:
        bg = _solid(240, 320, 100)
        noisy = bg.copy()
        # Add ±5 noise — below pixel_threshold=25
        noisy[:, :] = np.clip(noisy[:, :].astype(int) + 4, 0, 255).astype(np.uint8)
        differ = BackgroundDiffer(bg, pixel_threshold=25, area_threshold=200)
        assert not differ.is_different(noisy)

    def test_large_bright_patch_is_different(self) -> None:
        bg = _solid(240, 320, 50)
        frame = bg.copy()
        # Paint a 60×60 bright block in the centre — large foreground
        frame[90:150, 130:190] = 255
        differ = BackgroundDiffer(bg, pixel_threshold=25, area_threshold=200)
        assert differ.is_different(frame)

    def test_tiny_patch_below_area_threshold_not_different(self) -> None:
        bg = _solid(240, 320, 50)
        frame = bg.copy()
        # Paint a 5×5 block — small noise, below area_threshold=200
        frame[100:105, 100:105] = 255
        differ = BackgroundDiffer(bg, pixel_threshold=25, area_threshold=200)
        assert not differ.is_different(frame)

    def test_downsample_half(self) -> None:
        bg = _solid(480, 640, 80)
        frame = bg.copy()
        frame[100:300, 100:400] = 240   # large foreground object
        differ = BackgroundDiffer(bg, pixel_threshold=25, area_threshold=200, downsample=0.5)
        # Should still detect at half resolution
        assert differ.is_different(frame)

    def test_invalid_downsample_raises(self) -> None:
        bg = _solid(100, 100)
        with pytest.raises(ValueError, match="downsample"):
            BackgroundDiffer(bg, downsample=0.0)
        with pytest.raises(ValueError, match="downsample"):
            BackgroundDiffer(bg, downsample=1.5)
