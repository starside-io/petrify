"""Tests for BackgroundDiffer."""

from __future__ import annotations

import numpy as np
import pytest

from petrify.background import LockedBackground
from petrify.core import (
    BackgroundDiffConfig,
    BackgroundNotReady,
    Detector,
    Frame,
    SourceInfo,
)
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


# ----------------------------------------------------------------------
# from_model
# ----------------------------------------------------------------------

_INFO = SourceInfo(fps=25.0, width=320, height=240, frame_count=None, seekable=False, live=True)


def _frame(image: np.ndarray) -> Frame:
    return Frame(image=image, index=0, timestamp=0.0, source=_INFO)


class _SwitchModel:
    """Model whose image and version the test sets directly."""

    def __init__(self) -> None:
        self.version = 0
        self.image: np.ndarray | None = None

    def current(self) -> np.ndarray | None:
        return self.image

    def update(self, frame: Frame, subject_present: bool) -> None:
        pass

    def set(self, image: np.ndarray) -> None:
        self.image = image
        self.version += 1


def _query_frames(bg: np.ndarray) -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    frames = [bg.copy(), np.clip(bg.astype(int) + 4, 0, 255).astype(np.uint8)]
    for x, size in ((10, 5), (50, 40), (130, 60), (200, 15)):
        f = bg.copy()
        f[60 : 60 + size, x : x + size] = 255
        frames.append(f)
    frames.append(rng.integers(0, 256, bg.shape, dtype=np.uint8))
    return frames


def test_from_model_not_ready() -> None:
    model = _SwitchModel()
    differ = BackgroundDiffer.from_model(model, BackgroundDiffConfig())
    assert isinstance(differ, Detector)
    out = differ.detect(_frame(_solid(240, 320)))
    assert out.background_diff is None
    assert not out.subject_present
    with pytest.raises(BackgroundNotReady):
        differ.is_different(_solid(240, 320))


@pytest.mark.parametrize("downsample", [0.5, 1.0])
def test_from_model_matches_fixed_reference(downsample: float) -> None:
    bg = np.random.default_rng(0).integers(0, 200, (240, 320, 3), dtype=np.uint8)
    cfg = BackgroundDiffConfig(downsample=downsample)
    fixed = BackgroundDiffer.from_config(bg, cfg)
    for model in (LockedBackground(bg), _SwitchModel()):
        if isinstance(model, _SwitchModel):
            model.set(bg)
        differ = BackgroundDiffer.from_model(model, cfg)
        results = []
        for f in _query_frames(bg):
            expected = fixed.is_different(f)
            assert differ.is_different(f) == expected
            assert differ.detect(_frame(f)).background_diff == expected
            results.append(expected)
        assert True in results and False in results


def test_from_model_recomputes_reference_only_on_version_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _SwitchModel()
    differ = BackgroundDiffer.from_model(model, BackgroundDiffConfig())
    calls: list[np.ndarray] = []
    original = BackgroundDiffer._reference_gray

    def spy(self: BackgroundDiffer, reference: np.ndarray) -> np.ndarray:
        calls.append(reference)
        return original(self, reference)

    monkeypatch.setattr(BackgroundDiffer, "_reference_gray", spy)
    dark, light = _solid(240, 320, 50), _solid(240, 320, 200)
    differ.detect(_frame(dark))
    assert calls == []
    model.set(dark)
    for _ in range(5):
        assert differ.detect(_frame(dark)).background_diff is False
        assert differ.is_different(dark) is False
    assert len(calls) == 1
    model.set(light)
    for _ in range(3):
        assert differ.detect(_frame(dark)).background_diff is True
        assert differ.detect(_frame(light)).background_diff is False
    assert len(calls) == 2
    assert calls[1] is light
