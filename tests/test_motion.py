"""Tests for MotionDetector."""

from __future__ import annotations

import numpy as np

from petrify.motion import MotionDetector


def _random_frame(h: int = 240, w: int = 320, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _solid(h: int, w: int, value: int = 100) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


class TestMotionDetector:
    def test_constructs_with_defaults(self) -> None:
        md = MotionDetector()
        assert md.threshold > 0

    def test_reset_clears_model(self) -> None:
        md = MotionDetector()
        frame = _solid(240, 320)
        md.is_motion(frame)
        md.reset()
        # After reset the subtractor model is fresh — smoke test only
        result = md.is_motion(frame)
        assert isinstance(result, bool)

    def test_motion_score_returns_float(self) -> None:
        md = MotionDetector()
        frame = _solid(240, 320)
        # Warm up the model
        for _ in range(5):
            md.motion_score(frame)
        score = md.motion_score(frame)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_stable_scene_low_score_after_warmup(self) -> None:
        md = MotionDetector(threshold=500, downsample=1.0)
        frame = _solid(240, 320, 128)
        # Warm up with many identical frames
        for _ in range(50):
            md.motion_score(frame)
        score = md.motion_score(frame)
        # A stable scene should produce near-zero motion score
        assert score < 100, f"Expected low score for static scene, got {score}"
