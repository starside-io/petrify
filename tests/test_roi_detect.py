"""Tests for auto ROI detection helpers."""

from __future__ import annotations

import numpy as np
import pytest


class TestEdgeXExtent:
    """Unit tests for _edge_x_extent via the public interface."""

    def _make_cap_stub(self, frames: list[np.ndarray]):
        """Return a minimal cv2.VideoCapture-like stub."""
        import unittest.mock as mock

        cap = mock.MagicMock()
        call_count = [0]

        def read_side_effect():
            idx = call_count[0]
            call_count[0] += 1
            if idx < len(frames):
                return True, frames[idx]
            return False, None

        cap.read.side_effect = read_side_effect
        cap.set.return_value = None
        return cap

    def test_finds_text_column_extent(self) -> None:
        from petrify.roi_detect import _edge_x_extent

        h, w = 30, 200
        # Simulate a timestamp strip: alternating black/white columns to mimic
        # text glyphs (each glyph produces a Laplacian edge at its left and right
        # boundary).  Columns 5–79 contain text; columns 80+ are empty background.
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Alternate 5-pixel black/white blocks in the "text" region
        for x in range(5, 80, 10):
            frame[:, x:x + 5] = 255   # white glyph block

        cap = self._make_cap_stub([frame, frame, frame])
        far_x = _edge_x_extent(cap, 3, 0, h, from_left=True, frame_w=w,
                                edge_threshold=50, gap_tolerance=15, n_edge_samples=3)
        # Should stop inside the text region (last hot column ≈ x=74–79)
        assert 65 <= far_x <= 85, f"Expected ~75, got {far_x}"

    def test_returns_fallback_when_no_edges(self) -> None:
        from petrify.roi_detect import _edge_x_extent

        h, w = 30, 200
        # Uniform grey frame — no edges at all
        frame = np.full((h, w, 3), 128, dtype=np.uint8)

        cap = self._make_cap_stub([frame])
        far_x = _edge_x_extent(cap, 1, 0, h, from_left=True, frame_w=w,
                                edge_threshold=50, gap_tolerance=10, n_edge_samples=1)
        assert far_x == w - 1   # fallback
