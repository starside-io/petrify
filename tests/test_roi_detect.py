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
        # boundary).  Columns 5 to 79 contain text; columns 80+ are empty background.
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Alternate 5-pixel black/white blocks in the "text" region
        for x in range(5, 80, 10):
            frame[:, x:x + 5] = 255   # white glyph block

        cap = self._make_cap_stub([frame, frame, frame])
        far_x = _edge_x_extent(cap, 3, 0, h, from_left=True, frame_w=w,
                                edge_threshold=50, gap_tolerance=15, n_edge_samples=3)
        # Should stop inside the text region (last hot column ≈ x=74 to 79)
        assert 65 <= far_x <= 85, f"Expected ~75, got {far_x}"

    def test_returns_fallback_when_no_edges(self) -> None:
        from petrify.roi_detect import _edge_x_extent

        h, w = 30, 200
        # Uniform grey frame: no edges at all
        frame = np.full((h, w, 3), 128, dtype=np.uint8)

        cap = self._make_cap_stub([frame])
        far_x = _edge_x_extent(cap, 1, 0, h, from_left=True, frame_w=w,
                                edge_threshold=50, gap_tolerance=10, n_edge_samples=1)
        assert far_x == w - 1   # fallback


# ---------------------------------------------------------------------------
# Pure steps on synthetic inputs (no capture)
# ---------------------------------------------------------------------------

FW, FH = 200, 100
# A (1, 1) close kernel is the identity, so a blob's anchor box is exactly the blob.
NO_CLOSE = {"close_kw": 1, "close_kh": 1}


def _freq_map(*blobs: tuple[int, int, int, int, float]) -> np.ndarray:
    """A FH x FW float32 map with each ``(x, y, w, h, freq)`` blob filled in."""
    freq = np.zeros((FH, FW), dtype=np.float32)
    for x, y, w, h, value in blobs:
        freq[y : y + h, x : x + w] = value
    return freq


class TestFindAnchor:
    def test_single_top_left_blob(self) -> None:
        from petrify.roi_detect import find_anchor

        plan = find_anchor(_freq_map((10, 5, 20, 10, 1.0)), **NO_CLOSE)
        assert plan is not None
        assert plan.anchor == (10, 5, 20, 10)
        assert (plan.frame_w, plan.frame_h) == (FW, FH)
        assert plan.side == "left"
        # Top snap: band from 0 to ay + ah + 2 * max(margin, ah // 2).
        assert (plan.y_final, plan.y_band_start) == (0, 0)
        assert plan.y_band_end == 5 + 10 + 2 * 8
        assert plan.h_final == plan.y_band_end
        assert plan.cx_rel == pytest.approx(20 / FW)
        assert plan.cy_rel == pytest.approx(10 / FH)

    def test_prefers_the_blob_nearest_a_corner(self) -> None:
        from petrify.roi_detect import find_anchor

        corner = (0, 0, 20, 10, 1.0)
        centre = (90, 45, 20, 10, 1.0)
        plan = find_anchor(_freq_map(centre, corner), **NO_CLOSE)
        assert plan is not None
        assert plan.anchor == (0, 0, 20, 10)

    def test_frequency_can_outweigh_corner_distance(self) -> None:
        from petrify.roi_detect import find_anchor

        # Score is mean_freq - 0.3 * corner_dist (corner_dist normalised by the diagonal).
        weak_corner = (0, 0, 20, 10, 0.61)
        strong_near = (30, 0, 20, 10, 1.0)
        plan = find_anchor(_freq_map(weak_corner, strong_near), **NO_CLOSE)
        assert plan is not None
        assert plan.anchor == (30, 0, 20, 10)

    def test_empty_map_returns_none_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        from petrify.roi_detect import find_anchor

        with caplog.at_level("WARNING", logger="petrify.roi_detect"):
            assert find_anchor(_freq_map()) is None
        assert len(caplog.records) == 1
        assert "no consistently-changing region found" in caplog.records[0].getMessage()

    def test_below_min_freq_is_ignored(self) -> None:
        from petrify.roi_detect import find_anchor

        assert find_anchor(_freq_map((10, 5, 20, 10, 0.59)), **NO_CLOSE) is None

    @pytest.mark.parametrize(
        "blob, kwargs",
        [
            ((10, 5, 5, 5, 1.0), {"min_area": 50}),  # contour area 16 < 50
            ((0, 0, 60, 40, 1.0), {"max_area_frac": 0.10}),  # area > 0.10 * FW * FH
        ],
        ids=["too-small", "too-large"],
    )
    def test_area_filters_return_none_with_warning(
        self,
        blob: tuple[int, int, int, int, float],
        kwargs: dict,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from petrify.roi_detect import find_anchor

        with caplog.at_level("WARNING", logger="petrify.roi_detect"):
            assert find_anchor(_freq_map(blob), **NO_CLOSE, **kwargs) is None
        assert len(caplog.records) == 1
        assert "all candidates rejected by area/size filters" in caplog.records[0].getMessage()

    @pytest.mark.parametrize(
        "x, side",
        [
            (40, "left"),  # cx_rel == 0.25 exactly: <= snap_h_thresh
            (41, "center"),
            (139, "center"),
            (140, "right"),  # cx_rel == 0.75 exactly: >= 1 - snap_h_thresh
        ],
    )
    def test_horizontal_snap_at_threshold(self, x: int, side: str) -> None:
        from petrify.roi_detect import find_anchor

        plan = find_anchor(_freq_map((x, 0, 20, 10, 1.0)), snap_h_thresh=0.25, **NO_CLOSE)
        assert plan is not None
        assert plan.side == side

    @pytest.mark.parametrize(
        "y, expected",
        [
            # (y_final, h_final, y_band_start, y_band_end); ah=10 so y_expand = margin = 8.
            (20, (0, 20 + 10 + 16, 0, 20 + 10 + 16)),  # cy_rel == 0.25: top snap
            (21, (21 - 8, 10 + 16, 21 - 8, 21 + 10 + 8)),  # middle: pad by y_expand
            (69, (69 - 8, 10 + 16, 69 - 8, 69 + 10 + 8)),
            (70, (70 - 16, FH - (70 - 16), 70 - 16, FH)),  # cy_rel == 0.75: bottom snap
        ],
    )
    def test_vertical_snap_at_threshold(
        self, y: int, expected: tuple[int, int, int, int]
    ) -> None:
        from petrify.roi_detect import find_anchor

        plan = find_anchor(_freq_map((0, y, 20, 10, 1.0)), snap_v_thresh=0.25, **NO_CLOSE)
        assert plan is not None
        assert (plan.y_final, plan.h_final, plan.y_band_start, plan.y_band_end) == expected

    def test_tall_anchor_widens_the_band(self) -> None:
        from petrify.roi_detect import find_anchor

        # ah // 2 = 15 > margin, so y_expand = 15.
        plan = find_anchor(_freq_map((0, 40, 20, 30, 1.0)), snap_v_thresh=0.1, **NO_CLOSE)
        assert plan is not None
        assert (plan.y_band_start, plan.y_band_end) == (40 - 15, 40 + 30 + 15)

    def test_close_merges_nearby_glyphs(self) -> None:
        from petrify.roi_detect import find_anchor

        # Two glyphs 6 px apart (x 20..27 and 34..41) become one blob with the default
        # 30 x 15 close kernel. The even kernel width puts OpenCV's anchor off centre,
        # which shifts the closed blob right by one pixel.
        plan = find_anchor(_freq_map((20, 20, 8, 12, 1.0), (34, 20, 8, 12, 1.0)))
        assert plan is not None
        assert plan.anchor == (21, 20, 22, 12)


class TestEdgeFarX:
    @staticmethod
    def _energy(hot: list[int], width: int = FW, value: float = 200.0) -> np.ndarray:
        energy = np.zeros(width, dtype=np.float64)
        energy[hot] = value
        return energy

    def test_bridges_gaps_up_to_tolerance(self) -> None:
        from petrify.roi_detect import edge_far_x

        # Hot runs separated by exactly gap_tolerance cold columns are one block.
        hot = list(range(10, 21)) + [51]
        assert edge_far_x(self._energy(hot), from_left=True, gap_tolerance=30) == 51

    def test_stops_at_first_gap_wider_than_tolerance(self) -> None:
        from petrify.roi_detect import edge_far_x

        hot = list(range(10, 21)) + [52, 53]
        assert edge_far_x(self._energy(hot), from_left=True, gap_tolerance=30) == 20

    def test_leading_cold_columns_are_not_a_gap(self) -> None:
        from petrify.roi_detect import edge_far_x

        hot = list(range(100, 110))
        assert edge_far_x(self._energy(hot), from_left=True, gap_tolerance=5) == 109

    def test_from_right_mirrors(self) -> None:
        from petrify.roi_detect import edge_far_x

        block = list(range(FW - 21, FW - 10))
        bridged = block + [FW - 21 - 31]  # 30 cold columns in between
        assert edge_far_x(self._energy(bridged), from_left=False, gap_tolerance=30) == FW - 52
        split = block + [FW - 21 - 32]  # 31 cold columns in between
        assert edge_far_x(self._energy(split), from_left=False, gap_tolerance=30) == FW - 21

    def test_threshold_is_strict(self) -> None:
        from petrify.roi_detect import edge_far_x

        energy = self._energy([5, 6, 7], value=80.0)
        assert edge_far_x(energy, from_left=True, edge_threshold=80.0) == FW - 1
        assert edge_far_x(energy, from_left=True, edge_threshold=79.9) == 7

    def test_fallback_when_no_column_is_hot(self) -> None:
        from petrify.roi_detect import edge_far_x

        energy = self._energy([])
        assert edge_far_x(energy, from_left=True) == FW - 1
        assert edge_far_x(energy, from_left=False) == 0

    def test_frame_w_defaults_to_vector_length(self) -> None:
        from petrify.roi_detect import edge_far_x

        energy = self._energy([], width=64)
        assert edge_far_x(energy, from_left=True) == 63
        assert edge_far_x(energy, from_left=True, frame_w=32) == 31


class TestAssembleRoi:
    @staticmethod
    def _plan(side: str, anchor: tuple[int, int, int, int] = (90, 0, 20, 10)):
        from petrify.roi_detect import AnchorPlan

        return AnchorPlan(
            frame_w=FW,
            frame_h=FH,
            anchor=anchor,
            score=1.0,
            cx_rel=0.5,
            cy_rel=0.05,
            y_final=0,
            h_final=26,
            y_band_start=0,
            y_band_end=26,
            side=side,  # type: ignore[arg-type]
        )

    def test_left(self) -> None:
        from petrify.roi_detect import assemble_roi

        assert assemble_roi(self._plan("left"), 120, margin=8) == (0, 0, 128, 26)
        assert assemble_roi(self._plan("left"), FW - 1, margin=8) == (0, 0, FW, 26)

    def test_right(self) -> None:
        from petrify.roi_detect import assemble_roi

        assert assemble_roi(self._plan("right"), 120, margin=8) == (112, 0, FW - 112, 26)
        assert assemble_roi(self._plan("right"), 0, margin=8) == (0, 0, FW, 26)

    def test_center_pads_anchor_and_ignores_far_x(self) -> None:
        from petrify.roi_detect import assemble_roi

        assert assemble_roi(self._plan("center"), None, margin=8) == (82, 0, 36, 26)
        assert assemble_roi(self._plan("center"), 5, margin=8) == (82, 0, 36, 26)
        near_left = self._plan("center", (3, 0, 20, 10))
        assert assemble_roi(near_left, None, margin=8) == (0, 0, 36, 26)
        near_right = self._plan("center", (185, 0, 15, 10))
        assert assemble_roi(near_right, None, margin=8) == (177, 0, FW - 177, 26)

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_snapped_side_requires_far_x(self, side: str) -> None:
        from petrify.roi_detect import assemble_roi

        with pytest.raises(ValueError, match="far_x"):
            assemble_roi(self._plan(side), None)
