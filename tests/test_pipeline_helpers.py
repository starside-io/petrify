"""Characterization tests for the pure pipeline helpers.

These pin the current behavior of ``_fill_gaps`` and ``_freeze_cold_start`` so a
replacement implementation (streaming lookahead buffer) can be checked against
them. Expected values are written by hand or computed with a tiny oracle that is
independent of the code under test.
"""

from __future__ import annotations

import pytest

from petrify.pipeline import _fill_gaps, _freeze_cold_start

T = True
F = False


def _bits(s: str) -> list[bool]:
    """Build a bool list from a string of ``1`` and ``.`` characters."""
    return [c == "1" for c in s]


def _oracle_fill(has_subject: list[bool], gap: int) -> list[bool]:
    """Independent reference: frame j is True if any detection lies within *gap* of j."""
    if gap <= 0:
        return list(has_subject)
    hits = [i for i, v in enumerate(has_subject) if v]
    return [any(abs(i - j) <= gap for i in hits) for j in range(len(has_subject))]


class TestFillGaps:
    @pytest.mark.parametrize("gap", [0, -1, -100])
    def test_non_positive_gap_returns_equal_new_list(self, gap: int) -> None:
        src = _bits("..1..1...")
        before = list(src)
        out = _fill_gaps(src, gap)
        assert out == before
        assert out is not src
        assert src == before

    def test_non_positive_gap_result_not_aliased(self) -> None:
        src = _bits("1..")
        out = _fill_gaps(src, 0)
        out[1] = True
        assert src == _bits("1..")

    def test_single_detection_expands_by_gap_both_sides(self) -> None:
        src = _bits("..........1..........")
        assert _fill_gaps(src, 3) == _bits(".......1111111.......")

    def test_gap_one_single_detection(self) -> None:
        assert _fill_gaps(_bits("...1..."), 1) == _bits("..111..")

    def test_detection_at_index_zero_clamps_left(self) -> None:
        assert _fill_gaps(_bits("1........"), 3) == _bits("1111.....")

    def test_detection_at_last_index_clamps_right(self) -> None:
        assert _fill_gaps(_bits("........1"), 3) == _bits(".....1111")

    def test_detections_at_both_ends(self) -> None:
        assert _fill_gaps(_bits("1..........1"), 2) == _bits("111......111")

    def test_close_detections_merge_into_one_run(self) -> None:
        # Detections at 5 and 9 with gap 3: distance 4 < 2 * gap.
        src = _bits(".....1...1.......")
        assert _fill_gaps(src, 3) == _bits("..11111111111....")

    def test_detections_exactly_two_gap_apart_touch(self) -> None:
        # Detections at 3 and 9 with gap 3: runs [0, 6] and [6, 12] share frame 6.
        src = _bits("...1.....1...")
        assert _fill_gaps(src, 3) == _bits("1111111111111")

    def test_adjacent_runs_leave_no_clean_frame(self) -> None:
        # Detections at 3 and 10 with gap 3: runs [0, 6] and [7, 13] are adjacent.
        src = _bits("...1......1...")
        assert _fill_gaps(src, 3) == _bits("11111111111111")

    def test_far_detections_stay_separate(self) -> None:
        # Detections at 3 and 14 with gap 3: runs [0, 6] and [11, 17], clean 7..10.
        src = _bits("...1..........1...")
        out = _fill_gaps(src, 3)
        assert out == _bits("1111111....1111111")
        assert out[7:11] == [F, F, F, F]

    def test_all_false_stays_all_false(self) -> None:
        assert _fill_gaps([F] * 20, 5) == [F] * 20

    def test_all_true_stays_all_true(self) -> None:
        assert _fill_gaps([T] * 20, 5) == [T] * 20

    @pytest.mark.parametrize("gap", [-1, 0, 1, 10])
    def test_empty_list(self, gap: int) -> None:
        assert _fill_gaps([], gap) == []

    def test_gap_larger_than_list_fills_everything(self) -> None:
        assert _fill_gaps(_bits("..1.."), 100) == [T] * 5

    def test_gap_larger_than_list_with_no_detection(self) -> None:
        assert _fill_gaps(_bits("....."), 100) == [F] * 5

    def test_input_not_mutated(self) -> None:
        src = _bits("...1.......1..")
        before = list(src)
        out = _fill_gaps(src, 2)
        assert src == before
        assert out is not src

    @pytest.mark.parametrize(
        ("pattern", "gap"),
        [
            ("1.1.1.....1....1", 1),
            (".....11.....1...........1.", 4),
            ("1..............................1", 7),
            ("..1...1...1...1...1...1..", 2),
        ],
    )
    def test_matches_independent_oracle(self, pattern: str, gap: int) -> None:
        src = _bits(pattern)
        assert _fill_gaps(src, gap) == _oracle_fill(src, gap)


class TestFreezeColdStart:
    def test_default_warmup_forces_first_fifty(self) -> None:
        src = [i % 3 == 0 for i in range(80)]
        out = _freeze_cold_start(src)
        assert out[:50] == [T] * 50
        assert out[50:] == [i % 3 == 0 for i in range(50, 80)]
        assert len(out) == 80

    def test_default_warmup_all_false_input(self) -> None:
        out = _freeze_cold_start([F] * 60)
        assert out == [T] * 50 + [F] * 10

    def test_exactly_fifty_becomes_all_true(self) -> None:
        assert _freeze_cold_start([F] * 50) == [T] * 50

    def test_shorter_than_fifty_becomes_all_true(self) -> None:
        assert _freeze_cold_start([F] * 7) == [T] * 7

    def test_warmup_zero_leaves_list_unchanged(self) -> None:
        src = _bits("..1..1..")
        out = _freeze_cold_start(src, warmup_frames=0)
        assert out == _bits("..1..1..")
        assert out is not src

    def test_negative_warmup_raises(self) -> None:
        with pytest.raises(ValueError, match="warmup_frames must be >= 0"):
            _freeze_cold_start(_bits("..1.."), warmup_frames=-3)

    @pytest.mark.parametrize(
        ("warmup", "expected"),
        [
            (1, "1...1..."),
            (3, "111.1..."),
            (5, "11111..."),
            (8, "11111111"),
            (20, "11111111"),
        ],
    )
    def test_explicit_warmup(self, warmup: int, expected: str) -> None:
        assert _freeze_cold_start(_bits("....1..."), warmup_frames=warmup) == _bits(expected)

    @pytest.mark.parametrize("warmup", [0, 1, 50])
    def test_empty_list(self, warmup: int) -> None:
        assert _freeze_cold_start([], warmup_frames=warmup) == []

    def test_input_not_mutated(self) -> None:
        src = [F] * 60
        out = _freeze_cold_start(src)
        assert src == [F] * 60
        assert out is not src


class TestComposition:
    def test_freeze_after_fill_matches_run_order(self) -> None:
        # Detections at 55, 70 and 90 in 100 frames, gap 4, default warmup 50.
        src = [F] * 100
        for i in (55, 70, 90):
            src[i] = T
        expected = [F] * 100
        for i in range(50):
            expected[i] = T
        for lo, hi in ((51, 59), (66, 74), (86, 94)):
            for i in range(lo, hi + 1):
                expected[i] = T
        assert _freeze_cold_start(_fill_gaps(src, 4)) == expected

    def test_fill_spilling_into_warmup_window(self) -> None:
        # Detection at 52 with gap 5 reaches back to 47, overlapping the warmup.
        src = [F] * 60
        src[52] = T
        expected = [T] * 58 + [F] * 2
        assert _freeze_cold_start(_fill_gaps(src, 5)) == expected

    def test_short_clip_composition(self) -> None:
        src = _bits("....1...")
        assert _freeze_cold_start(_fill_gaps(src, 1), warmup_frames=2) == _bits("11.111..")
