"""Tests for the streaming lookahead buffer in ``petrify.core.engine``.

The file pipeline helpers ``_fill_gaps`` and ``_freeze_cold_start`` serve as the
oracle; hand-written cases cover the edges without relying on them.
"""

from __future__ import annotations

import random

import pytest

from petrify.core.engine import Decision, LookaheadBuffer
from petrify.pipeline import _fill_gaps, _freeze_cold_start


def _bits(s: str) -> list[bool]:
    """Build a bool list from a string of ``1`` and ``.`` characters."""
    return [c == "1" for c in s]


def _run(
    raws: list[bool], lookahead: int, cold_start: int = 0
) -> tuple[list[object], list[Decision[object]], list[list[Decision[object]]]]:
    """Push one fresh object per raw value, then flush.

    Returns the pushed items, all decisions in emission order, and the list
    returned by each push (flush output appended as the last entry).
    """
    buf: LookaheadBuffer[object] = LookaheadBuffer(lookahead, cold_start)
    items = [object() for _ in raws]
    per_call: list[list[Decision[object]]] = []
    for item, raw in zip(items, raws):
        per_call.append(buf.push(item, raw))
        assert buf.pending <= lookahead + 1
    per_call.append(buf.flush())
    assert buf.pending == 0
    decisions = [d for batch in per_call for d in batch]
    return items, decisions, per_call


# ---------------------------------------------------------------------------
# Oracle equivalence
# ---------------------------------------------------------------------------

LOOKAHEADS = [0, 1, 2, 3, 10, 25, 400]
COLD_STARTS = [0, 1, 50, 400]
DENSITIES = [0.0, 0.01, 0.05, 0.2, 0.5, 0.9, 1.0]


@pytest.mark.parametrize("lookahead", LOOKAHEADS)
@pytest.mark.parametrize("cold_start", COLD_STARTS)
def test_matches_pipeline_helpers(lookahead: int, cold_start: int) -> None:
    for seed in range(40):
        rng = random.Random(seed * 1000 + lookahead * 10 + cold_start)
        length = rng.randint(0, 300)
        density = DENSITIES[seed % len(DENSITIES)]
        raws = [rng.random() < density for _ in range(length)]

        items, decisions, _ = _run(raws, lookahead, cold_start)

        expected_filled = _fill_gaps(raws, lookahead)
        expected_final = _freeze_cold_start(expected_filled, cold_start)
        assert [d.index for d in decisions] == list(range(length))
        assert all(d.item is it for d, it in zip(decisions, items))
        assert [d.raw for d in decisions] == raws
        assert [d.filled for d in decisions] == expected_filled
        assert [d.final for d in decisions] == expected_final


def test_matches_pipeline_helpers_short_sequences_exhaustive() -> None:
    for length in range(0, 9):
        for mask in range(1 << length):
            raws = [bool(mask >> i & 1) for i in range(length)]
            for lookahead in (0, 1, 2, 3, 10):
                _, decisions, _ = _run(raws, lookahead, 2)
                filled = _fill_gaps(raws, lookahead)
                assert [d.filled for d in decisions] == filled
                assert [d.final for d in decisions] == _freeze_cold_start(filled, 2)


# ---------------------------------------------------------------------------
# Hand-written cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raws", "lookahead", "expected"),
    [
        # detection at index 0
        ("1.......", 2, "111....."),
        # detection at the last index
        (".......1", 2, ".....111"),
        # two detections exactly 2 * lookahead apart: the fill meets in the middle
        ("..1...1..", 2, "111111111"),
        # one further apart: still contiguous because the fills touch
        ("..1....1..", 2, "1111111111"),
        # two further apart: a single unfilled frame between the fills
        ("..1.....1..", 2, "11111.11111"),
        # lookahead 0 is the identity
        ("1..11.1", 0, "1..11.1"),
        # nothing detected
        ("......", 3, "......"),
    ],
)
def test_hand_written_fill(raws: str, lookahead: int, expected: str) -> None:
    _, decisions, _ = _run(_bits(raws), lookahead)
    assert [d.filled for d in decisions] == _bits(expected)
    assert [d.final for d in decisions] == _bits(expected)


def test_cold_start_forces_final_but_not_filled() -> None:
    _, decisions, _ = _run(_bits("......1."), 1, 3)
    assert [d.filled for d in decisions] == _bits(".....111")
    assert [d.final for d in decisions] == _bits("111..111")


def test_empty_stream() -> None:
    buf: LookaheadBuffer[object] = LookaheadBuffer(5, 10)
    assert buf.pending == 0
    assert buf.flush() == []
    assert buf.pending == 0


@pytest.mark.parametrize("lookahead", [0, 1, 7])
@pytest.mark.parametrize("raw", [True, False])
def test_single_item(lookahead: int, raw: bool) -> None:
    items, decisions, per_call = _run([raw], lookahead)
    assert len(decisions) == 1
    d = decisions[0]
    assert d.item is items[0]
    assert d.index == 0
    assert d.raw is raw
    assert d.filled is raw
    assert d.final is raw
    # emitted by the push when lookahead is 0, by flush otherwise
    assert (per_call[0] != []) == (lookahead == 0)


def test_single_item_cold_start() -> None:
    _, decisions, _ = _run([False], 3, 1)
    assert decisions[0].filled is False
    assert decisions[0].final is True


def test_item_is_passed_through_untouched() -> None:
    buf: LookaheadBuffer[dict[str, int]] = LookaheadBuffer(1)
    payload = {"a": 1}
    buf.push(payload, True)
    (d,) = buf.flush()
    assert d.item is payload
    assert payload == {"a": 1}


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lookahead", [0, 1, 2, 5, 25])
def test_each_index_emitted_by_push_of_index_plus_lookahead(lookahead: int) -> None:
    rng = random.Random(lookahead)
    length = 120
    raws = [rng.random() < 0.1 for _ in range(length)]
    _, _, per_call = _run(raws, lookahead)

    pushes, flushed = per_call[:-1], per_call[-1]
    for push_idx, batch in enumerate(pushes):
        expected = [push_idx - lookahead] if push_idx >= lookahead else []
        assert [d.index for d in batch] == expected
    assert [d.index for d in flushed] == list(range(max(0, length - lookahead), length))


def test_lookahead_zero_emits_immediately() -> None:
    buf: LookaheadBuffer[int] = LookaheadBuffer(0)
    for i in range(10):
        out = buf.push(i, i % 3 == 0)
        assert len(out) == 1
        assert out[0].index == i
        assert out[0].item == i
        assert buf.pending == 0
    assert buf.flush() == []


def test_lookahead_longer_than_stream_emits_only_on_flush() -> None:
    _, decisions, per_call = _run(_bits("..1.."), 10)
    assert all(batch == [] for batch in per_call[:-1])
    assert len(per_call[-1]) == 5
    assert [d.filled for d in decisions] == [True] * 5


# ---------------------------------------------------------------------------
# Memory bound and counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lookahead", [0, 1, 3, 50])
def test_pending_never_exceeds_lookahead_plus_one(lookahead: int) -> None:
    rng = random.Random(99)
    buf: LookaheadBuffer[int] = LookaheadBuffer(lookahead, 50)
    emitted = 0
    pushed = 5000
    for i in range(pushed):
        emitted += len(buf.push(i, rng.random() < 0.05))
        assert buf.pending <= lookahead + 1
        assert buf.pending == min(i + 1, lookahead)
    emitted += len(buf.flush())
    assert emitted == pushed
    assert buf.pending == 0


# ---------------------------------------------------------------------------
# Validation and reuse
# ---------------------------------------------------------------------------


def test_negative_lookahead_raises() -> None:
    with pytest.raises(ValueError, match="lookahead"):
        LookaheadBuffer(-1)


def test_negative_cold_start_raises() -> None:
    with pytest.raises(ValueError, match="cold_start_frames"):
        LookaheadBuffer(2, -1)


def test_reuse_after_flush_starts_new_stream_at_index_zero() -> None:
    buf: LookaheadBuffer[str] = LookaheadBuffer(2, 1)
    first = [d for i in range(4) for d in buf.push(f"a{i}", i == 3)] + buf.flush()
    assert [d.index for d in first] == [0, 1, 2, 3]
    assert [d.filled for d in first] == [False, True, True, True]

    # the True at the end of the first stream must not leak into the second
    second = [d for i in range(4) for d in buf.push(f"b{i}", False)] + buf.flush()
    assert [d.index for d in second] == [0, 1, 2, 3]
    assert [d.item for d in second] == ["b0", "b1", "b2", "b3"]
    assert [d.filled for d in second] == [False] * 4
    # cold start applies again to the new stream
    assert [d.final for d in second] == [True, False, False, False]


def test_negative_start_index_raises() -> None:
    with pytest.raises(ValueError, match="start_index"):
        LookaheadBuffer(2, 0, -1)


# ---------------------------------------------------------------------------
# Start index (mid-stream buffers)
# ---------------------------------------------------------------------------


def test_default_start_index_is_zero() -> None:
    buf: LookaheadBuffer[int] = LookaheadBuffer(2, 3)
    assert buf.start_index == 0
    assert buf.next_index == 0
    assert buf.cold_start_end == 3


@pytest.mark.parametrize("lookahead", [0, 1, 3, 10])
@pytest.mark.parametrize("cold_start", [0, 2, 50])
@pytest.mark.parametrize("start", [1, 7, 1000])
def test_start_index_only_shifts_indices(lookahead: int, cold_start: int, start: int) -> None:
    for seed in range(20):
        rng = random.Random(seed * 7 + start + lookahead)
        raws = [rng.random() < DENSITIES[seed % len(DENSITIES)] for _ in range(rng.randint(0, 120))]
        _, base, base_calls = _run(raws, lookahead, cold_start)

        buf: LookaheadBuffer[object] = LookaheadBuffer(lookahead, cold_start, start)
        items = [object() for _ in raws]
        calls = [buf.push(item, raw) for item, raw in zip(items, raws)]
        assert buf.next_index == start + len(raws)
        calls.append(buf.flush())
        shifted = [d for batch in calls for d in batch]

        assert [len(c) for c in calls] == [len(c) for c in base_calls]
        assert [d.index for d in shifted] == [d.index + start for d in base]
        assert all(d.item is it for d, it in zip(shifted, items))
        assert [(d.raw, d.filled, d.final) for d in shifted] == [
            (d.raw, d.filled, d.final) for d in base
        ]


def test_cold_start_and_fill_at_non_zero_start() -> None:
    buf: LookaheadBuffer[int] = LookaheadBuffer(1, 2, 10)
    assert buf.cold_start_end == 12
    raws = _bits("....1...")
    decisions = [d for i, raw in enumerate(raws) for d in buf.push(i, raw)] + buf.flush()
    assert [d.index for d in decisions] == list(range(10, 18))
    assert [d.filled for d in decisions] == _bits("...111..")
    assert [d.final for d in decisions] == _bits("11.111..")


def test_fill_never_reaches_before_start() -> None:
    buf: LookaheadBuffer[int] = LookaheadBuffer(3, 0, 5)
    decisions = buf.push(0, True) + buf.push(1, False) + buf.flush()
    assert [d.index for d in decisions] == [5, 6]
    assert [d.filled for d in decisions] == [True, True]


def test_flush_rewinds_to_start_index() -> None:
    buf: LookaheadBuffer[str] = LookaheadBuffer(1, 1, 4)
    first = buf.push("a", False) + buf.push("b", True) + buf.flush()
    assert [d.index for d in first] == [4, 5]
    assert buf.next_index == 4
    second = buf.push("c", False) + buf.push("d", False) + buf.flush()
    assert [d.index for d in second] == [4, 5]
    assert [d.filled for d in second] == [False, False]
    assert [d.final for d in second] == [True, False]


def test_flush_twice_is_empty() -> None:
    buf: LookaheadBuffer[int] = LookaheadBuffer(3)
    buf.push(0, True)
    assert len(buf.flush()) == 1
    assert buf.flush() == []
