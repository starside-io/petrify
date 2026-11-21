"""Tests for the greedy IoU box tracker.

Each test pins one behavior: id stability, greedy association, the hold
window, deterministic tie breaking, input filtering, validation, and the
``max_tracks`` cap.
"""

from __future__ import annotations

import math

import pytest

from petrify.core.detections import Box
from petrify.detectors.tracker import IouTracker, Track, iou


def _ids(tracks: tuple[Track, ...]) -> list[int]:
    return [t.track_id for t in tracks]


def _by_id(tracks: tuple[Track, ...]) -> dict[int, Track]:
    return {t.track_id: t for t in tracks}


# ----------------------------------------------------------------------
# iou helper
# ----------------------------------------------------------------------


def test_iou_values() -> None:
    a = Box(0, 0, 10, 10)
    assert iou(a, a) == 1.0
    assert iou(a, Box(10, 0, 10, 10)) == 0.0
    assert iou(a, Box(5, 0, 10, 10)) == pytest.approx(50 / 150)
    assert iou(a, Box(0, 0, 0, 10)) == 0.0


# ----------------------------------------------------------------------
# Association
# ----------------------------------------------------------------------


def test_first_frame_starts_tracks_immediately() -> None:
    boxes = [Box(0, 0, 10, 10, score=0.9), Box(100, 0, 10, 10), Box(200, 0, 10, 10)]
    out = IouTracker().update(boxes)
    assert _ids(out) == [1, 2, 3]
    assert [t.box for t in out] == boxes
    assert all(t.age == 1 and t.missed == 0 and not t.held for t in out)


def test_moving_box_keeps_its_id() -> None:
    tracker = IouTracker()
    for frame in range(10):
        box = Box(frame * 5, frame * 3, 40, 40)
        out = tracker.update([box])
        assert len(out) == 1
        assert out[0].track_id == 1
        assert out[0].box == box
        assert out[0].age == frame + 1
        assert out[0].missed == 0


def test_crossing_boxes_keep_their_ids() -> None:
    tracker = IouTracker()
    for step in range(11):
        a = Box(step * 10, 0, 40, 40)  # moves right
        b = Box(100 - step * 10, 10, 40, 40)  # moves left, slightly lower
        # Alternate the input order so detection index cannot carry the ids.
        dets = [a, b] if step % 2 == 0 else [b, a]
        out = _by_id(tracker.update(dets))
        assert set(out) == {1, 2}
        assert out[1].box == a
        assert out[2].box == b


def test_greedy_takes_highest_iou_pair_first() -> None:
    tracker = IouTracker(iou_threshold=0.3)
    tracker.update([Box(0, 0, 10, 10), Box(4, 0, 10, 10)])
    d = Box(3, 0, 10, 10)  # iou 0.54 with track 1, 0.82 with track 2
    e = Box(-4, 0, 10, 10)  # iou 0.43 with track 1, 0 with track 2
    # A per-track best match in id order would give d to track 1 and orphan track 2.
    out = _by_id(tracker.update([d, e]))
    assert set(out) == {1, 2}
    assert out[1].box == e
    assert out[2].box == d


def test_below_threshold_starts_a_new_track() -> None:
    tracker = IouTracker(iou_threshold=0.5)
    tracker.update([Box(0, 0, 10, 10)])
    out = tracker.update([Box(5, 0, 10, 10)])  # iou 0.33
    assert _ids(out) == [2]


def test_iou_exactly_at_threshold_matches() -> None:
    tracker = IouTracker(iou_threshold=50 / 150)
    tracker.update([Box(0, 0, 10, 10)])
    out = tracker.update([Box(5, 0, 10, 10)])
    assert _ids(out) == [1]


def test_each_detection_matches_at_most_one_track() -> None:
    tracker = IouTracker(hold_frames=0)
    tracker.update([Box(0, 0, 10, 10), Box(2, 0, 10, 10)])
    out = tracker.update([Box(1, 0, 10, 10)])
    assert len(out) == 1


def test_output_sorted_by_track_id() -> None:
    tracker = IouTracker()
    a, b = Box(0, 0, 10, 10), Box(100, 0, 10, 10)
    tracker.update([a, b])
    out = tracker.update([Box(300, 0, 10, 10), b, a])
    assert _ids(out) == [1, 2, 3]


# ----------------------------------------------------------------------
# Determinism on ties
# ----------------------------------------------------------------------


def test_tied_detections_lower_index_wins() -> None:
    left, right = Box(-5, 0, 10, 10), Box(5, 0, 10, 10)
    assert iou(Box(0, 0, 10, 10), left) == iou(Box(0, 0, 10, 10), right)

    tracker = IouTracker()
    tracker.update([Box(0, 0, 10, 10)])
    out = _by_id(tracker.update([left, right]))
    assert out[1].box == left and out[2].box == right

    tracker = IouTracker()
    tracker.update([Box(0, 0, 10, 10)])
    out = _by_id(tracker.update([right, left]))
    assert out[1].box == right and out[2].box == left


def test_tied_tracks_lower_id_wins() -> None:
    tracker = IouTracker(hold_frames=1)
    tracker.update([Box(0, 0, 10, 10), Box(10, 0, 10, 10)])
    det = Box(5, 0, 10, 10)
    out = _by_id(tracker.update([det]))
    assert out[1].box == det and out[1].missed == 0
    assert out[2].missed == 1


def test_same_input_sequence_gives_same_output() -> None:
    frames = [
        [Box(0, 0, 10, 10), Box(10, 0, 10, 10)],
        [Box(5, 0, 10, 10)],
        [Box(-5, 0, 10, 10), Box(5, 0, 10, 10)],
        [],
        [Box(5, 0, 10, 10), Box(15, 0, 10, 10)],
    ]

    def run() -> list[tuple[Track, ...]]:
        tracker = IouTracker(hold_frames=2)
        return [tracker.update(f) for f in frames]

    assert run() == run()


# ----------------------------------------------------------------------
# Hold window
# ----------------------------------------------------------------------


def test_hold_for_exactly_n_frames_then_drop() -> None:
    tracker = IouTracker(hold_frames=2)
    box = Box(0, 0, 10, 10, score=0.7)
    tracker.update([box])

    out = tracker.update([])
    assert len(out) == 1
    assert out[0].box == box and out[0].missed == 1 and out[0].age == 2 and out[0].held

    out = tracker.update([])
    assert len(out) == 1
    assert out[0].box == box and out[0].missed == 2 and out[0].age == 3

    assert tracker.update([]) == ()


def test_rematch_during_hold_keeps_id_and_resets_missed() -> None:
    tracker = IouTracker(hold_frames=3)
    tracker.update([Box(0, 0, 10, 10)])
    tracker.update([])
    tracker.update([])
    moved = Box(2, 0, 10, 10)
    out = tracker.update([moved])
    assert len(out) == 1
    assert out[0].track_id == 1
    assert out[0].box == moved
    assert out[0].missed == 0 and not out[0].held
    assert out[0].age == 4

    # The miss counter restarts, so a full hold window is available again.
    for expected_missed in (1, 2, 3):
        assert tracker.update([])[0].missed == expected_missed
    assert tracker.update([]) == ()


def test_new_id_after_track_expired() -> None:
    tracker = IouTracker(hold_frames=1)
    box = Box(0, 0, 10, 10)
    tracker.update([box])
    tracker.update([])
    assert tracker.update([]) == ()
    out = tracker.update([box])
    assert _ids(out) == [2]
    assert out[0].age == 1


def test_hold_zero_is_pass_through_with_ids() -> None:
    tracker = IouTracker(hold_frames=0)
    a = Box(0, 0, 10, 10)
    assert _ids(tracker.update([a])) == [1]
    assert _ids(tracker.update([Box(1, 0, 10, 10)])) == [1]
    assert tracker.update([]) == ()
    assert _ids(tracker.update([a])) == [2]


def test_held_and_detected_tracks_together() -> None:
    tracker = IouTracker(hold_frames=2)
    a, b = Box(0, 0, 10, 10), Box(100, 0, 10, 10)
    tracker.update([a, b])
    out = _by_id(tracker.update([b]))
    assert out[1].held and out[1].box == a
    assert not out[2].held and out[2].box == b


# ----------------------------------------------------------------------
# Input handling
# ----------------------------------------------------------------------


def test_empty_input() -> None:
    tracker = IouTracker(hold_frames=2)
    assert tracker.update([]) == ()
    assert tracker.update(()) == ()
    assert _ids(tracker.update([Box(0, 0, 10, 10)])) == [1]


@pytest.mark.parametrize(
    "bad", [Box(0, 0, 0, 10), Box(0, 0, 10, 0), Box(0, 0, -5, 10), Box(0, 0, 10, -1)]
)
def test_degenerate_boxes_are_ignored(bad: Box) -> None:
    tracker = IouTracker(hold_frames=1)
    assert tracker.update([bad]) == ()
    good = Box(0, 0, 10, 10)
    out = tracker.update([bad, good, bad])
    assert _ids(out) == [1]
    assert out[0].box == good
    # A degenerate box at the same spot does not refresh the track.
    out = tracker.update([bad])
    assert out[0].missed == 1


# ----------------------------------------------------------------------
# reset
# ----------------------------------------------------------------------


def test_reset_clears_tracks_and_restarts_ids() -> None:
    tracker = IouTracker(hold_frames=5)
    tracker.update([Box(0, 0, 10, 10), Box(100, 0, 10, 10)])
    tracker.reset()
    assert tracker.update([]) == ()
    out = tracker.update([Box(0, 0, 10, 10)])
    assert _ids(out) == [1]
    assert out[0].age == 1


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.01, math.nan])
def test_invalid_iou_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        IouTracker(iou_threshold=threshold)


def test_iou_threshold_of_one_is_valid() -> None:
    tracker = IouTracker(iou_threshold=1.0)
    box = Box(0, 0, 10, 10)
    tracker.update([box])
    assert _ids(tracker.update([box])) == [1]
    assert _ids(tracker.update([Box(1, 0, 10, 10)])) == [2]


def test_negative_hold_frames() -> None:
    with pytest.raises(ValueError, match="hold_frames"):
        IouTracker(hold_frames=-1)


@pytest.mark.parametrize("max_tracks", [0, -3])
def test_non_positive_max_tracks(max_tracks: int) -> None:
    with pytest.raises(ValueError, match="max_tracks"):
        IouTracker(max_tracks=max_tracks)


# ----------------------------------------------------------------------
# max_tracks cap
# ----------------------------------------------------------------------


def _spaced(i: int) -> Box:
    return Box(i * 100, 0, 10, 10)


def test_cap_evicts_longest_missed_held_track_first() -> None:
    tracker = IouTracker(hold_frames=10, max_tracks=3)
    a, b, c, d = (_spaced(i) for i in range(4))
    tracker.update([a, b, c])  # ids 1, 2, 3
    tracker.update([a, c])  # 2 missed 1
    tracker.update([a])  # 2 missed 2, 3 missed 1
    out = _by_id(tracker.update([a, d]))  # 4 tracks wanted, cap 3
    assert set(out) == {1, 3, 4}
    assert out[3].missed == 2


def test_cap_ties_evict_lower_id_first() -> None:
    tracker = IouTracker(hold_frames=10, max_tracks=3)
    a, b, c, d = (_spaced(i) for i in range(4))
    tracker.update([a, b, c])
    tracker.update([a])  # 2 and 3 both missed 1
    out = tracker.update([a, d])
    assert _ids(out) == [1, 3, 4]


def test_cap_never_drops_current_detections() -> None:
    tracker = IouTracker(hold_frames=10, max_tracks=2)
    tracker.update([_spaced(0), _spaced(1)])
    boxes = [_spaced(i) for i in range(2, 7)]
    out = tracker.update(boxes)
    # Both held tracks are evicted, every detection is returned past the cap.
    assert _ids(out) == [3, 4, 5, 6, 7]
    assert [t.box for t in out] == boxes
    assert not any(t.held for t in out)


def test_under_cap_keeps_all_held_tracks() -> None:
    tracker = IouTracker(hold_frames=10, max_tracks=4)
    tracker.update([_spaced(0), _spaced(1), _spaced(2)])
    out = tracker.update([_spaced(3)])
    assert _ids(out) == [1, 2, 3, 4]
