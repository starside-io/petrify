"""Tests for the greedy IoU box tracker.

Each test pins one behavior: id stability, greedy association, the hold
window, deterministic tie breaking, input filtering, validation, the
``max_tracks`` cap, and the :class:`TrackerRefiner` that runs the tracker in the
engine.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterator, Sequence

import numpy as np
import pytest

from petrify.core import (
    Engine,
    EngineConfig,
    Frame,
    FrameMeta,
    FrameTrace,
    Refiner,
    SourceInfo,
    TrackerConfig,
)
from petrify.core.detections import (
    Box,
    DetectionKind,
    Detections,
    Face,
    MalformedDetections,
    check_detections,
)
from petrify.detectors import TrackerRefiner as ExportedTrackerRefiner
from petrify.detectors.tracker import IouTracker, Track, TrackerRefiner, iou


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


# ----------------------------------------------------------------------
# TrackerRefiner
# ----------------------------------------------------------------------

P, FACES, LANDMARKS, M = (
    DetectionKind.PERSONS,
    DetectionKind.FACES,
    DetectionKind.LANDMARKS,
    DetectionKind.MOTION,
)
FRAME_H, FRAME_W = 40, 60
INFO = SourceInfo(
    fps=10.0, width=FRAME_W, height=FRAME_H, frame_count=None, seekable=False, live=False
)
SHAPE = (FRAME_H, FRAME_W, 3)


def _frame(i: int) -> Frame:
    image = np.full(SHAPE, 100, dtype=np.uint8)
    return Frame(image=image, index=i, timestamp=i / INFO.fps, source=INFO)


def _refiner(kind: DetectionKind = P, hold: int = 2, *, landmarks: bool = False) -> TrackerRefiner:
    return TrackerRefiner(
        kind, iou_threshold=0.3, hold_frames=hold, max_tracks=16, landmarks=landmarks
    )


def _persons(refiner: TrackerRefiner, i: int, *boxes: Box) -> tuple[Box, ...] | None:
    return refiner.refine(_frame(i), Detections(persons=tuple(boxes))).persons


def _faces(refiner: TrackerRefiner, i: int, *faces: Face) -> tuple[Face, ...] | None:
    return refiner.refine(_frame(i), Detections(faces=tuple(faces))).faces


BOX_A = Box(2, 2, 10, 10, score=0.9)
BOX_B = Box(30, 20, 10, 10)
LM_1 = ((3.0, 4.0), (8.0, 4.0))
LM_2 = ((4.0, 5.0), (9.0, 5.0))


def _held(box: Box, missed: int) -> Box:
    """``box`` as the refiner returns it after ``missed`` frames without a match."""
    return dataclasses.replace(box, held=missed)


def test_refiner_satisfies_protocol_and_is_exported() -> None:
    assert ExportedTrackerRefiner is TrackerRefiner
    assert isinstance(_refiner(), Refiner)


def test_refiner_declarations() -> None:
    persons = _refiner(P)
    assert persons.consumes == persons.provides == frozenset({P})
    plain = _refiner(FACES)
    assert plain.consumes == plain.provides == frozenset({FACES})
    marked = _refiner(FACES, landmarks=True)
    assert marked.consumes == marked.provides == frozenset({FACES, LANDMARKS})


def test_refiner_holds_persons_for_exactly_hold_frames_then_drops() -> None:
    refiner = _refiner(P, hold=3)
    assert _persons(refiner, 0, BOX_A) is None  # Detected, nothing held.
    for i in (1, 2, 3):
        assert _persons(refiner, i) == (_held(BOX_A, i),)
    assert _persons(refiner, 4) is None
    assert _persons(refiner, 5) is None


def test_refiner_returns_only_held_boxes() -> None:
    refiner = _refiner(P)
    _persons(refiner, 0, BOX_A, BOX_B)
    # BOX_B is detected again this frame, so only BOX_A is added by the refiner.
    assert _persons(refiner, 1, BOX_B) == (_held(BOX_A, 1),)


def test_refiner_redetection_during_hold_stops_the_hold() -> None:
    refiner = _refiner(P, hold=3)
    _persons(refiner, 0, BOX_A)
    assert _persons(refiner, 1) == (_held(BOX_A, 1),)
    moved = Box(3, 2, 10, 10)
    assert _persons(refiner, 2, moved) is None  # No duplicate of the re-detected box.
    # The hold restarts from the re-detected box, with one box for the track.
    assert _persons(refiner, 3) == (_held(moved, 1),)


def test_refiner_treats_missing_persons_as_empty() -> None:
    refiner = _refiner(P, hold=2)
    refiner.refine(_frame(0), Detections(persons=(BOX_A,)))
    # persons=None ("not computed") ages the track like an empty tuple.
    assert refiner.refine(_frame(1), Detections()).persons == (_held(BOX_A, 1),)
    assert refiner.refine(_frame(2), Detections(motion=True)).persons == (_held(BOX_A, 2),)
    out = refiner.refine(_frame(3), Detections())
    # Nothing held: the result leaves every field unset.
    assert out.persons is None and out.faces is None


def test_refiner_never_holds_a_zero_area_box() -> None:
    refiner = _refiner(P, hold=2)
    _persons(refiner, 0, Box(FRAME_W, 5, 0, 10))
    assert _persons(refiner, 1) is None


def test_refiner_holds_faces_with_the_last_landmarks() -> None:
    refiner = _refiner(FACES, hold=2, landmarks=True)
    _faces(refiner, 0, Face(BOX_A, LM_1))
    moved = Box(3, 2, 10, 10)
    assert _faces(refiner, 1, Face(moved, LM_2)) is None
    assert _faces(refiner, 2) == (Face(_held(moved, 1), LM_2),)
    assert _faces(refiner, 3) == (Face(_held(moved, 2), LM_2),)
    assert _faces(refiner, 4) is None


def test_refiner_holds_faces_without_landmarks() -> None:
    refiner = _refiner(FACES, hold=1)
    # Landmarks from the detector are dropped, since LANDMARKS is not provided.
    _faces(refiner, 0, Face(BOX_A, LM_1), Face(BOX_B))
    assert _faces(refiner, 1) == (Face(_held(BOX_A, 1)), Face(_held(BOX_B, 1)))
    assert _faces(refiner, 2) is None


def test_refiner_matches_each_face_to_its_own_landmarks() -> None:
    refiner = _refiner(FACES, hold=1, landmarks=True)
    _faces(refiner, 0, Face(BOX_B, LM_2), Face(BOX_A, LM_1))
    held = _faces(refiner, 1)
    assert held is not None
    assert {face.box: face.landmarks for face in held} == {
        _held(BOX_A, 1): LM_1,
        _held(BOX_B, 1): LM_2,
    }


def test_refiner_keeps_landmarks_when_a_match_has_none() -> None:
    refiner = _refiner(FACES, hold=1, landmarks=True)
    _faces(refiner, 0, Face(BOX_A, LM_1))
    _faces(refiner, 1, Face(BOX_A))  # For example a second face detector without landmarks.
    assert _faces(refiner, 2) == (Face(_held(BOX_A, 1), LM_1),)


def test_refiner_rejects_a_held_face_that_never_had_landmarks() -> None:
    refiner = _refiner(FACES, hold=1, landmarks=True)
    _faces(refiner, 0, Face(BOX_A))
    with pytest.raises(ValueError, match="no landmarks"):
        _faces(refiner, 1)


@pytest.mark.parametrize(
    ("kind", "landmarks", "first", "field"),
    [
        (P, False, Detections(persons=(BOX_A, BOX_B)), "persons"),
        (FACES, False, Detections(faces=(Face(BOX_A, LM_1), Face(BOX_B))), "faces"),
        (FACES, True, Detections(faces=(Face(BOX_A, LM_1), Face(BOX_B, LM_2))), "faces"),
    ],
)
def test_refiner_result_passes_check_detections(
    kind: DetectionKind, landmarks: bool, first: Detections, field: str
) -> None:
    refiner = _refiner(kind, hold=2, landmarks=landmarks)
    results = [refiner.refine(_frame(0), first)]
    results += [refiner.refine(_frame(i), Detections()) for i in (1, 2, 3)]
    for result in results:
        check_detections(result, refiner.provides, SHAPE, allow_held=True)
    assert len(getattr(results[1], field)) == 2


def test_refiner_result_with_held_boxes_needs_allow_held() -> None:
    # A held box is a refiner's claim; the same result from a detector is malformed.
    for kind, first in ((P, Detections(persons=(BOX_A,))), (FACES, Detections(faces=(Face(BOX_A),)))):
        refiner = _refiner(kind, hold=1)
        refiner.refine(_frame(0), first)
        result = refiner.refine(_frame(1), Detections())
        with pytest.raises(MalformedDetections, match="held"):
            check_detections(result, refiner.provides, SHAPE)


def test_refiner_person_held_count_follows_missed() -> None:
    refiner = _refiner(P, hold=4)
    _persons(refiner, 0, BOX_A, BOX_B)
    # BOX_B is seen again on frame 2, so its count restarts while BOX_A keeps counting.
    expected = {
        1: (_held(BOX_A, 1), _held(BOX_B, 1)),
        2: (_held(BOX_A, 2),),
        3: (_held(BOX_A, 3), _held(BOX_B, 1)),
        4: (_held(BOX_A, 4), _held(BOX_B, 2)),
    }
    for i in (1, 2, 3, 4):
        boxes = (BOX_B,) if i == 2 else ()
        out = _persons(refiner, i, *boxes)
        tracks = refiner._tracker._tracks
        assert out == expected[i]
        # Every returned count is the missed count of a live track.
        missed = sorted(t.missed for t in tracks if t.held)
        assert sorted(b.held for b in out) == missed


def test_refiner_face_held_count_follows_missed() -> None:
    refiner = _refiner(FACES, hold=3, landmarks=True)
    _faces(refiner, 0, Face(BOX_A, LM_1))
    for i in (1, 2, 3):
        assert _faces(refiner, i) == (Face(_held(BOX_A, i), LM_1),)
    # Re-detected: nothing is held on that frame, and the next hold starts at 1.
    assert _faces(refiner, 4, Face(BOX_A, LM_2)) is None
    assert _faces(refiner, 5) == (Face(_held(BOX_A, 1), LM_2),)


def test_refiner_held_boxes_keep_position_and_score() -> None:
    scored = Box(4, 6, 12, 14, score=0.75)
    for kind in (P, FACES):
        refiner = _refiner(kind, hold=2)
        if kind is P:
            _persons(refiner, 0, scored)
            out = _persons(refiner, 1)
            assert out is not None
            box = out[0]
        else:
            _faces(refiner, 0, Face(scored))
            faces = _faces(refiner, 1)
            assert faces is not None
            box = faces[0].box
        assert (box.x, box.y, box.w, box.h, box.score) == (4, 6, 12, 14, 0.75)
        assert box.held == 1


def test_tracker_stores_the_detector_boxes_unchanged() -> None:
    # The face refiner maps tracks to faces by box identity, so the tracker must
    # keep the very object it was given; the held count goes on a copy.
    refiner = _refiner(FACES, hold=2)
    face = Face(BOX_A)
    _faces(refiner, 0, face)
    _faces(refiner, 1)
    _faces(refiner, 2)
    (track,) = refiner._tracker._tracks
    assert track.box is BOX_A and track.box.held == 0
    persons = _refiner(P, hold=2)
    _persons(persons, 0, BOX_B)
    _persons(persons, 1)
    (track,) = persons._tracker._tracks
    assert track.box is BOX_B and track.box.held == 0


def test_refiner_frees_payload_of_expired_tracks() -> None:
    refiner = _refiner(FACES, hold=1, landmarks=True)
    for i in range(50):
        # A face away from the previous one every frame: one new track per frame.
        _faces(refiner, i, Face(Box((i % 5) * 12, (i % 2) * 20, 5, 5), LM_1))
        assert len(refiner._payload) <= 2  # This frame's track plus one held.
    _faces(refiner, 50)
    _faces(refiner, 51)
    assert refiner._payload == {}


def test_refiner_reset_drops_tracks_and_payload() -> None:
    refiner = _refiner(FACES, hold=3, landmarks=True)
    _faces(refiner, 0, Face(BOX_A, LM_1))
    refiner.reset()
    assert refiner._payload == {}
    assert _faces(refiner, 1) is None


def test_refiner_reset_drops_held_persons() -> None:
    refiner = _refiner(P, hold=3)
    _persons(refiner, 0, BOX_A)
    refiner.reset()
    assert _persons(refiner, 1) is None


def test_refiner_does_not_mutate_its_input() -> None:
    refiner = _refiner(FACES, hold=2, landmarks=True)
    faces = (Face(BOX_A, LM_1),)
    given = Detections(persons=(BOX_B,), faces=faces)
    refiner.refine(_frame(0), given)
    refiner.refine(_frame(1), Detections(persons=(BOX_B,)))
    assert given.faces is faces and given.faces == (Face(BOX_A, LM_1),)
    assert given.persons == (BOX_B,)


def test_refiner_from_config_uses_every_config_value() -> None:
    config = TrackerConfig(iou_threshold=0.5, hold_frames=4, max_tracks=7)
    refiner = TrackerRefiner.from_config(FACES, config, landmarks=True)
    assert refiner.provides == frozenset({FACES, LANDMARKS})
    tracker = refiner._tracker
    assert (tracker.iou_threshold, tracker.hold_frames, tracker.max_tracks) == (0.5, 4, 7)
    # The config default of 15 is passed, not the tracker default of 0.
    assert TrackerRefiner.from_config(P, TrackerConfig())._tracker.hold_frames == 15


@pytest.mark.parametrize("kind", [M, LANDMARKS, DetectionKind.MASK, "persons"])
def test_refiner_rejects_other_kinds(kind: object) -> None:
    with pytest.raises(ValueError, match="kind"):
        _refiner(kind)  # type: ignore[arg-type]


def test_refiner_rejects_landmarks_for_persons() -> None:
    with pytest.raises(ValueError, match="landmarks"):
        _refiner(P, landmarks=True)


def test_refiner_validates_tracker_settings() -> None:
    with pytest.raises(ValueError, match="hold_frames"):
        _refiner(P, hold=-1)


# ----------------------------------------------------------------------
# TrackerRefiner in the engine
# ----------------------------------------------------------------------


class _ListSource:
    def __init__(self, n: int) -> None:
        self.info = INFO
        self.n = n

    def __iter__(self) -> Iterator[Frame]:
        for i in range(self.n):
            yield _frame(i)

    def close(self) -> None:
        pass


class _PersonsOnFrames:
    """Persons detector that finds BOX_A on the frames in ``hits``."""

    provides = frozenset({P})
    stateful = False

    def __init__(self, hits: Sequence[int]) -> None:
        self.hits = set(hits)

    def detect(self, frame: Frame) -> Detections:
        return Detections(persons=(BOX_A,) if frame.index in self.hits else ())

    def reset(self) -> None:
        pass


class _ResultOnly:
    """Detector that provides ``provides`` and returns an empty result."""

    stateful = False

    def __init__(self, provides: frozenset[DetectionKind]) -> None:
        self.provides = provides

    def detect(self, frame: Frame) -> Detections:
        return Detections()

    def reset(self) -> None:
        pass


class _RecordingMode:
    """Records the persons it renders with and the final decision."""

    name = "recording"
    detect_policy = "all"

    def __init__(self, requires: frozenset[DetectionKind]) -> None:
        self.requires = requires
        self.seen: list[tuple[int, tuple[Box, ...] | None, bool]] = []

    def render(self, frame: Frame, detections: Detections, subject_present: bool) -> np.ndarray:
        self.seen.append((frame.index, detections.persons, subject_present))
        return frame.image.copy()


class _NullSink:
    critical = True

    def open(self, info: SourceInfo) -> None:
        pass

    def write(self, image: np.ndarray, meta: FrameMeta) -> None:
        pass

    def close(self) -> None:
        pass

    def abort(self) -> None:
        pass


def _tracker_engine(
    detectors: Sequence[object],
    mode: _RecordingMode,
    refiner: TrackerRefiner,
    observer: Callable[[FrameTrace], None] | None = None,
) -> Engine:
    return Engine(
        detectors,  # type: ignore[arg-type]
        mode,
        [],
        [_NullSink()],
        EngineConfig(lookahead=0, cold_start_frames=0),
        observer,
        refiners=[refiner],
    )


def test_engine_holds_a_missed_person_and_counts_it_as_present() -> None:
    mode = _RecordingMode(frozenset({P}))
    traces: list[FrameTrace] = []
    engine = _tracker_engine([_PersonsOnFrames([0])], mode, _refiner(P, hold=3), traces.append)
    engine.run(_ListSource(6))
    assert [persons for _, persons, _ in mode.seen] == [
        (BOX_A,),
        (_held(BOX_A, 1),),
        (_held(BOX_A, 2),),
        (_held(BOX_A, 3),),
        (),
        (),
    ]
    assert [t.raw for t in traces] == [True, True, True, True, False, False]
    assert [present for _, _, present in mode.seen] == [True] * 4 + [False, False]


def _chained_engine(
    detectors: Sequence[object], mode: _RecordingMode, refiners: Sequence[object]
) -> Engine:
    return Engine(
        detectors,  # type: ignore[arg-type]
        mode,
        [],
        [_NullSink()],
        EngineConfig(lookahead=0, cold_start_frames=0),
        refiners=refiners,  # type: ignore[arg-type]
    )


def test_chained_trackers_keep_the_cumulative_held_count() -> None:
    # The second tracker matches the first one's held boxes; its own count must
    # continue from theirs, not restart at the match.
    mode = _RecordingMode(frozenset({P}))
    refiners = [_refiner(P, hold=2), _refiner(P, hold=4)]
    _chained_engine([_PersonsOnFrames([0])], mode, refiners).run(_ListSource(8))
    held = [[b.held for b in persons or ()] for _, persons, _ in mode.seen]
    assert held == [[0], [1], [2], [3], [4], [5], [6], []]
    assert [persons for _, persons, _ in mode.seen][3] == (_held(BOX_A, 3),)


class _UpstreamHolder:
    """Refiner that emits BOX_A held 1, 2, 3 on frames 1 to 3, then nothing."""

    consumes = frozenset({P})
    provides = frozenset({P})

    def refine(self, frame: Frame, detections: Detections) -> Detections:
        if 1 <= frame.index <= 3:
            return Detections(persons=(_held(BOX_A, frame.index),))
        return Detections()

    def reset(self) -> None:
        pass


def test_tracker_counts_on_from_an_upstream_held_box() -> None:
    mode = _RecordingMode(frozenset({P}))
    refiners = [_UpstreamHolder(), _refiner(P, hold=2)]
    _chained_engine([_PersonsOnFrames([0])], mode, refiners).run(_ListSource(7))
    assert [persons for _, persons, _ in mode.seen] == [
        (BOX_A,),
        (_held(BOX_A, 1),),
        (_held(BOX_A, 2),),
        (_held(BOX_A, 3),),
        (_held(BOX_A, 4),),
        (_held(BOX_A, 5),),
        (),
    ]


class _FailsOnFrames(_PersonsOnFrames):
    """Finds BOX_A on ``hits`` and raises on ``fails``."""

    def __init__(self, hits: Sequence[int], fails: Sequence[int]) -> None:
        super().__init__(hits)
        self.fails = set(fails)

    def detect(self, frame: Frame) -> Detections:
        if frame.index in self.fails:
            raise RuntimeError("detector broke")
        return super().detect(frame)


def test_held_does_not_count_frames_that_failed_closed() -> None:
    # Pinned on purpose: refiners do not run on a failed frame, so the count is
    # refined frames since the match, not stream frames.
    mode = _RecordingMode(frozenset({P}))
    traces: list[FrameTrace] = []
    engine = _tracker_engine(
        [_FailsOnFrames([0], [2, 3, 4])], mode, _refiner(P, hold=5), traces.append
    )
    engine.run(_ListSource(6))
    assert [t.index for t in traces if t.failed_closed] == [2, 3, 4]
    assert [(i, persons) for i, persons, _ in mode.seen] == [
        (0, (BOX_A,)),
        (1, (_held(BOX_A, 1),)),
        (5, (_held(BOX_A, 2),)),
    ]


def test_engine_rejects_the_refiner_without_a_detector_for_its_kind() -> None:
    motion_only = [_ResultOnly(frozenset({M}))]
    with pytest.raises(ValueError, match="consumes"):
        _tracker_engine(motion_only, _RecordingMode(frozenset({M, P})), _refiner(P))
    persons_only = [_PersonsOnFrames([])]
    with pytest.raises(ValueError, match="consumes"):
        _tracker_engine(persons_only, _RecordingMode(frozenset({P, FACES})), _refiner(FACES))


def test_engine_rejects_landmark_refiner_without_a_landmark_detector() -> None:
    mode = _RecordingMode(frozenset({FACES}))
    faces_only = _ResultOnly(frozenset({FACES}))
    assert _tracker_engine([faces_only], mode, _refiner(FACES)).refiners != ()
    with pytest.raises(ValueError, match="consumes"):
        _tracker_engine([faces_only], mode, _refiner(FACES, landmarks=True))


def test_engine_resets_the_refiner_when_reconfigured_with_new_detectors() -> None:
    mode = _RecordingMode(frozenset({P}))
    refiner = _refiner(P, hold=5)
    engine = _tracker_engine([_PersonsOnFrames([0])], mode, refiner)
    engine.open(INFO)
    engine.step(_frame(0))
    engine.step(_frame(1))
    assert mode.seen[-1][1] == (_held(BOX_A, 1),)  # Held.
    engine.reconfigure(detectors=[_PersonsOnFrames([])], refiners=[refiner])
    engine.step(_frame(2))
    engine.close()
    assert mode.seen[-1][:2] == (2, ())  # The hold from the old detector is gone.


def test_engine_resets_the_refiner_on_open() -> None:
    mode = _RecordingMode(frozenset({P}))
    refiner = _refiner(P, hold=5)
    refiner.refine(_frame(0), Detections(persons=(BOX_B,)))  # State from an earlier stream.
    engine = _tracker_engine([_PersonsOnFrames([])], mode, refiner)
    engine.run(_ListSource(2))
    assert [persons for _, persons, _ in mode.seen] == [(), ()]
