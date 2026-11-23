# Copyright 2026 petrify contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-License-Identifier: Apache-2.0

"""Greedy IoU box tracker that keeps ids stable and holds boxes through short misses.

:class:`TrackerRefiner` puts the tracker into the engine as a refiner.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from petrify.core.config import TrackerConfig
from petrify.core.detections import Box, DetectionKind, Detections, Face
from petrify.core.frame import Frame


@dataclass(frozen=True)
class Track:
    """One tracked subject as reported for a single frame.

    Attributes
    ----------
    track_id:
        Identifier assigned when the track was created. Ids start at 1, grow by
        one per new track, and are never reused until :meth:`IouTracker.reset`.
    box:
        The detection matched this frame, or the last matched box unchanged
        while the track is held. The tracker stores the box it was given and
        never sets :attr:`Box.held` itself; :class:`TrackerRefiner` does that.
    age:
        Frames since the track was created, counting the creation frame as 1.
    missed:
        Consecutive frames without a matching detection, 0 when matched this frame.
    """

    track_id: int
    box: Box
    age: int
    missed: int

    @property
    def held(self) -> bool:
        """True when the box is carried over from an earlier frame."""
        return self.missed > 0


def iou(a: Box, b: Box) -> float:
    """Return the intersection over union of two boxes, 0.0 when either is degenerate."""
    if a.w <= 0 or a.h <= 0 or b.w <= 0 or b.h <= 0:
        return 0.0
    ix = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
    iy = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union


class IouTracker:
    """Associate per-frame boxes into tracks by greedy IoU matching.

    Call :meth:`update` once per frame, in frame order, with that frame's
    detections. Each call returns every live track sorted by ascending
    ``track_id``.

    Association is greedy: all (track, detection) pairs with IoU at or above
    ``iou_threshold`` are visited from highest IoU to lowest, and a pair is
    accepted when neither side is already matched. Equal IoU values are broken
    by lower ``track_id`` first, then by lower detection index, so the same
    input sequence always yields the same ids.

    Every detection that matches no track starts a new track on the same frame.
    There is no confirmation delay, so a new subject is covered on its first frame.

    A track that finds no match keeps its last matched box (no motion
    extrapolation) and is returned for up to ``hold_frames`` consecutive
    frames, then dropped. A later detection overlapping a held track resumes
    it with the same id and resets ``missed`` to 0. With ``hold_frames=0`` the
    tracker is a pass-through that only adds ids.

    Boxes with non-positive width or height are ignored: they never match,
    never start a track, and are not returned, because IoU is undefined for them
    and they cover no pixels.

    ``max_tracks`` bounds the number of live tracks, but it never drops a box
    detected this frame. When the total would exceed the cap, held tracks are
    evicted first, the one with the most consecutive misses first (lower
    ``track_id`` breaks ties). If this frame's detections alone exceed the cap,
    all of them are still returned and every held track is evicted.

    Parameters
    ----------
    iou_threshold:
        Minimum IoU, in (0, 1], for a detection to continue a track.
    hold_frames:
        Number of consecutive missed frames a track is still returned for (>= 0).
    max_tracks:
        Soft upper bound on live tracks (> 0), enforced only by evicting held tracks.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        hold_frames: int = 0,
        max_tracks: int = 256,
    ) -> None:
        if not (0.0 < iou_threshold <= 1.0):
            raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
        if hold_frames < 0:
            raise ValueError(f"hold_frames must be >= 0, got {hold_frames}")
        if max_tracks <= 0:
            raise ValueError(f"max_tracks must be > 0, got {max_tracks}")

        self.iou_threshold = iou_threshold
        self.hold_frames = hold_frames
        self.max_tracks = max_tracks

        self._tracks: list[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        """Drop every track and restart ids at 1."""
        self._tracks = []
        self._next_id = 1

    def update(self, boxes: Sequence[Box]) -> tuple[Track, ...]:
        """Advance one frame with that frame's detections and return the live tracks.

        The result is sorted by ascending ``track_id``.
        """
        detections = [b for b in boxes if b.w > 0 and b.h > 0]
        tracks = self._tracks  # kept sorted by ascending track_id

        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(tracks):
            for di, det in enumerate(detections):
                score = iou(track.box, det)
                if score >= self.iou_threshold:
                    pairs.append((score, ti, di))
        # Highest IoU first; ties go to the lower track index (lower id), then
        # the lower detection index.
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))

        track_match: dict[int, int] = {}
        used_dets: set[int] = set()
        for _, ti, di in pairs:
            if ti in track_match or di in used_dets:
                continue
            track_match[ti] = di
            used_dets.add(di)

        detected: list[Track] = []
        held: list[Track] = []
        for ti, track in enumerate(tracks):
            if ti in track_match:
                detected.append(
                    Track(track.track_id, detections[track_match[ti]], track.age + 1, 0)
                )
            elif track.missed + 1 <= self.hold_frames:
                held.append(Track(track.track_id, track.box, track.age + 1, track.missed + 1))

        for di, det in enumerate(detections):
            if di not in used_dets:
                detected.append(Track(self._next_id, det, 1, 0))
                self._next_id += 1

        overflow = len(detected) + len(held) - self.max_tracks
        if overflow > 0:
            evict_order = sorted(held, key=lambda t: (-t.missed, t.track_id))
            evicted = {t.track_id for t in evict_order[:overflow]}
            held = [t for t in held if t.track_id not in evicted]

        self._tracks = sorted(detected + held, key=lambda t: t.track_id)
        return tuple(self._tracks)


def _aged(track: Track) -> Box:
    """The track's box with ``held`` counted from the last real detection."""
    return replace(track.box, held=track.box.held + track.missed)


class TrackerRefiner:
    """Engine refiner that keeps person boxes or faces covered through short misses.

    It feeds one kind of box to an :class:`IouTracker` on every frame it sees and
    returns only the boxes of held tracks, so the engine's union is "detected
    this frame plus held from earlier frames" without duplicating this frame's
    detections. A held box is the last matched box, with position and score
    unchanged and :attr:`Box.held` set to the matched box's own ``held`` plus the
    track's ``missed`` count. The count is cumulative so that a refiner placed
    after another one, which may match a box that was itself held, keeps the
    staleness instead of restarting it. It counts refined frames, not stream
    frames: see the note on failed frames below. There is no motion extrapolation, so when the camera or
    the subject moves the held box covers where the subject was, not where it is.

    ``kind`` selects what is tracked:

    - ``DetectionKind.PERSONS``: ``consumes = provides = {PERSONS}``. The result
      sets ``persons`` to the held boxes.
    - ``DetectionKind.FACES``: the result sets ``faces`` to held :class:`Face`
      objects. With ``landmarks=True``, ``consumes = provides = {FACES,
      LANDMARKS}``, so the engine only accepts the refiner when an active
      detector provides landmarks, and every held face carries the last
      landmarks seen for its track. With ``landmarks=False``, ``consumes =
      provides = {FACES}`` and held faces carry no landmarks, even when the
      detector provided some, so the result always passes
      :func:`petrify.core.detections.check_detections` for its ``provides``.

    A field that is None ("not computed") is treated as an empty tuple: the frame
    counts as a miss for every track, so held boxes keep ageing out instead of
    being held forever. When no track is held the result is an empty
    ``Detections()``, so the refiner never turns a "not computed" field into
    "nothing found".

    Boxes with non-positive width or height are ignored by the tracker, so a
    zero-area box (for example one clipped to the frame edge) is never held.

    The engine does not run refiners on a frame whose detection failed, so such
    frames do not age the tracks and are not counted in ``held``: a hold then
    lasts more wall-clock frames, which keeps a region covered longer. The input ``Detections`` is never mutated.

    Parameters
    ----------
    kind:
        ``DetectionKind.PERSONS`` or ``DetectionKind.FACES``.
    iou_threshold, hold_frames, max_tracks:
        Passed to the internal :class:`IouTracker`.
    landmarks:
        Faces only: True when the face detector provides landmarks.
    """

    def __init__(
        self,
        kind: DetectionKind,
        *,
        iou_threshold: float,
        hold_frames: int,
        max_tracks: int,
        landmarks: bool = False,
    ) -> None:
        if kind is DetectionKind.PERSONS:
            if landmarks:
                raise ValueError("landmarks=True is only valid for DetectionKind.FACES")
            kinds = frozenset({DetectionKind.PERSONS})
        elif kind is DetectionKind.FACES:
            kinds = (
                frozenset({DetectionKind.FACES, DetectionKind.LANDMARKS})
                if landmarks
                else frozenset({DetectionKind.FACES})
            )
        else:
            raise ValueError(
                f"kind must be DetectionKind.PERSONS or DetectionKind.FACES, got {kind!r}"
            )
        self.kind = kind
        self.landmarks = landmarks
        self.consumes: frozenset[DetectionKind] = kinds
        self.provides: frozenset[DetectionKind] = kinds
        self._tracker = IouTracker(
            iou_threshold=iou_threshold, hold_frames=hold_frames, max_tracks=max_tracks
        )
        # Last landmarks seen per live face track, keyed by track_id.
        self._payload: dict[int, tuple[tuple[float, float], ...] | None] = {}

    @classmethod
    def from_config(
        cls, kind: DetectionKind, config: TrackerConfig, *, landmarks: bool = False
    ) -> TrackerRefiner:
        """Build a refiner with every tracker setting taken from ``config``."""
        return cls(
            kind,
            iou_threshold=config.iou_threshold,
            hold_frames=config.hold_frames,
            max_tracks=config.max_tracks,
            landmarks=landmarks,
        )

    def reset(self) -> None:
        """Drop every track and every stored face payload."""
        self._tracker.reset()
        self._payload = {}

    def refine(self, frame: Frame, detections: Detections) -> Detections:
        """Advance the tracker one frame and return only the held boxes or faces."""
        if self.kind is DetectionKind.PERSONS:
            held = tuple(
                _aged(t)
                for t in self._tracker.update(detections.persons or ())
                if t.held
            )
            return Detections(persons=held) if held else Detections()
        return self._refine_faces(detections.faces or ())

    def _refine_faces(self, faces: tuple[Face, ...]) -> Detections:
        # The tracker stores the detection Box object itself, so a matched track
        # finds its face by the identity of its box.
        face_of_box = {id(face.box): face for face in faces}
        tracks = self._tracker.update([face.box for face in faces])
        payload: dict[int, tuple[tuple[float, float], ...] | None] = {}
        held: list[Face] = []
        for track in tracks:
            previous = self._payload.get(track.track_id)
            if track.held:
                points = previous
                if self.landmarks:
                    if points is None:
                        raise ValueError(
                            f"held face track {track.track_id} has no landmarks but "
                            "LANDMARKS is provided"
                        )
                    held.append(Face(box=_aged(track), landmarks=points))
                else:
                    held.append(Face(box=_aged(track)))
            elif self.landmarks:
                seen = face_of_box[id(track.box)].landmarks
                points = previous if seen is None else seen
            else:
                points = None
            payload[track.track_id] = points
        # Rebuilt from live tracks only, so payloads of dropped tracks are freed.
        self._payload = payload
        return Detections(faces=tuple(held)) if held else Detections()
