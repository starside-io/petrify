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

"""Greedy IoU box tracker that keeps ids stable and holds boxes through short misses."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from petrify.core.detections import Box


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
        while the track is held.
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
