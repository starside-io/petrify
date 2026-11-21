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

"""Per-frame detection results produced by detectors and consumed by modes."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TypeVar

import numpy as np

_T = TypeVar("_T")


class MalformedDetections(ValueError):
    """A detection result breaks the :class:`Detections` contract."""


class DetectionKind(str, Enum):
    """Kinds of detection a detector can provide and a mode can require."""

    MOTION = "motion"
    PERSONS = "persons"
    BACKGROUND_DIFF = "background_diff"
    FACES = "faces"
    LANDMARKS = "landmarks"
    MASK = "mask"


FIELD_OF: Mapping[DetectionKind, str] = MappingProxyType({
    DetectionKind.MOTION: "motion",
    DetectionKind.PERSONS: "persons",
    DetectionKind.BACKGROUND_DIFF: "background_diff",
    DetectionKind.FACES: "faces",
    # Landmarks live on Face.landmarks, so they travel in the faces field.
    DetectionKind.LANDMARKS: "faces",
    DetectionKind.MASK: "mask",
})
"""The :class:`Detections` field that carries each kind of detection."""


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in full-resolution pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    score: float | None = None


@dataclass(frozen=True)
class Face:
    """A detected face with optional landmark points."""

    box: Box
    landmarks: tuple[tuple[float, float], ...] | None = None


@dataclass(frozen=True, eq=False)
class Detections:
    """Detection results for one frame.

    Every field defaults to None, and None always means "not computed", never
    "nothing found". An empty tuple or False means the detector ran and found nothing.

    Field contract (enforced by :func:`check_detections`, which returns a new,
    normalized ``Detections`` rather than the detector's own object):

    - ``motion`` and ``background_diff``: ``bool`` or ``numpy.bool_`` on input,
      Python ``bool`` after the check.
    - ``persons``: a tuple of :class:`Box` with integer coordinates, ``w >= 0``
      and ``h >= 0``, and ``score`` None or a finite real number. Detectors may
      return boxes that start before 0 or extend past the frame edge; the check
      clips them to the frame. A box wholly outside the frame becomes a zero-area
      box on the nearest edge and is never dropped.
    - ``faces``: a tuple of :class:`Face`, each with a box as above (clipped the
      same way) and ``landmarks`` None or a non-empty tuple of ``(x, y)`` float
      points. Landmarks are set exactly when LANDMARKS is provided. Landmark
      points are not clipped: they are geometry hints, and consumers must
      tolerate points outside the frame.
    - ``mask``: a numpy array of dtype ``bool`` with shape ``(H, W)``, at the full
      resolution of the frame it was computed on. True marks a subject pixel.

    Results that have passed the check hold only immutable values: tuples of
    frozen dataclasses and a read-only, C-contiguous copy of the mask. A mask
    produced by :meth:`merged` from two masks is read-only as well.
    """

    motion: bool | None = None
    persons: tuple[Box, ...] | None = None
    background_diff: bool | None = None
    faces: tuple[Face, ...] | None = None
    mask: np.ndarray | None = None

    def merged(self, other: Detections) -> Detections:
        """Return the union of this result and ``other``, mutating neither.

        A None field yields to the other side, and the result then holds the other
        side's object itself. When both sides are set: ``motion`` and
        ``background_diff`` are OR-ed, ``persons`` and ``faces`` are concatenated
        (this side first), and ``mask`` is a pixelwise logical OR in a new,
        read-only array.

        Raises
        ------
        MalformedDetections
            If both sides carry a mask and the shapes differ.
        """
        return Detections(
            motion=_union(self.motion, other.motion, _or_flag),
            persons=_union(self.persons, other.persons, _concat),
            background_diff=_union(self.background_diff, other.background_diff, _or_flag),
            faces=_union(self.faces, other.faces, _concat),
            mask=_union(self.mask, other.mask, _or_mask),
        )

    @property
    def subject_present(self) -> bool:
        """True when motion, a person, a background difference, a face, or a mask pixel."""
        return (
            bool(self.motion)
            or bool(self.persons)
            or bool(self.background_diff)
            or bool(self.faces)
            or (self.mask is not None and bool(np.asarray(self.mask).any()))
        )


def _union(a: _T | None, b: _T | None, combine: Callable[[_T, _T], _T]) -> _T | None:
    if a is None:
        return b
    if b is None:
        return a
    return combine(a, b)


def _or_flag(a: bool, b: bool) -> bool:
    return bool(a) or bool(b)


def _concat(a: tuple[_T, ...], b: tuple[_T, ...]) -> tuple[_T, ...]:
    return (*a, *b)


def _or_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise MalformedDetections(
            f"cannot merge masks of different shapes {a.shape} and {b.shape}"
        )
    out = np.logical_or(a, b)
    out.setflags(write=False)
    return out


def _is_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _finite_float(value: object) -> float | None:
    """Return ``value`` as a finite Python float, or None if it is not a finite real."""
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(
        value, (bool, np.bool_)
    ):
        return None
    try:
        out = float(value)
    except OverflowError:
        return None
    return out if math.isfinite(out) else None


def _clip(start: int, length: int, limit: int) -> tuple[int, int]:
    """Clip the span ``[start, start + length)`` into ``[0, limit]``; return (start, length)."""
    lo = min(max(start, 0), limit)
    hi = min(max(start + length, 0), limit)
    return lo, hi - lo


def _checked_box(box: object, where: str, height: int, width: int) -> Box:
    """Return ``box`` rebuilt as a base :class:`Box` clipped to the frame."""
    if not isinstance(box, Box):
        raise MalformedDetections(f"{where}: expected a Box, got {type(box).__name__}")
    coords: list[int] = []
    for name in ("x", "y", "w", "h"):
        value = getattr(box, name)
        if not _is_int(value):
            raise MalformedDetections(
                f"{where}: Box.{name} must be an integer, got {type(value).__name__}"
            )
        coords.append(int(value))
    x, y, w, h = coords
    if w < 0 or h < 0:
        raise MalformedDetections(f"{where}: Box size must be non-negative, got w={w} h={h}")
    score: float | None = None
    if box.score is not None:
        score = _finite_float(box.score)
        if score is None:
            raise MalformedDetections(
                f"{where}: Box.score must be None or a finite real number, got {box.score!r}"
            )
    x, w = _clip(x, w, width)
    y, h = _clip(y, h, height)
    return Box(x=x, y=y, w=w, h=h, score=score)


def _checked_landmarks(landmarks: object, where: str) -> tuple[tuple[float, float], ...]:
    """Return ``landmarks`` as a non-empty tuple of ``(float, float)`` points."""
    if not isinstance(landmarks, (tuple, list)) or type(landmarks) not in (tuple, list):
        raise MalformedDetections(
            f"{where}: landmarks must be a tuple or list, got {type(landmarks).__name__}"
        )
    if not landmarks:
        raise MalformedDetections(f"{where}: landmarks must not be empty")
    points: list[tuple[float, float]] = []
    for j, point in enumerate(landmarks):
        if type(point) is np.ndarray:
            if point.ndim != 1:
                raise MalformedDetections(
                    f"{where}: landmarks[{j}] must be a 1-d array, got shape {point.shape}"
                )
        elif type(point) not in (tuple, list):
            raise MalformedDetections(
                f"{where}: landmarks[{j}] must be a 2-sequence, got {type(point).__name__}"
            )
        if len(point) != 2:
            raise MalformedDetections(
                f"{where}: landmarks[{j}] must have 2 coordinates, got {len(point)}"
            )
        px, py = _finite_float(point[0]), _finite_float(point[1])
        if px is None or py is None:
            raise MalformedDetections(
                f"{where}: landmarks[{j}] must hold finite real numbers, "
                f"got ({point[0]!r}, {point[1]!r})"
            )
        points.append((px, py))
    return tuple(points)


def check_detections(
    result: object,
    provides: frozenset[DetectionKind],
    frame_shape: tuple[int, ...],
) -> Detections:
    """Validate ``result`` and return a new, normalized base :class:`Detections`.

    The input object is never returned: the output is built only from validated
    values (Python bools, tuples of base :class:`Box` and :class:`Face` with boxes
    clipped to the frame, float landmark points, and a read-only C-contiguous copy
    of the mask), so nothing downstream holds an object the detector can still
    mutate. ``result`` may be a ``Detections`` subclass; only the five known
    fields are read.

    ``provides`` is the set of :class:`DetectionKind` values the detector declared;
    a field may be set only when a declared kind maps to it in :data:`FIELD_OF`.
    ``frame_shape`` is the shape of the frame the result was computed on (height
    first); boxes are clipped to it and a mask must match its first two dimensions.

    Raises
    ------
    MalformedDetections
        With a message naming the first problem found.
    """
    if not isinstance(result, Detections):
        raise MalformedDetections(f"expected Detections, got {type(result).__name__}")
    try:
        kinds = list(provides)
    except TypeError as exc:
        raise MalformedDetections(f"provides must be a set of DetectionKind: {exc}") from exc
    for kind in kinds:
        if not isinstance(kind, DetectionKind):
            raise MalformedDetections(f"unknown detection kind in provides: {kind!r}")
    allowed = {FIELD_OF[kind] for kind in kinds}
    for name in ("motion", "persons", "background_diff", "faces", "mask"):
        if getattr(result, name) is not None and name not in allowed:
            raise MalformedDetections(
                f"field {name!r} is set but no provided kind maps to it"
            )
    height, width = int(frame_shape[0]), int(frame_shape[1])

    flags: dict[str, bool | None] = {}
    for name in ("motion", "background_diff"):
        value = getattr(result, name)
        if value is not None and not isinstance(value, (bool, np.bool_)):
            raise MalformedDetections(f"{name} must be a bool, got {type(value).__name__}")
        flags[name] = None if value is None else bool(value)

    persons_in = result.persons
    persons: tuple[Box, ...] | None = None
    if persons_in is not None:
        if type(persons_in) is not tuple:
            raise MalformedDetections(
                f"persons must be a tuple, got {type(persons_in).__name__}"
            )
        persons = tuple(
            _checked_box(box, f"persons[{i}]", height, width)
            for i, box in enumerate(persons_in)
        )

    faces_in = result.faces
    faces: tuple[Face, ...] | None = None
    if faces_in is not None:
        if type(faces_in) is not tuple:
            raise MalformedDetections(f"faces must be a tuple, got {type(faces_in).__name__}")
        need_landmarks = DetectionKind.LANDMARKS in kinds
        out_faces: list[Face] = []
        for i, face in enumerate(faces_in):
            where = f"faces[{i}]"
            if not isinstance(face, Face):
                raise MalformedDetections(f"{where}: expected a Face, got {type(face).__name__}")
            box = _checked_box(face.box, f"{where}.box", height, width)
            landmarks: tuple[tuple[float, float], ...] | None = None
            if face.landmarks is None:
                if need_landmarks:
                    raise MalformedDetections(
                        f"{where} has no landmarks but LANDMARKS is provided"
                    )
            elif not need_landmarks:
                raise MalformedDetections(
                    f"{where} has landmarks but LANDMARKS is not provided"
                )
            else:
                landmarks = _checked_landmarks(face.landmarks, where)
            out_faces.append(Face(box=box, landmarks=landmarks))
        faces = tuple(out_faces)

    mask_in = result.mask
    mask: np.ndarray | None = None
    if mask_in is not None:
        if type(mask_in) is not np.ndarray:
            raise MalformedDetections(
                f"mask must be a numpy.ndarray, got {type(mask_in).__name__}"
            )
        if mask_in.dtype != np.bool_:
            raise MalformedDetections(f"mask dtype must be bool, got {mask_in.dtype}")
        expected = (height, width)
        if mask_in.shape != expected:
            raise MalformedDetections(
                f"mask shape must be {expected} (frame H by W), got {mask_in.shape}"
            )
        mask = np.array(mask_in, dtype=np.bool_, order="C", copy=True)
        mask.setflags(write=False)

    return Detections(
        motion=flags["motion"],
        persons=persons,
        background_diff=flags["background_diff"],
        faces=faces,
        mask=mask,
    )
