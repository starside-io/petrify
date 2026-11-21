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

    Field contract (enforced by :func:`check_detections`):

    - ``motion`` and ``background_diff``: ``bool`` or ``numpy.bool_``.
    - ``persons``: a tuple of :class:`Box` with integer coordinates, ``w >= 0``
      and ``h >= 0``.
    - ``faces``: a tuple of :class:`Face`, each with a valid box as above.
    - ``mask``: a numpy array of dtype ``bool`` with shape ``(H, W)``, at the full
      resolution of the frame it was computed on. True marks a subject pixel.
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
        (this side first), and ``mask`` is a pixelwise logical OR.

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
            or (self.mask is not None and bool(np.any(self.mask)))
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
    return np.logical_or(a, b)


def _is_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _box_problem(box: object) -> str | None:
    """Return why ``box`` is not a valid :class:`Box`, or None when it is."""
    if not isinstance(box, Box):
        return f"expected a Box, got {type(box).__name__}"
    for name in ("x", "y", "w", "h"):
        value = getattr(box, name)
        if not _is_int(value):
            return f"Box.{name} must be an integer, got {type(value).__name__}"
    if box.w < 0 or box.h < 0:
        return f"Box size must be non-negative, got w={box.w} h={box.h}"
    return None


def check_detections(
    result: object,
    provides: frozenset[DetectionKind],
    frame_shape: tuple[int, ...],
) -> Detections:
    """Return ``result`` unchanged if it is a well formed :class:`Detections`.

    ``provides`` is the set of kinds the detector declared; a field may be set only
    when a declared kind maps to it in :data:`FIELD_OF`. ``frame_shape`` is the
    shape of the frame the result was computed on; a mask must match its first two
    dimensions.

    Raises
    ------
    MalformedDetections
        With a message naming the first problem found.
    """
    if not isinstance(result, Detections):
        raise MalformedDetections(f"expected Detections, got {type(result).__name__}")
    allowed = {FIELD_OF[kind] for kind in provides}
    for name in ("motion", "persons", "background_diff", "faces", "mask"):
        if getattr(result, name) is not None and name not in allowed:
            raise MalformedDetections(
                f"field {name!r} is set but no provided kind maps to it"
            )

    for name in ("motion", "background_diff"):
        value = getattr(result, name)
        if value is not None and not isinstance(value, (bool, np.bool_)):
            raise MalformedDetections(f"{name} must be a bool, got {type(value).__name__}")

    persons = result.persons
    if persons is not None:
        if not isinstance(persons, tuple):
            raise MalformedDetections(f"persons must be a tuple, got {type(persons).__name__}")
        for i, box in enumerate(persons):
            problem = _box_problem(box)
            if problem is not None:
                raise MalformedDetections(f"persons[{i}]: {problem}")

    faces = result.faces
    if faces is not None:
        if not isinstance(faces, tuple):
            raise MalformedDetections(f"faces must be a tuple, got {type(faces).__name__}")
        need_landmarks = DetectionKind.LANDMARKS in provides
        for i, face in enumerate(faces):
            if not isinstance(face, Face):
                raise MalformedDetections(
                    f"faces[{i}]: expected a Face, got {type(face).__name__}"
                )
            problem = _box_problem(face.box)
            if problem is not None:
                raise MalformedDetections(f"faces[{i}].box: {problem}")
            if need_landmarks and face.landmarks is None:
                raise MalformedDetections(
                    f"faces[{i}] has no landmarks but LANDMARKS is provided"
                )

    mask = result.mask
    if mask is not None:
        if not isinstance(mask, np.ndarray):
            raise MalformedDetections(f"mask must be a numpy array, got {type(mask).__name__}")
        if mask.dtype != np.bool_:
            raise MalformedDetections(f"mask dtype must be bool, got {mask.dtype}")
        expected = tuple(frame_shape[:2])
        if mask.shape != expected:
            raise MalformedDetections(
                f"mask shape must be {expected} (frame H by W), got {mask.shape}"
            )
    return result
