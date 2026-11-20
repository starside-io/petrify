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

from dataclasses import dataclass, fields, replace
from enum import Enum

import numpy as np


class DetectionKind(str, Enum):
    """Kinds of detection a detector can provide and a mode can require."""

    MOTION = "motion"
    PERSONS = "persons"
    BACKGROUND_DIFF = "background_diff"
    FACES = "faces"
    LANDMARKS = "landmarks"
    MASK = "mask"


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
    """

    motion: bool | None = None
    persons: tuple[Box, ...] | None = None
    background_diff: bool | None = None
    faces: tuple[Face, ...] | None = None
    mask: np.ndarray | None = None

    def merged(self, other: Detections) -> Detections:
        """Return a copy where every non-None field of ``other`` overrides this one."""
        updates = {
            f.name: getattr(other, f.name)
            for f in fields(other)
            if getattr(other, f.name) is not None
        }
        return replace(self, **updates)

    @property
    def subject_present(self) -> bool:
        """True when motion, a person box, or a background difference was detected."""
        return bool(self.motion) or bool(self.persons) or bool(self.background_diff)
