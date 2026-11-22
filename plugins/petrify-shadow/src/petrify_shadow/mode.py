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

"""The shadow mode: covers the frame while a subject is present.

This is still the skeleton render: it draws no style yet and covers the whole
frame black on every frame with a subject, which is the safe result when no
region is drawn. What it requires already follows the selected styles.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from petrify.core.detections import DetectionKind, Detections
from petrify.core.frame import Frame
from petrify.core.plugin import DetectPolicy
from petrify_shadow.options import FACE_STYLES, MASK_STYLES, ShadowParams

BASE_REQUIRES: frozenset[DetectionKind] = frozenset(
    {DetectionKind.MOTION, DetectionKind.PERSONS}
)
"""Presence signals the mode always requires, so a missed face turns black, not visible."""


FACE_KINDS: frozenset[DetectionKind] = frozenset(
    {DetectionKind.FACES, DetectionKind.LANDMARKS}
)
"""What any face style adds. Landmarks always come with faces, so the tracker holds them too."""


def requires_for(styles: frozenset[str]) -> frozenset[DetectionKind]:
    """Return the kinds a mode with ``styles`` requires: the base plus each group's kinds."""
    kinds = set(BASE_REQUIRES)
    if styles & FACE_STYLES:
        kinds |= FACE_KINDS
    if styles & MASK_STYLES:
        kinds.add(DetectionKind.MASK)
    return frozenset(kinds)


def mandatory_for(styles: frozenset[str]) -> frozenset[DetectionKind]:
    """Return the kinds a mode with ``styles`` cannot run without.

    Motion and persons stay optional in general: under the engine's lenient rule
    only the presence signals ``--detector`` builds run. Person boxes become
    mandatory only for bodybox, which has nothing to draw on without them.
    """
    kinds: set[DetectionKind] = set()
    if styles & FACE_STYLES:
        kinds |= FACE_KINDS
    if styles & MASK_STYLES:
        kinds.add(DetectionKind.MASK)
    if "bodybox" in styles:
        kinds.add(DetectionKind.PERSONS)
    return frozenset(kinds)


class ShadowMode:
    """Return the frame unchanged without a subject, a full black frame with one."""

    name: ClassVar[str] = "shadow"
    # A region mode needs every detector on every frame; "first_hit" would hand it
    # incomplete regions.
    detect_policy: ClassVar[DetectPolicy] = "all"

    def __init__(self, params: ShadowParams, *, rng: np.random.Generator | None = None) -> None:
        self.params = params
        # Per instance, since they follow the selected styles; the registry checks
        # that they equal what the factory reported for the same options.
        self.requires: frozenset[DetectionKind] = requires_for(params.styles)
        self.mandatory: frozenset[DetectionKind] = mandatory_for(params.styles)
        # No seed: the noise style must not be reproducible, so the generator draws
        # its seed from OS entropy. Only tests pass a seeded one.
        self._rng = rng if rng is not None else np.random.default_rng()

    def render(self, frame: Frame, detections: Detections, subject_present: bool) -> np.ndarray:
        """Return ``frame.image`` itself without a subject, else a new black image."""
        if not subject_present:
            return frame.image
        # A new array: the engine fails the frame closed when the output shares
        # memory with the input on a frame with a subject.
        return np.zeros_like(frame.image)
