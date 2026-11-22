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

This is the skeleton. It has no styles yet, so it covers the whole frame black on
every frame with a subject, which is the safe result when no region is drawn.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from petrify.core.detections import DetectionKind, Detections
from petrify.core.frame import Frame
from petrify.core.plugin import DetectPolicy

BASE_REQUIRES: frozenset[DetectionKind] = frozenset(
    {DetectionKind.MOTION, DetectionKind.PERSONS}
)
"""Presence signals the mode always requires, so a missed face turns black, not visible."""


class ShadowMode:
    """Return the frame unchanged without a subject, a full black frame with one."""

    name: ClassVar[str] = "shadow"
    requires: ClassVar[frozenset[DetectionKind]] = BASE_REQUIRES
    mandatory: ClassVar[frozenset[DetectionKind]] = frozenset()
    # A region mode needs every detector on every frame; "first_hit" would hand it
    # incomplete regions.
    detect_policy: ClassVar[DetectPolicy] = "all"

    def render(self, frame: Frame, detections: Detections, subject_present: bool) -> np.ndarray:
        """Return ``frame.image`` itself without a subject, else a new black image."""
        if not subject_present:
            return frame.image
        # A new array: the engine fails the frame closed when the output shares
        # memory with the input on a frame with a subject.
        return np.zeros_like(frame.image)
