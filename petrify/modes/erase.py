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

"""Erase mode: show the locked background while a subject is present."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from petrify.core.detections import DetectionKind, Detections
from petrify.core.frame import Frame


class EraseMode:
    """Show the locked background while a subject is present, the live frame otherwise.

    Parameters
    ----------
    background:
        The locked clean background frame, a 3-channel image (H x W x 3).
    """

    name: ClassVar[str] = "erase"
    requires: ClassVar[frozenset[DetectionKind]] = frozenset(
        {DetectionKind.MOTION, DetectionKind.PERSONS, DetectionKind.BACKGROUND_DIFF}
    )

    def __init__(self, background: np.ndarray) -> None:
        if background.ndim != 3 or background.shape[2] != 3:
            raise ValueError(
                f"background must be a 3-channel image (H x W x 3), got shape {background.shape}"
            )
        self.background = background

    def render(self, frame: Frame, detections: Detections, subject_present: bool) -> np.ndarray:
        """Return a copy of the background when *subject_present*, else ``frame.image``."""
        if frame.image.shape != self.background.shape:
            raise ValueError(
                f"frame shape {frame.image.shape} does not match background shape "
                f"{self.background.shape}"
            )
        if subject_present:
            return self.background.copy()
        return frame.image
