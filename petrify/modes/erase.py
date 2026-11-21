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

"""Erase mode: show the clean background while a subject is present."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from petrify.background.locked import LockedBackground
from petrify.core.detections import DetectionKind, Detections
from petrify.core.frame import Frame
from petrify.core.plugin import BackgroundModel, BackgroundNotReady


class EraseMode:
    """Show the background while a subject is present, the live frame otherwise.

    Parameters
    ----------
    background:
        Either a fixed clean background frame, a 3-channel image (H x W x 3), which
        is wrapped in a :class:`LockedBackground`, or a :class:`BackgroundModel`.
        The model is updated with every rendered frame and its final decision.
    require_background_diff:
        When True, a frame on which no detector fired and the background differ did
        not run (``detections.background_diff is None``) is failed closed with
        :class:`BackgroundNotReady` instead of passing through. This happens to
        frames detected while the model was not ready but rendered after it became
        ready (detection runs ``lookahead`` frames ahead of rendering). Set it
        whenever a model-backed differ (``BackgroundDiffer.from_model``) is in the
        detector list. A frame where another detector fired is not affected: the
        engine skips the differ for it and the frame is frozen anyway.
    """

    name: ClassVar[str] = "erase"
    requires: ClassVar[frozenset[DetectionKind]] = frozenset(
        {DetectionKind.MOTION, DetectionKind.PERSONS, DetectionKind.BACKGROUND_DIFF}
    )

    def __init__(
        self,
        background: np.ndarray | BackgroundModel,
        *,
        require_background_diff: bool = False,
    ) -> None:
        if isinstance(background, np.ndarray):
            _check_three_channel(background)
            background = LockedBackground(background)
        self.model: BackgroundModel = background
        self.require_background_diff = require_background_diff

    @property
    def background(self) -> np.ndarray | None:
        """The model's current background image, or None when it is not ready."""
        return self.model.current()

    def render(self, frame: Frame, detections: Detections, subject_present: bool) -> np.ndarray:
        """Return a copy of the background when *subject_present*, else ``frame.image``.

        While the model has no background, raise :class:`BackgroundNotReady` for
        every frame (the engine then fails it closed), and likewise for a frame the
        differ did not vouch for when ``require_background_diff`` is set. The model
        is updated with the frame in every case.
        """
        model = self.model
        bg = model.current()
        if bg is None:
            model.update(frame, subject_present)
            raise BackgroundNotReady("background model is not ready yet")
        _check_three_channel(bg)
        if frame.image.shape != bg.shape:
            raise ValueError(
                f"frame shape {frame.image.shape} does not match background shape {bg.shape}"
            )
        if (
            self.require_background_diff
            and detections.background_diff is None
            and not detections.subject_present
        ):
            model.update(frame, subject_present)
            raise BackgroundNotReady(
                "no detector fired and the background differ did not run on this frame"
            )
        out = bg.copy() if subject_present else frame.image
        model.update(frame, subject_present)
        return out


def _check_three_channel(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"background must be a 3-channel image (H x W x 3), got shape {image.shape}"
        )
