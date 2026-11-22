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

from collections.abc import Mapping
from typing import ClassVar

import numpy as np

from petrify.background.locked import LockedBackground
from petrify.core.config import OptionValue
from petrify.core.detections import DetectionKind, Detections
from petrify.core.frame import Frame
from petrify.core.plugin import (
    BackgroundModel,
    BackgroundNotReady,
    DetectPolicy,
    Mode,
    ModeContext,
    OptionSpec,
)


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
    detect_policy: ClassVar[DetectPolicy] = "first_hit"

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


class _EraseFactory:
    """The :class:`ModeFactory` of the built-in erase mode (see :data:`ERASE`)."""

    name = "erase"
    summary = "Show the clean background while a subject is present, the live frame otherwise."
    options: tuple[OptionSpec, ...] = ()
    needs_background = True
    # The ROI stage pastes the live timestamp over the frozen background, as wanted.
    roi_overlay = True

    def requires(self, options: Mapping[str, OptionValue]) -> frozenset[DetectionKind]:
        """Return :attr:`EraseMode.requires`."""
        return EraseMode.requires

    def mandatory(self, options: Mapping[str, OptionValue]) -> frozenset[DetectionKind]:
        """Return the empty set: erase has no mandatory kind."""
        return frozenset()

    def weakened(self, options: Mapping[str, OptionValue]) -> tuple[str, ...]:
        """Return nothing: erase has no option that reduces protection."""
        return ()

    def create(self, ctx: ModeContext, previous: Mode | None = None) -> Mode:
        """Return an :class:`EraseMode` around ``ctx.background``, or ``previous``.

        ``require_background_diff`` is True exactly when a background differ is
        configured and the background is an online model (the differ is then
        model-backed), as the pipeline decides it. ``previous`` is returned when it
        is an :class:`EraseMode` over the same background object with the same
        ``require_background_diff``. Raises ``ValueError`` without a background.
        """
        background = ctx.background
        if background is None:
            raise ValueError("the erase mode needs a background image or model")
        require = ctx.config.background_diff is not None and not isinstance(
            background, np.ndarray
        )
        if (
            isinstance(previous, EraseMode)
            and previous.require_background_diff == require
            and _same_background(previous, background)
        ):
            return previous
        return EraseMode(background, require_background_diff=require)


def _same_background(mode: EraseMode, background: np.ndarray | BackgroundModel) -> bool:
    """True when ``mode`` renders from ``background`` itself."""
    if isinstance(background, np.ndarray):
        # EraseMode wraps an image in a LockedBackground of its own.
        return isinstance(mode.model, LockedBackground) and mode.model.current() is background
    return mode.model is background


ERASE = _EraseFactory()
"""The built-in erase mode factory, registered as ``erase``."""


def _check_three_channel(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"background must be a 3-channel image (H x W x 3), got shape {image.shape}"
        )
