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

"""ROI crop-and-paste compositor for the live timestamp overlay."""

from __future__ import annotations

import logging

import numpy as np

from petrify.core.frame import Frame
from petrify.roi_detect import WarmupRoiDetector

logger = logging.getLogger(__name__)


class Compositor:
    """Composite a live ROI region from the current frame onto a background.

    When *roi* is ``None`` the compositor is a pass-through: ``compose``
    simply returns the background unchanged.

    Parameters
    ----------
    roi:
        A ``(x, y, w, h)`` tuple describing the bounding box of the region
        to keep live (e.g. a timestamp overlay burned into the video).
        Coordinates are in pixels relative to the full-resolution frame.
        Pass ``None`` to disable ROI compositing entirely.
    """

    def __init__(self, roi: tuple[int, int, int, int] | None) -> None:
        self.roi = roi
        if roi is not None:
            x, y, w, h = roi
            if w <= 0 or h <= 0:
                raise ValueError(f"ROI width and height must be positive, got w={w} h={h}")
            logger.debug("Compositor: roi=(x=%d y=%d w=%d h=%d)", x, y, w, h)
        else:
            logger.debug("Compositor: roi=None (pass-through mode)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_roi(self, frame: np.ndarray) -> np.ndarray | None:
        """Crop the ROI region from *frame*.

        Parameters
        ----------
        frame:
            Full-resolution BGR frame.

        Returns
        -------
        numpy.ndarray or None
            Cropped ROI patch, or ``None`` if no ROI is configured or the
            bounding box falls entirely outside the frame.
        """
        if self.roi is None:
            return None

        x, y, w, h = self.roi
        fh, fw = frame.shape[:2]

        # Clamp to frame bounds to avoid index errors on edge ROIs.
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(fw, x + w)
        y2 = min(fh, y + h)

        if x2 <= x1 or y2 <= y1:
            logger.warning(
                "extract_roi: ROI (%d,%d,%d,%d) is outside frame (%dx%d); skipping.",
                x, y, w, h, fw, fh,
            )
            return None

        patch = frame[y1:y2, x1:x2].copy()
        logger.debug(
            "extract_roi: extracted patch shape=%s from frame shape=%s",
            patch.shape,
            frame.shape,
        )
        return patch

    def compose(
        self,
        background: np.ndarray,
        live_roi: np.ndarray | None,
    ) -> np.ndarray:
        """Paste *live_roi* onto *background* at the configured ROI position.

        If *live_roi* is ``None`` or no ROI is configured, *background* is
        returned as-is (no copy unless a paste is performed).

        Parameters
        ----------
        background:
            The frozen (or current) background frame to paste onto.
        live_roi:
            The live ROI patch extracted from the real current frame.

        Returns
        -------
        numpy.ndarray
            The composited output frame (may be *background* itself if no
            paste was necessary).
        """
        if self.roi is None or live_roi is None:
            return background

        x, y, w, h = self.roi
        fh, fw = background.shape[:2]

        # Destination slice — clamp to frame bounds.
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(fw, x + w)
        y2 = min(fh, y + h)

        if x2 <= x1 or y2 <= y1:
            logger.warning(
                "compose: ROI (%d,%d,%d,%d) is outside background (%dx%d); skipping paste.",
                x, y, w, h, fw, fh,
            )
            return background

        # The extracted patch may be smaller than (w, h) if it was clamped.
        patch_h = y2 - y1
        patch_w = x2 - x1

        output = background.copy()

        # Guard against shape mismatch between the patch and destination slice.
        src = live_roi[:patch_h, :patch_w]
        output[y1:y2, x1:x2] = src

        logger.debug(
            "compose: pasted roi patch shape=%s at (%d,%d)", src.shape, x1, y1
        )
        return output


class RoiOverlay:
    """Stage that pastes the live ROI from the real frame over the mode's output.

    Parameters
    ----------
    roi:
        ``(x, y, w, h)`` of the region to keep live, or ``None`` to return the
        image unchanged.
    """

    def __init__(self, roi: tuple[int, int, int, int] | None) -> None:
        self.roi = roi
        self._compositor = Compositor(roi)

    def apply(self, frame: Frame, image: np.ndarray) -> np.ndarray:
        """Return *image* with the ROI region taken from ``frame.image``."""
        return self._compositor.compose(image, self._compositor.extract_roi(frame.image))


class WarmupRoiOverlay:
    """Stage that detects the ROI on a warmup window, then behaves like :class:`RoiOverlay`.

    While the detector is not done, every frame's image is fed to it and the mode's
    output is returned unchanged: nothing from the live frame is pasted during
    warmup, so a frozen frame shows the frozen timestamp. That is the safe
    direction, since pasting an unverified region could expose part of a person.
    Once the detector is done the stage locks onto ``RoiOverlay(detector.roi)``
    (identity when the ROI is None), starting with the frame that completed it.

    Stages see frames in release order after the lookahead, which is still stream
    order, so feeding the detector from here is valid.
    """

    def __init__(self, detector: WarmupRoiDetector) -> None:
        self._detector = detector
        self._overlay: RoiOverlay | None = None

    @property
    def locked(self) -> bool:
        """True once the detector is done and the ROI is fixed."""
        return self._overlay is not None

    @property
    def roi(self) -> tuple[int, int, int, int] | None:
        """The locked ROI, or None while warming up or when nothing was found."""
        return None if self._overlay is None else self._overlay.roi

    def apply(self, frame: Frame, image: np.ndarray) -> np.ndarray:
        """Feed the detector during warmup (returning *image*), then paste the live ROI."""
        overlay = self._overlay
        if overlay is None:
            detector = self._detector
            if not detector.done:
                detector.update(frame.image)
                if not detector.done:
                    return image
            overlay = self._lock(detector.roi)
        return overlay.apply(frame, image)

    def _lock(self, roi: tuple[int, int, int, int] | None) -> RoiOverlay:
        """Create the overlay for *roi* and log it once."""
        overlay = RoiOverlay(roi)
        self._overlay = overlay
        if roi is None:
            logger.info("Warmup ROI detection found nothing; the ROI overlay is disabled.")
        else:
            logger.info("Auto ROI detected: x=%d y=%d w=%d h=%d", *roi)
        return overlay
