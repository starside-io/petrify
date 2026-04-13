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

"""Static background differencing — compare every frame against a locked reference.

Why this is needed
------------------
MOG2 is an *adaptive* background model: after ``history`` frames of a pixel
being static it absorbs that pixel into its background estimate.  A person who
walks in and then stands still for ~30 s will be silently absorbed, making MOG2
report zero motion even though the scene is clearly different from the empty
corridor.

This module takes the *locked* clean background frame (already determined by
the pipeline) and computes a straightforward absolute difference against every
incoming frame.  Because the reference never changes:

* A person standing still is **always** detected (they are always different
  from the empty-corridor reference).
* The check is cheap — one ``absdiff`` + threshold + contour area per frame.
* Lighting-noise is suppressed by a per-pixel grey threshold (default 25) and
  a minimum contour-area threshold (default 200 px² in downsampled space).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BackgroundDiffer:
    """Detect foreground objects by differencing against a locked reference frame.

    Parameters
    ----------
    reference:
        The clean background frame (BGR, any resolution).  It is downsampled
        internally alongside each query frame.
    pixel_threshold:
        Per-pixel grayscale absolute difference (0–255) above which a pixel is
        marked as changed.  25 suppresses sensor noise while catching even low-
        contrast foreground (dark clothing against a light corridor wall).
    area_threshold:
        Total changed-pixel contour area (in downsampled px²) above which the
        frame is classified as containing a foreground object.  200 is enough
        to catch a person 30+ m down a corridor without tripping on noise.
    downsample:
        Scale factor applied to both the reference and query frame before
        differencing.  Matches the MOG2 downsample so comparisons are apples-
        to-apples.
    """

    def __init__(
        self,
        reference: np.ndarray,
        pixel_threshold: int = 25,
        area_threshold: int = 200,
        downsample: float = 0.5,
    ) -> None:
        if not (0.0 < downsample <= 1.0):
            raise ValueError(f"downsample must be in (0, 1], got {downsample}")

        self.pixel_threshold = pixel_threshold
        self.area_threshold = area_threshold
        self.downsample = downsample

        h, w = reference.shape[:2]
        new_w = max(1, int(w * downsample))
        new_h = max(1, int(h * downsample))
        ref_small = cv2.resize(reference, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        self._ref_gray = cv2.cvtColor(ref_small, cv2.COLOR_BGR2GRAY)

        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        logger.debug(
            "BackgroundDiffer created: pixel_threshold=%d area_threshold=%d "
            "downsample=%.2f ref_size=%dx%d",
            pixel_threshold, area_threshold, downsample, new_w, new_h,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_different(self, frame: np.ndarray) -> bool:
        """Return True if *frame* differs significantly from the locked reference.

        Parameters
        ----------
        frame:
            Full-resolution BGR frame.

        Returns
        -------
        bool
            ``True`` when the changed-pixel contour area exceeds
            ``area_threshold``.
        """
        small = self._maybe_downsample(frame)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray, self._ref_gray)
        _, binary = cv2.threshold(diff, self.pixel_threshold, 255, cv2.THRESH_BINARY)

        # Opening removes isolated noise specks.
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = sum(cv2.contourArea(c) for c in contours)

        result = total_area > self.area_threshold
        logger.debug(
            "BackgroundDiffer: area=%.1f threshold=%d -> %s",
            total_area, self.area_threshold, result,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _maybe_downsample(self, frame: np.ndarray) -> np.ndarray:
        if self.downsample >= 1.0:
            return frame
        h, w = frame.shape[:2]
        new_w = max(1, int(w * self.downsample))
        new_h = max(1, int(h * self.downsample))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
