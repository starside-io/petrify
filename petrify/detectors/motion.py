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

"""Motion detection via OpenCV MOG2 background subtraction."""

from __future__ import annotations

import logging
from typing import ClassVar

import cv2
import numpy as np

from petrify.core.config import MotionConfig
from petrify.core.detections import DetectionKind, Detections
from petrify.core.frame import Frame

logger = logging.getLogger(__name__)


class MotionDetector:
    """Detect motion in video frames using the MOG2 background subtractor.

    Parameters
    ----------
    history:
        Number of frames used to build the background model.
    var_threshold:
        Mahalanobis distance threshold for deciding whether a pixel is
        well-described by the background model.
    detect_shadows:
        If True, MOG2 also detects and marks shadow pixels (grey in the
        foreground mask).  This slightly increases CPU cost.
    threshold:
        Total foreground-contour area (in pixels) above which a frame is
        considered to contain motion.  Evaluated *after* downsampling, so
        set accordingly.
    downsample:
        Float in (0, 1].  Frames are scaled by this factor before being fed
        to the subtractor to speed up processing.  The contour areas are
        measured in the downsampled coordinate space, so keep that in mind
        when choosing ``threshold``.
    """

    provides: ClassVar[frozenset[DetectionKind]] = frozenset({DetectionKind.MOTION})
    stateful: ClassVar[bool] = True

    def __init__(
        self,
        history: int = 200,
        var_threshold: float = 16.0,
        detect_shadows: bool = False,
        threshold: int = 500,
        downsample: float = 0.5,
    ) -> None:
        if not (0.0 < downsample <= 1.0):
            raise ValueError(f"downsample must be in (0, 1], got {downsample}")

        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.threshold = threshold
        self.downsample = downsample

        self._subtractor: cv2.BackgroundSubtractorMOG2 = self._make_subtractor()
        logger.debug(
            "MotionDetector created: history=%d var_threshold=%.1f "
            "detect_shadows=%s threshold=%d downsample=%.2f",
            history,
            var_threshold,
            detect_shadows,
            threshold,
            downsample,
        )

    @classmethod
    def from_config(cls, cfg: MotionConfig) -> MotionDetector:
        """Build a detector from a :class:`MotionConfig`."""
        return cls(
            history=cfg.history,
            var_threshold=cfg.var_threshold,
            detect_shadows=cfg.detect_shadows,
            threshold=cfg.threshold,
            downsample=cfg.downsample,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: Frame) -> Detections:
        """Return detections with only ``motion`` set (see :meth:`is_motion`)."""
        return Detections(motion=self.is_motion(frame.image))

    def motion_score(self, frame: np.ndarray) -> float:
        """Return raw foreground contour area without thresholding.

        Useful for background scanning — compare scores across candidate frames
        to pick the one with the least detectable motion.
        """
        small = self._maybe_downsample(frame)
        fg_mask = self._subtractor.apply(small)
        _, binary = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return float(sum(cv2.contourArea(c) for c in contours))

    def is_motion(self, frame: np.ndarray) -> bool:
        """Return True if *frame* contains motion above the pixel-area threshold.

        Parameters
        ----------
        frame:
            BGR frame as a ``numpy.ndarray`` with shape ``(H, W, 3)``.

        Returns
        -------
        bool
            ``True`` when significant foreground contour area is detected.
        """
        small = self._maybe_downsample(frame)
        fg_mask = self._subtractor.apply(small)

        # MOG2 marks shadows as 127 when detect_shadows=True.
        # Threshold to pure binary: foreground=255, everything else=0.
        _, binary = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological opening to remove tiny noise specks before measuring area.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_area = sum(cv2.contourArea(c) for c in contours)
        motion = total_area > self.threshold

        logger.debug(
            "is_motion: total_contour_area=%.1f threshold=%d -> %s",
            total_area,
            self.threshold,
            motion,
        )
        return motion

    def reset(self) -> None:
        """Discard the background model and start fresh."""
        self._subtractor = self._make_subtractor()
        logger.debug("MotionDetector reset: background model cleared.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_subtractor(self) -> cv2.BackgroundSubtractorMOG2:
        return cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows,
        )

    def _maybe_downsample(self, frame: np.ndarray) -> np.ndarray:
        if self.downsample >= 1.0:
            return frame
        h, w = frame.shape[:2]
        new_w = max(1, int(w * self.downsample))
        new_h = max(1, int(h * self.downsample))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
