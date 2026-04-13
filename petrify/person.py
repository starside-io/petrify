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

"""Human presence detection using OpenCV's built-in HOG person detector."""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PersonDetector:
    """Detect human presence in a frame using the OpenCV HOG + SVM descriptor.

    No external model files are required — OpenCV ships the default people
    detector weights.  The detector is optimised for upright pedestrians and
    works on both colour and grayscale frames.

    Parameters
    ----------
    downsample:
        Scale factor (0 < x <= 1) applied before detection.  0.5 (half
        resolution) gives a large speed-up with minimal accuracy loss for
        full-HD footage.
    win_stride:
        HOG sliding-window step size ``(dx, dy)`` in pixels (evaluated in
        *downsampled* space).  Smaller = more thorough but slower.
    padding:
        HOG window padding ``(px, py)`` in pixels.
    scale:
        Image pyramid scale factor.  1.05 = fine pyramid (slower, more
        accurate); 1.1–1.2 = coarser pyramid (faster).
    min_confidence:
        Minimum detection weight (SVM score) required to accept a box.
        Higher values reduce false positives at the cost of recall.
    min_box_area:
        Minimum bounding-box area in *downsampled* pixels.  Rejects tiny
        spurious detections caused by partial views at frame edges.
    """

    def __init__(
        self,
        downsample: float = 0.5,
        win_stride: tuple[int, int] = (8, 8),
        padding: tuple[int, int] = (4, 4),
        scale: float = 1.05,
        min_confidence: float = 0.3,
        min_box_area: int = 1500,
    ) -> None:
        if not (0.0 < downsample <= 1.0):
            raise ValueError(f"downsample must be in (0, 1], got {downsample}")

        self.downsample = downsample
        self.win_stride = win_stride
        self.padding = padding
        self.scale = scale
        self.min_confidence = min_confidence
        self.min_box_area = min_box_area

        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        logger.debug(
            "PersonDetector created: downsample=%.2f win_stride=%s padding=%s "
            "scale=%.2f min_confidence=%.2f min_box_area=%d",
            downsample,
            win_stride,
            padding,
            scale,
            min_confidence,
            min_box_area,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_person(self, frame: np.ndarray) -> bool:
        """Return True if at least one person is detected in *frame*.

        Parameters
        ----------
        frame:
            Full-resolution BGR or grayscale frame.

        Returns
        -------
        bool
            ``True`` when a confident person detection is found.
        """
        small = self._maybe_downsample(frame)

        boxes, weights = self._hog.detectMultiScale(
            small,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale,
        )

        if len(weights) == 0:
            logger.debug("PersonDetector: no detections")
            return False

        for i, (x, y, w, h) in enumerate(boxes):
            confidence = float(weights[i])
            area = w * h
            if confidence >= self.min_confidence and area >= self.min_box_area:
                logger.debug(
                    "PersonDetector: person detected box=(%d,%d,%d,%d) "
                    "confidence=%.3f area=%d",
                    x, y, w, h, confidence, area,
                )
                return True

        logger.debug(
            "PersonDetector: %d raw detections rejected (below confidence/area thresholds)",
            len(weights),
        )
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _maybe_downsample(self, frame: np.ndarray) -> np.ndarray:
        if self.downsample >= 1.0:
            return frame
        h, w = frame.shape[:2]
        new_w = max(64, int(w * self.downsample))
        new_h = max(64, int(h * self.downsample))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
