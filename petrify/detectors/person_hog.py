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
from typing import ClassVar

import cv2
import numpy as np

from petrify.core.config import PersonHogConfig
from petrify.core.detections import Box, DetectionKind, Detections
from petrify.core.frame import Frame

logger = logging.getLogger(__name__)


class PersonDetector:
    """Detect human presence in a frame using the OpenCV HOG + SVM descriptor.

    No external model files are required: OpenCV ships the default people
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
        accurate); 1.1 to 1.2 = coarser pyramid (faster).
    min_confidence:
        Minimum detection weight (SVM score) required to accept a box.
        Higher values reduce false positives at the cost of recall.
    min_box_area:
        Minimum bounding-box area in *downsampled* pixels.  Rejects tiny
        spurious detections caused by partial views at frame edges.
    """

    provides: ClassVar[frozenset[DetectionKind]] = frozenset({DetectionKind.PERSONS})
    stateful: ClassVar[bool] = False

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

    @classmethod
    def from_config(cls, cfg: PersonHogConfig) -> PersonDetector:
        """Build a detector from a :class:`PersonHogConfig` (padding keeps its default)."""
        return cls(
            downsample=cfg.downsample,
            win_stride=cfg.win_stride,
            scale=cfg.scale,
            min_confidence=cfg.min_confidence,
            min_box_area=cfg.min_box_area,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: Frame) -> Detections:
        """Return detections with only ``persons`` set (see :meth:`detect_boxes`)."""
        return Detections(persons=self.detect_boxes(frame.image))

    def reset(self) -> None:
        """No-op: the detector keeps no per-frame state."""

    def detect_boxes(self, frame: np.ndarray) -> tuple[Box, ...]:
        """Return every detection that passes the confidence and area filters.

        Boxes are scaled back to full-resolution coordinates using the actual
        resize ratio on each axis; ``score`` is the SVM weight. The result is
        non-empty exactly when :meth:`is_person` returns True.

        Raises
        ------
        ValueError
            If the frame is smaller than the HOG detection window.
        """
        small, boxes, weights = self._raw_detections(frame)
        frame_h, frame_w = frame.shape[:2]
        small_h, small_w = small.shape[:2]
        sx = frame_w / small_w
        sy = frame_h / small_h

        out: list[Box] = []
        for i, (x, y, w, h) in enumerate(boxes):
            confidence = float(weights[i])
            if self._passes(confidence, w * h):
                out.append(
                    Box(
                        x=int(round(x * sx)),
                        y=int(round(y * sy)),
                        w=int(round(w * sx)),
                        h=int(round(h * sy)),
                        score=confidence,
                    )
                )
        return tuple(out)

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

        Raises
        ------
        ValueError
            If the frame is smaller than the HOG detection window, in which
            case detection is impossible (OpenCV would crash the process).
        """
        _, boxes, weights = self._raw_detections(frame)

        if len(weights) == 0:
            logger.debug("PersonDetector: no detections")
            return False

        for i, (x, y, w, h) in enumerate(boxes):
            confidence = float(weights[i])
            area = w * h
            if self._passes(confidence, area):
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

    def _raw_detections(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Downsample *frame* and run HOG; return ``(small, boxes, weights)`` unfiltered.

        Raises ``ValueError`` before calling OpenCV when the downsampled frame is
        smaller than the HOG window.
        """
        small = self._maybe_downsample(frame)

        win_w, win_h = self._hog.winSize
        small_h, small_w = small.shape[:2]
        if small_w < win_w or small_h < win_h:
            raise ValueError(
                f"frame too small for person detection: {small_w}x{small_h} "
                f"(source {frame.shape[1]}x{frame.shape[0]}), minimum is "
                f"{win_w}x{win_h} (width x height)"
            )

        boxes, weights = self._hog.detectMultiScale(
            small,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale,
        )
        return small, boxes, weights

    def _passes(self, confidence: float, area: int) -> bool:
        """Return True when a raw detection clears both the confidence and area filters."""
        return confidence >= self.min_confidence and area >= self.min_box_area

    def _maybe_downsample(self, frame: np.ndarray) -> np.ndarray:
        if self.downsample >= 1.0:
            return frame
        h, w = frame.shape[:2]
        win_w, win_h = self._hog.winSize
        # One scale for both axes (keeps aspect ratio), raised just enough that
        # neither axis goes under the HOG window, and never above 1.0.
        scale = max(self.downsample, win_w / w, win_h / h)
        if scale >= 1.0:
            return frame
        # scale < 1.0 implies w > win_w and h > win_h, so these max() calls
        # only undo float rounding and never upscale past the source.
        new_w = max(win_w, int(w * scale))
        new_h = max(win_h, int(h * scale))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
