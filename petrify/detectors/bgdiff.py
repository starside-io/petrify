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
from typing import ClassVar

import cv2
import numpy as np

from petrify.core.config import BackgroundDiffConfig
from petrify.core.detections import DetectionKind, Detections
from petrify.core.frame import Frame
from petrify.core.plugin import BackgroundModel, BackgroundNotReady

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

    provides: ClassVar[frozenset[DetectionKind]] = frozenset({DetectionKind.BACKGROUND_DIFF})
    stateful: ClassVar[bool] = False

    def __init__(
        self,
        reference: np.ndarray,
        pixel_threshold: int = 25,
        area_threshold: int = 200,
        downsample: float = 0.5,
    ) -> None:
        self._configure(pixel_threshold, area_threshold, downsample)
        self._ref_gray: np.ndarray | None = self._reference_gray(reference)
        h, w = self._ref_gray.shape[:2]

        logger.debug(
            "BackgroundDiffer created: pixel_threshold=%d area_threshold=%d "
            "downsample=%.2f ref_size=%dx%d",
            pixel_threshold, area_threshold, downsample, w, h,
        )

    def _configure(self, pixel_threshold: int, area_threshold: int, downsample: float) -> None:
        """Validate and store the thresholds; no reference and no model yet."""
        if not (0.0 < downsample <= 1.0):
            raise ValueError(f"downsample must be in (0, 1], got {downsample}")

        self.pixel_threshold = pixel_threshold
        self.area_threshold = area_threshold
        self.downsample = downsample
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._model: BackgroundModel | None = None
        self._model_version: int | None = None

    def _reference_gray(self, reference: np.ndarray) -> np.ndarray:
        """Return the downsampled grayscale version of *reference*."""
        h, w = reference.shape[:2]
        new_w = max(1, int(w * self.downsample))
        new_h = max(1, int(h * self.downsample))
        ref_small = cv2.resize(reference, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(ref_small, cv2.COLOR_BGR2GRAY)

    @classmethod
    def from_model(cls, model: BackgroundModel, cfg: BackgroundDiffConfig) -> BackgroundDiffer:
        """Build a differ that compares against ``model.current()``.

        While the model is not ready, :meth:`detect` returns ``background_diff=None``
        (not computed) and :meth:`is_different` raises :class:`BackgroundNotReady`.
        The cached reference is recomputed only when ``model.version`` changes.
        """
        differ = cls.__new__(cls)
        differ._configure(cfg.pixel_threshold, cfg.area_threshold, cfg.downsample)
        differ._ref_gray = None
        differ._model = model
        logger.debug(
            "BackgroundDiffer created from model %s: pixel_threshold=%d area_threshold=%d "
            "downsample=%.2f",
            type(model).__name__, cfg.pixel_threshold, cfg.area_threshold, cfg.downsample,
        )
        return differ

    @classmethod
    def from_config(cls, reference: np.ndarray, cfg: BackgroundDiffConfig) -> BackgroundDiffer:
        """Build a differ against *reference* from a :class:`BackgroundDiffConfig`."""
        return cls(
            reference=reference,
            pixel_threshold=cfg.pixel_threshold,
            area_threshold=cfg.area_threshold,
            downsample=cfg.downsample,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: Frame) -> Detections:
        """Return detections with only ``background_diff`` set (see :meth:`is_different`).

        With a model that is not ready yet, ``background_diff`` is None (not computed).
        """
        ref = self._current_reference()
        if ref is None:
            return Detections(background_diff=None)
        return Detections(background_diff=self._compare(frame.image, ref))

    def reset(self) -> None:
        """No-op: the differ keeps no per-frame state."""

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

        Raises
        ------
        BackgroundNotReady
            When built with :meth:`from_model` and the model is not ready yet.
        """
        ref = self._current_reference()
        if ref is None:
            raise BackgroundNotReady("background model is not ready yet")
        return self._compare(frame, ref)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _current_reference(self) -> np.ndarray | None:
        """Return the cached reference, refreshed from the model when its version changed."""
        model = self._model
        if model is None:
            return self._ref_gray
        image = model.current()
        if image is None:
            return None
        version = model.version
        if self._ref_gray is None or version != self._model_version:
            self._ref_gray = self._reference_gray(image)
            self._model_version = version
        return self._ref_gray

    def _compare(self, frame: np.ndarray, ref_gray: np.ndarray) -> bool:
        """Return True if *frame* differs significantly from *ref_gray*."""
        small = self._maybe_downsample(frame)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray, ref_gray)
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

    def _maybe_downsample(self, frame: np.ndarray) -> np.ndarray:
        if self.downsample >= 1.0:
            return frame
        h, w = frame.shape[:2]
        new_w = max(1, int(w * self.downsample))
        new_h = max(1, int(h * self.downsample))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
