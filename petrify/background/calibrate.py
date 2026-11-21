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

"""Calibrate background strategy: median of the first frames while the user is out of view."""

from __future__ import annotations

import logging
import math

import numpy as np

from petrify.background.running import (
    SampleFilter,
    _check_positive,
    _SampleVetting,
    median_image,
)
from petrify.core.frame import Frame

logger = logging.getLogger(__name__)

MAX_SAMPLES = 30
REJECTION_WARNING_EVERY = 100


def frames_for(seconds: float, fps: float, min_samples: int = 5) -> int:
    """Return the calibration length in frames for ``seconds`` at ``fps``, at least ``min_samples``."""
    if not seconds > 0:
        raise ValueError(f"seconds must be > 0, got {seconds}")
    if not fps > 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    _check_positive("min_samples", min_samples)
    return max(min_samples, math.ceil(seconds * fps))


class CalibratedBackground:
    """Background built once from the first ``frames`` accepted updates, then fixed.

    ``subject_present`` is ignored: the user is expected to be out of view. With no
    ``sample_filter`` every update is accepted, so the first ``frames`` updates are
    used. With a filter, it is called with the image of every update until the
    model is ready, on the calling thread; a False result or an exception rejects
    the frame, which then does not advance the calibration (the first filter
    exception is logged as a warning, none propagates). Calibration therefore lasts
    until ``frames`` accepted frames were seen, and never ends while every frame is
    rejected; a warning is logged every 100 consecutive rejections so that this is
    visible. At most 30 accepted frames, spread evenly, are kept. After the last one
    the per-pixel median is computed, ``version`` becomes 1, and the model never
    changes again. ``current`` is None until then.

    Known limitation: nothing but the filter checks the calibration frames. A
    person in view during calibration is absorbed into the background (and shown
    on frozen frames after they leave) unless ``sample_filter`` rejects those
    frames. With no filter nothing checks them.
    """

    def __init__(
        self, frames: int, min_samples: int = 5, sample_filter: SampleFilter | None = None
    ) -> None:
        _check_positive("frames", frames)
        _check_positive("min_samples", min_samples)
        if min_samples > MAX_SAMPLES:
            raise ValueError(f"min_samples must be <= {MAX_SAMPLES}, got {min_samples}")
        if frames < min_samples:
            raise ValueError(f"frames ({frames}) must be >= min_samples ({min_samples})")
        self.frames = frames
        self.min_samples = min_samples
        self.version = 0
        count = min(frames, MAX_SAMPLES)
        picks = np.linspace(0, frames - 1, count).round().astype(int)
        self._picks = {int(i): slot for slot, i in enumerate(picks)}
        self._buffer: np.ndarray | None = None
        self._seen = 0
        self._current: np.ndarray | None = None
        self._vetting = _SampleVetting(sample_filter, "CalibratedBackground")
        self._consecutive_rejections = 0

    @property
    def ready(self) -> bool:
        """True once the background has been computed."""
        return self._current is not None

    @property
    def remaining(self) -> int:
        """Accepted calibration frames still needed before the background is ready."""
        return self.frames - self._seen

    @property
    def rejected(self) -> int:
        """Number of frames the sample filter rejected (including filter errors)."""
        return self._vetting.rejected

    def current(self) -> np.ndarray | None:
        """Return the calibrated background, or None during calibration."""
        return self._current

    def update(self, frame: Frame, subject_present: bool) -> None:
        """Collect ``frame`` during calibration if accepted; no-op once ready."""
        if self._current is not None:
            return
        image = frame.image
        if self._buffer is not None and image.shape != self._buffer.shape[1:]:
            raise ValueError(
                f"frame shape {image.shape} does not match calibration sample shape "
                f"{self._buffer.shape[1:]}"
            )
        if not self._vetting.accepts(image):
            self._consecutive_rejections += 1
            if self._consecutive_rejections % REJECTION_WARNING_EVERY == 0:
                logger.warning(
                    "CalibratedBackground: calibration is waiting for a scene without a "
                    "person (%d consecutive frames rejected, %d accepted frames still "
                    "needed)",
                    self._consecutive_rejections,
                    self.remaining,
                )
            return
        self._consecutive_rejections = 0
        if self._buffer is None:
            self._buffer = np.empty((len(self._picks), *image.shape), dtype=image.dtype)
        slot = self._picks.get(self._seen)
        if slot is not None:
            self._buffer[slot] = image
        self._seen += 1
        if self._seen == self.frames:
            self._current = median_image(self._buffer)
            self._buffer = None
            self.version = 1
