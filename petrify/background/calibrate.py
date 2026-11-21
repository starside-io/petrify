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

import math

import numpy as np

from petrify.background.running import _check_positive, median_image
from petrify.core.frame import Frame

MAX_SAMPLES = 30


def frames_for(seconds: float, fps: float, min_samples: int = 5) -> int:
    """Return the calibration length in frames for ``seconds`` at ``fps``, at least ``min_samples``."""
    if not seconds > 0:
        raise ValueError(f"seconds must be > 0, got {seconds}")
    if not fps > 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    _check_positive("min_samples", min_samples)
    return max(min_samples, math.ceil(seconds * fps))


class CalibratedBackground:
    """Background built once from the first ``frames`` updates, then fixed.

    Every one of the first ``frames`` updates counts, whatever ``subject_present``
    says: the user is expected to be out of view. At most 30 of them, spread evenly,
    are kept. After the last one the per-pixel median is computed, ``version``
    becomes 1, and the model never changes again. ``current`` is None until then.
    """

    def __init__(self, frames: int, min_samples: int = 5) -> None:
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

    @property
    def ready(self) -> bool:
        """True once the background has been computed."""
        return self._current is not None

    @property
    def remaining(self) -> int:
        """Calibration frames still to be fed before the background is ready."""
        return self.frames - self._seen

    def current(self) -> np.ndarray | None:
        """Return the calibrated background, or None during calibration."""
        return self._current

    def update(self, frame: Frame, subject_present: bool) -> None:
        """Collect ``frame`` during calibration; no-op once ready."""
        if self._current is not None:
            return
        image = frame.image
        if self._buffer is None:
            self._buffer = np.empty((len(self._picks), *image.shape), dtype=image.dtype)
        elif image.shape != self._buffer.shape[1:]:
            raise ValueError(
                f"frame shape {image.shape} does not match calibration sample shape "
                f"{self._buffer.shape[1:]}"
            )
        slot = self._picks.get(self._seen)
        if slot is not None:
            self._buffer[slot] = image
        self._seen += 1
        if self._seen == self.frames:
            self._current = median_image(self._buffer)
            self._buffer = None
            self.version = 1
