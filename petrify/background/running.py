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

"""Running background strategy: rolling per-pixel median of recent clean frames."""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from petrify.core.frame import Frame


def median_image(samples: np.ndarray) -> np.ndarray:
    """Return the per-pixel median (uint8) of ``samples`` stacked on axis 0.

    For an even number of samples this is the lower of the two middle values, so
    the result is always one of the sampled pixel values. ``samples`` is not modified.
    """
    k = (samples.shape[0] - 1) // 2
    # Copy so the result does not keep the whole partitioned stack alive.
    return np.partition(samples, k, axis=0)[k].copy()


def _check_positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")


class RunningBackground:
    """Rolling median of recent frames the engine classified as clean.

    Only frames with ``subject_present == False`` are candidates. The first candidate
    and then every ``interval``-th one is copied into a ring buffer of at most
    ``window`` samples. Each time a sample is added and at least ``min_samples`` are
    stored, the background is recomputed as the per-pixel median and ``version``
    increases. ``current`` is None until the first median exists.

    With ``recompute_in_background=False`` (the default) the median is computed
    inside :meth:`update`, which is deterministic but costs hundreds of milliseconds
    per recompute at camera resolutions. With True, :meth:`update` only stores the
    sample and the median is computed on a short-lived daemon thread, at most one at
    a time: samples that arrive during a recompute trigger exactly one follow-up
    recompute on the newest samples. ``current`` then lags the newest samples by one
    recompute, and stays None until the first recompute has finished. An exception
    in the recompute is stored and raised by every later :meth:`update`. Use
    :meth:`wait_idle` to wait for pending recomputes. :meth:`update` itself must be
    called from one thread at a time.

    Known limitation: a subject who stands still long enough to be classified as
    absent by the detectors enters the samples. The median resists this until such
    frames are the majority of the window, after which the subject becomes part of
    the background.
    """

    def __init__(
        self,
        window: int = 30,
        interval: int = 5,
        min_samples: int = 5,
        *,
        recompute_in_background: bool = False,
    ) -> None:
        _check_positive("window", window)
        _check_positive("interval", interval)
        _check_positive("min_samples", min_samples)
        if min_samples > window:
            raise ValueError(f"min_samples ({min_samples}) must be <= window ({window})")
        self.window = window
        self.interval = interval
        self.min_samples = min_samples
        self.recompute_in_background = recompute_in_background
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        # Each stored sample is a private copy that is never mutated afterwards, so
        # a snapshot of this deque is safe to read without the lock.
        self._samples: deque[np.ndarray] = deque(maxlen=window)
        self._shape: tuple[int, ...] | None = None
        self._candidates = 0
        self._current: np.ndarray | None = None
        self._version = 0
        self._running = False
        self._dirty = False
        self._error: Exception | None = None
        self._thread: threading.Thread | None = None

    @property
    def version(self) -> int:
        """Increases every time :meth:`current` starts returning a new image."""
        with self._lock:
            return self._version

    @property
    def samples(self) -> int:
        """Number of samples currently stored."""
        with self._lock:
            return len(self._samples)

    def current(self) -> np.ndarray | None:
        """Return the latest published median, or None before the first one exists."""
        with self._lock:
            return self._current

    def update(self, frame: Frame, subject_present: bool) -> None:
        """Consider ``frame`` as a sample; recompute the median when one is added.

        Raises the stored exception if an earlier background recompute failed.
        """
        image = frame.image
        if self._shape is not None and image.shape != self._shape:
            raise ValueError(
                f"frame shape {image.shape} does not match background sample shape "
                f"{self._shape}"
            )
        with self._lock:
            if self._error is not None:
                raise self._error
        if subject_present:
            return
        take = self._candidates % self.interval == 0
        self._candidates += 1
        if not take:
            return
        sample = image.copy()
        if self._shape is None:
            self._shape = image.shape

        if not self.recompute_in_background:
            with self._lock:
                self._samples.append(sample)
                snapshot = list(self._samples)
            if len(snapshot) >= self.min_samples:
                median = median_image(np.stack(snapshot))
                with self._lock:
                    self._current = median
                    self._version += 1
            return

        with self._lock:
            self._samples.append(sample)
            if len(self._samples) < self.min_samples:
                return
            if self._running:
                self._dirty = True
                return
            self._running = True
            previous = self._thread
            thread = threading.Thread(
                target=self._recompute_loop, name="petrify-running-background", daemon=True
            )
            self._thread = thread
        if previous is not None:
            # It already cleared _running, so it is about to exit: keep one alive.
            previous.join()
        thread.start()

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until no recompute is running or pending; return False on timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._idle:
            if not self._idle.wait_for(lambda: not self._running, timeout):
                return False
            thread = self._thread
        if thread is not None:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            thread.join(remaining)
            if thread.is_alive():
                return False
        return True

    def _recompute_loop(self) -> None:
        """Recompute until no new sample arrived during the last recompute."""
        while True:
            with self._lock:
                snapshot = list(self._samples)
                self._dirty = False
            try:
                median = median_image(np.stack(snapshot))
            except Exception as exc:
                with self._lock:
                    self._error = exc
                    self._running = False
                    self._dirty = False
                    self._idle.notify_all()
                return
            with self._lock:
                self._current = median
                self._version += 1
                if not self._dirty:
                    self._running = False
                    self._idle.notify_all()
                    return
