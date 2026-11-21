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

"""Webcam frame source."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from types import TracebackType
from typing import Any

import cv2

from petrify.core.frame import Frame, SourceInfo
from petrify.sources.stream import _LatestFrameReader, _usable_fps

logger = logging.getLogger(__name__)


class CameraSource:
    """Live frames from a webcam by index via ``cv2.VideoCapture``.

    Requested ``width``, ``height`` and ``fps`` are passed to the driver, and
    ``info`` reports what the driver actually delivers. A capture thread keeps
    only the newest frame; frames the consumer is too slow to take are dropped
    and counted in ``dropped``.
    """

    def __init__(
        self,
        index: int = 0,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        max_read_failures: int = 30,
        capture_factory: Callable[[int], Any] = cv2.VideoCapture,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_read_failures < 1:
            raise ValueError("max_read_failures must be at least 1")
        cap = capture_factory(index)
        if not cap.isOpened():
            cap.release()
            raise OSError(f"Cannot open camera {index}")

        if width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps is not None:
            cap.set(cv2.CAP_PROP_FPS, fps)

        # Drivers silently ignore requests they cannot satisfy: trust the read-back.
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not _usable_fps(actual_fps):
            actual_fps = fps if fps is not None and _usable_fps(fps) else 30.0
        uri = f"camera:{index}"
        self.info = SourceInfo(
            fps=actual_fps,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=None,
            seekable=False,
            live=True,
            uri=uri,
        )
        self._reader = _LatestFrameReader(
            cap,
            self.info,
            label=uri,
            max_read_failures=max_read_failures,
            clock=clock,
        )
        logger.info(
            "Input: %s  |  %dx%d @ %.2f fps  |  live",
            uri,
            self.info.width,
            self.info.height,
            self.info.fps,
        )

    @property
    def dropped(self) -> int:
        """Frames captured but never yielded."""
        return self._reader.dropped

    def __iter__(self) -> Iterator[Frame]:
        """Yield the newest frame each time one is available, until the device stops or closes."""
        return self._reader.frames()

    def close(self) -> None:
        """Stop capturing and release the device; safe to call more than once."""
        self._reader.close()

    def __enter__(self) -> CameraSource:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
