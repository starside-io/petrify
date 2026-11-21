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

"""Network stream frame source (RTSP, HTTP MJPEG) and the shared live-capture reader."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Iterator
from types import TracebackType
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

from petrify.core.frame import Frame, SourceInfo

logger = logging.getLogger(__name__)

_RETRY_DELAY = 0.005  # Seconds to wait after a failed read before retrying.
_JOIN_TIMEOUT = 2.0  # Seconds close() waits for the capture thread to exit.


def redact_url(url: str) -> str:
    """Return ``url`` with userinfo replaced by ``***`` and the query and fragment dropped.

    ``rtsp://user:secret@host:554/path?token=abc`` becomes ``rtsp://***@host:554/path``.
    A string without a scheme and network location (not a URL) is returned unchanged.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        # Malformed URL: it may still carry credentials, so reveal nothing.
        return "<unparseable url>"
    if not parts.scheme or not parts.netloc:
        return url
    netloc = parts.netloc
    if "@" in netloc:
        netloc = "***@" + netloc.rsplit("@", 1)[1]
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _usable_fps(value: float) -> bool:
    return math.isfinite(value) and value > 0


class _LatestFrameReader:
    """Capture thread that keeps only the most recent frame, plus the consumer side.

    The thread reads ``capture`` in a loop and publishes each frame into a single
    slot guarded by ``_cond``. Publishing over a frame the consumer has not taken
    yet overwrites it and counts it as dropped, so latency never accumulates.
    The consumer (``frames``) waits on the same condition for a new frame, the
    end of the stream, an error, or ``close``.
    """

    def __init__(
        self,
        capture: Any,
        info: SourceInfo,
        *,
        label: str,
        max_read_failures: int,
        clock: Callable[[], float],
    ) -> None:
        self._capture = capture
        self._info = info
        self._label = label
        self._max_read_failures = max_read_failures
        self._clock = clock

        self._cond = threading.Condition()
        # State below is guarded by _cond.
        self._latest: tuple[np.ndarray, float] | None = None
        self._stopped = False  # Capture thread finished (end of stream or error).
        self._closed = False
        self._error: BaseException | None = None
        self._dropped = 0
        self._iterating = False
        self._next_index = 0

        self._closing = threading.Event()  # Interrupts the retry wait in the thread.
        self._release_lock = threading.Lock()
        self._released = False
        self._thread: threading.Thread | None = None
        self._start_time = 0.0

    @property
    def dropped(self) -> int:
        """Frames captured but never yielded."""
        with self._cond:
            return self._dropped

    def frames(self) -> Iterator[Frame]:
        """Return an iterator over live frames; only one may be active at a time."""
        with self._cond:
            if self._iterating:
                raise RuntimeError(f"{self._label} is already being iterated")
            self._iterating = True
        try:
            self._start()
        except BaseException:
            with self._cond:
                self._iterating = False
            raise
        return _FrameIterator(self)

    def _start(self) -> None:
        with self._cond:
            if self._thread is not None or self._closed:
                return
            self._start_time = self._clock()
            self._thread = threading.Thread(
                target=self._run, name=f"petrify-capture {self._label}", daemon=True
            )
            self._thread.start()

    def _run(self) -> None:
        failures = 0
        try:
            while not self._closing.is_set():
                try:
                    ok, image = self._capture.read()
                except BaseException as exc:  # Surfaced in the consumer.
                    with self._cond:
                        self._error = exc
                        self._stopped = True
                        self._cond.notify_all()
                    return
                if not ok or image is None:
                    failures += 1
                    if failures >= self._max_read_failures:
                        logger.warning(
                            "%s: stopping after %d consecutive failed reads",
                            self._label,
                            failures,
                        )
                        with self._cond:
                            self._stopped = True
                            self._cond.notify_all()
                        return
                    self._closing.wait(_RETRY_DELAY)
                    continue
                failures = 0
                # OpenCV backends may hand out a buffer they reuse on the next
                # read(), so publish a private copy that nothing else touches.
                frame_image = np.array(image, copy=True)
                timestamp = self._clock() - self._start_time
                with self._cond:
                    if self._latest is not None:
                        self._dropped += 1
                    self._latest = (frame_image, timestamp)
                    self._cond.notify_all()
        finally:
            with self._cond:
                self._stopped = True
                self._cond.notify_all()
            if self._closing.is_set():
                # close() may have given up joining; releasing here keeps the
                # capture from being released while read() is still running.
                self._release()

    def _take(self) -> tuple[np.ndarray, float] | None:
        """Block until a new frame is available and take it; None when the stream is over."""
        with self._cond:
            while self._latest is None and not self._stopped and not self._closed:
                self._cond.wait()
            if self._closed:
                return None
            if self._latest is not None:
                item = self._latest
                self._latest = None
                return item
            if self._error is not None:
                raise self._error
            return None

    def _next_frame(self) -> Frame | None:
        item = self._take()
        if item is None:
            return None
        image, timestamp = item
        info = self._info
        expected = (info.height, info.width, 3)
        if image.shape != expected:
            with self._cond:
                self._dropped += 1
            raise ValueError(
                f"{self._label}: frame shape {image.shape} does not match the source "
                f"({expected}); the device may have changed resolution"
            )
        with self._cond:
            index = self._next_index
            self._next_index += 1
        return Frame(image=image, index=index, timestamp=timestamp, source=info)

    def _end_iteration(self) -> None:
        with self._cond:
            self._iterating = False

    def _release(self) -> None:
        with self._release_lock:
            if self._released:
                return
            self._released = True
        self._capture.release()

    def close(self) -> None:
        """Stop the thread, release the capture once, and end any waiting iteration."""
        self._closing.set()
        with self._cond:
            if self._closed:
                return
            self._closed = True
            if self._latest is not None:
                self._dropped += 1
                self._latest = None
            thread = self._thread
            self._cond.notify_all()
        if thread is None:
            self._release()
            return
        if thread is not threading.current_thread():
            thread.join(_JOIN_TIMEOUT)
        if thread.is_alive():
            logger.warning(
                "%s: capture thread did not stop within %.1f s; it releases the capture on exit",
                self._label,
                _JOIN_TIMEOUT,
            )
            return
        self._release()


class _FrameIterator:
    """Iterator handed out by ``_LatestFrameReader.frames``; frees the single iteration slot."""

    def __init__(self, reader: _LatestFrameReader) -> None:
        self._reader: _LatestFrameReader | None = reader

    def __iter__(self) -> _FrameIterator:
        return self

    def __next__(self) -> Frame:
        reader = self._reader
        if reader is None:
            raise StopIteration
        try:
            frame = reader._next_frame()
        except BaseException:
            self.close()
            raise
        if frame is None:
            self.close()
            raise StopIteration
        return frame

    def close(self) -> None:
        """Give up the iteration slot; safe to call more than once."""
        reader, self._reader = self._reader, None
        if reader is not None:
            reader._end_iteration()

    def __del__(self) -> None:
        self.close()


class StreamSource:
    """Live frames from a network stream (RTSP, HTTP MJPEG) via ``cv2.VideoCapture``.

    A capture thread keeps only the newest frame; frames the consumer is too slow
    to take are dropped and counted in ``dropped``. Credentials in ``url`` never
    appear in ``info``, logs, or exception messages.
    """

    def __init__(
        self,
        url: str,
        max_read_failures: int = 30,
        capture_factory: Callable[[str], Any] = cv2.VideoCapture,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_read_failures < 1:
            raise ValueError("max_read_failures must be at least 1")
        redacted = redact_url(url)
        cap = capture_factory(url)
        if not cap.isOpened():
            cap.release()
            raise OSError(f"Cannot open stream {redacted}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        self.info = SourceInfo(
            fps=fps if _usable_fps(fps) else 25.0,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=None,
            seekable=False,
            live=True,
            uri=redacted,
        )
        self._reader = _LatestFrameReader(
            cap,
            self.info,
            label=f"stream {redacted}",
            max_read_failures=max_read_failures,
            clock=clock,
        )
        logger.info(
            "Input: %s  |  %dx%d @ %.2f fps  |  live",
            redacted,
            self.info.width,
            self.info.height,
            self.info.fps,
        )

    @property
    def dropped(self) -> int:
        """Frames captured but never yielded."""
        return self._reader.dropped

    def __iter__(self) -> Iterator[Frame]:
        """Yield the newest frame each time one is available, until the stream ends or closes."""
        return self._reader.frames()

    def close(self) -> None:
        """Stop capturing and release the stream; safe to call more than once."""
        self._reader.close()

    def __enter__(self) -> StreamSource:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
