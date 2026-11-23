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
import re
import threading
import time
from collections.abc import Callable, Iterator
from types import TracebackType
from typing import Any, TypeVar

import cv2
import numpy as np

from petrify.core.frame import Frame, SourceInfo

logger = logging.getLogger(__name__)

_RETRY_DELAY = 0.005  # Seconds to wait after a failed read before retrying.
_JOIN_TIMEOUT = 2.0  # Seconds close() waits for the capture thread to exit.

REDACTED = "<redacted>"
_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*")
_HOST_PORT = re.compile(
    r"(?P<host>\[[0-9A-Fa-f:.]+(?:%[A-Za-z0-9._~%-]+)?\]|[A-Za-z0-9._~%-]+)"
    r"(?::(?P<port>[0-9]{1,5})?)?"
)
_AUTHORITY_END = re.compile(r"[/?#\\]")

_T = TypeVar("_T")


def redact_url(url: str) -> str:
    """Return a form of ``url`` that identifies only its scheme, host and port.

    ``rtsp://user:secret@host:554/path?token=abc`` becomes ``rtsp://***@host:554``.
    The path, parameters, query and fragment are always dropped, since they can
    carry tokens, and ``***@`` marks that userinfo was present. The authority
    ends at the first ``/``, ``?``, ``#`` or ``\\`` after ``scheme://``. When
    every ``@`` lies before that point, the host follows the last ``@`` (so a
    password may contain ``@``). When any ``@`` lies at or after it, the input
    is ambiguous (a password with an unencoded delimiter, or an ``@`` in the
    path, query or fragment) and the result is ``scheme://<redacted>``: the host
    is hidden, never guessed. The scheme is lowercased and surrounding
    whitespace is stripped.

    A string without ``://`` and without ``@`` (a device path, a file name) is
    returned unchanged. Anything else that cannot be parsed confidently becomes
    ``"<redacted>"``. Never raises.
    """
    try:
        return _redact(url)
    except Exception:
        return REDACTED


def _redact(url: str) -> str:
    text = url.strip()
    scheme, sep, rest = text.partition("://")
    if not sep:
        return REDACTED if "@" in text else text
    if not _SCHEME.fullmatch(scheme):
        return REDACTED
    end = _AUTHORITY_END.search(rest)
    d = end.start() if end else len(rest)
    last_at = rest.rfind("@")
    if last_at >= d:
        # Ambiguous: a password holding a delimiter, or an "@" in the path, query or
        # fragment. Guessing the host could publish secret text, so hide it.
        return f"{scheme.lower()}://{REDACTED}"
    authority = rest[last_at + 1 : d]
    match = _HOST_PORT.fullmatch(authority)
    if match is None:
        return REDACTED
    result = f"{scheme.lower()}://"
    if last_at >= 0:
        result += "***@"
    result += match["host"]
    if match["port"]:
        result += f":{match['port']}"
    return result


def _guarded(action: Callable[[], _T], message: str) -> _T:
    """Run ``action``; turn an ``Exception`` into ``OSError`` that names only its type.

    The original exception is neither the cause nor the context of the
    ``OSError``, so its message (which may hold a credential-bearing URL) is not
    reachable from it. ``KeyboardInterrupt`` and ``SystemExit`` pass through.
    """
    failure = ""
    try:
        return action()
    except Exception as exc:
        failure = type(exc).__name__
    # Raised outside the except block so the original is not attached as context.
    raise OSError(f"{message} ({failure})") from None


def _release_quietly(capture: Any, label: str) -> None:
    failure = ""
    try:
        capture.release()
    except Exception as exc:
        failure = type(exc).__name__
    if failure:
        logger.warning("%s: releasing the capture failed (%s)", label, failure)


def _open_live(
    open_capture: Callable[[], Any],
    *,
    label: str,
    open_message: str,
    max_read_failures: int,
    configure: Callable[[Any], None] | None = None,
) -> tuple[Any, float, int, int]:
    """Open a live capture and return it with its reported fps, width and height.

    Every error is an ``OSError`` built by ``_guarded``. When the reported width
    or height is not positive, frames are read (up to ``max_read_failures``
    failed reads) until one succeeds and the size is taken from it; that probe
    frame is discarded. The capture is released when anything here fails.
    """
    cap = _guarded(open_capture, open_message)
    try:
        if not _guarded(cap.isOpened, open_message):
            raise OSError(open_message)
        if configure is not None:
            _guarded(lambda: configure(cap), open_message)
        fps = float(_guarded(lambda: cap.get(cv2.CAP_PROP_FPS), open_message))
        width = int(_guarded(lambda: cap.get(cv2.CAP_PROP_FRAME_WIDTH), open_message))
        height = int(_guarded(lambda: cap.get(cv2.CAP_PROP_FRAME_HEIGHT), open_message))
        if width <= 0 or height <= 0:
            height, width = _probe_size(cap, label=label, max_read_failures=max_read_failures)
    except BaseException:
        _release_quietly(cap, label)
        raise
    return cap, fps, width, height


def _probe_size(cap: Any, *, label: str, max_read_failures: int) -> tuple[int, int]:
    """Read until a frame arrives and return its (height, width); the frame is discarded."""
    message = f"{label}: cannot read a frame to determine the frame size"
    for attempt in range(max_read_failures):
        if attempt:
            time.sleep(_RETRY_DELAY)
        ok, image = _guarded(cap.read, message)
        if ok and image is not None and getattr(image, "ndim", 0) >= 2:
            return int(image.shape[0]), int(image.shape[1])
    raise OSError(f"{message} after {max_read_failures} failed reads")


def _check_read_timeout(read_timeout: float | None) -> None:
    if read_timeout is not None and not (math.isfinite(read_timeout) and read_timeout > 0):
        raise ValueError(f"read_timeout must be a finite number above 0, got {read_timeout}")


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
        join_timeout: float = _JOIN_TIMEOUT,
        read_timeout: float | None = None,
    ) -> None:
        self._capture = capture
        self._info = info
        self._label = label
        self._max_read_failures = max_read_failures
        self._clock = clock
        self._join_timeout = join_timeout
        self._read_timeout = read_timeout

        self._cond = threading.Condition()
        # State below is guarded by _cond.
        self._latest: tuple[np.ndarray, float] | None = None
        self._stopped = False  # Capture thread finished (end of stream or error).
        self._closed = False
        # A read error is kept as its message only, never the original exception,
        # whose text may hold the URL. KeyboardInterrupt and SystemExit are kept
        # as is and re-raised untouched.
        self._error_message: str | None = None
        self._error: BaseException | None = None
        self._dropped = 0
        self._iterating = False
        self._next_index = 0
        # time.monotonic() when the last frame was published (or the thread
        # started), for read_timeout. Not ``clock``: that one makes the frame
        # timestamps and may be a fake in tests.
        self._last_arrival = 0.0
        self._timeout_message: str | None = None

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
            self._last_arrival = time.monotonic()
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
                except Exception as exc:  # Surfaced in the consumer, by type name only.
                    with self._cond:
                        self._error_message = f"{self._label}: read failed ({type(exc).__name__})"
                        self._stopped = True
                        self._cond.notify_all()
                    return
                except BaseException as exc:  # KeyboardInterrupt, SystemExit: surfaced as is.
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
                    self._last_arrival = time.monotonic()
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
        """Block until a new frame is available and take it; None when the stream is over.

        With ``read_timeout``, raises ``OSError`` when no frame has arrived for that
        many seconds (since the last frame, or since the thread started).
        """
        with self._cond:
            while self._latest is None and not self._stopped and not self._closed:
                if self._timeout_message is not None:
                    break
                if self._read_timeout is None:
                    self._cond.wait()
                    continue
                remaining = self._last_arrival + self._read_timeout - time.monotonic()
                if remaining <= 0:
                    self._timeout_message = (
                        f"{self._label}: no frame received within {self._read_timeout:g} s"
                    )
                    # The capture thread may be stuck in read() for good: let it end
                    # (and release the capture) as soon as read() returns, even if
                    # close() is never called.
                    self._closing.set()
                    break
                self._cond.wait(remaining)
            if self._closed:
                return None
            if self._timeout_message is not None:
                # Checked before the slot: a frame that a stuck read() delivers after
                # the timeout does not restart a stream that was declared dead.
                raise OSError(self._timeout_message) from None
            if self._latest is not None:
                item = self._latest
                self._latest = None
                return item
            if self._error is not None:
                raise self._error
            if self._error_message is not None:
                raise OSError(self._error_message) from None
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
        # A failing release() must not escape the capture thread or close().
        _release_quietly(self._capture, self._label)

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
            thread.join(self._join_timeout)
        if thread.is_alive():
            logger.warning(
                "%s: capture thread did not stop within %.1f s; it releases the capture on exit",
                self._label,
                self._join_timeout,
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
    to take are dropped and counted in ``dropped``.

    The full ``url`` is passed to ``capture_factory`` and kept nowhere else.
    ``info.uri``, the messages this module logs, and the exceptions this class
    raises carry only ``redact_url(url)``. An error while opening or reading
    the stream is raised as ``OSError`` whose message holds the redacted URL and
    the original exception's type name; the original exception is not attached
    as cause or context. ``KeyboardInterrupt`` and ``SystemExit`` pass through
    untouched. OpenCV and FFmpeg may print the URL to stderr themselves, which
    this class cannot prevent.

    When the backend reports a width or height that is not positive, frames are
    read during construction until one arrives and its size is used; that frame
    is discarded.

    ``read_timeout`` (seconds, above 0) ends iteration with ``OSError`` when no
    frame arrives for that long, the first frame included, so a stalled stream
    ends the run instead of blocking it. ``None`` waits without limit.
    """

    def __init__(
        self,
        url: str,
        max_read_failures: int = 30,
        capture_factory: Callable[[str], Any] = cv2.VideoCapture,
        clock: Callable[[], float] = time.monotonic,
        *,
        read_timeout: float | None = None,
        _join_timeout: float = _JOIN_TIMEOUT,
    ) -> None:
        if max_read_failures < 1:
            raise ValueError("max_read_failures must be at least 1")
        _check_read_timeout(read_timeout)
        redacted = redact_url(url)
        label = f"stream {redacted}"
        cap, fps, width, height = _open_live(
            lambda: capture_factory(url),
            label=label,
            open_message=f"Cannot open stream {redacted}",
            max_read_failures=max_read_failures,
        )
        self.info = SourceInfo(
            fps=fps if _usable_fps(fps) else 25.0,
            width=width,
            height=height,
            frame_count=None,
            seekable=False,
            live=True,
            uri=redacted,
        )
        self._reader = _LatestFrameReader(
            cap,
            self.info,
            label=label,
            max_read_failures=max_read_failures,
            clock=clock,
            join_timeout=_join_timeout,
            read_timeout=read_timeout,
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
