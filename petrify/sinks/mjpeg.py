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

"""MJPEG sink: hold the latest processed frame as a JPEG for a local web page."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from petrify.core.frame import FrameMeta, SourceInfo

DEFAULT_BOUNDARY = "frame"
# Response Content-Type matching multipart_chunk(frame) with the default boundary.
MULTIPART_CONTENT_TYPE = f"multipart/x-mixed-replace; boundary={DEFAULT_BOUNDARY}"


@dataclass(frozen=True)
class MjpegFrame:
    """One stored JPEG and where it came from."""

    jpeg: bytes
    sequence: int  # 1 for the first frame stored after open, then +1 per stored frame.
    index: int  # FrameMeta.index of the frame it was encoded from.
    timestamp: float  # FrameMeta.timestamp of that frame.
    width: int
    height: int


class MjpegSink:
    """Keep the most recent output image as a JPEG for any number of readers.

    The engine calls ``open``, ``write``, ``close`` and ``abort`` from its thread;
    HTTP handler threads read with :meth:`latest`, :meth:`wait_next` and
    :meth:`frames`. The sink does no networking itself.

    This sink must only ever be fed processed output images, never the raw input
    frame (the engine guarantees this for its sinks). Exposing the stream beyond
    localhost is the server's decision, not the sink's.

    ``write`` encodes the image with ``cv2.imencode`` at ``quality`` and replaces
    the stored frame. Only the encoded bytes are kept, so nothing aliases the
    array passed in. When ``max_fps`` is set and less than ``1 / max_fps`` seconds
    (per ``clock``) passed since the last stored frame, the frame is skipped
    before encoding and counted in :attr:`skipped`. Frames keep being stored at
    that rate even with no subscriber, so a page that connects gets a current
    image at once. ``write`` never waits for readers: it only swaps the stored
    frame and notifies. An encoding failure raises ``OSError``; a frame whose
    size differs from the opened ``SourceInfo`` raises ``ValueError``.

    ``close`` and ``abort`` mark the sink closed, wake every waiter, and drop the
    stored frame: :meth:`latest` returns None afterwards, so a page never keeps
    showing a stale frame as if it were live once the run has ended. Both are
    idempotent and safe before ``open``. ``open`` after a close starts again with
    a fresh sequence. Every reader method is safe from any thread.
    """

    critical = False

    def __init__(
        self,
        *,
        quality: int = 80,
        max_fps: float | None = 15.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(quality, bool) or not isinstance(quality, int) or not 1 <= quality <= 100:
            raise ValueError(f"MjpegSink quality must be an int in 1..100, got {quality!r}")
        if max_fps is not None and not (
            isinstance(max_fps, (int, float))
            and not isinstance(max_fps, bool)
            and math.isfinite(max_fps)
            and max_fps > 0
        ):
            raise ValueError(f"MjpegSink max_fps must be > 0 or None, got {max_fps!r}")
        self.quality = quality
        self.max_fps = max_fps
        self._interval = None if max_fps is None else 1.0 / max_fps
        self._clock = clock
        self._cond = threading.Condition()
        # Guarded by _cond.
        self._open = False
        self._epoch = 0  # Changes on every open, close and abort.
        self._latest: MjpegFrame | None = None
        self._sequence = 0
        self._subscribers = 0
        # Writer side only (the engine thread).
        self._info: SourceInfo | None = None
        self._last_stored: float | None = None
        self._skipped = 0

    # ------------------------------------------------------------------
    # Sink side
    # ------------------------------------------------------------------

    def open(self, info: SourceInfo) -> None:
        """Start accepting frames for ``info``; the sequence restarts at 1."""
        with self._cond:
            if self._open:
                raise RuntimeError("MjpegSink is already open")
            self._info = info
            self._last_stored = None
            self._skipped = 0
            self._latest = None
            self._sequence = 0
            self._epoch += 1
            self._open = True
            self._cond.notify_all()

    def write(self, image: np.ndarray, meta: FrameMeta) -> None:
        """Encode ``image`` and store it as the latest frame, unless rate limited."""
        with self._cond:
            if not self._open or self._info is None:
                raise RuntimeError("MjpegSink is not open")
            info = self._info
            epoch = self._epoch
        if image.ndim < 2 or image.shape[0] != info.height or image.shape[1] != info.width:
            raise ValueError(
                f"MjpegSink expected {info.width}x{info.height} frames, got shape {image.shape}"
            )
        now = self._clock()
        if (
            self._interval is not None
            and self._last_stored is not None
            and now - self._last_stored < self._interval
        ):
            self._skipped += 1
            return
        # Encode outside the lock so readers are never held up by it.
        try:
            ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        except Exception as exc:
            raise OSError(f"JPEG encoding failed for frame {meta.index}: {exc!r}") from exc
        if not ok:
            raise OSError(f"JPEG encoding failed for frame {meta.index}")
        jpeg = buffer.tobytes()
        with self._cond:
            if not self._open or self._epoch != epoch:
                return  # Closed while encoding; never store into a later run.
            self._sequence += 1
            self._latest = MjpegFrame(
                jpeg=jpeg,
                sequence=self._sequence,
                index=meta.index,
                timestamp=meta.timestamp,
                width=int(image.shape[1]),
                height=int(image.shape[0]),
            )
            self._cond.notify_all()
        self._last_stored = now

    def close(self) -> None:
        """Mark the sink closed, drop the stored frame and wake every reader. Idempotent."""
        self._shut()

    def abort(self) -> None:
        """Same as :meth:`close`: there is no partial output to discard. Idempotent."""
        self._shut()

    def _shut(self) -> None:
        with self._cond:
            if self._open:
                self._epoch += 1
            self._open = False
            self._latest = None
            self._cond.notify_all()

    # ------------------------------------------------------------------
    # Reader side (any thread)
    # ------------------------------------------------------------------

    @property
    def closed(self) -> bool:
        """True when the sink is not open (before ``open``, or after ``close`` or ``abort``)."""
        with self._cond:
            return not self._open

    @property
    def skipped(self) -> int:
        """Frames skipped by the rate limit since ``open`` (never encoded)."""
        return self._skipped

    @property
    def subscribers(self) -> int:
        """Number of :meth:`frames` generators currently started and not yet finished."""
        with self._cond:
            return self._subscribers

    def latest(self) -> MjpegFrame | None:
        """Return the most recent stored frame, or None (none yet, or the sink is closed)."""
        with self._cond:
            return self._latest

    def wait_next(self, after_sequence: int, timeout: float | None = None) -> MjpegFrame | None:
        """Block until a frame with ``sequence > after_sequence`` is stored and return it.

        Returns the newest frame at once when one already qualifies. Returns None
        after ``timeout`` seconds (``None`` waits without limit), or as soon as the
        sink is closed or aborted (:attr:`closed` is then True, unless the sink was
        already reopened). A call that started on a closed sink returns None at once.
        """
        with self._cond:
            if not self._open:
                return None
            epoch = self._epoch
        return self._wait(after_sequence, timeout, epoch)

    def _wait(self, after_sequence: int, timeout: float | None, epoch: int) -> MjpegFrame | None:
        """Wait for a newer frame of run ``epoch``; None on timeout or when that run ended."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while True:
                if not self._open or self._epoch != epoch:
                    return None
                frame = self._latest
                if frame is not None and frame.sequence > after_sequence:
                    return frame
                if deadline is None:
                    self._cond.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._cond.wait(remaining)

    def frames(self, *, timeout: float = 5.0) -> Iterator[MjpegFrame]:
        """Yield each new stored frame to one subscriber until the sink closes.

        The first ``next`` registers the subscriber (see :attr:`subscribers`) and
        yields the current frame at once if one is stored. Each later ``next``
        yields the newest frame with a higher sequence than the last one yielded,
        so a sequence is never yielded twice and a slow reader skips straight to
        the newest frame. The generator ends (``StopIteration``) when the sink is
        closed or aborted, including a close that is followed by a reopen, and
        when it is started on a closed sink.

        ``timeout`` only bounds each internal wait: when it expires with no new
        frame, nothing is yielded and the generator re-checks for closure and
        waits again. Control does not return to the caller until a frame arrives
        or the sink closes, so a server that needs to notice a vanished client
        while the feed is idle should call :meth:`wait_next` itself instead.

        The subscriber count drops when the generator is exhausted, closed, or
        garbage collected while suspended.
        """
        if not timeout > 0:
            raise ValueError(f"frames timeout must be > 0, got {timeout!r}")
        return self._frames(timeout)

    def _frames(self, timeout: float) -> Iterator[MjpegFrame]:
        with self._cond:
            if not self._open:
                return
            epoch = self._epoch
            self._subscribers += 1
        try:
            last = 0
            while True:
                frame = self._wait(last, timeout, epoch)
                if frame is None:
                    with self._cond:
                        if not self._open or self._epoch != epoch:
                            return
                    continue  # Timed out with no new frame: wait again.
                last = frame.sequence
                yield frame
        finally:
            with self._cond:
                self._subscribers -= 1


def multipart_chunk(frame: MjpegFrame, boundary: str = DEFAULT_BOUNDARY) -> bytes:
    """Return one ``multipart/x-mixed-replace`` part carrying ``frame``'s JPEG."""
    if not boundary or any(c in boundary for c in "\r\n"):
        raise ValueError(f"Invalid multipart boundary {boundary!r}")
    header = (
        f"--{boundary}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(frame.jpeg)}\r\n\r\n"
    )
    return header.encode("ascii") + frame.jpeg + b"\r\n"
