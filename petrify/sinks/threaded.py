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

"""Sink wrapper that writes on a worker thread so a slow sink cannot stall the engine."""

from __future__ import annotations

import logging
import threading
from collections import deque

import numpy as np

from petrify.core.frame import FrameMeta, SourceInfo
from petrify.core.plugin import Sink

logger = logging.getLogger(__name__)

# Lifecycle states. IDLE: not open (never opened, or a failed open); abort forwards.
_IDLE = "idle"
_OPEN = "open"
_CLOSING = "closing"  # A close or abort claimed the transition and is running.
_DONE = "done"  # A close or abort has finished; open may be called again.


class ThreadedSink:
    """Forward writes to ``sink`` on a worker thread through a bounded queue.

    ``open`` runs synchronously on the caller's thread, then starts the worker.
    ``write`` copies the image before queueing it: the engine may hand over an
    array it does not own (the erase mode can return ``frame.image`` itself, which
    a live source may reuse for the next frame), so the caller is free to reuse
    the array as soon as ``write`` returns.

    When the queue holds ``max_queue`` frames, drop mode drops the oldest queued
    frame (counted in :attr:`dropped`); block mode makes ``write`` wait until the
    worker frees a slot, so no frame is lost, and raise ``TimeoutError`` after
    ``write_timeout`` seconds (``None`` waits without limit). ``drop_when_full=None``
    picks drop mode only for a non-critical wrapped sink; ``True`` with a critical
    wrapped sink raises ``ValueError``, since a recording must not lose frames
    while the run reports success.

    An exception raised by the wrapped sink on the worker is stored and re-raised
    from the next ``write`` or from ``close``; a ``BaseException`` that is not an
    ``Exception`` is surfaced as a ``RuntimeError`` chained to it. ``close`` drains
    the queue, waits for the worker and closes the wrapped sink; ``abort`` discards
    the queue, waits for the worker and aborts the wrapped sink. Both wait at most
    ``join_timeout`` seconds and are idempotent; exactly one of them reaches the
    wrapped sink, even when called concurrently.

    No two methods of the wrapped sink ever run at the same time. If the worker is
    still inside a wrapped ``write`` when the wait times out, the wrapped ``abort``
    is handed over to the worker, which runs it as soon as that call returns;
    ``abort`` then logs a warning and returns, and ``close`` raises ``TimeoutError``.

    If ``close`` fails before the wrapped ``close`` completed (a stored worker
    error, the wrapped ``close`` raising, or an interruption such as
    ``KeyboardInterrupt``), the wrapped sink is aborted exactly once (handed over
    to the worker if it is still running) and the original error propagates; a
    later ``abort`` does nothing.

    After a completed ``close`` or ``abort`` the sink may be opened again, with a
    fresh queue, worker and counters. ``open`` while open raises ``RuntimeError``.
    """

    def __init__(
        self,
        sink: Sink,
        max_queue: int = 8,
        drop_when_full: bool | None = None,
        *,
        join_timeout: float = 10.0,
        write_timeout: float | None = 5.0,
    ) -> None:
        if max_queue < 1:
            raise ValueError(f"max_queue must be >= 1, got {max_queue}")
        if not join_timeout > 0:
            raise ValueError(f"join_timeout must be > 0, got {join_timeout}")
        if write_timeout is not None and not write_timeout > 0:
            raise ValueError(f"write_timeout must be > 0 or None, got {write_timeout}")
        if drop_when_full is None:
            drop_when_full = not sink.critical
        elif drop_when_full and sink.critical:
            raise ValueError(
                f"drop_when_full=True would silently lose frames of the critical sink "
                f"{type(sink).__name__}; use drop_when_full=False or None"
            )
        self._sink = sink
        self._max_queue = max_queue
        self._drop_when_full = drop_when_full
        self._join_timeout = join_timeout
        self._write_timeout = write_timeout
        self._cond = threading.Condition()
        self._state = _IDLE
        self._thread: threading.Thread | None = None
        self._worker_exited = True  # No worker is running (or touching the sink).
        self._queue: deque[tuple[np.ndarray, FrameMeta]] = deque()
        self._stopping = False  # close requested: drain, then exit.
        self._aborting = False  # abort requested: exit without draining.
        self._handoff_abort = False  # The worker must abort the wrapped sink on exit.
        self._error: BaseException | None = None
        self._dropped = 0

    @property
    def critical(self) -> bool:
        """Mirrors the wrapped sink."""
        return self._sink.critical

    @property
    def sink(self) -> Sink:
        """The wrapped sink."""
        return self._sink

    @property
    def dropped(self) -> int:
        """Number of frames dropped because the queue was full (since the last open)."""
        with self._cond:
            return self._dropped

    def open(self, info: SourceInfo) -> None:
        """Open the wrapped sink, then start a fresh worker thread."""
        with self._cond:
            if self._state in (_OPEN, _CLOSING):
                raise RuntimeError("ThreadedSink is already open")
            if not self._worker_exited:
                raise RuntimeError(
                    "ThreadedSink worker from the previous run is still running"
                )
            self._state = _IDLE
            self._queue = deque()
            self._stopping = False
            self._aborting = False
            self._handoff_abort = False
            self._error = None
            self._dropped = 0
        self._sink.open(info)
        thread = threading.Thread(
            target=self._work,
            name=f"ThreadedSink-{type(self._sink).__name__}",
            daemon=True,
        )
        with self._cond:
            self._thread = thread
            self._worker_exited = False
            self._state = _OPEN
        thread.start()

    def write(self, image: np.ndarray, meta: FrameMeta) -> None:
        """Queue a copy of ``image``; raise a stored worker error first."""
        copy = np.array(image, copy=True)
        with self._cond:
            self._check_writable()
            if len(self._queue) >= self._max_queue:
                if self._drop_when_full:
                    self._queue.popleft()
                    self._dropped += 1
                else:
                    freed = self._cond.wait_for(
                        lambda: len(self._queue) < self._max_queue
                        or self._error is not None
                        or self._aborting
                        or self._stopping,
                        self._write_timeout,
                    )
                    if not freed:
                        raise TimeoutError(
                            f"ThreadedSink write blocked for more than {self._write_timeout} s; "
                            f"{type(self._sink).__name__} is not keeping up"
                        )
                    self._check_writable()
            self._queue.append((copy, meta))
            self._cond.notify_all()

    def close(self) -> None:
        """Drain the queue, wait for the worker, then close the wrapped sink."""
        with self._cond:
            state = self._state
            if state in (_CLOSING, _DONE):
                return
            if state == _IDLE:
                self._state = _DONE
                return
            self._state = _CLOSING
            self._stopping = True
            self._cond.notify_all()
        # Set once the wrapped sink was closed, or its abort was run or handed over.
        settled = False
        try:
            with self._cond:
                exited = self._wait_for_worker()
                error = self._error
            if not exited:
                settled = True  # _wait_for_worker handed the abort to the worker.
                self._warn_handoff()
                raise TimeoutError(
                    f"ThreadedSink worker did not finish within {self._join_timeout} s; "
                    "the wrapped sink will be aborted when its current write returns"
                )
            self._join_thread()
            if error is not None:
                raise error
            self._sink.close()
            settled = True
        except BaseException:
            if not settled:
                self._abort_after_failed_close()
            raise
        finally:
            with self._cond:
                self._state = _DONE

    def abort(self) -> None:
        """Discard queued frames, wait for the worker, then abort the wrapped sink."""
        with self._cond:
            state = self._state
            if state in (_CLOSING, _DONE):
                return
            self._state = _CLOSING
            exited = True
            if state == _OPEN:
                self._aborting = True
                self._queue.clear()
                self._cond.notify_all()
                exited = self._wait_for_worker()
        try:
            if not exited:
                self._warn_handoff()
                return
            if state == _OPEN:
                self._join_thread()
            self._sink.abort()
        finally:
            with self._cond:
                self._state = _DONE

    def _abort_after_failed_close(self) -> None:
        """Abort the wrapped sink once after ``close`` failed before closing it.

        If the worker is still running (the wait was interrupted), the abort is
        handed over to it, so two wrapped calls never overlap. An ``Exception``
        from the wrapped abort is logged; the caller propagates the original error.
        """
        with self._cond:
            if self._handoff_abort:
                return  # Already handed over before the interruption.
            if not self._worker_exited:
                self._aborting = True
                self._queue.clear()
                self._handoff_abort = True
                self._cond.notify_all()
                return
        self._join_thread()
        try:
            self._sink.abort()
        except Exception:
            logger.exception("Sink %s failed to abort", type(self._sink).__name__)

    def _wait_for_worker(self) -> bool:
        """With the lock held: wait for the worker to exit, or hand it the abort.

        Returns True when the worker exited and no longer touches the wrapped sink.
        On timeout the worker is told to stop and to abort the wrapped sink itself
        once its in-flight call returns, so the two never run concurrently.
        """
        if self._cond.wait_for(lambda: self._worker_exited, self._join_timeout):
            return True
        self._aborting = True
        self._queue.clear()
        self._handoff_abort = True
        self._cond.notify_all()
        return False

    def _warn_handoff(self) -> None:
        logger.warning(
            "ThreadedSink worker for %s still running after %.1f s; it will abort the "
            "wrapped sink when its current write returns",
            type(self._sink).__name__,
            self._join_timeout,
        )

    def _join_thread(self) -> None:
        """Reap the worker after it reported exit (it only has to return)."""
        thread = self._thread
        if thread is not None:
            thread.join(self._join_timeout)

    def _check_writable(self) -> None:
        """Raise the stored worker error, or if the sink is not accepting frames."""
        if self._error is not None:
            raise self._error
        if self._state == _IDLE:
            raise RuntimeError("ThreadedSink is not open")
        if self._state != _OPEN or self._stopping or self._aborting:
            raise RuntimeError("ThreadedSink is closed")

    def _work(self) -> None:
        """Worker: write queued frames, then perform a handed-over abort, if any."""
        try:
            self._write_loop()
        finally:
            with self._cond:
                handoff = self._handoff_abort
                if not handoff:
                    self._worker_exited = True
                    self._cond.notify_all()
            if handoff:
                try:
                    self._sink.abort()
                except BaseException:
                    logger.exception("Sink %s failed to abort", type(self._sink).__name__)
                finally:
                    with self._cond:
                        self._worker_exited = True
                        self._cond.notify_all()

    def _write_loop(self) -> None:
        """Write queued frames in order until closed, aborted, or failed."""
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._queue or self._stopping or self._aborting)
                if self._aborting or not self._queue:
                    return
                image, meta = self._queue.popleft()
                self._cond.notify_all()
            try:
                self._sink.write(image, meta)
            except BaseException as exc:
                error: BaseException = exc
                if not isinstance(exc, Exception):
                    error = RuntimeError(
                        f"ThreadedSink worker for {type(self._sink).__name__} raised {exc!r}"
                    )
                    error.__cause__ = exc
                with self._cond:
                    self._error = error
                    self._queue.clear()
                    self._cond.notify_all()
                return
