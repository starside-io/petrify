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


class ThreadedSink:
    """Forward writes to ``sink`` on a worker thread through a bounded queue.

    ``open`` runs synchronously on the caller's thread, then starts the worker.
    ``write`` copies the image before queueing it: the engine may hand over an
    array it does not own (the erase mode can return ``frame.image`` itself, which
    a live source may reuse for the next frame), so the caller is free to reuse
    the array as soon as ``write`` returns.

    When the queue holds ``max_queue`` frames, ``drop_when_full=True`` drops the
    oldest queued frame (counted in :attr:`dropped`); ``False`` makes ``write``
    block until the worker frees a slot, so no frame is lost.

    An exception raised by the wrapped sink on the worker is stored and re-raised
    from the next ``write`` or from ``close``. ``close`` drains the queue, joins
    the worker and closes the wrapped sink; ``abort`` discards the queue, joins the
    worker and aborts the wrapped sink. Both join with ``join_timeout`` seconds
    and are idempotent; the first of them to run wins.
    """

    def __init__(
        self,
        sink: Sink,
        max_queue: int = 8,
        drop_when_full: bool = True,
        *,
        join_timeout: float = 10.0,
    ) -> None:
        if max_queue < 1:
            raise ValueError(f"max_queue must be >= 1, got {max_queue}")
        if not join_timeout > 0:
            raise ValueError(f"join_timeout must be > 0, got {join_timeout}")
        self._sink = sink
        self._max_queue = max_queue
        self._drop_when_full = drop_when_full
        self._join_timeout = join_timeout
        self._cond = threading.Condition()
        self._queue: deque[tuple[np.ndarray, FrameMeta]] = deque()
        self._thread: threading.Thread | None = None
        self._stopping = False  # close requested: drain, then exit.
        self._aborting = False  # abort requested: exit without draining.
        self._error: BaseException | None = None
        self._dropped = 0
        self._done = False  # close or abort has run.

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
        """Number of frames dropped because the queue was full."""
        with self._cond:
            return self._dropped

    def open(self, info: SourceInfo) -> None:
        """Open the wrapped sink, then start the worker thread."""
        if self._thread is not None or self._done:
            raise RuntimeError("ThreadedSink can only be opened once")
        self._sink.open(info)
        thread = threading.Thread(
            target=self._work,
            name=f"ThreadedSink-{type(self._sink).__name__}",
            daemon=True,
        )
        self._thread = thread
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
                    self._cond.wait_for(
                        lambda: len(self._queue) < self._max_queue
                        or self._error is not None
                        or self._aborting
                        or self._stopping
                    )
                    self._check_writable()
            self._queue.append((copy, meta))
            self._cond.notify_all()

    def close(self) -> None:
        """Drain the queue, join the worker, then close the wrapped sink."""
        if self._done:
            return
        self._done = True
        thread = self._thread
        if thread is None:
            return
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        thread.join(self._join_timeout)
        if thread.is_alive():
            with self._cond:
                self._aborting = True
                self._queue.clear()
                self._cond.notify_all()
            self._sink.abort()
            raise TimeoutError(
                f"ThreadedSink worker did not finish within {self._join_timeout} s; "
                "the wrapped sink was aborted"
            )
        if self._error is not None:
            self._sink.abort()
            raise self._error
        self._sink.close()

    def abort(self) -> None:
        """Discard queued frames, join the worker, then abort the wrapped sink."""
        if self._done:
            return
        self._done = True
        thread = self._thread
        if thread is not None:
            with self._cond:
                self._aborting = True
                self._queue.clear()
                self._cond.notify_all()
            thread.join(self._join_timeout)
            if thread.is_alive():
                logger.warning(
                    "ThreadedSink worker for %s still running after %.1f s; aborting anyway",
                    type(self._sink).__name__,
                    self._join_timeout,
                )
        self._sink.abort()

    def _check_writable(self) -> None:
        """Raise the stored worker error, or if the sink is not accepting frames."""
        if self._error is not None:
            raise self._error
        if self._thread is None:
            raise RuntimeError("ThreadedSink is not open")
        if self._done or self._stopping or self._aborting:
            raise RuntimeError("ThreadedSink is closed")

    def _work(self) -> None:
        """Worker loop: write queued frames in order until closed, aborted, or failed."""
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
                with self._cond:
                    self._error = exc
                    self._queue.clear()
                    self._cond.notify_all()
                return
