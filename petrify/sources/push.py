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

"""Push API: feed in-memory frames to an engine one by one and get processed frames back."""

from __future__ import annotations

import math
from types import TracebackType

import numpy as np

from petrify.core.engine import Engine, EngineStats
from petrify.core.frame import Frame, SourceInfo

_OPEN = "open"
_FINISHED = "finished"
_ABORTED = "aborted"
_FAILED = "failed"


class PushSession:
    """Thin facade over an :class:`Engine` for callers that already hold raw images.

    The engine is opened in the constructor with a live :class:`SourceInfo` built
    from the arguments, so a sink that fails to open raises here and ``finish``
    is well defined even if nothing was pushed. The caller builds the engine
    (detectors, mode, stages, sinks); an engine with zero sinks works, its
    outputs are then only returned.

    Latency: with lookahead K (:attr:`delay`), the n-th :meth:`push` returns the
    processed image of input n - K, or ``None`` for the first K pushes;
    :meth:`finish` returns the last K.

    Ownership: every pushed image is copied before the engine sees it, so the
    caller may reuse or mutate its array as soon as :meth:`push` returns, and no
    returned array shares memory with an array the caller passed in. Returned
    arrays may be the same objects the engine's sinks received; a sink that keeps
    an image beyond ``write`` copies it (as the threaded sink does).

    Fail closed is left to the engine: a frame that failed closed comes back as an
    all-black image, like any other output.

    Always end a session with :meth:`finish` or :meth:`abort`, or use it as a
    context manager. A session is not thread-safe: call it from one thread.
    """

    def __init__(self, engine: Engine, width: int, height: int, fps: float = 30.0) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be > 0, got {width}x{height}")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"fps must be a finite number > 0, got {fps}")
        self._engine = engine
        self._info = SourceInfo(
            fps=float(fps),
            width=width,
            height=height,
            frame_count=None,
            seekable=False,
            live=True,
            uri="push",
        )
        self._shape = (height, width, 3)
        self._delay = int(engine.config.lookahead)
        self._next_index = 0
        self._last_timestamp: float | None = None
        self._stats: EngineStats | None = None
        self._state = _FAILED
        engine.open(self._info)  # On failure the engine aborts itself.
        self._state = _OPEN

    @property
    def info(self) -> SourceInfo:
        """The stream description the engine and its sinks were opened with."""
        return self._info

    @property
    def delay(self) -> int:
        """Frames of latency between a push and its output: the engine's lookahead."""
        return self._delay

    @property
    def stats(self) -> EngineStats | None:
        """Engine counts, set by a successful :meth:`finish`; ``None`` before."""
        return self._stats

    def push(self, image: np.ndarray, timestamp: float | None = None) -> np.ndarray | None:
        """Process one BGR uint8 image of shape ``(height, width, 3)``.

        ``timestamp`` is in seconds from stream start; it defaults to
        ``index / fps`` and must not be lower than the previous frame's.
        Returns the image released by this push, or ``None`` while the lookahead
        fills. Invalid input raises ``ValueError`` before the engine is called and
        leaves the session usable. An engine error propagates and fails the session.
        """
        self._require_open("push")
        self._validate(image)
        index = self._next_index
        if timestamp is None:
            ts = index / self._info.fps
        else:
            ts = float(timestamp)
            if not math.isfinite(ts):
                raise ValueError(f"timestamp must be finite, got {timestamp}")
        if self._last_timestamp is not None and ts < self._last_timestamp:
            raise ValueError(
                f"timestamp {ts} for frame {index} is lower than the previous one "
                f"({self._last_timestamp})"
            )
        # Private contiguous copy: the engine buffers frames and a pass-through
        # mode may return the frame's own array as the output.
        frame = Frame(
            image=np.array(image, dtype=np.uint8, copy=True, order="C"),
            index=index,
            timestamp=ts,
            source=self._info,
        )
        try:
            outputs = self._engine.step(frame)
            if len(outputs) > 1:
                raise RuntimeError(
                    f"Engine released {len(outputs)} frames for one push; expected at most 1"
                )
        except BaseException:
            self._fail()
            raise
        self._next_index = index + 1
        self._last_timestamp = ts
        return outputs[0].image if outputs else None

    def finish(self) -> list[np.ndarray]:
        """Flush the lookahead and close the engine; return the tail images in order.

        Stores the engine stats in :attr:`stats`. Raises ``RuntimeError`` if the
        session already finished, was aborted, or failed.
        """
        self._require_open("finish")
        try:
            outputs = self._engine.flush()
            stats = self._engine.close()
        except BaseException:
            self._fail()
            raise
        self._state = _FINISHED
        self._stats = stats
        return [output.image for output in outputs]

    def abort(self) -> None:
        """Abort the engine and its sinks, discarding buffered frames. Idempotent.

        Safe in any state, including after :meth:`finish` or a failure.
        """
        if self._state == _OPEN:
            self._state = _ABORTED
        self._engine.abort()

    def __enter__(self) -> PushSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Abort on an exception, otherwise finish if still open (the tail is discarded).

        The tail still reaches the engine's sinks; call :meth:`finish` inside the
        block to get the tail images back.
        """
        if exc_type is not None or self._state != _OPEN:
            self.abort()
        else:
            self.finish()

    def _require_open(self, what: str) -> None:
        """Raise ``RuntimeError`` unless the session accepts ``what`` now."""
        if self._state != _OPEN:
            raise RuntimeError(f"PushSession is {self._state}; cannot {what}")

    def _validate(self, image: object) -> None:
        """Raise ``ValueError`` unless ``image`` is a uint8 array of the session's shape."""
        if not isinstance(image, np.ndarray):
            raise ValueError(f"image must be a numpy array, got {type(image).__name__}")
        if image.dtype != np.uint8:
            raise ValueError(f"image dtype must be uint8, got {image.dtype}")
        if image.shape != self._shape:
            raise ValueError(f"image shape must be {self._shape}, got {image.shape}")

    def _fail(self) -> None:
        """Mark the session failed and make sure the engine is aborted."""
        self._state = _FAILED
        self._engine.abort()
