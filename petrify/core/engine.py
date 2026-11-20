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

"""Streaming engine: lazy detectors, lookahead buffer, mode, stages and sinks in one pass."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from petrify.core.config import EngineConfig
from petrify.core.detections import Detections
from petrify.core.frame import Frame
from petrify.core.plugin import Detector, Mode, Sink, Source, Stage

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class Decision(Generic[T]):
    """Final per-item verdict emitted by :class:`LookaheadBuffer`."""

    item: T
    index: int
    """0-based position in the stream."""
    raw: bool
    """The value pushed for this item."""
    filled: bool
    """``raw`` after gap fill: any raw True within +/- ``lookahead`` positions."""
    final: bool
    """``filled``, or ``index < cold_start_frames``."""


class LookaheadBuffer(Generic[T]):
    """Delay items by ``lookahead`` positions so gap fill can be decided in one pass.

    Item ``i`` is emitted by the push of item ``i + lookahead`` (or by :meth:`flush`
    at end of stream). The result matches ``_fill_gaps`` followed by
    ``_freeze_cold_start`` from the file pipeline, index for index.

    After :meth:`flush` the buffer is reset and may be reused for a new stream,
    whose first item gets index 0.
    """

    def __init__(self, lookahead: int, cold_start_frames: int = 0) -> None:
        if lookahead < 0:
            raise ValueError(f"lookahead must be >= 0, got {lookahead}")
        if cold_start_frames < 0:
            raise ValueError(f"cold_start_frames must be >= 0, got {cold_start_frames}")
        self._lookahead = lookahead
        self._cold_start = cold_start_frames
        self._pending: deque[tuple[T, bool]] = deque()
        self._next_index = 0
        self._last_true: int | None = None

    @property
    def lookahead(self) -> int:
        """Number of items of delay (equal to the gap fill)."""
        return self._lookahead

    @property
    def cold_start_frames(self) -> int:
        """Number of leading items forced to ``final=True``."""
        return self._cold_start

    @property
    def pending(self) -> int:
        """Number of items pushed but not yet emitted."""
        return len(self._pending)

    def push(self, item: T, raw: bool) -> list[Decision[T]]:
        """Add the next item; return the decisions this push made final, in order."""
        index = self._next_index
        self._next_index += 1
        if raw:
            self._last_true = index
        self._pending.append((item, raw))
        if len(self._pending) > self._lookahead:
            return [self._emit()]
        return []

    def flush(self) -> list[Decision[T]]:
        """Emit every pending item (end of stream) and reset for a new stream."""
        out = [self._emit() for _ in range(len(self._pending))]
        self._next_index = 0
        self._last_true = None
        return out

    def _emit(self) -> Decision[T]:
        """Pop and decide the oldest pending item.

        Every raw value up to ``index + lookahead`` (or end of stream) has been seen,
        and ``_last_true`` is the most recent True among them, so the item is filled
        exactly when that True is no more than ``lookahead`` positions before it.
        """
        item, raw = self._pending.popleft()
        index = self._next_index - 1 - len(self._pending)
        filled = self._last_true is not None and self._last_true >= index - self._lookahead
        final = filled or index < self._cold_start
        return Decision(item=item, index=index, raw=raw, filled=filled, final=final)


_MAX_LOGGED_FAILURES = 5


@dataclass(frozen=True)
class FrameTrace:
    """What the engine decided for one emitted frame."""

    index: int
    raw: bool
    """``subject_present`` from the detectors (True when detection failed closed)."""
    filled: bool
    """After lookahead gap fill."""
    final: bool
    """After cold start; the value the mode received."""
    failed_closed: bool
    """True when the frame was replaced by a black image."""


@dataclass
class EngineStats:
    """Counts for one engine run.

    ``frozen`` and ``passed_through`` count frames that rendered normally, so
    ``frozen + passed_through + failed_closed == frames``.
    """

    frames: int = 0
    frozen: int = 0
    passed_through: int = 0
    failed_closed: int = 0


@dataclass(frozen=True)
class _Pending:
    """One frame waiting in the lookahead buffer."""

    frame: Frame
    detections: Detections
    detect_failed: bool


class Engine:
    """Single-threaded, single-pass streaming engine.

    Per frame: run the detectors the mode requires (lazily), push the result into a
    :class:`LookaheadBuffer`, and for every frame the buffer releases render it with
    the mode, apply the stages in order, write it to every sink, and report a
    :class:`FrameTrace` to the observer. A detector, mode, or stage error replaces
    that frame with a black image (fail closed); source and sink errors abort the run.
    """

    def __init__(
        self,
        source: Source,
        detectors: Sequence[Detector],
        mode: Mode,
        stages: Sequence[Stage],
        sinks: Sequence[Sink],
        config: EngineConfig,
        observer: Callable[[FrameTrace], None] | None = None,
    ) -> None:
        self._source = source
        self._detectors = tuple(d for d in detectors if d.provides & mode.requires)
        self._mode = mode
        self._stages = tuple(stages)
        self._sinks = tuple(sinks)
        self._config = config
        self._observer = observer
        self._stats = EngineStats()
        self._failures_seen = 0

    @property
    def detectors(self) -> tuple[Detector, ...]:
        """The detectors this engine runs, in order (only those the mode requires)."""
        return self._detectors

    def run(self) -> EngineStats:
        """Process the whole source and return the run's counts.

        Every sink is opened with ``source.info``. On success every sink is closed;
        on any exception (``KeyboardInterrupt`` included) every opened sink is
        aborted and the exception propagates. The source is not closed.
        """
        self._stats = EngineStats()
        self._failures_seen = 0
        buffer: LookaheadBuffer[_Pending] = LookaheadBuffer(
            self._config.lookahead, self._config.cold_start_frames
        )
        for detector in self._detectors:
            detector.reset()

        opened: list[Sink] = []
        try:
            for sink in self._sinks:
                sink.open(self._source.info)
                opened.append(sink)
            for frame in self._source:
                pending = self._detect(frame)
                raw = pending.detect_failed or pending.detections.subject_present
                for decision in buffer.push(pending, raw):
                    self._emit(decision)
            for decision in buffer.flush():
                self._emit(decision)
        except BaseException:
            _abort_all(opened)
            raise

        _close_all(opened)

        if self._failures_seen > _MAX_LOGGED_FAILURES:
            logger.warning(
                "%d further fail-closed failures were not logged individually",
                self._failures_seen - _MAX_LOGGED_FAILURES,
            )
        return self._stats

    def _detect(self, frame: Frame) -> _Pending:
        """Run the detectors lazily: stateful ones always, others until a subject is found.

        After a detector raises, the remaining stateful detectors still see the frame
        so their models stay in step with the stream; their results are discarded.
        """
        detections = Detections()
        failed = False
        for detector in self._detectors:
            if not detector.stateful and (failed or detections.subject_present):
                continue
            try:
                result = detector.detect(frame)
            except Exception as exc:
                if not failed:
                    self._log_failure(frame.index, f"detector {type(detector).__name__}", exc)
                failed = True
                continue
            if not failed:
                detections = detections.merged(result)
        return _Pending(frame=frame, detections=detections, detect_failed=failed)

    def _emit(self, decision: Decision[_Pending]) -> None:
        """Render one released frame, write it to every sink, and report it."""
        pending = decision.item
        frame = pending.frame
        failed = pending.detect_failed
        image: np.ndarray | None = None
        if not failed:
            where = f"mode {self._mode.name}"
            try:
                image = self._mode.render(frame, pending.detections, decision.final)
                for stage in self._stages:
                    where = f"stage {type(stage).__name__}"
                    image = stage.apply(frame, image)
            except Exception as exc:
                self._log_failure(frame.index, where, exc)
                failed = True
        if failed or image is None:
            failed = True
            image = np.zeros_like(frame.image)

        for sink in self._sinks:
            sink.write(frame, image)

        stats = self._stats
        stats.frames += 1
        if failed:
            stats.failed_closed += 1
        elif decision.final:
            stats.frozen += 1
        else:
            stats.passed_through += 1

        if self._observer is not None:
            self._observer(
                FrameTrace(
                    index=decision.index,
                    raw=decision.raw,
                    filled=decision.filled,
                    final=decision.final,
                    failed_closed=failed,
                )
            )

    def _log_failure(self, index: int, where: str, exc: Exception) -> None:
        """Log the first few fail-closed failures; later ones are only counted."""
        self._failures_seen += 1
        if self._failures_seen <= _MAX_LOGGED_FAILURES:
            logger.warning("Frame %d failed closed (%s): %r", index, where, exc)


def _close_all(sinks: Sequence[Sink]) -> None:
    """Close every sink in order; if one raises, abort the rest and re-raise."""
    for i, sink in enumerate(sinks):
        try:
            sink.close()
        except BaseException:
            _abort_all(sinks[i + 1 :])
            raise


def _abort_all(sinks: Sequence[Sink]) -> None:
    """Abort every sink, logging (not raising) any error so the original one propagates.

    Every sink is attempted even if an abort raises a non-``Exception`` error such as
    ``KeyboardInterrupt``; the first such error is re-raised once all were attempted.
    """
    interrupt: BaseException | None = None
    for sink in sinks:
        try:
            sink.abort()
        except Exception:
            logger.exception("Sink %s failed to abort", type(sink).__name__)
        except BaseException as exc:
            if interrupt is None:
                interrupt = exc
    if interrupt is not None:
        raise interrupt
