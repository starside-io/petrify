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
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from petrify.core.config import EngineConfig
from petrify.core.detections import Detections
from petrify.core.frame import Frame, FrameMeta, SourceInfo
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


@dataclass(frozen=True, eq=False)
class Output:
    """One released output frame: the image the sinks received, its metadata, its trace.

    ``image`` is not copied: on a pass-through frame the mode may return the input
    frame's own array, so copy it before keeping it if the source reuses buffers.
    """

    image: np.ndarray
    meta: FrameMeta
    trace: FrameTrace


@dataclass
class EngineStats:
    """Counts for one engine run.

    ``frozen`` and ``passed_through`` count frames that rendered normally, so
    ``frozen + passed_through + failed_closed == frames``. ``over_budget`` frames
    are a subset of ``failed_closed``. ``sink_failures`` counts non-critical sinks
    that failed and were dropped.
    """

    frames: int = 0
    frozen: int = 0
    passed_through: int = 0
    failed_closed: int = 0
    over_budget: int = 0
    sink_failures: int = 0


@dataclass(frozen=True)
class _Pending:
    """One frame waiting in the lookahead buffer."""

    frame: Frame
    detections: Detections
    detect_failed: bool
    over_budget: bool
    detect_seconds: float


class Engine:
    """Single-threaded, single-pass streaming engine driven one frame at a time.

    Lifecycle: :meth:`open`, then :meth:`step` per frame, then :meth:`flush`
    (optional, returns the tail outputs) and :meth:`close`; or :meth:`abort` at
    any point. :meth:`run` is that loop over a :class:`Source`.

    Per frame: run the detectors the mode requires (lazily), push the result into a
    :class:`LookaheadBuffer`, and for every frame the buffer releases render it with
    the mode, apply the stages in order, write the output image and the frame's
    :class:`FrameMeta` to every active sink, and report a :class:`FrameTrace` to the
    observer. Sinks never receive the input frame.

    Fail closed: a detector, mode, or stage error replaces that frame with a black
    image, and so does exceeding ``config.frame_budget_ms`` (detection plus render
    time, measured with ``clock``). The budget is checked after the fact: a running
    detector or mode is never interrupted.

    A failure of a critical sink, of the observer, or any ``BaseException`` aborts
    the engine (every active sink aborted, pending frames discarded) and propagates.
    A non-critical sink that fails is aborted and dropped while the run continues;
    if no sink is left the engine aborts and raises.
    """

    def __init__(
        self,
        detectors: Sequence[Detector],
        mode: Mode,
        stages: Sequence[Stage],
        sinks: Sequence[Sink],
        config: EngineConfig,
        observer: Callable[[FrameTrace], None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._detectors = tuple(d for d in detectors if d.provides & mode.requires)
        self._mode = mode
        self._stages = tuple(stages)
        self._sinks = tuple(sinks)
        self._config = config
        self._observer = observer
        self._clock = clock
        budget_ms = config.frame_budget_ms
        self._budget_s = None if budget_ms is None else budget_ms / 1000.0
        self._stats = EngineStats()
        self._failures_seen = 0
        self._buffer: LookaheadBuffer[_Pending] | None = None
        self._flushed = False
        self._active: list[Sink] = []

    @property
    def detectors(self) -> tuple[Detector, ...]:
        """The detectors this engine runs, in order (only those the mode requires)."""
        return self._detectors

    @property
    def is_open(self) -> bool:
        """True between a successful :meth:`open` and :meth:`close` or :meth:`abort`."""
        return self._buffer is not None

    # ------------------------------------------------------------------
    # Step API
    # ------------------------------------------------------------------

    def open(self, info: SourceInfo) -> None:
        """Reset the detectors, create the lookahead buffer, and open every sink.

        Raises ``RuntimeError`` if the engine is already open. On failure every
        opened sink is aborted, the engine stays closed, and the error propagates.
        """
        if self._buffer is not None:
            raise RuntimeError("Engine is already open; close or abort it first")
        self._stats = EngineStats()
        self._failures_seen = 0
        buffer: LookaheadBuffer[_Pending] = LookaheadBuffer(
            self._config.lookahead, self._config.cold_start_frames
        )
        for detector in self._detectors:
            detector.reset()
        self._buffer = buffer
        self._flushed = False
        self._active = []
        try:
            for sink in self._sinks:
                try:
                    sink.open(info)
                except Exception as exc:
                    if sink.critical:
                        raise
                    self._drop_sink(sink, "open", exc)
                    continue
                self._active.append(sink)
            self._require_some_sink()
        except BaseException:
            self.abort()
            raise

    def step(self, frame: Frame) -> list[Output]:
        """Process one frame; return the outputs it released, in order.

        Returns an empty list while the lookahead buffer fills. On failure the
        engine is aborted and the error propagates.
        """
        buffer = self._require_streaming("step")
        try:
            pending = self._detect(frame)
            raw = (
                pending.detect_failed
                or pending.over_budget
                or pending.detections.subject_present
            )
            return [self._emit(decision) for decision in buffer.push(pending, raw)]
        except BaseException:
            self.abort()
            raise

    def flush(self) -> list[Output]:
        """End the stream: render and write the frames still in the lookahead buffer.

        Returns the tail outputs, in order. After this only :meth:`close` or
        :meth:`abort` may be called. On failure the engine is aborted.
        """
        buffer = self._require_streaming("flush")
        self._flushed = True
        try:
            return [self._emit(decision) for decision in buffer.flush()]
        except BaseException:
            self.abort()
            raise

    def close(self) -> EngineStats:
        """Flush the tail (if :meth:`flush` was not called), close every sink, return stats.

        Callers that need the tail outputs call :meth:`flush` first; otherwise the
        tail is still written to the sinks but its outputs are not returned. On
        failure of a critical sink the remaining sinks are aborted and the error
        propagates.
        """
        if self._buffer is None:
            raise RuntimeError("Engine is not open")
        if not self._flushed:
            self.flush()
        try:
            closed = 0
            while self._active:
                sink = self._active.pop(0)
                try:
                    sink.close()
                except Exception as exc:
                    if sink.critical:
                        raise
                    self._stats.sink_failures += 1
                    logger.warning(
                        "Non-critical sink %s failed to close and was dropped: %r",
                        type(sink).__name__,
                        exc,
                    )
                    _abort_all([sink])
                    continue
                closed += 1
            if self._sinks and closed == 0:
                raise RuntimeError("Every sink failed; no output was written")
        except BaseException:
            self.abort()
            raise

        self._buffer = None
        if self._failures_seen > _MAX_LOGGED_FAILURES:
            logger.warning(
                "%d further fail-closed failures were not logged individually",
                self._failures_seen - _MAX_LOGGED_FAILURES,
            )
        return self._stats

    def abort(self) -> None:
        """Abort every active sink and discard pending frames. Idempotent.

        Safe to call at any time, including after a failed :meth:`open`.
        """
        sinks, self._active = self._active, []
        self._buffer = None
        _abort_all(sinks)

    def run(self, source: Source) -> EngineStats:
        """Process the whole source with :meth:`open`, :meth:`step` and :meth:`close`.

        On any exception (``KeyboardInterrupt`` included) the engine is aborted and
        the exception propagates. The source is not closed.
        """
        try:
            self.open(source.info)
            for frame in source:
                self.step(frame)
            return self.close()
        except BaseException:
            self.abort()
            raise

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_streaming(self, what: str) -> LookaheadBuffer[_Pending]:
        """Return the buffer, or raise if the engine cannot accept ``what`` now."""
        if self._buffer is None:
            raise RuntimeError(f"Engine is not open; cannot {what}")
        if self._flushed:
            raise RuntimeError(f"Engine was flushed; cannot {what}")
        return self._buffer

    def _require_some_sink(self) -> None:
        """Raise when sinks were configured but none is still active."""
        if self._sinks and not self._active:
            raise RuntimeError("Every sink failed; no output is being written")

    def _drop_sink(self, sink: Sink, phase: str, exc: Exception) -> None:
        """Abort a failed non-critical sink and count it; the caller removes it."""
        self._stats.sink_failures += 1
        logger.warning(
            "Non-critical sink %s failed to %s and was dropped: %r",
            type(sink).__name__,
            phase,
            exc,
        )
        _abort_all([sink])

    def _detect(self, frame: Frame) -> _Pending:
        """Run the detectors lazily: stateful ones always, others until a subject is found.

        After a detector raises, the remaining stateful detectors still see the frame
        so their models stay in step with the stream; their results are discarded.
        """
        budget = self._budget_s
        start = self._clock() if budget is not None else 0.0
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

        elapsed = 0.0
        over = False
        if budget is not None:
            elapsed = self._clock() - start
            if not failed and elapsed > budget:
                over = True
                self._log_over_budget(frame.index, "detection", elapsed)
        return _Pending(
            frame=frame,
            detections=detections,
            detect_failed=failed,
            over_budget=over,
            detect_seconds=elapsed,
        )

    def _emit(self, decision: Decision[_Pending]) -> Output:
        """Render one released frame, write it to every active sink, and report it."""
        pending = decision.item
        frame = pending.frame
        over = pending.over_budget
        failed = pending.detect_failed or over
        image: np.ndarray | None = None
        if not failed:
            budget = self._budget_s
            start = self._clock() if budget is not None else 0.0
            where = f"mode {self._mode.name}"
            try:
                image = self._mode.render(frame, pending.detections, decision.final)
                for stage in self._stages:
                    where = f"stage {type(stage).__name__}"
                    image = stage.apply(frame, image)
            except Exception as exc:
                self._log_failure(frame.index, where, exc)
                failed = True
            if not failed and budget is not None:
                total = pending.detect_seconds + (self._clock() - start)
                if total > budget:
                    over = failed = True
                    self._log_over_budget(frame.index, "detection and render", total)
        if failed or image is None:
            failed = True
            image = np.zeros_like(frame.image)

        meta = frame.meta
        self._write(image, meta)

        stats = self._stats
        stats.frames += 1
        if failed:
            stats.failed_closed += 1
            if over:
                stats.over_budget += 1
        elif decision.final:
            stats.frozen += 1
        else:
            stats.passed_through += 1

        trace = FrameTrace(
            index=decision.index,
            raw=decision.raw,
            filled=decision.filled,
            final=decision.final,
            failed_closed=failed,
        )
        if self._observer is not None:
            self._observer(trace)
        return Output(image=image, meta=meta, trace=trace)

    def _write(self, image: np.ndarray, meta: FrameMeta) -> None:
        """Write to every active sink; drop a failing non-critical sink, raise for a critical one."""
        for sink in tuple(self._active):
            try:
                sink.write(image, meta)
            except Exception as exc:
                if sink.critical:
                    raise
                self._active.remove(sink)
                self._drop_sink(sink, "write", exc)
        self._require_some_sink()

    def _log_failure(self, index: int, where: str, exc: Exception) -> None:
        """Log the first few fail-closed failures; later ones are only counted."""
        self._failures_seen += 1
        if self._failures_seen <= _MAX_LOGGED_FAILURES:
            logger.warning("Frame %d failed closed (%s): %r", index, where, exc)

    def _log_over_budget(self, index: int, what: str, seconds: float) -> None:
        """Log an over-budget frame with the same rate limit as other fail-closed events."""
        self._failures_seen += 1
        if self._failures_seen <= _MAX_LOGGED_FAILURES:
            logger.warning(
                "Frame %d failed closed (over budget): %s took %.1f ms, budget %.1f ms",
                index,
                what,
                seconds * 1000.0,
                self._config.frame_budget_ms,
            )


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
