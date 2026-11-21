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
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Generic, TypeVar

import numpy as np

from petrify.core.config import EngineConfig
from petrify.core.detections import DetectionKind, Detections, check_detections
from petrify.core.frame import Frame, FrameMeta, SourceInfo
from petrify.core.plugin import Detector, DetectPolicy, Mode, Refiner, Sink, Source, Stage

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class Decision(Generic[T]):
    """Final per-item verdict emitted by :class:`LookaheadBuffer`."""

    item: T
    index: int
    """Position in the stream; the first item pushed gets ``start_index`` (0 by default)."""
    raw: bool
    """The value pushed for this item."""
    filled: bool
    """``raw`` after gap fill: any raw True within +/- ``lookahead`` positions."""
    final: bool
    """``filled``, or ``index < start_index + cold_start_frames``."""


class LookaheadBuffer(Generic[T]):
    """Delay items by ``lookahead`` positions so gap fill can be decided in one pass.

    Item ``i`` is emitted by the push of item ``i + lookahead`` (or by :meth:`flush`
    at end of stream). The result matches ``_fill_gaps`` followed by
    ``_freeze_cold_start`` from the file pipeline, index for index.

    ``start_index`` numbers the first item, so a buffer created mid-stream (for
    example by :meth:`Engine.reconfigure`) continues the stream's numbering. The
    cold-start window covers the first ``cold_start_frames`` items pushed, that is
    indices ``start_index`` to ``start_index + cold_start_frames - 1``, and gap fill
    never looks before the first item pushed.

    After :meth:`flush` the buffer is reset and may be reused for a new stream,
    whose first item gets index ``start_index`` again.
    """

    def __init__(self, lookahead: int, cold_start_frames: int = 0, start_index: int = 0) -> None:
        if lookahead < 0:
            raise ValueError(f"lookahead must be >= 0, got {lookahead}")
        if cold_start_frames < 0:
            raise ValueError(f"cold_start_frames must be >= 0, got {cold_start_frames}")
        if start_index < 0:
            raise ValueError(f"start_index must be >= 0, got {start_index}")
        self._lookahead = lookahead
        self._cold_start = cold_start_frames
        self._start = start_index
        self._cold_end = start_index + cold_start_frames
        self._pending: deque[tuple[T, bool]] = deque()
        self._next_index = start_index
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
    def start_index(self) -> int:
        """Index of the first item pushed (after construction or :meth:`flush`)."""
        return self._start

    @property
    def next_index(self) -> int:
        """Index the next pushed item will get."""
        return self._next_index

    @property
    def cold_start_end(self) -> int:
        """First index after the cold-start window (``start_index + cold_start_frames``)."""
        return self._cold_end

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
        self._next_index = self._start
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
        final = filled or index < self._cold_end
        return Decision(item=item, index=index, raw=raw, filled=filled, final=final)


_MAX_LOGGED_FAILURES = 5
_SUMMARY_EVERY = 100  # After the first few, one summary line per this many fail-closed frames.


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
    that failed and were dropped. ``reconfigurations`` counts successful
    :meth:`Engine.reconfigure` calls.
    """

    frames: int = 0
    frozen: int = 0
    passed_through: int = 0
    failed_closed: int = 0
    over_budget: int = 0
    sink_failures: int = 0
    reconfigurations: int = 0


_TIMING_ALPHA = 0.1  # Weight of the newest sample in the moving averages.


@dataclass(frozen=True)
class StageTiming:
    """Rolling duration of one per-frame stage, in milliseconds (None before any sample)."""

    mean_ms: float | None
    """Exponentially weighted moving average (alpha 0.1)."""
    last_ms: float | None
    """The most recent sample."""


@dataclass(frozen=True)
class EngineTimings:
    """Snapshot of the engine's rolling per-frame timings, from :attr:`Engine.timings`.

    ``detect`` is the detector and refiner time of each emitted frame, ``render`` the mode plus
    stages (only frames that were rendered: a frame whose detection failed or went
    over budget is not rendered), ``write`` the time spent writing to every sink.
    ``frames_per_second`` is the inverse of the moving average of the interval
    between two emitted frames; None until two frames were emitted, or while that
    average is zero.
    """

    detect: StageTiming
    render: StageTiming
    write: StageTiming
    frames_per_second: float | None


class _Rolling:
    """Mutable exponentially weighted moving average, in seconds."""

    __slots__ = ("last", "mean")

    def __init__(self) -> None:
        self.mean: float | None = None
        self.last: float | None = None

    def add(self, seconds: float) -> None:
        self.last = seconds
        mean = self.mean
        self.mean = seconds if mean is None else mean + _TIMING_ALPHA * (seconds - mean)

    def snapshot(self) -> StageTiming:
        mean, last = self.mean, self.last
        return StageTiming(
            mean_ms=None if mean is None else mean * 1000.0,
            last_ms=None if last is None else last * 1000.0,
        )


@dataclass(frozen=True)
class _Pending:
    """One frame waiting in the lookahead buffer."""

    frame: Frame
    detections: Detections
    subject_present: bool
    """``detections.subject_present``, computed while detection could still fail closed."""
    detect_failed: bool
    over_budget: bool
    detect_seconds: float


class Engine:
    """Single-threaded, single-pass streaming engine driven one frame at a time.

    Lifecycle: :meth:`open`, then :meth:`step` per frame, then :meth:`flush`
    (optional, returns the tail outputs) and :meth:`close`; or :meth:`abort` at
    any point. :meth:`run` is that loop over a :class:`Source`.

    Per frame: run the detectors the mode requires (lazily under the mode's
    ``"first_hit"`` detection policy, all of them under ``"all"``), then the active
    refiners in order, push the result into a :class:`LookaheadBuffer`, and for
    every frame the buffer releases render it with the mode, apply the stages in
    order, write the output image and the frame's :class:`FrameMeta` to every
    active sink, and report a :class:`FrameTrace` to the observer. Sinks never
    receive the input frame. Every detector and refiner result is validated and
    normalized with :func:`~petrify.core.detections.check_detections` (boxes
    clipped to the frame, a read-only copy of the mask) before it is merged, so
    the mode sees new objects, never the detector's own.

    Fail closed: a detector, refiner, mode, or stage error replaces that frame with
    a black image, and so do a malformed detector or refiner result, a mode or
    stage image that is not a ``numpy.ndarray`` with the input frame's shape and
    dtype (or, on a frame whose final decision is subject present, that may share
    memory with the input frame image), and exceeding ``config.frame_budget_ms`` (detection plus render time,
    measured with ``clock``). The budget is checked after the fact: a running
    detector or mode is never interrupted. A detection overrun is known before the
    frame enters the lookahead buffer, so it counts as a subject and freezes its
    neighbors. A frame that goes over budget only during render is blacked without
    freezing its neighbors: its raw decision was already pushed.

    Construction raises ``ValueError`` when the mode requires detections and no
    detector or refiner provides any of them (every frame would otherwise pass
    through), when the mode's ``detect_policy`` is not a :data:`DetectPolicy`,
    when a kind in the mode's ``mandatory`` has no active detector or refiner,
    when ``mandatory`` is non-empty or a refiner is active under the
    ``"first_hit"`` policy, when an active refiner consumes a kind no active
    detector or earlier active refiner provides, when a refiner's ``consumes`` or
    ``provides`` is not a set or frozenset of :class:`DetectionKind`, when the
    same refiner object is given twice, and when reading any of these attributes
    raises.

    A failure of a critical sink, of the observer, or any ``BaseException`` aborts
    the engine (every active sink aborted, pending frames discarded) and propagates.
    A non-critical sink that fails is aborted and dropped while the run continues;
    if no sink is left the engine aborts and raises.

    Buffer ownership: the engine keeps a reference to each frame's image while the
    frame waits in the lookahead buffer, and on a pass-through frame it may hand
    that same array to the sinks and return it in :class:`Output`. A source must
    therefore not modify or reuse an image array after yielding it (the built-in
    sources and ``PushSession`` already copy). The engine does not copy.

    Reconfiguration: :meth:`reconfigure` swaps detectors, refiners, mode, stages, or
    config between two :meth:`step` calls, that is at a frame boundary. :attr:`timings`
    reports rolling detect, render and write times and the output frame rate.

    Threading: the engine is single-threaded and not thread-safe; drive it from one
    thread. Separately, it is not re-entrant: :meth:`open`, :meth:`step`,
    :meth:`reconfigure`, :meth:`flush`, :meth:`close` and :meth:`run` raise
    ``RuntimeError`` when called (for example by a sink, a stage, or the observer)
    while one of them is running.
    :meth:`abort` may be called re-entrantly: no sink receives a write after it, and
    the call it interrupted raises ``RuntimeError``.
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
        *,
        refiners: Sequence[Refiner] = (),
    ) -> None:
        self._all_detectors = tuple(detectors)
        self._all_refiners = tuple(refiners)
        self._detectors, self._refiners, self._first_hit = _resolve(
            self._all_detectors, self._all_refiners, mode
        )
        self._mode = mode
        self._stages = tuple(stages)
        self._sinks = tuple(sinks)
        self._config = config
        self._observer = observer
        self._clock = clock
        self._budget_s = _budget_seconds(config)
        self._stats = EngineStats()
        self._failures_seen = 0
        self._buffer: LookaheadBuffer[_Pending] | None = None
        self._flushed = False
        self._active: list[Sink] = []
        self._busy: str | None = None  # Name of the public call in progress.
        self._aborted_in_call = False  # abort() ran during that call.
        self._reset_timings()

    @property
    def config(self) -> EngineConfig:
        """The current configuration (read-only; change it with :meth:`reconfigure`)."""
        return self._config

    @property
    def detectors(self) -> tuple[Detector, ...]:
        """The detectors this engine runs, in order (only those the mode requires)."""
        return self._detectors

    @property
    def refiners(self) -> tuple[Refiner, ...]:
        """The refiners this engine runs, in order (only those whose output the mode requires)."""
        return self._refiners

    @property
    def mode(self) -> Mode:
        """The current mode."""
        return self._mode

    @property
    def stages(self) -> tuple[Stage, ...]:
        """The current stages, in order."""
        return self._stages

    @property
    def timings(self) -> EngineTimings:
        """Rolling per-frame timings since :meth:`open`, as an immutable snapshot.

        Measured with the engine's ``clock`` on every emitted frame, whether or not
        a frame budget is set; they never influence the budget check.
        """
        interval = self._interval.mean
        return EngineTimings(
            detect=self._detect_time.snapshot(),
            render=self._render_time.snapshot(),
            write=self._write_time.snapshot(),
            frames_per_second=1.0 / interval if interval else None,
        )

    @property
    def is_open(self) -> bool:
        """True between a successful :meth:`open` and :meth:`close` or :meth:`abort`."""
        return self._buffer is not None

    # ------------------------------------------------------------------
    # Step API
    # ------------------------------------------------------------------

    def open(self, info: SourceInfo) -> None:
        """Reset the detectors and refiners, create the lookahead buffer, and open every sink.

        Raises ``RuntimeError`` if the engine is already open. On failure every
        opened sink, and a critical sink that failed to open, is aborted once, the
        engine stays closed, and the error propagates.
        """
        with self._call("open"):
            self._open(info)

    def _open(self, info: SourceInfo) -> None:
        if self._buffer is not None:
            raise RuntimeError("Engine is already open; close or abort it first")
        self._stats = EngineStats()
        self._failures_seen = 0
        self._reset_timings()
        buffer: LookaheadBuffer[_Pending] = LookaheadBuffer(
            self._config.lookahead, self._config.cold_start_frames
        )
        for detector in self._detectors:
            detector.reset()
        for refiner in self._refiners:
            refiner.reset()
        self._buffer = buffer
        self._flushed = False
        self._active = []
        try:
            for sink in self._sinks:
                try:
                    sink.open(info)
                except Exception as exc:
                    if sink.critical:
                        # abort() is safe after a failed open: abort this one too.
                        self._active.insert(0, sink)
                        raise
                    self._drop_sink(sink, "open", exc)
                    self._check_not_aborted()
                    continue
                self._active.append(sink)
                self._check_not_aborted()
            self._require_some_sink()
        except BaseException:
            self.abort()
            raise

    def step(self, frame: Frame) -> list[Output]:
        """Process one frame; return the outputs it released, in order.

        Returns an empty list while the lookahead buffer fills. On failure the
        engine is aborted and the error propagates.
        """
        with self._call("step"):
            return self._step(frame)

    def _step(self, frame: Frame) -> list[Output]:
        buffer = self._require_streaming("step")
        try:
            pending = self._detect(frame)
            self._check_not_aborted()
            raw = pending.detect_failed or pending.over_budget or pending.subject_present
            return [self._emit(decision) for decision in buffer.push(pending, raw)]
        except BaseException:
            self.abort()
            raise

    def flush(self) -> list[Output]:
        """End the stream: render and write the frames still in the lookahead buffer.

        Returns the tail outputs, in order. After this only :meth:`close` or
        :meth:`abort` may be called. On failure the engine is aborted.
        """
        with self._call("flush"):
            return self._flush()

    def _flush(self) -> list[Output]:
        buffer = self._require_streaming("flush")
        self._flushed = True
        try:
            return [self._emit(decision) for decision in buffer.flush()]
        except BaseException:
            self.abort()
            raise

    def reconfigure(
        self,
        *,
        detectors: Sequence[Detector] | None = None,
        mode: Mode | None = None,
        stages: Sequence[Stage] | None = None,
        config: EngineConfig | None = None,
        refiners: Sequence[Refiner] | None = None,
    ) -> list[Output]:
        """Swap components at a frame boundary; return the outputs of the flushed tail.

        Call it between two :meth:`step` calls while the engine is open and not
        flushed. An argument left as None keeps the current component; ``detectors``
        (and likewise ``refiners``) replaces the whole list given to the
        constructor, and the mode in effect then picks the ones it requires,
        exactly as at construction. Build a changed config with
        ``dataclasses.replace(engine.config, lookahead=3)``. Sinks and the observer
        are not changed, and stats keep accumulating.

        Validation runs first, with the constructor's rules (required kinds,
        detection policy, mandatory kinds, refiner policy and ``consumes``): on a
        violation ``ValueError`` is raised, nothing changes, and the engine stays
        usable.

        Every reconfiguration freezes the frames around the switch: the buffered
        tail and at least one lookahead window after it, because the lookahead
        cannot see across the switch in either direction. The only exception is a
        call that changes nothing affecting decisions or rendering (every argument
        None, or a ``config`` that differs only in ``frame_budget_ms``): it applies
        the new budget, keeps the buffer as it is, and freezes nothing.

        Transition, in order:

        1. The frames still in the lookahead buffer were detected by the old
           detectors. They are rendered with the old mode and stages and written to
           the sinks, exactly as :meth:`flush` would, and returned. All of them are
           forced to ``final=True`` (``raw`` and ``filled`` keep their values).
        2. The new components are installed. Detectors that became active
           (``detectors`` given, or a new mode that activates a detector the old
           mode did not run) are ``reset()``, and so are refiners that became
           active by the same rule (``refiners`` given, or newly activated by the
           mode). Installing a refiner counts as installing a detector for the
           cold start below. A new lookahead buffer continues the
           stream's frame numbering, so indices stay contiguous for sinks and the
           observer.
        3. The frames after the switch are forced to ``final=True`` for the
           longest of: ``max(old lookahead, new lookahead)``; the full
           ``cold_start_frames`` of the new config when detectors or refiners became
           active or the config changes ``lookahead`` or ``cold_start_frames``; the
           rest of a cold-start window already in progress.

        If writing the tail fails in a way that aborts the engine (critical sink,
        observer, or abort from a callback), or a detector's or refiner's
        ``reset()`` raises,
        the engine is aborted, the error propagates, and the new components are not
        installed.

        Threading: like every other call this runs on the thread that drives the
        engine. A caller on another thread (for example a web request) must hand
        the change to that thread, which applies it between two steps; the engine
        has no locking.
        """
        with self._call("reconfigure"):
            return self._reconfigure(detectors, mode, stages, config, refiners)

    def _reconfigure(
        self,
        detectors: Sequence[Detector] | None,
        mode: Mode | None,
        stages: Sequence[Stage] | None,
        config: EngineConfig | None,
        refiners: Sequence[Refiner] | None,
    ) -> list[Output]:
        buffer = self._require_streaming("reconfigure")
        new_all = self._all_detectors if detectors is None else tuple(detectors)
        new_all_refiners = self._all_refiners if refiners is None else tuple(refiners)
        new_mode = self._mode if mode is None else mode
        new_stages = self._stages if stages is None else tuple(stages)
        new_config = self._config if config is None else config
        # Raises before any change.
        new_detectors, new_refiners, new_first_hit = _resolve(
            new_all, new_all_refiners, new_mode
        )

        old_config = self._config
        if (
            detectors is None
            and refiners is None
            and mode is None
            and stages is None
            and replace(new_config, frame_budget_ms=old_config.frame_budget_ms) == old_config
        ):
            # Nothing that affects decisions or rendering changes: keep the buffer.
            self._config = new_config
            self._budget_s = _budget_seconds(new_config)
            self._stats.reconfigurations += 1
            return []

        old_ids = {id(d) for d in self._detectors}
        if detectors is not None:
            installed = new_detectors
        else:
            installed = tuple(d for d in new_detectors if id(d) not in old_ids)
        old_refiner_ids = {id(r) for r in self._refiners}
        if refiners is not None:
            installed_refiners = new_refiners
        else:
            installed_refiners = tuple(r for r in new_refiners if id(r) not in old_refiner_ids)
        cold = bool(installed) or bool(installed_refiners) or (
            new_config.lookahead != old_config.lookahead
            or new_config.cold_start_frames != old_config.cold_start_frames
        )

        start = buffer.next_index  # Read before flush(), which rewinds it.
        cold_frames = max(buffer.cold_start_end - start, 0)  # Window already in progress.
        if cold:
            cold_frames = max(cold_frames, new_config.cold_start_frames)
        # The lookahead cannot see across the switch in either direction.
        cold_frames = max(cold_frames, old_config.lookahead, new_config.lookahead)
        try:
            decisions = [d if d.final else replace(d, final=True) for d in buffer.flush()]
            outputs = [self._emit(decision) for decision in decisions]
            for detector in installed:
                detector.reset()
            for refiner in installed_refiners:
                refiner.reset()
        except BaseException:
            self.abort()
            raise

        self._all_detectors = new_all
        self._detectors = new_detectors
        self._all_refiners = new_all_refiners
        self._refiners = new_refiners
        self._first_hit = new_first_hit
        self._mode = new_mode
        self._stages = new_stages
        self._config = new_config
        self._budget_s = _budget_seconds(new_config)
        self._buffer = LookaheadBuffer(new_config.lookahead, cold_frames, start)
        self._stats.reconfigurations += 1
        return outputs

    def close(self) -> EngineStats:
        """Flush the tail (if :meth:`flush` was not called), close every sink, return stats.

        Callers that need the tail outputs call :meth:`flush` first; otherwise the
        tail is still written to the sinks but its outputs are not returned. On
        failure of a critical sink that sink and the remaining sinks are aborted and
        the error propagates.
        """
        with self._call("close"):
            return self._close()

    def _close(self) -> EngineStats:
        if self._buffer is None:
            raise RuntimeError("Engine is not open")
        if not self._flushed:
            self._flush()
        try:
            closed = 0
            while self._active:
                sink = self._active.pop(0)
                try:
                    sink.close()
                except Exception as exc:
                    if sink.critical:
                        # Abort it along with the remaining sinks.
                        self._active.insert(0, sink)
                        raise
                    self._stats.sink_failures += 1
                    logger.warning(
                        "Non-critical sink %s failed to close and was dropped: %r",
                        type(sink).__name__,
                        exc,
                    )
                    _abort_all([sink])
                    self._check_not_aborted()
                    continue
                closed += 1
                self._check_not_aborted()
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

        Safe to call at any time, including after a failed :meth:`open`, and from
        inside another engine call (a sink, a stage, or the observer): that call
        then writes nothing more and raises ``RuntimeError``.
        """
        if self._busy is not None:
            self._aborted_in_call = True
        sinks, self._active = self._active, []
        self._buffer = None
        _abort_all(sinks)

    def run(self, source: Source) -> EngineStats:
        """Process the whole source with :meth:`open`, :meth:`step` and :meth:`close`.

        On any exception (``KeyboardInterrupt`` included) the engine is aborted and
        the exception propagates. The source is not closed.
        """
        self._reject_reentry("run")  # Before the try: must not abort the outer call.
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

    @contextmanager
    def _call(self, what: str) -> Iterator[None]:
        """Mark a public call in progress; reject a re-entrant one."""
        self._reject_reentry(what)
        self._busy = what
        self._aborted_in_call = False
        try:
            yield
        finally:
            self._busy = None
            self._aborted_in_call = False

    def _reject_reentry(self, what: str) -> None:
        """Raise when ``what`` is called while another engine call is running."""
        if self._busy is not None:
            raise RuntimeError(
                f"Engine.{what}() called re-entrantly during Engine.{self._busy}(); "
                "the engine is not re-entrant"
            )

    def _reset_timings(self) -> None:
        self._detect_time = _Rolling()
        self._render_time = _Rolling()
        self._write_time = _Rolling()
        self._interval = _Rolling()
        self._last_emit: float | None = None

    def _check_not_aborted(self) -> None:
        """Raise when :meth:`abort` was called from inside the current call."""
        if self._aborted_in_call:
            raise RuntimeError(f"Engine was aborted during Engine.{self._busy}()")

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
        """Run the detectors, then the refiners, and merge their checked results.

        Stateful detectors always run. Under the "first_hit" policy a stateless
        detector is skipped once a subject was found on the frame; under "all" every
        detector runs. After a detector raises or returns a malformed result, the
        remaining stateful detectors still see the frame so their models stay in
        step with the stream; their results are discarded, and refiners do not run.
        """
        budget = self._budget_s
        start = self._clock()
        shape = frame.image.shape
        first_hit = self._first_hit
        detections = Detections()
        present = False
        failed = False
        for detector in self._detectors:
            if not detector.stateful and (failed or (first_hit and present)):
                continue
            try:
                result = detector.detect(frame)
                if not failed:
                    merged = detections.merged(check_detections(result, detector.provides, shape))
                    present = merged.subject_present
                    detections = merged
            except Exception as exc:
                if not failed:
                    self._log_failure(frame.index, f"detector {type(detector).__name__}", exc)
                failed = True

        if not failed:
            for refiner in self._refiners:
                try:
                    result = refiner.refine(frame, detections)
                    merged = detections.merged(check_detections(result, refiner.provides, shape))
                    present = merged.subject_present
                    detections = merged
                except Exception as exc:
                    self._log_failure(frame.index, f"refiner {type(refiner).__name__}", exc)
                    failed = True
                    break

        elapsed = self._clock() - start
        over = False
        if budget is not None and not failed and elapsed > budget:
            over = True
            self._log_over_budget(frame.index, "detection", elapsed)
        return _Pending(
            frame=frame,
            detections=detections,
            subject_present=present,
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
        clock = self._clock
        self._detect_time.add(pending.detect_seconds)
        image: np.ndarray | None = None
        if not failed:
            budget = self._budget_s
            start = clock()
            where = f"mode {self._mode.name}"
            try:
                image = self._mode.render(frame, pending.detections, decision.final)
                _check_image(image, frame.image, decision.final)
                for stage in self._stages:
                    where = f"stage {type(stage).__name__}"
                    image = stage.apply(frame, image)
                    _check_image(image, frame.image, decision.final)
            except Exception as exc:
                self._log_failure(frame.index, where, exc)
                failed = True
            rendered = clock() - start
            self._render_time.add(rendered)
            if not failed and budget is not None:
                total = pending.detect_seconds + rendered
                if total > budget:
                    over = failed = True
                    self._log_over_budget(frame.index, "detection and render", total)
        if failed or image is None:
            failed = True
            image = np.zeros_like(frame.image)

        meta = frame.meta
        start = clock()
        self._write(image, meta)
        end = clock()
        self._write_time.add(end - start)
        if self._last_emit is not None:
            self._interval.add(end - self._last_emit)
        self._last_emit = end

        stats = self._stats
        stats.frames += 1
        if failed:
            stats.failed_closed += 1
            if over:
                stats.over_budget += 1
            beyond = stats.failed_closed - _MAX_LOGGED_FAILURES
            if beyond > 0 and beyond % _SUMMARY_EVERY == 0:
                logger.warning(
                    "%d frames failed closed so far (%d over budget)",
                    stats.failed_closed,
                    stats.over_budget,
                )
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
            self._check_not_aborted()
        return Output(image=image, meta=meta, trace=trace)

    def _write(self, image: np.ndarray, meta: FrameMeta) -> None:
        """Write to every active sink; drop a failing non-critical sink, raise for a critical one."""
        for sink in tuple(self._active):
            self._check_not_aborted()
            try:
                sink.write(image, meta)
            except Exception as exc:
                if sink.critical:
                    raise
                self._active.remove(sink)
                self._drop_sink(sink, "write", exc)
        self._check_not_aborted()
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


_POLICIES: tuple[DetectPolicy, ...] = ("first_hit", "all")


def _kind_names(kinds: Iterable[object]) -> str:
    """Return the kinds' values for a message; never raises on odd members."""
    try:
        return ", ".join(sorted(str(getattr(kind, "value", kind)) for kind in kinds))
    except Exception:
        return repr(kinds)


def _read(owner: str, obj: object, attr: str, *default: object) -> object:
    """Return ``getattr(obj, attr, *default)``; turn any error into ``ValueError``."""
    try:
        return getattr(obj, attr, *default)
    except Exception as exc:
        raise ValueError(f"{owner}: reading {attr} raised {exc!r}") from exc


def _kind_set(owner: str, obj: object, attr: str) -> frozenset[DetectionKind]:
    """Return ``obj.attr`` after checking it is a set or frozenset of DetectionKind."""
    value = _read(owner, obj, attr)
    if not isinstance(value, (set, frozenset)):
        raise ValueError(
            f"{owner} {attr} must be a set or frozenset of DetectionKind, "
            f"got {type(value).__name__}"
        )
    for kind in value:
        if not isinstance(kind, DetectionKind):
            raise ValueError(f"{owner} {attr} holds {kind!r}, which is not a DetectionKind")
    return frozenset(value)


def _resolve(
    detectors: Sequence[Detector], refiners: Sequence[Refiner], mode: Mode
) -> tuple[tuple[Detector, ...], tuple[Refiner, ...], bool]:
    """Return the active detectors, the active refiners, and whether the policy is first_hit.

    Raises ``ValueError`` for every configuration the engine rejects (see
    :class:`Engine`), including any error raised while reading a component's
    attributes; nothing is mutated, so callers can validate before a change.
    """
    mode_label = f"Mode {_read('Mode', mode, 'name')!r}"
    requires = _read(mode_label, mode, "requires")

    seen: set[int] = set()
    refiner_kinds: list[tuple[Refiner, str, frozenset[DetectionKind], frozenset[DetectionKind]]]
    refiner_kinds = []
    for refiner in refiners:
        label = f"Refiner {type(refiner).__name__}"
        if id(refiner) in seen:
            raise ValueError(f"{label} is given more than once; pass each refiner once")
        seen.add(id(refiner))
        consumes = _kind_set(label, refiner, "consumes")
        provides = _kind_set(label, refiner, "provides")
        refiner_kinds.append((refiner, label, consumes, provides))

    active_list: list[Detector] = []
    available: set[DetectionKind] = set()
    for detector in detectors:
        try:
            if detector.provides & requires:
                active_list.append(detector)
                available.update(detector.provides)
        except Exception as exc:
            raise ValueError(
                f"Detector {type(detector).__name__}: cannot match provides against "
                f"{mode_label} requires: {exc!r}"
            ) from exc
    active = tuple(active_list)
    try:
        active_refiners = tuple(entry for entry in refiner_kinds if entry[3] & requires)
        needs = bool(requires)
    except Exception as exc:
        raise ValueError(f"{mode_label}: requires cannot be used as a set: {exc!r}") from exc
    if needs and not active and not active_refiners:
        raise ValueError(
            f"{mode_label} requires {_kind_names(requires)} but no detector or "  # type: ignore[arg-type]
            "refiner provides any of them; every frame would pass through unprocessed"
        )

    policy = _read(mode_label, mode, "detect_policy", "first_hit")
    try:
        valid = isinstance(policy, str) and policy in _POLICIES
        first_hit = bool(policy == "first_hit")
    except Exception:
        valid = False
    if not valid:
        raise ValueError(
            f"{mode_label} has detect_policy {policy!r}; expected one of "
            + ", ".join(repr(p) for p in _POLICIES)
        )

    for _, label, consumes, provides in active_refiners:
        if first_hit:
            raise ValueError(
                f"{label} is active but {mode_label} uses the 'first_hit' detection "
                "policy, which skips detectors; refiners need 'all'"
            )
        missing = consumes - available
        if missing:
            raise ValueError(
                f"{label} consumes {_kind_names(missing)} but no active detector "
                "or earlier refiner provides it"
            )
        available.update(provides)

    mandatory = _read(mode_label, mode, "mandatory", frozenset())
    try:
        kinds = frozenset(mandatory)  # type: ignore[arg-type]
    except Exception as exc:
        raise ValueError(f"{mode_label} mandatory must be a set of DetectionKind: {exc!r}") from exc
    for kind in kinds:
        if not isinstance(kind, DetectionKind):
            raise ValueError(f"{mode_label} mandatory holds {kind!r}, which is not a DetectionKind")
    if kinds and first_hit:
        raise ValueError(
            f"{mode_label} has mandatory kinds ({_kind_names(kinds)}) but uses the "
            "'first_hit' detection policy, which can skip their detectors; the mode must "
            "declare detect_policy = \"all\""
        )
    uncovered = kinds - available
    if uncovered:
        raise ValueError(
            f"{mode_label} requires {_kind_names(uncovered)} as mandatory but no "
            "active detector or refiner provides it"
        )
    return active, tuple(entry[0] for entry in active_refiners), first_hit


def _check_image(image: object, like: np.ndarray, subject_present: bool) -> None:
    """Raise ``ValueError`` unless ``image`` is a valid output for the input image ``like``.

    It must be an ndarray with ``like``'s shape and dtype, and on a frame whose
    final decision is subject present it must not possibly share memory with
    ``like`` (``np.may_share_memory``, a bounds check), so the input pixels of a
    frozen frame cannot reach the sinks.
    """
    if type(image) is not np.ndarray:
        raise ValueError(f"output image must be a numpy.ndarray, got {type(image).__name__}")
    if image.shape != like.shape or image.dtype != like.dtype:
        raise ValueError(
            f"output image has shape {image.shape} and dtype {image.dtype}; "
            f"the input frame has shape {like.shape} and dtype {like.dtype}"
        )
    if subject_present and np.may_share_memory(image, like):
        raise ValueError(
            "output image may share memory with the input frame on a frame where a "
            "subject is present"
        )


def _budget_seconds(config: EngineConfig) -> float | None:
    """Return the per-frame budget in seconds, or None when disabled."""
    budget_ms = config.frame_budget_ms
    return None if budget_ms is None else budget_ms / 1000.0


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
