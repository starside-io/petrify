"""Tests for petrify.sinks.threaded.ThreadedSink (event-gated fakes, no timing sleeps)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator

import numpy as np
import pytest

from petrify.core import (
    DetectionKind,
    Detections,
    Engine,
    EngineConfig,
    Frame,
    FrameMeta,
    Sink,
    SourceInfo,
)
from petrify.sinks import ThreadedSink

INFO = SourceInfo(fps=10.0, width=4, height=2, frame_count=None, seekable=False, live=True)
WAIT = 5.0  # Generous upper bound for event waits; only reached if something is broken.


def _meta(i: int) -> FrameMeta:
    return FrameMeta(index=i, timestamp=i / INFO.fps, source=INFO)


def _image(i: int) -> np.ndarray:
    return np.full((2, 4, 3), i, dtype=np.uint8)


class GatedSink:
    """Records writes; each write waits for ``gate`` and signals ``entered`` first."""

    def __init__(self, *, critical: bool = False, gated: bool = False) -> None:
        self.critical = critical
        self.gate = threading.Event()
        if not gated:
            self.gate.set()
        self.entered = threading.Event()
        self.events: list[str] = []
        self.written: list[tuple[int, np.ndarray]] = []
        self.fail_at: int | None = None
        self.threads: set[str] = set()

    def open(self, info: SourceInfo) -> None:
        self.events.append("open")

    def write(self, image: np.ndarray, meta: FrameMeta) -> None:
        self.threads.add(threading.current_thread().name)
        self.entered.set()
        assert self.gate.wait(WAIT)
        if meta.index == self.fail_at:
            raise OSError("preview broke")
        self.written.append((meta.index, image.copy()))

    def close(self) -> None:
        self.events.append("close")

    def abort(self) -> None:
        self.events.append("abort")


def _worker_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("ThreadedSink")]


@pytest.fixture(autouse=True)
def _no_leftover_threads() -> Iterator[None]:
    assert _worker_threads() == []
    yield
    assert _worker_threads() == []


def _wait_entered(inner: GatedSink) -> None:
    assert inner.entered.wait(WAIT)
    inner.entered.clear()


def test_satisfies_protocol_and_critical_mirrors() -> None:
    assert isinstance(ThreadedSink(GatedSink()), Sink)
    assert ThreadedSink(GatedSink(critical=True)).critical is True
    assert ThreadedSink(GatedSink(critical=False)).critical is False


def test_invalid_queue_size_rejected() -> None:
    with pytest.raises(ValueError, match="max_queue"):
        ThreadedSink(GatedSink(), max_queue=0)


def test_forwards_everything_in_order_on_worker_thread() -> None:
    inner = GatedSink()
    sink = ThreadedSink(inner, max_queue=64)
    sink.open(INFO)
    for i in range(20):
        sink.write(_image(i), _meta(i))
    sink.close()
    assert [i for i, _ in inner.written] == list(range(20))
    for i, image in inner.written:
        assert np.array_equal(image, _image(i))
    assert inner.events == ["open", "close"]
    assert sink.dropped == 0
    assert threading.current_thread().name not in inner.threads


def test_write_copies_the_image() -> None:
    inner = GatedSink(gated=True)
    sink = ThreadedSink(inner, max_queue=4)
    sink.open(INFO)
    shared = _image(0)
    sink.write(shared, _meta(0))
    _wait_entered(inner)
    sink.write(shared, _meta(1))
    shared[:] = 99  # The caller reuses its buffer right after write returns.
    inner.gate.set()
    sink.close()
    assert [i for i, _ in inner.written] == [0, 1]
    for _, image in inner.written:
        assert not image.any()


def test_drops_oldest_when_full() -> None:
    inner = GatedSink(gated=True)
    sink = ThreadedSink(inner, max_queue=2, drop_when_full=True)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    _wait_entered(inner)  # The worker holds frame 0; the queue is empty.
    for i in range(1, 6):
        sink.write(_image(i), _meta(i))
    assert sink.dropped == 3  # Frames 1, 2, 3 were the oldest queued.
    inner.gate.set()
    sink.close()
    assert [i for i, _ in inner.written] == [0, 4, 5]
    assert inner.events == ["open", "close"]


def test_blocks_instead_of_dropping() -> None:
    inner = GatedSink(gated=True)
    sink = ThreadedSink(inner, max_queue=1, drop_when_full=False)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    _wait_entered(inner)
    sink.write(_image(1), _meta(1))  # Fills the queue.

    returned = threading.Event()

    def writer() -> None:
        sink.write(_image(2), _meta(2))
        returned.set()

    thread = threading.Thread(target=writer)
    thread.start()
    assert not returned.wait(0.02)  # Still blocked on the full queue.
    inner.gate.set()
    assert returned.wait(WAIT)
    thread.join(WAIT)
    sink.close()
    assert [i for i, _ in inner.written] == [0, 1, 2]
    assert sink.dropped == 0


def test_worker_error_surfaces_on_next_write() -> None:
    inner = GatedSink()
    inner.fail_at = 0
    sink = ThreadedSink(inner, max_queue=4)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    assert sink._thread is not None
    sink._thread.join(WAIT)  # The worker exits right after storing the error.
    assert not sink._thread.is_alive()
    with pytest.raises(OSError, match="preview broke"):
        sink.write(_image(1), _meta(1))
    with pytest.raises(OSError, match="preview broke"):
        sink.write(_image(2), _meta(2))
    sink.abort()
    assert inner.events == ["open", "abort"]


def test_worker_error_surfaces_on_close() -> None:
    inner = GatedSink(gated=True)
    inner.fail_at = 0
    sink = ThreadedSink(inner, max_queue=4)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    inner.gate.set()
    with pytest.raises(OSError, match="preview broke"):
        sink.close()
    assert inner.events == ["open", "abort"]
    sink.close()
    sink.abort()
    assert inner.events == ["open", "abort"]


def test_close_drains_queue() -> None:
    inner = GatedSink(gated=True)
    sink = ThreadedSink(inner, max_queue=8)
    sink.open(INFO)
    for i in range(6):
        sink.write(_image(i), _meta(i))
    inner.gate.set()
    sink.close()
    assert [i for i, _ in inner.written] == list(range(6))
    assert inner.events == ["open", "close"]
    sink.close()
    sink.abort()
    assert inner.events == ["open", "close"]


def test_abort_discards_queue() -> None:
    inner = GatedSink(gated=True)
    sink = ThreadedSink(inner, max_queue=8)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    _wait_entered(inner)
    for i in range(1, 6):
        sink.write(_image(i), _meta(i))

    def release_after_abort_starts() -> None:
        # Abort sets the flag and empties the queue under the lock; only then let the
        # in-flight write of frame 0 finish.
        with sink._cond:
            sink._cond.wait_for(lambda: sink._aborting, WAIT)
        inner.gate.set()

    releaser = threading.Thread(target=release_after_abort_starts)
    releaser.start()
    sink.abort()
    releaser.join(WAIT)
    assert [i for i, _ in inner.written] == [0]
    assert inner.events == ["open", "abort"]
    sink.abort()
    sink.close()
    assert inner.events == ["open", "abort"]


def test_write_after_close_or_before_open_raises() -> None:
    sink = ThreadedSink(GatedSink())
    with pytest.raises(RuntimeError, match="not open"):
        sink.write(_image(0), _meta(0))
    sink.open(INFO)
    sink.close()
    with pytest.raises(RuntimeError, match="closed"):
        sink.write(_image(0), _meta(0))


def test_abort_without_open_forwards_abort() -> None:
    inner = GatedSink()
    sink = ThreadedSink(inner)
    sink.abort()
    sink.abort()
    assert inner.events == ["abort"]


def test_threaded_non_critical_sink_dropped_by_engine() -> None:
    from petrify.core import DetectionKind, Detections, Engine, EngineConfig, Frame

    class Mode:
        name = "black"
        requires = frozenset({DetectionKind.MOTION})

        def render(self, frame: Frame, detections: Detections, present: bool) -> np.ndarray:
            return np.zeros_like(frame.image)

    class Detector:
        provides = frozenset({DetectionKind.MOTION})
        stateful = False

        def detect(self, frame: Frame) -> Detections:
            return Detections(motion=False)

        def reset(self) -> None:
            pass

    recording = GatedSink(critical=True)
    preview_inner = GatedSink()
    preview_inner.fail_at = 0
    preview = ThreadedSink(preview_inner, max_queue=1, drop_when_full=False)
    engine = Engine([Detector()], Mode(), [], [recording, preview], EngineConfig(lookahead=0))
    engine.open(INFO)
    for i in range(50):
        engine.step(Frame(image=_image(i), index=i, timestamp=0.0, source=INFO))
    stats = engine.close()
    assert [i for i, _ in recording.written] == list(range(50))
    assert recording.events == ["open", "close"]
    assert preview_inner.events == ["open", "abort"]
    assert stats.sink_failures == 1


# ---------------------------------------------------------------------------
# Review fixes (P2-8a)
# ---------------------------------------------------------------------------

SHORT = 0.01  # A join or write timeout that is certain to expire: the fake is blocked on an event.


class OverlapSink:
    """Records every call; ``write`` blocks on ``release``. Flags any two overlapping calls."""

    def __init__(self, *, critical: bool = False) -> None:
        self.critical = critical
        self.release = threading.Event()
        self.entered = threading.Event()
        self.calls: list[str] = []
        self.overlap = False
        self._lock = threading.Lock()
        self._active = 0

    def _enter(self, name: str) -> None:
        with self._lock:
            self._active += 1
            if self._active > 1:
                self.overlap = True
            self.calls.append(name)

    def _leave(self) -> None:
        with self._lock:
            self._active -= 1

    def open(self, info: SourceInfo) -> None:
        self._enter("open")
        self._leave()

    def write(self, image: np.ndarray, meta: FrameMeta) -> None:
        self._enter("write")
        try:
            self.entered.set()
            assert self.release.wait(WAIT)
        finally:
            self._leave()

    def close(self) -> None:
        self._enter("close")
        self._leave()

    def abort(self) -> None:
        self._enter("abort")
        self._leave()


class _MotionDetector:
    provides = frozenset({DetectionKind.MOTION})
    stateful = False

    def detect(self, frame: Frame) -> Detections:
        return Detections(motion=False)

    def reset(self) -> None:
        pass


class _BlackMode:
    name = "black"
    requires = frozenset({DetectionKind.MOTION})

    def render(self, frame: Frame, detections: Detections, present: bool) -> np.ndarray:
        return np.zeros_like(frame.image)


def _black_engine(sinks: list[Sink]) -> Engine:
    return Engine([_MotionDetector()], _BlackMode(), [], sinks, EngineConfig(lookahead=0))


def _frame(i: int) -> Frame:
    return Frame(image=_image(i), index=i, timestamp=0.0, source=INFO)


def _join_worker(sink: ThreadedSink) -> None:
    assert sink._thread is not None
    sink._thread.join(WAIT)
    assert not sink._thread.is_alive()


def test_t1_abort_timeout_hands_abort_to_worker(caplog: pytest.LogCaptureFixture) -> None:
    inner = OverlapSink()
    sink = ThreadedSink(inner, max_queue=4, join_timeout=SHORT)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    assert inner.entered.wait(WAIT)
    with caplog.at_level(logging.WARNING, logger="petrify.sinks.threaded"):
        sink.abort()  # The worker is stuck inside write: abort must not run concurrently.
    assert "abort" not in inner.calls
    assert any("still running" in r.getMessage() for r in caplog.records)
    inner.release.set()
    _join_worker(sink)
    sink.abort()
    sink.close()
    assert inner.calls == ["open", "write", "abort"]
    assert not inner.overlap


def test_t1_close_timeout_hands_abort_to_worker() -> None:
    inner = OverlapSink(critical=True)
    sink = ThreadedSink(inner, max_queue=4, join_timeout=SHORT)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    sink.write(_image(1), _meta(1))
    assert inner.entered.wait(WAIT)
    with pytest.raises(TimeoutError):
        sink.close()
    assert "abort" not in inner.calls
    inner.release.set()
    _join_worker(sink)
    sink.close()
    sink.abort()
    assert inner.calls == ["open", "write", "abort"]  # Frame 1 was discarded, not written.
    assert not inner.overlap


def test_t2_blocked_write_times_out() -> None:
    inner = GatedSink(gated=True, critical=True)
    sink = ThreadedSink(inner, max_queue=1, write_timeout=SHORT)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    _wait_entered(inner)
    sink.write(_image(1), _meta(1))  # Fills the queue.
    with pytest.raises(TimeoutError, match="blocked"):
        sink.write(_image(2), _meta(2))
    inner.gate.set()
    sink.close()
    assert [i for i, _ in inner.written] == [0, 1]
    assert sink.dropped == 0


def test_t2_write_timeout_none_and_validation() -> None:
    with pytest.raises(ValueError, match="write_timeout"):
        ThreadedSink(GatedSink(), write_timeout=0.0)
    inner = GatedSink(gated=True)
    sink = ThreadedSink(inner, max_queue=1, drop_when_full=False, write_timeout=None)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    _wait_entered(inner)
    sink.write(_image(1), _meta(1))
    returned = threading.Event()

    def writer() -> None:
        sink.write(_image(2), _meta(2))
        returned.set()

    thread = threading.Thread(target=writer)
    thread.start()
    assert not returned.wait(0.02)  # Unbounded wait: still blocked.
    inner.gate.set()
    assert returned.wait(WAIT)
    thread.join(WAIT)
    assert not thread.is_alive()
    sink.close()
    assert [i for i, _ in inner.written] == [0, 1, 2]


def test_t2_engine_drops_stuck_non_critical_block_mode_sink() -> None:
    recording = GatedSink(critical=True)
    stuck = OverlapSink()
    preview = ThreadedSink(
        stuck, max_queue=1, drop_when_full=False, join_timeout=SHORT, write_timeout=SHORT
    )
    engine = _black_engine([recording, preview])
    engine.open(INFO)
    for i in range(5):
        engine.step(_frame(i))
    stats = engine.close()
    assert [i for i, _ in recording.written] == list(range(5))
    assert recording.events == ["open", "close"]
    assert stats.sink_failures == 1
    stuck.release.set()
    _join_worker(preview)
    assert stuck.calls == ["open", "write", "abort"]
    assert not stuck.overlap


def test_t3_critical_sink_blocks_by_default() -> None:
    inner = GatedSink(gated=True, critical=True)
    sink = ThreadedSink(inner, max_queue=1, write_timeout=SHORT)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    _wait_entered(inner)
    sink.write(_image(1), _meta(1))
    with pytest.raises(TimeoutError):
        sink.write(_image(2), _meta(2))  # Blocks (then times out) instead of dropping.
    assert sink.dropped == 0
    inner.gate.set()
    sink.close()
    assert [i for i, _ in inner.written] == [0, 1]


def test_t3_non_critical_sink_drops_by_default() -> None:
    inner = GatedSink(gated=True)
    sink = ThreadedSink(inner, max_queue=1)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    _wait_entered(inner)
    sink.write(_image(1), _meta(1))
    sink.write(_image(2), _meta(2))
    assert sink.dropped == 1
    inner.gate.set()
    sink.close()
    assert [i for i, _ in inner.written] == [0, 2]


def test_t3_drop_mode_rejected_for_critical_sink() -> None:
    with pytest.raises(ValueError, match="critical"):
        ThreadedSink(GatedSink(critical=True), drop_when_full=True)
    ThreadedSink(GatedSink(critical=True), drop_when_full=False)


class _RaceProbe(ThreadedSink):
    """Before the fix, close and abort read ``_done`` and then set it, outside the lock.

    Once armed, the first two reads wait for each other, forcing that interleaving.
    The fixed sink does not use ``_done``, so there the probe is inert.
    """

    def __init__(self, sink: Sink) -> None:
        self._probe = threading.Barrier(2)
        self._probe_reads = 0
        self._probe_lock = threading.Lock()
        self._done_value = False
        self.armed = False
        super().__init__(sink, max_queue=4)

    @property
    def _done(self) -> bool:
        with self._probe_lock:
            if self.armed:
                self._probe_reads += 1
            wait = self.armed and self._probe_reads <= 2
            value = self._done_value
        if wait:
            try:
                self._probe.wait(WAIT)
            except threading.BrokenBarrierError:
                pass
        return value

    @_done.setter
    def _done(self, value: bool) -> None:
        self._done_value = value


def _close_and_abort_concurrently(sink: ThreadedSink) -> list[BaseException]:
    errors: list[BaseException] = []
    start = threading.Barrier(2)

    def call(fn: Callable[[], None]) -> None:
        start.wait(WAIT)
        try:
            fn()
        except BaseException as exc:  # Recorded and asserted on by the caller.
            errors.append(exc)

    threads = [threading.Thread(target=call, args=(f,)) for f in (sink.close, sink.abort)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(WAIT)
        assert not thread.is_alive()
    return errors


def test_t4_close_abort_race_forced_interleaving() -> None:
    inner = GatedSink()
    sink = _RaceProbe(inner)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    sink.armed = True
    assert _close_and_abort_concurrently(sink) == []
    assert inner.events in (["open", "close"], ["open", "abort"])


def test_t4_close_abort_race_exactly_one_terminal_call() -> None:
    for _ in range(50):
        inner = GatedSink()
        sink = ThreadedSink(inner, max_queue=4)
        sink.open(INFO)
        sink.write(_image(0), _meta(0))
        assert _close_and_abort_concurrently(sink) == []
        assert inner.events in (["open", "close"], ["open", "abort"])


def test_t5_reopen_after_close_and_after_abort() -> None:
    inner = GatedSink(gated=True)
    sink = ThreadedSink(inner, max_queue=1)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    _wait_entered(inner)
    sink.write(_image(1), _meta(1))
    sink.write(_image(2), _meta(2))
    assert sink.dropped == 1
    inner.gate.set()
    sink.close()
    assert [i for i, _ in inner.written] == [0, 2]
    sink.open(INFO)
    assert sink.dropped == 0
    sink.abort()
    sink.open(INFO)
    sink.write(_image(4), _meta(4))
    sink.close()
    assert inner.events == ["open", "close", "open", "abort", "open", "close"]
    assert [i for i, _ in inner.written] == [0, 2, 4]


def test_t5_open_while_open_raises() -> None:
    inner = GatedSink()
    sink = ThreadedSink(inner)
    sink.open(INFO)
    with pytest.raises(RuntimeError, match="already open"):
        sink.open(INFO)
    sink.close()
    assert inner.events == ["open", "close"]


def test_t5_engine_reuse_keeps_non_critical_threaded_sink() -> None:
    inner = GatedSink()
    preview = ThreadedSink(inner, max_queue=64)
    engine = _black_engine([preview])
    for _ in range(2):
        engine.open(INFO)
        for i in range(3):
            engine.step(_frame(i))
        stats = engine.close()
        assert stats.sink_failures == 0
    assert inner.events == ["open", "close", "open", "close"]
    assert [i for i, _ in inner.written] == [0, 1, 2, 0, 1, 2]


class _Fatal(BaseException):
    """A non-Exception error raised by the wrapped sink."""


class _FatalSink(GatedSink):
    def write(self, image: np.ndarray, meta: FrameMeta) -> None:
        raise _Fatal("fatal")


def test_t6_base_exception_in_worker_surfaces_as_runtime_error() -> None:
    inner = _FatalSink()
    sink = ThreadedSink(inner, max_queue=4)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    _join_worker(sink)
    with pytest.raises(RuntimeError) as info:
        sink.write(_image(1), _meta(1))
    assert isinstance(info.value.__cause__, _Fatal)
    with pytest.raises(RuntimeError):
        sink.close()
    assert inner.events == ["open", "abort"]


def test_t6_engine_drops_non_critical_sink_on_worker_base_exception() -> None:
    recording, inner = GatedSink(critical=True), _FatalSink()
    preview = ThreadedSink(inner, max_queue=1, drop_when_full=False)
    engine = _black_engine([recording, preview])
    engine.open(INFO)
    for i in range(20):
        engine.step(_frame(i))
    stats = engine.close()
    assert [i for i, _ in recording.written] == list(range(20))
    assert recording.events == ["open", "close"]
    assert inner.events == ["open", "abort"]
    assert stats.sink_failures == 1


# ---------------------------------------------------------------------------
# Review fixes (P2-9b)
# ---------------------------------------------------------------------------


class _InterruptingSink(ThreadedSink):
    """Raises ``KeyboardInterrupt`` from one internal step of ``close`` once armed.

    ``step`` names the method that stands in for the interrupted code: the wait for
    the worker, or the join of the worker thread.
    """

    def __init__(self, sink: Sink, step: str, **kwargs: object) -> None:
        super().__init__(sink, **kwargs)  # type: ignore[arg-type]
        self.step = step
        self.armed = False

    def _interrupt(self, step: str) -> None:
        if self.armed and self.step == step:
            self.armed = False
            raise KeyboardInterrupt

    def _wait_for_worker(self) -> bool:
        self._interrupt("wait")
        return super()._wait_for_worker()

    def _join_thread(self) -> None:
        self._interrupt("join")
        super()._join_thread()


class _CloseFailsSink(OverlapSink):
    def __init__(self, error: BaseException, abort_error: Exception | None = None) -> None:
        super().__init__(critical=True)
        self.release.set()
        self.error = error
        self.abort_error = abort_error

    def close(self) -> None:
        super().close()
        raise self.error

    def abort(self) -> None:
        super().abort()
        if self.abort_error is not None:
            raise self.abort_error


def test_p2_9b_t1_interrupt_while_waiting_hands_abort_to_worker() -> None:
    inner = OverlapSink(critical=True)
    sink = _InterruptingSink(inner, "wait", max_queue=4)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    sink.write(_image(1), _meta(1))
    assert inner.entered.wait(WAIT)  # The worker is inside the wrapped write.
    sink.armed = True
    with pytest.raises(KeyboardInterrupt):
        sink.close()
    assert "abort" not in inner.calls  # Never concurrently with the in-flight write.
    sink.abort()  # No-op: the abort is already handed to the worker.
    inner.release.set()
    _join_worker(sink)
    sink.abort()
    sink.close()
    assert inner.calls == ["open", "write", "abort"]  # Frame 1 discarded, not written.
    assert not inner.overlap


def test_p2_9b_t1_interrupt_after_worker_exit_aborts_on_caller() -> None:
    inner = OverlapSink(critical=True)
    inner.release.set()
    sink = _InterruptingSink(inner, "join", max_queue=4)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    sink.armed = True
    with pytest.raises(KeyboardInterrupt):
        sink.close()
    assert inner.calls == ["open", "write", "abort"]
    assert sink._thread is not None and not sink._thread.is_alive()
    sink.abort()
    sink.close()
    assert inner.calls == ["open", "write", "abort"]
    assert not inner.overlap


def test_p2_9b_t1_interrupt_inside_wrapped_close_aborts_once() -> None:
    inner = _CloseFailsSink(KeyboardInterrupt())
    sink = ThreadedSink(inner, max_queue=4)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    with pytest.raises(KeyboardInterrupt):
        sink.close()
    sink.abort()
    sink.close()
    assert inner.calls == ["open", "write", "close", "abort"]
    assert not inner.overlap


def test_p2_9b_t2_wrapped_close_error_aborts_once_then_propagates() -> None:
    inner = _CloseFailsSink(OSError("disk full"))
    sink = ThreadedSink(inner, max_queue=4)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    with pytest.raises(OSError, match="disk full"):
        sink.close()
    sink.abort()
    sink.close()
    assert inner.calls == ["open", "write", "close", "abort"]


def test_p2_9b_t2_failing_abort_does_not_hide_the_close_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    inner = _CloseFailsSink(OSError("disk full"), abort_error=RuntimeError("abort broke"))
    sink = ThreadedSink(inner, max_queue=4)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    with caplog.at_level(logging.ERROR, logger="petrify.sinks.threaded"):
        with pytest.raises(OSError, match="disk full"):
            sink.close()
    assert inner.calls == ["open", "write", "close", "abort"]
    assert any("failed to abort" in r.getMessage() for r in caplog.records)


def test_p2_9b_t2_engine_drops_sink_whose_close_fails_and_it_is_aborted_once() -> None:
    recording = GatedSink(critical=True)
    inner = _CloseFailsSink(OSError("preview gone"))
    inner.critical = False
    preview = ThreadedSink(inner, max_queue=64)
    engine = _black_engine([recording, preview])
    engine.open(INFO)
    for i in range(3):
        engine.step(_frame(i))
    stats = engine.close()
    assert stats.sink_failures == 1
    assert recording.events == ["open", "close"]
    assert inner.calls == ["open", "write", "write", "write", "close", "abort"]
