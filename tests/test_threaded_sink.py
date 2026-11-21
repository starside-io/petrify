"""Tests for petrify.sinks.threaded.ThreadedSink (event-gated fakes, no timing sleeps)."""

from __future__ import annotations

import threading
from collections.abc import Iterator

import numpy as np
import pytest

from petrify.core import FrameMeta, Sink, SourceInfo
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
