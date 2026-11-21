"""Tests for petrify.sinks.mjpeg.MjpegSink (no networking, fake clock, joined threads)."""

from __future__ import annotations

import gc
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any, ClassVar

import cv2
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
from petrify.sinks import MjpegFrame, MjpegSink
from petrify.sinks.mjpeg import MULTIPART_CONTENT_TYPE, multipart_chunk

H, W = 48, 64
INFO = SourceInfo(fps=10.0, width=W, height=H, frame_count=None, seekable=False, live=True)
JOIN = 5.0  # Upper bound for joins; tests normally finish in milliseconds.


def _image(i: int) -> np.ndarray:
    return np.full((H, W, 3), (10 + i) % 256, dtype=np.uint8)


def _meta(i: int) -> FrameMeta:
    return FrameMeta(index=i, timestamp=i / INFO.fps, source=INFO)


def _decode(frame: MjpegFrame) -> np.ndarray:
    decoded = cv2.imdecode(np.frombuffer(frame.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    return decoded


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _open_sink(**kwargs: Any) -> MjpegSink:
    kwargs.setdefault("max_fps", None)
    sink = MjpegSink(**kwargs)
    sink.open(INFO)
    return sink


class _Worker:
    """Runs ``fn`` on a thread and records its result or exception."""

    def __init__(self, fn: Callable[[], Any]) -> None:
        self.result: Any = None
        self.error: BaseException | None = None
        self.done = threading.Event()
        self._fn = fn
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            self.result = self._fn()
        except BaseException as exc:
            self.error = exc
        finally:
            self.done.set()

    def join(self) -> None:
        self.thread.join(JOIN)
        assert not self.thread.is_alive()


@pytest.fixture
def count_encodes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    real = cv2.imencode

    def counting(ext: str, image: np.ndarray, params: Any) -> Any:
        calls.append(1)
        return real(ext, image, params)

    monkeypatch.setattr(cv2, "imencode", counting)
    return calls


# ----------------------------------------------------------------------
# Protocol and validation
# ----------------------------------------------------------------------


def test_is_a_non_critical_sink() -> None:
    sink = MjpegSink()
    assert isinstance(sink, Sink)
    assert sink.critical is False
    assert MjpegSink.critical is False


@pytest.mark.parametrize("quality", [0, 101, -5, 50.0, True])
def test_bad_quality(quality: Any) -> None:
    with pytest.raises(ValueError, match="quality"):
        MjpegSink(quality=quality)


@pytest.mark.parametrize("max_fps", [0, -1.0, float("nan"), float("inf"), True])
def test_bad_max_fps(max_fps: Any) -> None:
    with pytest.raises(ValueError, match="max_fps"):
        MjpegSink(max_fps=max_fps)


def test_valid_bounds_accepted() -> None:
    MjpegSink(quality=1, max_fps=None)
    MjpegSink(quality=100, max_fps=0.5)


def test_shape_mismatch() -> None:
    sink = _open_sink()
    with pytest.raises(ValueError, match="64x48"):
        sink.write(np.zeros((H, W + 1, 3), dtype=np.uint8), _meta(0))
    with pytest.raises(ValueError):
        sink.write(np.zeros((H + 1, W, 3), dtype=np.uint8), _meta(0))
    assert sink.latest() is None


def test_write_before_open_and_double_open() -> None:
    sink = MjpegSink()
    with pytest.raises(RuntimeError, match="not open"):
        sink.write(_image(0), _meta(0))
    sink.open(INFO)
    with pytest.raises(RuntimeError, match="already open"):
        sink.open(INFO)


def test_encoder_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = _open_sink()
    monkeypatch.setattr(cv2, "imencode", lambda *a: (False, None))
    with pytest.raises(OSError, match="encoding failed"):
        sink.write(_image(0), _meta(0))
    assert sink.latest() is None


def test_encoder_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = _open_sink()

    def broken(*args: Any) -> Any:
        raise cv2.error("boom")

    monkeypatch.setattr(cv2, "imencode", broken)
    with pytest.raises(OSError, match="encoding failed") as info:
        sink.write(_image(0), _meta(0))
    assert isinstance(info.value.__cause__, cv2.error)


def test_frames_bad_timeout() -> None:
    with pytest.raises(ValueError, match="timeout"):
        _open_sink().frames(timeout=0)


# ----------------------------------------------------------------------
# Stored frames
# ----------------------------------------------------------------------


def test_stored_jpeg_decodes_to_input() -> None:
    sink = _open_sink(quality=90)
    image = np.empty((H, W, 3), dtype=np.uint8)
    image[:] = (40, 120, 200)
    sink.write(image, FrameMeta(index=7, timestamp=0.7, source=INFO))
    frame = sink.latest()
    assert frame is not None
    assert frame.sequence == 1
    assert frame.index == 7
    assert frame.timestamp == pytest.approx(0.7)
    assert (frame.width, frame.height) == (W, H)
    assert frame.jpeg[:2] == b"\xff\xd8"
    decoded = _decode(frame)
    assert decoded.shape == (H, W, 3)
    np.testing.assert_allclose(decoded.reshape(-1, 3).mean(axis=0), (40, 120, 200), atol=3)


def test_sequence_increments_and_replaces() -> None:
    sink = _open_sink()
    for i in range(3):
        sink.write(_image(i), _meta(i))
    frame = sink.latest()
    assert frame is not None
    assert (frame.sequence, frame.index) == (3, 2)


def test_no_aliasing_of_input() -> None:
    sink = _open_sink(quality=95)
    image = np.full((H, W, 3), 50, dtype=np.uint8)
    sink.write(image, _meta(0))
    image[:] = 250
    frame = sink.latest()
    assert frame is not None
    assert abs(float(_decode(frame).mean()) - 50) < 3


def test_rate_limit_skips_without_encoding(count_encodes: list[int]) -> None:
    clock = FakeClock()
    sink = MjpegSink(max_fps=10.0, clock=clock)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))  # t=0: stored.
    clock.t = 0.05
    sink.write(_image(1), _meta(1))  # Skipped.
    clock.t = 0.0999
    sink.write(_image(2), _meta(2))  # Skipped.
    assert len(count_encodes) == 1
    assert sink.skipped == 2
    latest = sink.latest()
    assert latest is not None and latest.index == 0
    clock.t = 0.1
    sink.write(_image(3), _meta(3))  # Interval elapsed: stored.
    clock.t = 0.15
    sink.write(_image(4), _meta(4))  # Skipped: measured from the last stored frame.
    clock.t = 0.25
    sink.write(_image(5), _meta(5))  # Stored.
    assert len(count_encodes) == 3
    assert sink.skipped == 3
    latest = sink.latest()
    assert latest is not None and (latest.sequence, latest.index) == (3, 5)


def test_max_fps_none_stores_everything(count_encodes: list[int]) -> None:
    clock = FakeClock()
    sink = MjpegSink(max_fps=None, clock=clock)
    sink.open(INFO)
    for i in range(5):
        sink.write(_image(i), _meta(i))
    assert len(count_encodes) == 5
    assert sink.skipped == 0
    latest = sink.latest()
    assert latest is not None and latest.sequence == 5


def test_reopen_resets_rate_limit_and_skipped() -> None:
    clock = FakeClock()
    sink = MjpegSink(max_fps=10.0, clock=clock)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    sink.write(_image(1), _meta(1))
    assert sink.skipped == 1
    sink.close()
    sink.open(INFO)
    assert sink.skipped == 0
    sink.write(_image(2), _meta(2))  # Same clock time, but a fresh run: stored.
    latest = sink.latest()
    assert latest is not None and latest.sequence == 1


# ----------------------------------------------------------------------
# latest and wait_next
# ----------------------------------------------------------------------


def test_latest_none_before_first_frame_and_after_close() -> None:
    sink = MjpegSink()
    assert sink.latest() is None
    sink.open(INFO)
    assert sink.latest() is None
    sink.write(_image(0), _meta(0))
    assert sink.latest() is not None
    sink.close()
    assert sink.latest() is None
    assert sink.closed


def test_wait_next_returns_immediately_when_newer_exists() -> None:
    sink = _open_sink()
    sink.write(_image(0), _meta(0))
    sink.write(_image(1), _meta(1))
    frame = sink.wait_next(0, timeout=0)
    assert frame is not None and frame.sequence == 2
    assert sink.wait_next(2, timeout=0) is None


def test_wait_next_blocks_until_a_frame_arrives() -> None:
    sink = _open_sink()
    sink.write(_image(0), _meta(0))
    worker = _Worker(lambda: sink.wait_next(1, timeout=JOIN))
    assert not worker.done.wait(0.02)  # Still waiting: nothing newer than 1 yet.
    sink.write(_image(1), _meta(1))
    assert worker.done.wait(JOIN)
    worker.join()
    assert worker.error is None
    assert worker.result is not None and worker.result.sequence == 2


def test_wait_next_timeout() -> None:
    sink = _open_sink()
    start = time.monotonic()
    assert sink.wait_next(0, timeout=0.01) is None
    assert time.monotonic() - start < 1.0
    assert not sink.closed


def test_wait_next_returns_none_on_close() -> None:
    sink = _open_sink()
    worker = _Worker(lambda: sink.wait_next(0, timeout=JOIN))
    assert not worker.done.wait(0.02)
    sink.close()
    assert worker.done.wait(JOIN)
    worker.join()
    assert worker.error is None and worker.result is None
    assert sink.closed


def test_wait_next_returns_none_on_abort_and_when_closed() -> None:
    sink = _open_sink()
    worker = _Worker(lambda: sink.wait_next(0, timeout=None))
    assert not worker.done.wait(0.02)
    sink.abort()
    assert worker.done.wait(JOIN)
    worker.join()
    assert worker.result is None and sink.closed
    assert sink.wait_next(0, timeout=None) is None  # Closed: returns at once.
    assert MjpegSink().wait_next(0) is None  # Never opened.


def test_write_does_not_block_on_a_stuck_reader() -> None:
    sink = _open_sink()
    sink.write(_image(0), _meta(0))
    got = threading.Event()
    release = threading.Event()

    def stuck_reader() -> MjpegFrame | None:
        frame = sink.wait_next(0, timeout=JOIN)
        got.set()
        release.wait(JOIN)  # Holds the returned frame and never reads again.
        return frame

    worker = _Worker(stuck_reader)
    try:
        assert got.wait(JOIN)
        start = time.monotonic()
        for i in range(1, 11):
            sink.write(_image(i), _meta(i))
        assert time.monotonic() - start < 2.0
        latest = sink.latest()
        assert latest is not None and latest.sequence == 11
    finally:
        release.set()
        worker.join()
    assert worker.result is not None and worker.result.sequence == 1


# ----------------------------------------------------------------------
# frames()
# ----------------------------------------------------------------------


def test_frames_yields_each_sequence_once() -> None:
    sink = _open_sink()
    sink.write(_image(0), _meta(0))
    gen = sink.frames(timeout=0.01)
    first = next(gen)
    assert first.sequence == 1
    sink.write(_image(1), _meta(1))
    assert next(gen).sequence == 2

    # No new frame: next() keeps waiting (timeouts yield nothing) until a write.
    worker = _Worker(lambda: next(gen))
    assert not worker.done.wait(0.05)  # Several 10 ms timeouts expired meanwhile.
    sink.write(_image(2), _meta(2))
    assert worker.done.wait(JOIN)
    worker.join()
    assert worker.error is None and worker.result.sequence == 3
    gen.close()


def test_frames_slow_reader_skips_to_newest() -> None:
    sink = _open_sink()
    sink.write(_image(0), _meta(0))
    gen = sink.frames()
    assert next(gen).sequence == 1
    for i in range(1, 5):
        sink.write(_image(i), _meta(i))
    frame = next(gen)
    assert (frame.sequence, frame.index) == (5, 4)
    gen.close()


def test_frames_ends_on_close() -> None:
    sink = _open_sink()
    sink.write(_image(0), _meta(0))
    gen = sink.frames()
    assert next(gen).sequence == 1
    sink.close()
    with pytest.raises(StopIteration):
        next(gen)
    assert sink.subscribers == 0


def test_frames_blocked_reader_ends_on_abort() -> None:
    sink = _open_sink()
    received: list[int] = []
    started = threading.Event()

    def consume() -> None:
        for frame in sink.frames(timeout=JOIN):
            received.append(frame.sequence)
            started.set()

    sink.write(_image(0), _meta(0))
    worker = _Worker(consume)
    assert started.wait(JOIN)
    assert sink.subscribers == 1
    sink.abort()
    assert worker.done.wait(JOIN)
    worker.join()
    assert worker.error is None
    assert received == [1]
    assert sink.subscribers == 0


def test_frames_ends_when_closed_and_reopened() -> None:
    sink = _open_sink()
    sink.write(_image(0), _meta(0))
    gen = sink.frames()
    assert next(gen).sequence == 1
    sink.close()
    sink.open(INFO)
    sink.write(_image(1), _meta(1))
    with pytest.raises(StopIteration):
        next(gen)  # Belongs to the previous run.


def test_frames_on_closed_sink_ends_immediately() -> None:
    sink = MjpegSink()
    assert list(sink.frames()) == []
    assert sink.subscribers == 0


def test_subscribers_count() -> None:
    sink = _open_sink()
    sink.write(_image(0), _meta(0))
    gen = sink.frames()
    assert sink.subscribers == 0  # Not counted before the first next.
    next(gen)
    assert sink.subscribers == 1
    other = sink.frames()
    next(other)
    assert sink.subscribers == 2
    gen.close()
    assert sink.subscribers == 1
    del other
    gc.collect()
    assert sink.subscribers == 0


def test_subscriber_dropped_on_garbage_collection_of_a_cycle() -> None:
    sink = _open_sink()
    sink.write(_image(0), _meta(0))

    cycle: list[Any] = [sink.frames()]
    cycle.append(cycle)  # Only the cycle collector can free the generator now.
    next(cycle[0])
    assert sink.subscribers == 1
    del cycle
    gc.collect()
    assert sink.subscribers == 0


def test_two_subscribers_both_receive() -> None:
    sink = _open_sink()
    received: dict[str, list[int]] = {"a": [], "b": []}
    ready = threading.Barrier(3)
    counted = threading.Condition()

    def consume(name: str) -> None:
        gen = sink.frames(timeout=JOIN)
        ready.wait(JOIN)
        for frame in gen:
            with counted:
                received[name].append(frame.sequence)
                counted.notify_all()

    workers = [_Worker(lambda n=n: consume(n)) for n in ("a", "b")]
    ready.wait(JOIN)
    try:
        for i in range(3):
            sink.write(_image(i), _meta(i))
            with counted:
                assert counted.wait_for(
                    lambda s=i + 1: all(r and r[-1] == s for r in received.values()), JOIN
                )
    finally:
        sink.close()
        for worker in workers:
            worker.join()
    for worker in workers:
        assert worker.error is None
    assert received == {"a": [1, 2, 3], "b": [1, 2, 3]}
    assert sink.subscribers == 0


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------


def test_close_and_abort_idempotent_and_safe_before_open() -> None:
    sink = MjpegSink()
    assert sink.closed
    sink.close()
    sink.abort()
    sink.open(INFO)
    assert not sink.closed
    sink.write(_image(0), _meta(0))
    sink.close()
    sink.close()
    sink.abort()
    assert sink.closed and sink.latest() is None
    with pytest.raises(RuntimeError):
        sink.write(_image(1), _meta(1))


def test_reopen_restarts_the_sequence() -> None:
    sink = _open_sink()
    for i in range(3):
        sink.write(_image(i), _meta(i))
    sink.abort()
    sink.open(INFO)
    assert sink.latest() is None
    sink.write(_image(9), _meta(9))
    frame = sink.latest()
    assert frame is not None and (frame.sequence, frame.index) == (1, 9)


# ----------------------------------------------------------------------
# multipart helper
# ----------------------------------------------------------------------


def test_multipart_chunk_exact_bytes() -> None:
    frame = MjpegFrame(jpeg=b"\xff\xd8ab", sequence=1, index=0, timestamp=0.0, width=1, height=1)
    assert multipart_chunk(frame) == (
        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 4\r\n\r\n\xff\xd8ab\r\n"
    )
    assert multipart_chunk(frame, "xyz").startswith(b"--xyz\r\n")
    assert MULTIPART_CONTENT_TYPE == "multipart/x-mixed-replace; boundary=frame"


@pytest.mark.parametrize("boundary", ["", "a\r\nb"])
def test_multipart_chunk_bad_boundary(boundary: str) -> None:
    frame = MjpegFrame(jpeg=b"x", sequence=1, index=0, timestamp=0.0, width=1, height=1)
    with pytest.raises(ValueError):
        multipart_chunk(frame, boundary)


# ----------------------------------------------------------------------
# Engine integration
# ----------------------------------------------------------------------


class _ListSource:
    def __init__(self, n: int) -> None:
        self.info = INFO
        self.n = n

    def __iter__(self) -> Iterator[Frame]:
        for i in range(self.n):
            yield Frame(image=_image(i), index=i, timestamp=i / INFO.fps, source=INFO)

    def close(self) -> None:
        pass


class _PassMode:
    name = "pass"
    requires: ClassVar[frozenset[DetectionKind]] = frozenset()

    def render(self, frame: Frame, detections: Detections, subject_present: bool) -> np.ndarray:
        return frame.image


class _ListSink:
    critical = True

    def __init__(self) -> None:
        self.indices: list[int] = []
        self.closed = False

    def open(self, info: SourceInfo) -> None:
        pass

    def write(self, image: np.ndarray, meta: FrameMeta) -> None:
        self.indices.append(meta.index)

    def close(self) -> None:
        self.closed = True

    def abort(self) -> None:
        pass


def _engine(sinks: list[Sink]) -> Engine:
    return Engine([], _PassMode(), [], sinks, EngineConfig(lookahead=0, cold_start_frames=0))


class _RecordingMjpeg(MjpegSink):
    def __init__(self) -> None:
        super().__init__(max_fps=None)
        self.stored: list[int] = []

    def write(self, image: np.ndarray, meta: FrameMeta) -> None:
        super().write(image, meta)
        frame = self.latest()
        assert frame is not None
        self.stored.append(frame.sequence)


def test_engine_run_with_mjpeg_sink() -> None:
    recording = _ListSink()
    mjpeg = _RecordingMjpeg()
    stats = _engine([recording, mjpeg]).run(_ListSource(6))
    assert recording.indices == list(range(6)) and recording.closed
    assert mjpeg.stored == [1, 2, 3, 4, 5, 6]
    assert mjpeg.closed and mjpeg.latest() is None
    assert stats.sink_failures == 0


def test_engine_drops_mjpeg_sink_on_encoder_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    real = cv2.imencode
    calls: list[int] = []

    def flaky(ext: str, image: np.ndarray, params: Any) -> Any:
        calls.append(1)
        if len(calls) == 3:
            return False, None
        return real(ext, image, params)

    monkeypatch.setattr(cv2, "imencode", flaky)
    recording = _ListSink()
    mjpeg = MjpegSink(max_fps=None)
    stats = _engine([recording, mjpeg]).run(_ListSource(6))
    assert recording.indices == list(range(6)) and recording.closed
    assert stats.sink_failures == 1
    assert len(calls) == 3  # Dropped after the failure: no further writes.
    assert mjpeg.closed and mjpeg.latest() is None
