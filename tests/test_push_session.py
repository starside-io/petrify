"""Unit tests for petrify.sources.push.PushSession with a real Engine and small fakes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import pytest

from petrify.core import (
    DetectionKind,
    Detections,
    Engine,
    EngineConfig,
    Frame,
    FrameMeta,
    Output,
    SourceInfo,
)
from petrify.sources.push import PushSession

H, W = 4, 6
FPS = 10.0
BACKGROUND_VALUE = 7


def _image(i: int) -> np.ndarray:
    """Input ``i``: every pixel is ``100 + i``, so real content is never zero."""
    return np.full((H, W, 3), 100 + i, dtype=np.uint8)


class FakeDetector:
    """Reports motion on the frames in ``hits``; raises on the frames in ``fail``."""

    provides: ClassVar[frozenset[DetectionKind]] = frozenset({DetectionKind.MOTION})
    stateful = True

    def __init__(self, hits: Sequence[int] = (), fail: Sequence[int] = ()) -> None:
        self.hits = set(hits)
        self.fail = set(fail)
        self.seen: list[int] = []

    def detect(self, frame: Frame) -> Detections:
        self.seen.append(frame.index)
        if frame.index in self.fail:
            raise RuntimeError(f"detector broke on {frame.index}")
        return Detections(motion=frame.index in self.hits)

    def reset(self) -> None:
        pass


class FakeMode:
    """Frozen frames become a constant background; others pass the input array through."""

    name = "fake"
    requires: ClassVar[frozenset[DetectionKind]] = frozenset({DetectionKind.MOTION})

    def __init__(self) -> None:
        self.frames: list[tuple[int, float]] = []

    def render(self, frame: Frame, detections: Detections, subject_present: bool) -> np.ndarray:
        self.frames.append((frame.index, frame.timestamp))
        if subject_present:
            return np.full_like(frame.image, BACKGROUND_VALUE)
        return frame.image


class FakeSink:
    def __init__(self, fail_write_at: int | None = None) -> None:
        self.critical = True
        self.events: list[str] = []
        self.written: list[np.ndarray] = []
        self.info: SourceInfo | None = None
        self.fail_write_at = fail_write_at

    def open(self, info: SourceInfo) -> None:
        self.info = info
        self.events.append("open")

    def write(self, image: np.ndarray, meta: FrameMeta) -> None:
        if meta.index == self.fail_write_at:
            raise OSError("sink broke")
        self.written.append(image.copy())

    def close(self) -> None:
        self.events.append("close")

    def abort(self) -> None:
        self.events.append("abort")


def _engine(
    *,
    lookahead: int = 0,
    detector: FakeDetector | None = None,
    mode: FakeMode | None = None,
    sinks: Sequence[FakeSink] = (),
) -> Engine:
    return Engine(
        [detector or FakeDetector()],
        mode or FakeMode(),
        [],
        list(sinks),
        EngineConfig(lookahead=lookahead, cold_start_frames=0),
    )


def _value(image: np.ndarray) -> int:
    """The single pixel value of a constant image."""
    values = np.unique(image)
    assert values.size == 1
    return int(values[0])


# ---------------------------------------------------------------------------
# Latency and ordering
# ---------------------------------------------------------------------------


def test_lookahead_zero_returns_one_image_per_push() -> None:
    session = PushSession(_engine(lookahead=0), W, H, FPS)
    for i in range(5):
        out = session.push(_image(i))
        assert out is not None
        assert _value(out) == 100 + i
    assert session.finish() == []


@pytest.mark.parametrize("k", [1, 3])
def test_lookahead_k_delays_outputs(k: int) -> None:
    n = 7
    session = PushSession(_engine(lookahead=k), W, H, FPS)
    returned: list[int] = []
    for i in range(n):
        out = session.push(_image(i))
        if i < k:
            assert out is None
        else:
            assert out is not None
            assert _value(out) == 100 + i - k
            returned.append(_value(out))
    tail = session.finish()
    assert len(tail) == k
    returned.extend(_value(img) for img in tail)
    assert returned == [100 + i for i in range(n)]


def test_returned_images_match_sink() -> None:
    sink = FakeSink()
    detector = FakeDetector(hits=[2])
    session = PushSession(_engine(lookahead=2, detector=detector, sinks=[sink]), W, H, FPS)
    returned: list[np.ndarray] = []
    for i in range(6):
        out = session.push(_image(i))
        if out is not None:
            returned.append(out.copy())
    returned.extend(img.copy() for img in session.finish())
    assert len(returned) == len(sink.written) == 6
    for got, written in zip(returned, sink.written, strict=True):
        np.testing.assert_array_equal(got, written)
    assert [_value(img) for img in returned] == [7, 7, 7, 7, 7, 105]
    assert sink.events == ["open", "close"]


def test_works_with_zero_sinks() -> None:
    session = PushSession(_engine(lookahead=1), W, H, FPS)
    assert session.push(_image(0)) is None
    out = session.push(_image(1))
    assert out is not None and _value(out) == 100
    assert [_value(img) for img in session.finish()] == [101]
    assert session.stats is not None and session.stats.frames == 2


# ---------------------------------------------------------------------------
# Index and timestamp
# ---------------------------------------------------------------------------


def test_default_timestamps_and_contiguous_indices() -> None:
    mode = FakeMode()
    session = PushSession(_engine(mode=mode), W, H, FPS)
    for i in range(4):
        session.push(_image(i))
    session.finish()
    assert mode.frames == [(i, i / FPS) for i in range(4)]


def test_explicit_timestamps_passed_through() -> None:
    mode = FakeMode()
    session = PushSession(_engine(mode=mode), W, H, FPS)
    for i, ts in enumerate([0.5, 0.5, 1.25, 9.0]):
        session.push(_image(i), timestamp=ts)
    session.finish()
    assert mode.frames == [(0, 0.5), (1, 0.5), (2, 1.25), (3, 9.0)]


def test_decreasing_timestamp_raises_without_consuming_index() -> None:
    mode = FakeMode()
    detector = FakeDetector()
    session = PushSession(_engine(mode=mode, detector=detector), W, H, FPS)
    session.push(_image(0), timestamp=2.0)
    with pytest.raises(ValueError, match="lower than the previous"):
        session.push(_image(1), timestamp=1.0)
    assert detector.seen == [0]
    session.push(_image(1), timestamp=3.0)
    session.finish()
    assert mode.frames == [(0, 2.0), (1, 3.0)]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((H + 1, W, 3), dtype=np.uint8),
        np.zeros((H, W, 4), dtype=np.uint8),
        np.zeros((H, W, 3), dtype=np.float32),
        np.zeros((H, W, 3), dtype=np.uint16),
        np.zeros((H, W), dtype=np.uint8),
        [[[0, 0, 0]] * W] * H,
        None,
    ],
    ids=["height", "channels", "float32", "uint16", "grayscale", "list", "none"],
)
def test_invalid_image_raises_and_session_stays_usable(bad: object) -> None:
    detector = FakeDetector()
    mode = FakeMode()
    session = PushSession(_engine(detector=detector, mode=mode), W, H, FPS)
    session.push(_image(0))
    with pytest.raises(ValueError):
        session.push(bad)  # type: ignore[arg-type]
    assert detector.seen == [0]
    out = session.push(_image(1))
    assert out is not None and _value(out) == 101
    session.finish()
    assert [index for index, _ in mode.frames] == [0, 1]


# ---------------------------------------------------------------------------
# Ownership
# ---------------------------------------------------------------------------


def test_mutating_input_after_push_does_not_leak() -> None:
    k = 2
    session = PushSession(_engine(lookahead=k), W, H, FPS)
    buffer = np.empty((H, W, 3), dtype=np.uint8)  # One array reused for every push.
    returned: list[np.ndarray] = []
    snapshots: list[np.ndarray] = []
    for i in range(6):
        buffer[...] = 100 + i
        out = session.push(buffer)
        buffer[...] = 0  # Caller scribbles over its array right after the push.
        if out is not None:
            assert not np.shares_memory(out, buffer)
            returned.append(out)
            snapshots.append(out.copy())
    for img in session.finish():
        assert not np.shares_memory(img, buffer)
        returned.append(img)
        snapshots.append(img.copy())
    buffer[...] = 255
    assert [_value(img) for img in returned] == [100 + i for i in range(6)]
    for img, snap in zip(returned, snapshots, strict=True):
        np.testing.assert_array_equal(img, snap)


def test_output_does_not_share_memory_with_input() -> None:
    session = PushSession(_engine(lookahead=0), W, H, FPS)
    inputs = [_image(i) for i in range(3)]
    outputs = [session.push(img) for img in inputs]
    for out in outputs:
        assert out is not None
        for img in inputs:
            assert not np.shares_memory(out, img)
    session.finish()


def test_non_contiguous_input_is_accepted() -> None:
    wide = np.full((H, W * 2, 3), 103, dtype=np.uint8)
    view = wide[:, ::2]
    assert not view.flags.c_contiguous
    session = PushSession(_engine(), W, H, FPS)
    out = session.push(view)
    assert out is not None and out.flags.c_contiguous and _value(out) == 103
    session.finish()


# ---------------------------------------------------------------------------
# Fail closed
# ---------------------------------------------------------------------------


def test_failed_detector_frame_comes_back_black() -> None:
    detector = FakeDetector(fail=[2])
    session = PushSession(_engine(lookahead=1, detector=detector), W, H, FPS)
    returned: list[np.ndarray] = []
    for i in range(5):
        out = session.push(_image(i))
        if out is not None:
            returned.append(out)
    returned.extend(session.finish())
    assert len(returned) == 5
    assert not returned[2].any()
    assert all(not np.any(img == 102) for img in returned)
    # Frames next to the failure are frozen by gap fill (raw True on failure).
    assert [_value(img) for img in returned] == [100, BACKGROUND_VALUE, 0, BACKGROUND_VALUE, 104]
    assert session.stats is not None and session.stats.failed_closed == 1


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_push_after_finish_raises() -> None:
    session = PushSession(_engine(), W, H, FPS)
    session.push(_image(0))
    session.finish()
    with pytest.raises(RuntimeError):
        session.push(_image(1))


def test_push_after_abort_raises() -> None:
    sink = FakeSink()
    session = PushSession(_engine(lookahead=2, sinks=[sink]), W, H, FPS)
    session.push(_image(0))
    session.abort()
    assert sink.events == ["open", "abort"]
    with pytest.raises(RuntimeError):
        session.push(_image(1))
    with pytest.raises(RuntimeError):
        session.finish()


def test_finish_twice_raises_and_abort_is_idempotent() -> None:
    sink = FakeSink()
    session = PushSession(_engine(sinks=[sink]), W, H, FPS)
    session.push(_image(0))
    session.finish()
    with pytest.raises(RuntimeError):
        session.finish()
    session.abort()
    session.abort()
    assert sink.events == ["open", "close"]
    assert session.stats is not None


def test_finish_without_push() -> None:
    sink = FakeSink()
    session = PushSession(_engine(lookahead=3, sinks=[sink]), W, H, FPS)
    assert session.finish() == []
    assert sink.events == ["open", "close"]
    assert session.stats is not None and session.stats.frames == 0


def test_engine_error_in_step_fails_session() -> None:
    sink = FakeSink(fail_write_at=1)
    session = PushSession(_engine(sinks=[sink]), W, H, FPS)
    session.push(_image(0))
    with pytest.raises(OSError, match="sink broke"):
        session.push(_image(1))
    assert sink.events == ["open", "abort"]
    with pytest.raises(RuntimeError):
        session.push(_image(2))
    with pytest.raises(RuntimeError):
        session.finish()
    session.abort()
    assert session.stats is None


def test_engine_error_in_finish_fails_session() -> None:
    sink = FakeSink(fail_write_at=1)
    session = PushSession(_engine(lookahead=2, sinks=[sink]), W, H, FPS)
    session.push(_image(0))
    session.push(_image(1))
    with pytest.raises(OSError, match="sink broke"):
        session.finish()
    assert sink.events == ["open", "abort"]
    with pytest.raises(RuntimeError):
        session.finish()
    session.abort()
    assert session.stats is None


def test_engine_releasing_two_frames_raises() -> None:
    class DoubleEngine(Engine):
        def step(self, frame: Frame) -> list[Output]:
            outputs = super().step(frame)
            return outputs + outputs

    sink = FakeSink()
    engine = DoubleEngine([FakeDetector()], FakeMode(), [], [sink], EngineConfig(lookahead=0))
    session = PushSession(engine, W, H, FPS)
    with pytest.raises(RuntimeError, match="at most 1"):
        session.push(_image(0))
    assert sink.events == ["open", "abort"]
    with pytest.raises(RuntimeError):
        session.push(_image(1))


def test_context_manager_finishes_on_success() -> None:
    sink = FakeSink()
    with PushSession(_engine(lookahead=2, sinks=[sink]), W, H, FPS) as session:
        for i in range(4):
            session.push(_image(i))
    assert sink.events == ["open", "close"]
    assert len(sink.written) == 4
    assert session.stats is not None and session.stats.frames == 4


def test_context_manager_aborts_on_exception() -> None:
    sink = FakeSink()
    with pytest.raises(KeyError):
        with PushSession(_engine(lookahead=2, sinks=[sink]), W, H, FPS) as session:
            session.push(_image(0))
            raise KeyError("caller broke")
    assert sink.events == ["open", "abort"]
    assert session.stats is None
    with pytest.raises(RuntimeError):
        session.push(_image(1))


def test_context_manager_after_explicit_finish() -> None:
    sink = FakeSink()
    with PushSession(_engine(lookahead=1, sinks=[sink]), W, H, FPS) as session:
        session.push(_image(0))
        tail = session.finish()
    assert [_value(img) for img in tail] == [100]
    assert sink.events == ["open", "close"]


# ---------------------------------------------------------------------------
# Info and delay
# ---------------------------------------------------------------------------


def test_info_and_delay() -> None:
    sink = FakeSink()
    session = PushSession(_engine(lookahead=4, sinks=[sink]), W, H, 25.0)
    assert session.info == SourceInfo(
        fps=25.0, width=W, height=H, frame_count=None, seekable=False, live=True, uri="push"
    )
    assert sink.info == session.info
    assert session.delay == 4
    assert session.stats is None
    session.abort()


def test_e6_delay_read_from_public_engine_config() -> None:
    class ConfigEngine(Engine):
        @property
        def config(self) -> EngineConfig:
            return EngineConfig(lookahead=7)

    engine = ConfigEngine([FakeDetector()], FakeMode(), [], [], EngineConfig(lookahead=0))
    session = PushSession(engine, W, H, FPS)
    assert session.delay == 7  # Taken from the public accessor, not the private attribute.
    session.abort()


def test_default_fps() -> None:
    session = PushSession(_engine(), W, H)
    assert session.info.fps == 30.0
    session.abort()


@pytest.mark.parametrize(
    ("width", "height", "fps"), [(0, H, FPS), (W, -1, FPS), (W, H, 0.0), (W, H, float("nan"))]
)
def test_invalid_constructor_arguments(width: int, height: int, fps: float) -> None:
    with pytest.raises(ValueError):
        PushSession(_engine(), width, height, fps)
