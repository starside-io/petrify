"""Unit tests for petrify.core.engine.Engine using small fakes (no OpenCV)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from typing import ClassVar

import numpy as np
import pytest

from petrify.core import (
    DetectionKind,
    Detections,
    Detector,
    Engine,
    EngineConfig,
    EngineStats,
    Frame,
    FrameMeta,
    FrameTrace,
    Mode,
    Output,
    Sink,
    Source,
    SourceInfo,
    Stage,
)

H, W = 4, 6
INFO = SourceInfo(fps=10.0, width=W, height=H, frame_count=None, seekable=False, live=False)
BACKGROUND_VALUE = 7


def _image(i: int) -> np.ndarray:
    """Frame ``i``: every pixel is ``100 + i``, so real content is never zero."""
    return np.full((H, W, 3), 100 + i, dtype=np.uint8)


class FakeSource:
    def __init__(self, n: int, fail_at: int | None = None) -> None:
        self.info = INFO
        self.n = n
        self.fail_at = fail_at
        self.closed = False

    def __iter__(self) -> Iterator[Frame]:
        for i in range(self.n):
            if i == self.fail_at:
                raise OSError("source broke")
            yield Frame(image=_image(i), index=i, timestamp=i / INFO.fps, source=INFO)

    def close(self) -> None:
        self.closed = True


class FakeDetector:
    """Reports motion on the frames in ``hits``; raises on the frames in ``fail``."""

    provides: frozenset[DetectionKind] = frozenset({DetectionKind.MOTION})

    def __init__(
        self,
        hits: Sequence[int] = (),
        *,
        stateful: bool = False,
        fail: Sequence[int] = (),
        kind: DetectionKind = DetectionKind.MOTION,
    ) -> None:
        self.provides = frozenset({kind})
        self.kind = kind
        self.stateful = stateful
        self.hits = set(hits)
        self.fail = set(fail)
        self.seen: list[int] = []
        self.resets = 0

    def detect(self, frame: Frame) -> Detections:
        self.seen.append(frame.index)
        if frame.index in self.fail:
            raise RuntimeError(f"detector broke on {frame.index}")
        hit = frame.index in self.hits
        if self.kind is DetectionKind.PERSONS:
            return Detections(persons=())  # Ran, found nothing; hits not used for persons.
        if self.kind is DetectionKind.BACKGROUND_DIFF:
            return Detections(background_diff=hit)
        return Detections(motion=hit)

    def reset(self) -> None:
        self.resets += 1


class FakeMode:
    """Frozen frames become the constant background; others pass through."""

    name = "fake"
    requires: ClassVar[frozenset[DetectionKind]] = frozenset(
        {DetectionKind.MOTION, DetectionKind.BACKGROUND_DIFF}
    )

    def __init__(self, fail: Sequence[int] = ()) -> None:
        self.fail = set(fail)
        self.calls: list[tuple[int, bool]] = []

    def render(self, frame: Frame, detections: Detections, subject_present: bool) -> np.ndarray:
        self.calls.append((frame.index, subject_present))
        if frame.index in self.fail:
            raise RuntimeError(f"mode broke on {frame.index}")
        if subject_present:
            return np.full_like(frame.image, BACKGROUND_VALUE)
        return frame.image


class FakeStage:
    """Adds 1 to every pixel; raises on the frames in ``fail``."""

    def __init__(self, fail: Sequence[int] = ()) -> None:
        self.fail = set(fail)
        self.applied: list[int] = []

    def apply(self, frame: Frame, image: np.ndarray) -> np.ndarray:
        if frame.index in self.fail:
            raise RuntimeError(f"stage broke on {frame.index}")
        self.applied.append(frame.index)
        return image + 1


class FakeSink:
    def __init__(self, fail_write_at: int | None = None, *, critical: bool = True) -> None:
        self.critical = critical
        self.events: list[str] = []
        self.written: list[tuple[int, np.ndarray]] = []
        self.received: list[np.ndarray] = []  # The arrays exactly as passed to write.
        self.metas: list[object] = []
        self.info: SourceInfo | None = None
        self.fail_write_at = fail_write_at

    def open(self, info: SourceInfo) -> None:
        self.info = info
        self.events.append("open")

    def write(self, image: np.ndarray, meta: FrameMeta) -> None:
        if meta.index == self.fail_write_at:
            raise OSError("sink broke")
        self.metas.append(meta)
        self.received.append(image)
        self.written.append((meta.index, image.copy()))

    def close(self) -> None:
        self.events.append("close")

    def abort(self) -> None:
        self.events.append("abort")


def _run(
    n: int,
    detectors: Sequence[Detector],
    *,
    mode: Mode | None = None,
    stages: Sequence[Stage] = (),
    sinks: Sequence[Sink] | None = None,
    lookahead: int = 0,
    cold_start: int = 0,
    source: Source | None = None,
    budget_ms: float | None = None,
    clock: Callable[[], float] | None = None,
) -> tuple[EngineStats, list[FrameTrace], list[FakeSink]]:
    traces: list[FrameTrace] = []
    sink_list = list(sinks) if sinks is not None else [FakeSink()]
    config = EngineConfig(
        lookahead=lookahead, cold_start_frames=cold_start, frame_budget_ms=budget_ms
    )
    engine = (
        Engine(detectors, mode or FakeMode(), stages, sink_list, config, traces.append)
        if clock is None
        else Engine(detectors, mode or FakeMode(), stages, sink_list, config, traces.append, clock)
    )
    stats = engine.run(source or FakeSource(n))
    return stats, traces, sink_list  # type: ignore[return-value]


def test_fakes_satisfy_protocols() -> None:
    assert isinstance(FakeSource(1), Source)
    assert isinstance(FakeDetector(), Detector)
    assert isinstance(FakeMode(), Mode)
    assert isinstance(FakeStage(), Stage)
    assert isinstance(FakeSink(), Sink)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_stateful_detector_runs_on_every_frame() -> None:
    first = FakeDetector(hits=[0, 1, 2], stateful=False)
    stateful = FakeDetector(stateful=True)
    _run(5, [first, stateful])
    assert stateful.seen == [0, 1, 2, 3, 4]


def test_stateless_detector_skipped_once_subject_present() -> None:
    motion = FakeDetector(hits=[1, 3], stateful=True)
    diff = FakeDetector(kind=DetectionKind.BACKGROUND_DIFF)
    _run(5, [motion, diff])
    assert motion.seen == [0, 1, 2, 3, 4]
    assert diff.seen == [0, 2, 4]


def test_stateless_detectors_short_circuit_in_order() -> None:
    a = FakeDetector(hits=[2])
    b = FakeDetector(hits=[3], kind=DetectionKind.BACKGROUND_DIFF)
    c = FakeDetector(kind=DetectionKind.BACKGROUND_DIFF)
    _, traces, _ = _run(5, [a, b, c])
    assert a.seen == [0, 1, 2, 3, 4]
    assert b.seen == [0, 1, 3, 4]
    assert c.seen == [0, 1, 4]
    assert [t.raw for t in traces] == [False, False, True, True, False]


def test_detector_not_required_by_mode_never_called() -> None:
    persons = FakeDetector(kind=DetectionKind.PERSONS, stateful=True)
    motion = FakeDetector(hits=[1])
    engine = Engine([persons, motion], FakeMode(), [], [FakeSink()], EngineConfig(lookahead=0))
    engine.run(FakeSource(3))
    assert persons.seen == []
    assert persons.resets == 0
    assert engine.detectors == (motion,)


def test_detectors_reset_at_run_start() -> None:
    motion = FakeDetector(stateful=True)
    _run(2, [motion])
    assert motion.resets == 1


# ---------------------------------------------------------------------------
# Lookahead, cold start, ordering
# ---------------------------------------------------------------------------


def test_lookahead_and_cold_start_reach_mode() -> None:
    mode = FakeMode()
    _, traces, _ = _run(10, [FakeDetector(hits=[6])], mode=mode, lookahead=2, cold_start=2)
    expected_raw = [i == 6 for i in range(10)]
    expected_filled = [4 <= i <= 8 for i in range(10)]
    expected_final = [i < 2 or 4 <= i <= 8 for i in range(10)]
    assert [t.raw for t in traces] == expected_raw
    assert [t.filled for t in traces] == expected_filled
    assert [t.final for t in traces] == expected_final
    assert mode.calls == list(enumerate(expected_final))


def test_frames_reach_sinks_in_order_once_including_tail() -> None:
    sinks = [FakeSink(), FakeSink()]
    _, traces, _ = _run(7, [FakeDetector(hits=[3])], sinks=sinks, lookahead=3)
    for sink in sinks:
        assert [i for i, _ in sink.written] == list(range(7))
        assert sink.info == INFO
        assert sink.events == ["open", "close"]
    assert [t.index for t in traces] == list(range(7))


def test_lookahead_longer_than_stream() -> None:
    _, traces, sinks = _run(3, [FakeDetector(hits=[2])], lookahead=10)
    assert [i for i, _ in sinks[0].written] == [0, 1, 2]
    assert all(t.final for t in traces)


def test_output_images_and_stats() -> None:
    stage = FakeStage()
    stats, traces, sinks = _run(5, [FakeDetector(hits=[2])], stages=[stage], lookahead=0)
    images = dict(sinks[0].written)
    assert np.all(images[2] == BACKGROUND_VALUE + 1)
    for i in (0, 1, 3, 4):
        assert np.array_equal(images[i], _image(i) + 1)
    assert stage.applied == [0, 1, 2, 3, 4]
    assert stats == EngineStats(frames=5, frozen=1, passed_through=4, failed_closed=0)
    assert not any(t.failed_closed for t in traces)


def test_zero_frame_source() -> None:
    mode = FakeMode()
    stats, traces, sinks = _run(0, [FakeDetector(stateful=True)], mode=mode, lookahead=3)
    assert stats == EngineStats()
    assert traces == []
    assert mode.calls == []
    assert sinks[0].written == []
    assert sinks[0].events == ["open", "close"]


def test_run_does_not_close_source() -> None:
    source = FakeSource(2)
    _run(2, [FakeDetector()], source=source)
    assert source.closed is False


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_source_error_aborts_sinks_and_propagates() -> None:
    sinks = [FakeSink(), FakeSink()]
    with pytest.raises(OSError, match="source broke"):
        _run(5, [FakeDetector()], sinks=sinks, source=FakeSource(5, fail_at=3))
    for sink in sinks:
        assert sink.events == ["open", "abort"]


def test_sink_error_aborts_sinks_and_propagates() -> None:
    sinks = [FakeSink(), FakeSink(fail_write_at=2)]
    with pytest.raises(OSError, match="sink broke"):
        _run(5, [FakeDetector()], sinks=sinks)
    for sink in sinks:
        assert sink.events == ["open", "abort"]


def test_sink_open_error_aborts_only_opened_sinks() -> None:
    class BadOpen(FakeSink):
        def open(self, info: SourceInfo) -> None:
            raise OSError("cannot open")

    good, bad, never = FakeSink(), BadOpen(), FakeSink()
    with pytest.raises(OSError, match="cannot open"):
        _run(3, [FakeDetector()], sinks=[good, bad, never])
    assert good.events == ["open", "abort"]
    assert bad.events == ["abort"]  # The failing sink is aborted too (safe after a failed open).
    assert never.events == []


def test_keyboard_interrupt_aborts_sinks() -> None:
    class Interrupting(FakeDetector):
        def detect(self, frame: Frame) -> Detections:
            if frame.index == 2:
                raise KeyboardInterrupt
            return super().detect(frame)

    sink = FakeSink()
    with pytest.raises(KeyboardInterrupt):
        _run(5, [Interrupting()], sinks=[sink])
    assert sink.events == ["open", "abort"]


def test_observer_error_aborts_sinks() -> None:
    def observer(trace: FrameTrace) -> None:
        raise ValueError("observer broke")

    sink = FakeSink()
    engine = Engine([FakeDetector()], FakeMode(), [], [sink], EngineConfig(lookahead=0), observer)
    with pytest.raises(ValueError, match="observer broke"):
        engine.run(FakeSource(3))
    assert sink.events == ["open", "abort"]


# ---------------------------------------------------------------------------
# Fail closed
# ---------------------------------------------------------------------------


def _is_black(image: np.ndarray, like: np.ndarray) -> bool:
    return image.shape == like.shape and image.dtype == like.dtype and not image.any()


def test_detector_failure_yields_black_and_freezes_neighbors() -> None:
    detector = FakeDetector(fail=[5, 12])
    mode = FakeMode()
    stage = FakeStage()
    stats, traces, sinks = _run(
        16, [detector], mode=mode, stages=[stage], lookahead=2, cold_start=0
    )
    images = dict(sinks[0].written)
    assert sorted(images) == list(range(16))
    for i in (5, 12):
        assert _is_black(images[i], _image(i))
        assert traces[i].failed_closed and traces[i].raw
    assert [t.index for t in traces if t.failed_closed] == [5, 12]
    frozen = [i for i in range(16) if i not in (5, 12) and traces[i].final]
    assert frozen == [3, 4, 6, 7, 10, 11, 13, 14]
    for i in frozen:
        assert np.all(images[i] == BACKGROUND_VALUE + 1)
    # The mode and stage never see a frame whose detection failed.
    assert 5 not in [i for i, _ in mode.calls] and 12 not in stage.applied
    assert stats == EngineStats(frames=16, frozen=8, passed_through=6, failed_closed=2)


def test_failed_frame_content_never_reaches_sink() -> None:
    detector = FakeDetector(fail=[1])
    _, _, sinks = _run(3, [detector], lookahead=0)
    image = dict(sinks[0].written)[1]
    assert not np.any(image == 100 + 1)
    assert not image.any()


def test_stateful_detector_still_sees_frame_after_earlier_failure() -> None:
    broken = FakeDetector(fail=[1])
    stateful = FakeDetector(stateful=True)
    stateless = FakeDetector(kind=DetectionKind.BACKGROUND_DIFF)
    _run(3, [broken, stateful, stateless])
    assert stateful.seen == [0, 1, 2]
    assert stateless.seen == [0, 2]


def test_mode_failure_yields_black() -> None:
    stage = FakeStage()
    stats, traces, sinks = _run(4, [FakeDetector()], mode=FakeMode(fail=[2]), stages=[stage])
    images = dict(sinks[0].written)
    assert _is_black(images[2], _image(2))
    assert [t.failed_closed for t in traces] == [False, False, True, False]
    assert 2 not in stage.applied
    assert stats.failed_closed == 1 and stats.frames == 4


def test_stage_failure_yields_black_and_skips_later_stages() -> None:
    later = FakeStage()
    stats, _, sinks = _run(4, [FakeDetector()], stages=[FakeStage(fail=[1]), later])
    images = dict(sinks[0].written)
    assert _is_black(images[1], _image(1))
    assert np.array_equal(images[0], _image(0) + 2)
    assert later.applied == [0, 2, 3]
    assert stats.failed_closed == 1


def test_failures_logged_at_most_five_times(caplog: pytest.LogCaptureFixture) -> None:
    detector = FakeDetector(fail=range(8))
    with caplog.at_level(logging.WARNING, logger="petrify.core.engine"):
        stats, _, _ = _run(8, [detector])
    per_frame = [r for r in caplog.records if "failed closed (" in r.getMessage()]
    assert len(per_frame) == 5
    assert "Frame 0" in per_frame[0].getMessage()
    assert any("3 further" in r.getMessage() for r in caplog.records)
    assert stats.failed_closed == 8


# ---------------------------------------------------------------------------
# Close and abort ordering
# ---------------------------------------------------------------------------


def test_close_error_aborts_remaining_sinks_and_propagates() -> None:
    class BadClose(FakeSink):
        def close(self) -> None:
            self.events.append("close")
            raise OSError("cannot close")

    first, second = BadClose(), FakeSink()
    with pytest.raises(OSError, match="cannot close"):
        _run(3, [FakeDetector()], sinks=[first, second])
    assert first.events == ["open", "close", "abort"]
    assert second.events == ["open", "abort"]


def test_abort_interrupt_still_aborts_every_sink_then_reraises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class InterruptAbort(FakeSink):
        def abort(self) -> None:
            self.events.append("abort")
            raise KeyboardInterrupt

    class ErrorAbort(FakeSink):
        def abort(self) -> None:
            self.events.append("abort")
            raise OSError("abort broke")

    sinks = [InterruptAbort(), ErrorAbort(), FakeSink(fail_write_at=1)]
    with caplog.at_level(logging.ERROR), pytest.raises(KeyboardInterrupt):
        _run(3, [FakeDetector()], sinks=sinks)
    for sink in sinks:
        assert sink.events == ["open", "abort"]
    assert any("failed to abort" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Step API
# ---------------------------------------------------------------------------


def _frame(i: int) -> Frame:
    return Frame(image=_image(i), index=i, timestamp=i / INFO.fps, source=INFO)


def _engine(
    detectors: Sequence[Detector],
    sinks: Sequence[Sink],
    *,
    mode: Mode | None = None,
    stages: Sequence[Stage] = (),
    lookahead: int = 0,
    cold_start: int = 0,
    observer: Callable[[FrameTrace], None] | None = None,
) -> Engine:
    config = EngineConfig(lookahead=lookahead, cold_start_frames=cold_start)
    return Engine(detectors, mode or FakeMode(), stages, sinks, config, observer)


def test_step_outputs_and_tail_match_sink_writes() -> None:
    sink = FakeSink()
    traces: list[FrameTrace] = []
    engine = _engine(
        [FakeDetector(hits=[3])], [sink], stages=[FakeStage()], lookahead=3, observer=traces.append
    )
    engine.open(INFO)
    outputs: list[Output] = []
    for i in range(7):
        released = engine.step(_frame(i))
        if i < 3:
            assert released == []
        else:
            assert [o.meta.index for o in released] == [i - 3]
        outputs.extend(released)
    tail = engine.flush()
    assert [o.meta.index for o in tail] == [4, 5, 6]
    outputs.extend(tail)
    stats = engine.close()

    assert stats.frames == 7
    assert [o.meta.index for o in outputs] == list(range(7))
    assert [o.trace for o in outputs] == traces
    assert [o.meta for o in outputs] == sink.metas
    for out, received, (index, image) in zip(outputs, sink.received, sink.written):
        assert out.image is received
        assert out.meta.index == index
        assert np.array_equal(out.image, image)
        assert out.meta == FrameMeta(index=index, timestamp=index / INFO.fps, source=INFO)
    assert sink.events == ["open", "close"]


def test_close_without_flush_still_writes_tail() -> None:
    sink = FakeSink()
    engine = _engine([FakeDetector()], [sink], lookahead=2)
    engine.open(INFO)
    for i in range(4):
        engine.step(_frame(i))
    assert [i for i, _ in sink.written] == [0, 1]
    stats = engine.close()
    assert [i for i, _ in sink.written] == [0, 1, 2, 3]
    assert stats.frames == 4
    assert sink.events == ["open", "close"]


def test_run_and_manual_steps_are_identical() -> None:
    def build() -> tuple[Engine, FakeSink, list[FrameTrace]]:
        sink = FakeSink()
        traces: list[FrameTrace] = []
        engine = _engine(
            [FakeDetector(hits=[4, 9], fail=[6]), FakeDetector(stateful=True)],
            [sink],
            mode=FakeMode(fail=[11]),
            stages=[FakeStage()],
            lookahead=2,
            cold_start=1,
            observer=traces.append,
        )
        return engine, sink, traces

    run_engine, run_sink, run_traces = build()
    run_stats = run_engine.run(FakeSource(14))

    step_engine, step_sink, step_traces = build()
    step_engine.open(INFO)
    for frame in FakeSource(14):
        step_engine.step(frame)
    step_engine.flush()
    step_stats = step_engine.close()

    assert run_stats == step_stats
    assert run_traces == step_traces
    assert run_sink.events == step_sink.events == ["open", "close"]
    assert run_sink.metas == step_sink.metas
    assert len(run_sink.written) == len(step_sink.written) == 14
    for (ri, rimg), (si, simg) in zip(run_sink.written, step_sink.written):
        assert ri == si
        assert np.array_equal(rimg, simg)


def test_abort_mid_stream_stops_all_writes() -> None:
    sinks = [FakeSink(), FakeSink(critical=False)]
    engine = _engine([FakeDetector()], sinks, lookahead=2)
    engine.open(INFO)
    for i in range(5):
        engine.step(_frame(i))
    engine.abort()
    assert not engine.is_open
    for sink in sinks:
        assert sink.events == ["open", "abort"]
        assert [i for i, _ in sink.written] == [0, 1, 2]
    with pytest.raises(RuntimeError, match="not open"):
        engine.step(_frame(5))
    with pytest.raises(RuntimeError, match="not open"):
        engine.close()
    engine.abort()
    for sink in sinks:
        assert sink.events == ["open", "abort"]
        assert len(sink.written) == 3


def test_double_open_raises() -> None:
    sink = FakeSink()
    engine = _engine([FakeDetector()], [sink])
    engine.open(INFO)
    with pytest.raises(RuntimeError, match="already open"):
        engine.open(INFO)
    assert engine.is_open
    assert sink.events == ["open"]
    engine.close()


def test_calls_before_open_raise_and_abort_is_safe() -> None:
    sink = FakeSink()
    engine = _engine([FakeDetector()], [sink])
    with pytest.raises(RuntimeError, match="not open"):
        engine.step(_frame(0))
    with pytest.raises(RuntimeError, match="not open"):
        engine.flush()
    with pytest.raises(RuntimeError, match="not open"):
        engine.close()
    engine.abort()
    engine.abort()
    assert sink.events == []


def test_step_after_flush_raises() -> None:
    engine = _engine([FakeDetector()], [FakeSink()], lookahead=1)
    engine.open(INFO)
    engine.step(_frame(0))
    engine.flush()
    with pytest.raises(RuntimeError, match="flushed"):
        engine.step(_frame(1))
    engine.close()


def test_reopen_after_close_starts_a_fresh_stream() -> None:
    sink = FakeSink()
    detector = FakeDetector(stateful=True)
    engine = _engine([detector], [sink])
    engine.open(INFO)
    engine.step(_frame(0))
    engine.close()
    engine.open(INFO)
    (out,) = engine.step(_frame(0))
    assert out.trace.index == 0
    stats = engine.close()
    assert stats.frames == 1
    assert detector.resets == 2
    assert sink.events == ["open", "close", "open", "close"]


def test_failed_open_leaves_engine_closed_and_abort_safe() -> None:
    class BadOpen(FakeSink):
        def open(self, info: SourceInfo) -> None:
            raise OSError("cannot open")

    good, bad = FakeSink(), BadOpen()
    engine = _engine([FakeDetector()], [good, bad])
    with pytest.raises(OSError, match="cannot open"):
        engine.open(INFO)
    assert not engine.is_open
    engine.abort()
    assert good.events == ["open", "abort"]
    assert bad.events == ["abort"]


def test_step_error_aborts_engine() -> None:
    sink = FakeSink(fail_write_at=1)
    engine = _engine([FakeDetector()], [sink])
    engine.open(INFO)
    engine.step(_frame(0))
    with pytest.raises(OSError, match="sink broke"):
        engine.step(_frame(1))
    assert not engine.is_open
    assert sink.events == ["open", "abort"]


def test_engine_without_sinks_returns_outputs() -> None:
    engine = _engine([FakeDetector(hits=[1])], [], stages=[FakeStage()])
    engine.open(INFO)
    outputs = [o for i in range(3) for o in engine.step(_frame(i))]
    stats = engine.close()
    assert [o.meta.index for o in outputs] == [0, 1, 2]
    assert np.all(outputs[1].image == BACKGROUND_VALUE + 1)
    assert np.array_equal(outputs[0].image, _image(0) + 1)
    assert stats.frames == 3


# ---------------------------------------------------------------------------
# Sinks never receive the raw frame
# ---------------------------------------------------------------------------


class ConstantMode:
    """Always renders a constant image, never the input pixels."""

    name = "constant"
    requires: ClassVar[frozenset[DetectionKind]] = frozenset({DetectionKind.MOTION})

    def render(self, frame: Frame, detections: Detections, subject_present: bool) -> np.ndarray:
        return np.full_like(frame.image, 42)


def test_sinks_never_receive_raw_pixels() -> None:
    frames = [_frame(i) for i in range(6)]
    sink = FakeSink()
    engine = _engine([FakeDetector(hits=[2], fail=[4])], [sink], mode=ConstantMode(), lookahead=1)
    engine.open(INFO)
    for frame in frames:
        engine.step(frame)
    engine.close()

    assert len(sink.received) == len(frames)
    for frame, image, meta in zip(frames, sink.received, sink.metas):
        assert not np.shares_memory(image, frame.image)
        assert not np.array_equal(image, frame.image)
        assert isinstance(meta, FrameMeta)
        assert not hasattr(meta, "image")
        assert meta == frame.meta


def test_frame_meta_property() -> None:
    frame = _frame(3)
    assert frame.meta == FrameMeta(index=3, timestamp=3 / INFO.fps, source=INFO)


# ---------------------------------------------------------------------------
# Sink failure isolation
# ---------------------------------------------------------------------------


def test_non_critical_write_failure_is_dropped(caplog: pytest.LogCaptureFixture) -> None:
    critical, flaky = FakeSink(), FakeSink(fail_write_at=2, critical=False)
    with caplog.at_level(logging.WARNING, logger="petrify.core.engine"):
        stats, traces, _ = _run(5, [FakeDetector()], sinks=[critical, flaky])
    assert [i for i, _ in critical.written] == list(range(5))
    assert critical.events == ["open", "close"]
    assert [i for i, _ in flaky.written] == [0, 1]
    assert flaky.events == ["open", "abort"]
    assert stats.sink_failures == 1
    assert stats.frames == 5 and stats.failed_closed == 0
    assert len(traces) == 5
    dropped = [r for r in caplog.records if "dropped" in r.getMessage()]
    assert len(dropped) == 1


def test_non_critical_open_failure_is_dropped() -> None:
    class BadOpen(FakeSink):
        def open(self, info: SourceInfo) -> None:
            raise OSError("cannot open")

    critical, flaky = FakeSink(), BadOpen(critical=False)
    stats, _, _ = _run(4, [FakeDetector()], sinks=[flaky, critical])
    assert [i for i, _ in critical.written] == list(range(4))
    assert critical.events == ["open", "close"]
    assert flaky.events == ["abort"]
    assert flaky.written == []
    assert stats.sink_failures == 1


def test_non_critical_close_failure_is_dropped() -> None:
    class BadClose(FakeSink):
        def close(self) -> None:
            self.events.append("close")
            raise OSError("cannot close")

    flaky, critical = BadClose(critical=False), FakeSink()
    stats, _, _ = _run(4, [FakeDetector()], sinks=[flaky, critical])
    assert [i for i, _ in critical.written] == list(range(4))
    assert critical.events == ["open", "close"]
    assert flaky.events == ["open", "close", "abort"]
    assert stats.sink_failures == 1


def test_critical_failure_aborts_non_critical_sinks_too() -> None:
    preview, recording = FakeSink(critical=False), FakeSink(fail_write_at=2)
    with pytest.raises(OSError, match="sink broke"):
        _run(5, [FakeDetector()], sinks=[preview, recording])
    assert preview.events == ["open", "abort"]
    assert recording.events == ["open", "abort"]
    assert [i for i, _ in preview.written] == [0, 1, 2]  # Written before the recording failed.


def test_all_sinks_failing_on_write_raises() -> None:
    sinks = [FakeSink(fail_write_at=1, critical=False), FakeSink(fail_write_at=3, critical=False)]
    with pytest.raises(RuntimeError, match="Every sink failed"):
        _run(6, [FakeDetector()], sinks=sinks)
    for sink in sinks:
        assert sink.events == ["open", "abort"]


def test_all_sinks_failing_on_open_raises() -> None:
    class BadOpen(FakeSink):
        def open(self, info: SourceInfo) -> None:
            raise OSError("cannot open")

    sinks = [BadOpen(critical=False), BadOpen(critical=False)]
    engine = _engine([FakeDetector()], sinks)
    with pytest.raises(RuntimeError, match="Every sink failed"):
        engine.open(INFO)
    assert not engine.is_open
    for sink in sinks:
        assert sink.events == ["abort"]


def test_all_sinks_failing_on_close_raises() -> None:
    class BadClose(FakeSink):
        def close(self) -> None:
            raise OSError("cannot close")

    sinks = [BadClose(critical=False), BadClose(critical=False)]
    with pytest.raises(RuntimeError, match="Every sink failed"):
        _run(2, [FakeDetector()], sinks=sinks)
    for sink in sinks:
        assert sink.events == ["open", "abort"]


# ---------------------------------------------------------------------------
# Per-frame time budget
# ---------------------------------------------------------------------------


class FakeClock:
    """Manual clock: only the fakes below advance it, so timings are exact."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class SlowDetector(FakeDetector):
    """Advances ``clock`` by ``costs[index]`` seconds when it runs on that frame."""

    def __init__(
        self, clock: FakeClock, costs: dict[int, float], hits: Sequence[int] = ()
    ) -> None:
        super().__init__(hits)
        self.clock = clock
        self.costs = costs

    def detect(self, frame: Frame) -> Detections:
        self.clock.now += self.costs.get(frame.index, 0.0)
        return super().detect(frame)


class SlowMode(FakeMode):
    """Advances ``clock`` by ``costs[index]`` seconds when rendering that frame."""

    def __init__(self, clock: FakeClock, costs: dict[int, float]) -> None:
        super().__init__()
        self.clock = clock
        self.costs = costs

    def render(self, frame: Frame, detections: Detections, subject_present: bool) -> np.ndarray:
        self.clock.now += self.costs.get(frame.index, 0.0)
        return super().render(frame, detections, subject_present)


def test_slow_detection_fails_closed_and_freezes_neighbors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = FakeClock()
    mode = FakeMode()
    with caplog.at_level(logging.WARNING, logger="petrify.core.engine"):
        stats, traces, sinks = _run(
            8,
            [SlowDetector(clock, {3: 0.050})],
            mode=mode,
            lookahead=2,
            budget_ms=20.0,
            clock=clock,
        )
    images = dict(sinks[0].written)
    assert _is_black(images[3], _image(3))
    assert not np.any(images[3] == 100 + 3)
    assert traces[3].failed_closed and traces[3].raw
    assert [t.index for t in traces if t.failed_closed] == [3]
    assert [i for i in range(8) if traces[i].final] == [1, 2, 3, 4, 5]
    for i in (1, 2, 4, 5):
        assert np.all(images[i] == BACKGROUND_VALUE)
    assert 3 not in [i for i, _ in mode.calls]
    assert stats == EngineStats(
        frames=8, frozen=4, passed_through=3, failed_closed=1, over_budget=1
    )
    assert any("over budget" in r.getMessage() for r in caplog.records)


def test_slow_render_fails_closed() -> None:
    clock = FakeClock()
    mode = SlowMode(clock, {2: 0.030})
    stats, traces, sinks = _run(4, [FakeDetector()], mode=mode, budget_ms=20.0, clock=clock)
    images = dict(sinks[0].written)
    assert _is_black(images[2], _image(2))
    assert not np.any(images[2] == 100 + 2)
    assert [t.failed_closed for t in traces] == [False, False, True, False]
    assert not traces[2].raw
    assert stats.over_budget == 1 and stats.failed_closed == 1
    assert stats.frozen + stats.passed_through + stats.failed_closed == stats.frames


def test_budget_counts_detection_plus_render() -> None:
    clock = FakeClock()
    detector = SlowDetector(clock, {1: 0.012})
    mode = SlowMode(clock, {1: 0.012})
    stats, traces, sinks = _run(3, [detector], mode=mode, budget_ms=20.0, clock=clock)
    assert [t.failed_closed for t in traces] == [False, True, False]
    assert not dict(sinks[0].written)[1].any()
    assert stats.over_budget == 1


def test_under_budget_is_unaffected() -> None:
    clock = FakeClock()
    costs = {i: 0.009 for i in range(6)}
    budget_run = _run(
        6,
        [SlowDetector(clock, costs, hits=[2])],
        mode=SlowMode(clock, costs),
        stages=[FakeStage()],
        lookahead=1,
        budget_ms=20.0,
        clock=clock,
    )
    plain_run = _run(6, [FakeDetector(hits=[2])], stages=[FakeStage()], lookahead=1)
    assert budget_run[0] == plain_run[0]
    assert budget_run[0].over_budget == 0
    assert budget_run[1] == plain_run[1]
    for (bi, bimg), (pi, pimg) in zip(budget_run[2][0].written, plain_run[2][0].written):
        assert bi == pi and np.array_equal(bimg, pimg)


def test_no_budget_disables_the_check() -> None:
    clock = FakeClock()
    costs = {i: 10.0 for i in range(4)}
    stats, traces, _ = _run(
        4,
        [SlowDetector(clock, costs)],
        mode=SlowMode(clock, costs),
        budget_ms=None,
        clock=clock,
    )
    assert stats.over_budget == 0 and stats.failed_closed == 0
    assert not any(t.failed_closed for t in traces)


# ---------------------------------------------------------------------------
# Review fixes (P2-8a)
# ---------------------------------------------------------------------------


def test_e1_no_detector_for_required_mode_rejected() -> None:
    with pytest.raises(ValueError, match="fake.*requires"):
        Engine([], FakeMode(), [], [FakeSink()], EngineConfig(lookahead=0))


def test_e1_no_matching_detector_for_required_mode_rejected() -> None:
    persons = FakeDetector(kind=DetectionKind.PERSONS)
    with pytest.raises(ValueError, match="motion"):
        Engine([persons], FakeMode(), [], [FakeSink()], EngineConfig(lookahead=0))


def test_e1_mode_without_requirements_allows_no_detector() -> None:
    class NoNeeds(FakeMode):
        requires: ClassVar[frozenset[DetectionKind]] = frozenset()

    sink = FakeSink()
    engine = Engine([], NoNeeds(), [], [sink], EngineConfig(lookahead=0, cold_start_frames=0))
    stats = engine.run(FakeSource(3))
    assert engine.detectors == ()
    assert stats.frames == 3 and stats.failed_closed == 0
    assert [i for i, _ in sink.written] == [0, 1, 2]


class _BadOpenSink(FakeSink):
    def open(self, info: SourceInfo) -> None:
        raise OSError("cannot open")


def test_e2_failing_critical_open_aborts_it_and_the_others_once() -> None:
    good, bad, never = FakeSink(), _BadOpenSink(), FakeSink(critical=False)
    engine = _engine([FakeDetector()], [good, bad, never])
    with pytest.raises(OSError, match="cannot open"):
        engine.open(INFO)
    assert not engine.is_open
    engine.abort()
    assert bad.events == ["abort"]
    assert good.events == ["open", "abort"]
    assert never.events == []


def test_e3_failing_critical_close_is_aborted_once() -> None:
    class BadClose(FakeSink):
        def close(self) -> None:
            self.events.append("close")
            raise OSError("cannot close")

    first, second = BadClose(), FakeSink(critical=False)
    engine = _engine([FakeDetector()], [first, second])
    with pytest.raises(OSError, match="cannot close"):
        engine.run(FakeSource(3))
    engine.abort()
    assert first.events == ["open", "close", "abort"]
    assert second.events == ["open", "abort"]


def test_e4_step_from_observer_rejected_without_interleaving() -> None:
    sink = FakeSink()
    inner_errors: list[BaseException] = []
    holder: list[Engine] = []

    def observer(trace: FrameTrace) -> None:
        try:
            holder[0].step(_frame(100 + trace.index))
        except RuntimeError as exc:
            inner_errors.append(exc)
            raise

    engine = _engine([FakeDetector()], [sink], observer=observer)
    holder.append(engine)
    engine.open(INFO)
    with pytest.raises(RuntimeError, match="re-entrant"):
        engine.step(_frame(0))
    assert len(inner_errors) == 1
    assert [i for i, _ in sink.written] == [0]
    assert sink.events == ["open", "abort"]


def test_e4_reentrant_calls_from_sink_rejected() -> None:
    class Reentrant(FakeSink):
        def __init__(self, call: str) -> None:
            super().__init__(critical=False)
            self.call = call
            self.errors: list[str] = []

        def write(self, image: np.ndarray, meta: FrameMeta) -> None:
            super().write(image, meta)
            try:
                if self.call == "open":
                    holder[0].open(INFO)
                elif self.call == "step":
                    holder[0].step(_frame(50))
                elif self.call == "flush":
                    holder[0].flush()
                elif self.call == "run":
                    holder[0].run(FakeSource(1))
                else:
                    holder[0].close()
            except RuntimeError as exc:
                self.errors.append(str(exc))

    for call in ("open", "step", "flush", "close", "run"):
        holder: list[Engine] = []
        reentrant, recording = Reentrant(call), FakeSink()
        engine = _engine([FakeDetector()], [reentrant, recording], lookahead=1)
        holder.append(engine)
        engine.open(INFO)
        for i in range(3):
            engine.step(_frame(i))
        stats = engine.close()
        assert len(reentrant.errors) == 3, call
        assert all("re-entrant" in e for e in reentrant.errors), call
        assert [i for i, _ in recording.written] == [0, 1, 2], call
        assert [i for i, _ in reentrant.written] == [0, 1, 2], call
        assert recording.events == ["open", "close"], call
        assert stats.frames == 3


def test_e4_abort_from_sink_stops_writes_and_raises() -> None:
    holder: list[Engine] = []

    class Aborting(FakeSink):
        def write(self, image: np.ndarray, meta: FrameMeta) -> None:
            super().write(image, meta)
            if meta.index == 1:
                holder[0].abort()

    first, aborting, last = FakeSink(), Aborting(critical=False), FakeSink(critical=False)
    engine = _engine([FakeDetector()], [first, aborting, last], lookahead=2)
    holder.append(engine)
    engine.open(INFO)
    for i in range(3):
        engine.step(_frame(i))
    with pytest.raises(RuntimeError, match="aborted during"):
        engine.flush()  # Releases frames 1 and 2; the abort lands while writing frame 1.
    assert not engine.is_open
    assert [i for i, _ in first.written] == [0, 1]
    assert [i for i, _ in aborting.written] == [0, 1]
    assert [i for i, _ in last.written] == [0]
    for sink in (first, aborting, last):
        assert sink.events == ["open", "abort"]


def test_e4_abort_from_observer_stops_later_frames() -> None:
    holder: list[Engine] = []

    def observer(trace: FrameTrace) -> None:
        if trace.index == 0:
            holder[0].abort()

    sink = FakeSink()
    engine = _engine([FakeDetector()], [sink], lookahead=2, observer=observer)
    holder.append(engine)
    engine.open(INFO)
    engine.step(_frame(0))
    engine.step(_frame(1))
    with pytest.raises(RuntimeError, match="aborted during"):
        engine.close()  # Flushes frames 0, 1, 2; the observer aborts after frame 0.
    assert [i for i, _ in sink.written] == [0]
    assert sink.events == ["open", "abort"]
    with pytest.raises(RuntimeError, match="not open"):
        engine.step(_frame(2))


def test_e5_sustained_failure_logs_periodic_summary(caplog: pytest.LogCaptureFixture) -> None:
    class LongSource(FakeSource):
        def __iter__(self) -> Iterator[Frame]:
            for i in range(self.n):  # _image() would overflow uint8 past 155 frames.
                yield Frame(image=_image(i % 100), index=i, timestamp=i / INFO.fps, source=INFO)

    detector = FakeDetector(fail=range(210))
    with caplog.at_level(logging.WARNING, logger="petrify.core.engine"):
        stats, _, _ = _run(210, [detector], source=LongSource(210))
    messages = [r.getMessage() for r in caplog.records]
    assert len([m for m in messages if "failed closed (" in m]) == 5
    summaries = [m for m in messages if "so far" in m]
    assert summaries == [
        "105 frames failed closed so far (0 over budget)",
        "205 frames failed closed so far (0 over budget)",
    ]
    assert any("205 further" in m for m in messages)
    assert stats.failed_closed == 210


def test_e5_summary_counts_over_budget(caplog: pytest.LogCaptureFixture) -> None:
    clock = FakeClock()
    costs = {i: 0.050 for i in range(60, 105)}
    detector = SlowDetector(clock, costs)
    detector.fail = set(range(60))
    with caplog.at_level(logging.WARNING, logger="petrify.core.engine"):
        stats, _, _ = _run(105, [detector], budget_ms=20.0, clock=clock)
    summaries = [r.getMessage() for r in caplog.records if "so far" in r.getMessage()]
    assert summaries == ["105 frames failed closed so far (45 over budget)"]
    assert stats.failed_closed == 105 and stats.over_budget == 45


def test_e6_config_property_is_read_only() -> None:
    config = EngineConfig(lookahead=3, cold_start_frames=1)
    engine = Engine([FakeDetector()], FakeMode(), [], [FakeSink()], config)
    assert engine.config is config
    with pytest.raises(AttributeError):
        engine.config = EngineConfig()  # type: ignore[misc]
