"""Unit tests for petrify.core.engine.Engine using small fakes (no OpenCV)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import replace
from typing import ClassVar

import numpy as np
import pytest

from petrify.core import (
    Box,
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
    Refiner,
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


# ---------------------------------------------------------------------------
# Reconfigure at a frame boundary (P3-1)
# ---------------------------------------------------------------------------


class TagMode:
    """Renders a constant ``tag`` (``tag + 1`` when frozen) so outputs show which mode ran."""

    name = "tag"
    requires: ClassVar[frozenset[DetectionKind]] = frozenset({DetectionKind.MOTION})

    def __init__(self, tag: int) -> None:
        self.tag = tag
        self.calls: list[tuple[int, bool]] = []

    def render(self, frame: Frame, detections: Detections, subject_present: bool) -> np.ndarray:
        self.calls.append((frame.index, subject_present))
        return np.full_like(frame.image, self.tag + (1 if subject_present else 0))


class PersonsMode(TagMode):
    name = "persons"
    requires: ClassVar[frozenset[DetectionKind]] = frozenset({DetectionKind.PERSONS})


def _stream(engine: Engine, indices: Iterable[int]) -> list[Output]:
    return [o for i in indices for o in engine.step(_frame(i))]


def _swap_engine(
    *,
    detectors: Sequence[Detector] | None = None,
    mode: Mode | None = None,
    stages: Sequence[Stage] = (),
    lookahead: int = 0,
    cold_start: int = 0,
    sinks: Sequence[Sink] | None = None,
) -> tuple[Engine, list[FakeSink], list[FrameTrace]]:
    traces: list[FrameTrace] = []
    sink_list = list(sinks) if sinks is not None else [FakeSink()]
    engine = _engine(
        detectors if detectors is not None else [FakeDetector()],
        sink_list,
        mode=mode or TagMode(10),
        stages=stages,
        lookahead=lookahead,
        cold_start=cold_start,
        observer=traces.append,
    )
    return engine, sink_list, traces  # type: ignore[return-value]


SWAPS: dict[str, Callable[[], dict[str, object]]] = {
    "detectors": lambda: {"detectors": [FakeDetector(stateful=True)]},
    "mode": lambda: {"mode": TagMode(50)},
    "stages": lambda: {"stages": [FakeStage()]},
    "config": lambda: {"config": EngineConfig(lookahead=1, cold_start_frames=0)},
    "budget": lambda: {"config": EngineConfig(lookahead=3, frame_budget_ms=1000.0)},
    "combined": lambda: {
        "detectors": [FakeDetector()],
        "mode": TagMode(50),
        "stages": [FakeStage(), FakeStage()],
        "config": EngineConfig(lookahead=5, cold_start_frames=2),
    },
    "nothing": dict,
}


@pytest.mark.parametrize("swap", sorted(SWAPS))
def test_reconfigure_keeps_stream_contiguous(swap: str) -> None:
    k, n, m = 3, 6, 9
    engine, (sink,), traces = _swap_engine(detectors=[FakeDetector(hits=[2, 8])], lookahead=k)
    engine.open(INFO)
    outputs = _stream(engine, range(n))
    tail = engine.reconfigure(**SWAPS[swap]())  # type: ignore[arg-type]
    expected_tail = [] if swap == "nothing" else list(range(n - k, n))
    assert [o.meta.index for o in tail] == expected_tail
    assert all(o.trace.final for o in tail)
    outputs += tail
    outputs += _stream(engine, range(n, n + m))
    outputs += engine.flush()
    stats = engine.close()

    assert [i for i, _ in sink.written] == list(range(n + m))
    assert [t.index for t in traces] == list(range(n + m))
    assert [o.trace for o in outputs] == traces
    assert [o.meta for o in outputs] == sink.metas
    assert stats.frames == n + m
    assert stats.reconfigurations == 1
    assert stats.frozen + stats.passed_through + stats.failed_closed == stats.frames


def test_reconfigure_with_no_arguments_keeps_every_component() -> None:
    detector, mode, stage = FakeDetector(), TagMode(10), FakeStage()
    engine, _, _ = _swap_engine(detectors=[detector], mode=mode, stages=[stage], lookahead=2)
    config = engine.config
    engine.open(INFO)
    _stream(engine, range(3))
    engine.reconfigure()
    assert engine.detectors == (detector,)
    assert engine.mode is mode
    assert engine.stages == (stage,)
    assert engine.config is config
    assert detector.resets == 1  # Only the reset of open().
    engine.close()


@pytest.mark.parametrize("which", ["detectors", "mode", "stages", "config"])
def test_reconfigure_single_argument_changes_only_that(which: str) -> None:
    detector, mode, stage = FakeDetector(), TagMode(10), FakeStage()
    engine, _, _ = _swap_engine(detectors=[detector], mode=mode, stages=[stage], lookahead=1)
    config = engine.config
    engine.open(INFO)
    _stream(engine, range(2))
    new_detector, new_mode, new_stage = FakeDetector(), TagMode(50), FakeStage()
    new_config = EngineConfig(lookahead=4, cold_start_frames=0)
    args: dict[str, object] = {
        "detectors": [new_detector],
        "mode": new_mode,
        "stages": [new_stage],
        "config": new_config,
    }
    engine.reconfigure(**{which: args[which]})  # type: ignore[arg-type]
    assert engine.detectors == ((new_detector,) if which == "detectors" else (detector,))
    assert engine.mode is (new_mode if which == "mode" else mode)
    assert engine.stages == ((new_stage,) if which == "stages" else (stage,))
    assert engine.config is (new_config if which == "config" else config)
    engine.close()


def test_old_components_render_tail_and_new_ones_the_rest() -> None:
    k, n, m = 2, 5, 4
    old_mode, new_mode = TagMode(10), TagMode(50)
    old_stage, new_stage = FakeStage(), FakeStage()
    engine, (sink,), _ = _swap_engine(mode=old_mode, stages=[old_stage], lookahead=k)
    engine.open(INFO)
    _stream(engine, range(n))
    tail = engine.reconfigure(mode=new_mode, stages=[new_stage])
    _stream(engine, range(n, n + m))
    engine.close()

    assert [i for i, _ in old_mode.calls] == list(range(n))
    assert [i for i, _ in new_mode.calls] == list(range(n, n + m))
    assert old_stage.applied == list(range(n))
    assert new_stage.applied == list(range(n, n + m))
    images = dict(sink.written)
    # TagMode(tag) renders tag, or tag + 1 when frozen; each stage adds 1. The tail
    # (frames n - k to n - 1) and the first k frames after the switch are frozen.
    for i in range(n):
        assert np.all(images[i] == 10 + (1 if i >= n - k else 0) + 1)
    for i in range(n, n + m):
        assert np.all(images[i] == 50 + (1 if i < n + k else 0) + 1)
    for out in tail:
        assert np.all(out.image == 12)


def _finals(traces: Sequence[FrameTrace]) -> list[int]:
    return [t.index for t in traces if t.final]


def test_detector_swap_restarts_cold_start_and_freezes_tail() -> None:
    engine, _, traces = _swap_engine(lookahead=1, cold_start=2)
    engine.open(INFO)
    _stream(engine, range(6))  # Frames 0 and 1 are the initial cold start.
    tail = engine.reconfigure(detectors=[FakeDetector()])
    _stream(engine, range(6, 12))
    engine.close()
    assert [o.trace.index for o in tail] == [5]
    assert tail[0].trace.final and not tail[0].trace.filled
    assert _finals(traces) == [0, 1, 5, 6, 7]
    assert not any(t.raw for t in traces)


@pytest.mark.parametrize(
    "swap",
    [{"mode": TagMode(50)}, {"stages": [FakeStage()]}],
    ids=["mode", "stages"],
)
def test_mode_or_stage_swap_freezes_only_one_lookahead_window(swap: dict[str, object]) -> None:
    engine, _, traces = _swap_engine(lookahead=1, cold_start=2)
    engine.open(INFO)
    _stream(engine, range(6))
    tail = engine.reconfigure(**swap)  # type: ignore[arg-type]
    _stream(engine, range(6, 12))
    engine.close()
    assert [o.trace.index for o in tail] == [5]
    assert tail[0].trace.final and not tail[0].trace.filled
    assert _finals(traces) == [0, 1, 5, 6]  # No cold-start restart: one window of 1.


@pytest.mark.parametrize(
    "config",
    [
        EngineConfig(lookahead=3, cold_start_frames=2),
        EngineConfig(lookahead=1, cold_start_frames=3),
    ],
    ids=["lookahead", "cold_start_frames"],
)
def test_lookahead_or_cold_start_change_restarts_cold_start(config: EngineConfig) -> None:
    engine, _, traces = _swap_engine(lookahead=1, cold_start=2)
    engine.open(INFO)
    _stream(engine, range(6))
    tail = engine.reconfigure(config=config)
    _stream(engine, range(6, 14))
    engine.close()
    assert tail[0].trace.final and not tail[0].trace.filled
    length = max(config.cold_start_frames, 1, config.lookahead)
    window = list(range(6, 6 + length))
    assert _finals(traces) == [0, 1, 5, *window]


def test_cold_start_in_progress_carries_over_a_mode_swap() -> None:
    engine, _, traces = _swap_engine(lookahead=0, cold_start=5)
    engine.open(INFO)
    _stream(engine, range(2))
    engine.reconfigure(mode=TagMode(50))
    _stream(engine, range(2, 8))
    engine.close()
    assert _finals(traces) == [0, 1, 2, 3, 4]


def test_detector_swap_inside_cold_start_keeps_the_longer_window() -> None:
    engine, _, traces = _swap_engine(lookahead=0, cold_start=5)
    engine.open(INFO)
    _stream(engine, range(1))
    engine.reconfigure(
        detectors=[FakeDetector()], config=EngineConfig(lookahead=0, cold_start_frames=2)
    )
    _stream(engine, range(1, 8))
    engine.close()
    assert _finals(traces) == [0, 1, 2, 3, 4]


def test_tail_detection_and_fill_survive_a_mode_swap() -> None:
    engine, _, traces = _swap_engine(detectors=[FakeDetector(hits=[4])], lookahead=2)
    engine.open(INFO)
    _stream(engine, range(5))
    tail = engine.reconfigure(mode=TagMode(50))
    _stream(engine, range(5, 9))
    engine.close()
    assert [o.trace.index for o in tail] == [3, 4]
    assert [o.trace.final for o in tail] == [True, True]
    assert [o.trace.filled for o in tail] == [True, True]  # Filled from the hit at 4.
    assert _finals(traces) == [2, 3, 4, 5, 6]  # The window after the swap covers 5 and 6.


def test_new_stateful_detector_is_reset_and_sees_every_later_frame() -> None:
    old, new = FakeDetector(stateful=True), FakeDetector(stateful=True)
    engine, _, _ = _swap_engine(detectors=[old], lookahead=2)
    engine.open(INFO)
    _stream(engine, range(4))
    assert new.resets == 0
    engine.reconfigure(detectors=[new])
    assert new.resets == 1
    assert old.resets == 1  # Only the reset of open().
    _stream(engine, range(4, 9))
    engine.close()
    assert old.seen == [0, 1, 2, 3]
    assert new.seen == [4, 5, 6, 7, 8]
    assert engine.detectors == (new,)


def test_mode_swap_that_activates_a_detector_resets_it_and_cold_starts() -> None:
    motion = FakeDetector()
    persons = FakeDetector(kind=DetectionKind.PERSONS, stateful=True)
    engine, _, traces = _swap_engine(detectors=[motion, persons], lookahead=1, cold_start=0)
    engine.open(INFO)
    assert engine.detectors == (motion,)
    _stream(engine, range(4))
    assert persons.seen == [] and persons.resets == 0
    tail = engine.reconfigure(mode=PersonsMode(50))
    assert engine.detectors == (persons,)
    assert persons.resets == 1
    assert [o.trace.index for o in tail] == [3]
    assert tail[0].trace.final  # A detector came in: the tail is forced frozen.
    _stream(engine, range(4, 9))
    engine.close()
    assert persons.seen == [4, 5, 6, 7, 8]
    assert _finals(traces) == [3, 4]  # cold_start_frames is 0: one lookahead window.


def test_mode_swap_that_activates_a_detector_applies_cold_start() -> None:
    motion = FakeDetector()
    persons = FakeDetector(kind=DetectionKind.PERSONS, stateful=True)
    engine, _, traces = _swap_engine(detectors=[motion, persons], lookahead=0, cold_start=2)
    engine.open(INFO)
    _stream(engine, range(4))
    engine.reconfigure(mode=PersonsMode(50))
    _stream(engine, range(4, 9))
    engine.close()
    assert _finals(traces) == [0, 1, 4, 5]


@pytest.mark.parametrize(
    "swap",
    [
        lambda: {"mode": PersonsMode(50)},
        lambda: {"detectors": [FakeDetector(kind=DetectionKind.PERSONS)]},
        lambda: {"detectors": []},
    ],
    ids=["mode", "detectors", "empty"],
)
def test_invalid_reconfigure_changes_nothing(swap: Callable[[], dict[str, object]]) -> None:
    detector, mode, stage = FakeDetector(hits=[3]), TagMode(10), FakeStage()
    engine, (sink,), traces = _swap_engine(
        detectors=[detector], mode=mode, stages=[stage], lookahead=2, cold_start=1
    )
    config = engine.config
    engine.open(INFO)
    _stream(engine, range(4))
    written = len(sink.written)
    with pytest.raises(ValueError, match="requires"):
        engine.reconfigure(stages=[], config=EngineConfig(lookahead=0), **swap())  # type: ignore[arg-type]
    assert engine.is_open
    assert len(sink.written) == written  # Nothing was flushed.
    assert engine.detectors == (detector,)
    assert engine.mode is mode and engine.stages == (stage,) and engine.config is config
    _stream(engine, range(4, 8))
    stats = engine.close()
    assert [i for i, _ in sink.written] == list(range(8))
    assert [i for i, _ in mode.calls] == list(range(8))
    assert stage.applied == list(range(8))
    assert _finals(traces) == [0, 1, 2, 3, 4, 5]
    assert stats.reconfigurations == 0
    assert detector.seen == list(range(8))


def test_reconfigure_when_not_open_or_flushed_raises() -> None:
    engine, _, _ = _swap_engine(lookahead=1)
    with pytest.raises(RuntimeError, match="not open"):
        engine.reconfigure(mode=TagMode(50))
    engine.open(INFO)
    engine.step(_frame(0))
    engine.flush()
    with pytest.raises(RuntimeError, match="flushed"):
        engine.reconfigure(mode=TagMode(50))
    engine.close()
    with pytest.raises(RuntimeError, match="not open"):
        engine.reconfigure(mode=TagMode(50))


def test_reconfigure_from_sink_is_rejected() -> None:
    holder: list[Engine] = []

    class Reentrant(FakeSink):
        def __init__(self) -> None:
            super().__init__(critical=False)
            self.errors: list[str] = []

        def write(self, image: np.ndarray, meta: FrameMeta) -> None:
            super().write(image, meta)
            try:
                holder[0].reconfigure(mode=TagMode(50))
            except RuntimeError as exc:
                self.errors.append(str(exc))

    reentrant, recording = Reentrant(), FakeSink()
    mode = TagMode(10)
    engine, _, _ = _swap_engine(mode=mode, sinks=[reentrant, recording], lookahead=1)
    holder.append(engine)
    engine.open(INFO)
    _stream(engine, range(3))
    engine.reconfigure(stages=[FakeStage()])  # Releases frame 2 through the sinks.
    stats = engine.close()
    assert len(reentrant.errors) == 3
    assert all("re-entrant" in e for e in reentrant.errors)
    assert engine.mode is mode
    assert [i for i, _ in recording.written] == [0, 1, 2]
    assert recording.events == ["open", "close"]
    assert stats.reconfigurations == 1


def test_reconfigure_from_observer_is_rejected() -> None:
    holder: list[Engine] = []
    errors: list[str] = []

    def observer(trace: FrameTrace) -> None:
        try:
            holder[0].reconfigure(mode=TagMode(50))
        except RuntimeError as exc:
            errors.append(str(exc))

    sink = FakeSink()
    mode = TagMode(10)
    engine = _engine([FakeDetector()], [sink], mode=mode, lookahead=0, observer=observer)
    holder.append(engine)
    engine.open(INFO)
    _stream(engine, range(3))
    engine.close()
    assert len(errors) == 3 and all("re-entrant" in e for e in errors)
    assert engine.mode is mode
    assert [i for i, _ in sink.written] == [0, 1, 2]


def test_critical_sink_failure_during_tail_aborts_and_installs_nothing() -> None:
    preview, recording = FakeSink(critical=False), FakeSink(fail_write_at=4)
    mode, new_mode = TagMode(10), TagMode(50)
    new_detector = FakeDetector(stateful=True)
    engine, _, _ = _swap_engine(mode=mode, sinks=[preview, recording], lookahead=3)
    engine.open(INFO)
    _stream(engine, range(6))
    with pytest.raises(OSError, match="sink broke"):
        engine.reconfigure(detectors=[new_detector], mode=new_mode)
    assert not engine.is_open
    assert engine.mode is mode
    assert new_detector.resets == 0
    assert new_mode.calls == []
    assert [i for i, _ in recording.written] == [0, 1, 2, 3]
    for sink in (preview, recording):
        assert sink.events == ["open", "abort"]
    with pytest.raises(RuntimeError, match="not open"):
        engine.step(_frame(6))


def test_stats_count_reconfigurations_and_keep_the_partition() -> None:
    engine, _, _ = _swap_engine(detectors=[FakeDetector(hits=[1, 9], fail=[5])], lookahead=2)
    engine.open(INFO)
    _stream(engine, range(4))
    engine.reconfigure(mode=TagMode(50))
    _stream(engine, range(4, 8))
    engine.reconfigure(detectors=[FakeDetector(hits=[9], fail=[10])])
    _stream(engine, range(8, 12))
    stats = engine.close()
    assert stats.reconfigurations == 2
    assert stats.frames == 12
    assert stats.failed_closed == 2  # Frame 5 (first detector) and frame 10 (second).
    assert stats.frozen + stats.passed_through + stats.failed_closed == stats.frames


def test_detector_failing_after_swap_fails_closed() -> None:
    frames = [_frame(i) for i in range(10)]
    sink = FakeSink()
    engine, _, traces = _swap_engine(sinks=[sink], mode=FakeMode(), lookahead=1)
    engine.open(INFO)
    for frame in frames[:4]:
        engine.step(frame)
    engine.reconfigure(detectors=[FakeDetector(fail=range(4, 10))])
    for frame in frames[4:]:
        engine.step(frame)
    stats = engine.close()
    assert len(sink.received) == len(frames)
    for frame, image, meta in zip(frames, sink.received, sink.metas):
        assert meta == frame.meta
        if frame.index >= 4:
            assert _is_black(image, frame.image)
            assert not np.shares_memory(image, frame.image)
            assert not np.any(image == 100 + frame.index)
    assert [t.index for t in traces if t.failed_closed] == list(range(4, 10))
    assert stats.failed_closed == 6


# ---------------------------------------------------------------------------
# Every reconfigure freezes the frames around the switch (P3-1 follow-up)
# ---------------------------------------------------------------------------

# (name, old lookahead, swap arguments, new lookahead)
FREEZE_SWAPS: list[tuple[str, int, Callable[[], dict[str, object]], int]] = [
    ("mode", 3, lambda: {"mode": TagMode(50)}, 3),
    ("stages", 3, lambda: {"stages": [FakeStage()]}, 3),
    ("lookahead_grows", 2, lambda: {"config": EngineConfig(lookahead=5, cold_start_frames=0)}, 5),
    (
        "lookahead_shrinks",
        5,
        lambda: {"config": EngineConfig(lookahead=2, cold_start_frames=0)},
        2,
    ),
]
FREEZE_IDS = [case[0] for case in FREEZE_SWAPS]


@pytest.mark.parametrize(("name", "old_k", "swap", "new_k"), FREEZE_SWAPS, ids=FREEZE_IDS)
def test_tail_frozen_when_subject_appears_just_after_swap(
    name: str, old_k: int, swap: Callable[[], dict[str, object]], new_k: int
) -> None:
    n = 10
    detector = FakeDetector(hits=[n])  # First frame after the switch.
    engine, _, traces = _swap_engine(detectors=[detector], lookahead=old_k)
    engine.open(INFO)
    _stream(engine, range(n))
    tail = engine.reconfigure(**swap())  # type: ignore[arg-type]
    _stream(engine, range(n, n + 12))
    engine.close()
    assert [o.trace.index for o in tail] == list(range(n - old_k, n))
    assert all(o.trace.final for o in tail)
    assert not any(o.trace.raw or o.trace.filled for o in tail)  # Only final is forced.
    # After the switch: the forced window, or the new gap fill around the hit at n.
    end = n + max(max(old_k, new_k), new_k + 1)
    assert _finals(traces) == list(range(n - old_k, end))


@pytest.mark.parametrize(("name", "old_k", "swap", "new_k"), FREEZE_SWAPS, ids=FREEZE_IDS)
def test_frames_after_swap_frozen_when_subject_was_just_before(
    name: str, old_k: int, swap: Callable[[], dict[str, object]], new_k: int
) -> None:
    n = 10
    detector = FakeDetector(hits=[n - 1])  # Last frame before the switch; none after.
    engine, _, traces = _swap_engine(detectors=[detector], lookahead=old_k)
    engine.open(INFO)
    _stream(engine, range(n))
    engine.reconfigure(**swap())  # type: ignore[arg-type]
    _stream(engine, range(n, n + 15))
    engine.close()
    window = max(old_k, new_k)
    assert _finals(traces) == list(range(n - 1 - old_k, n + window))
    assert [t.index for t in traces] == list(range(n + 15))


@pytest.mark.parametrize(
    "make_config",
    [lambda c: replace(c, frame_budget_ms=500.0), lambda c: None],
    ids=["budget_only", "all_none"],
)
def test_no_op_reconfigure_neither_flushes_nor_freezes(
    make_config: Callable[[EngineConfig], EngineConfig | None],
) -> None:
    engine, (sink,), traces = _swap_engine(detectors=[FakeDetector(hits=[2])], lookahead=2)
    engine.open(INFO)
    _stream(engine, range(6))
    written = len(sink.written)
    config = make_config(engine.config)
    assert engine.reconfigure(config=config) == []
    assert len(sink.written) == written  # Nothing flushed.
    if config is not None:
        assert engine.config is config
    _stream(engine, range(6, 12))
    stats = engine.close()
    assert [t.index for t in traces] == list(range(12))
    assert [i for i, _ in sink.written] == list(range(12))
    assert _finals(traces) == [0, 1, 2, 3, 4]  # Only the gap fill around the hit at 2.
    assert stats.reconfigurations == 1


def test_budget_only_reconfigure_applies_the_new_budget() -> None:
    clock = FakeClock()
    traces: list[FrameTrace] = []
    engine = Engine(
        [SlowDetector(clock, {5: 0.030})],
        FakeMode(),
        [],
        [FakeSink()],
        EngineConfig(lookahead=0, cold_start_frames=0),
        traces.append,
        clock,
    )
    engine.open(INFO)
    _stream(engine, range(3))
    engine.reconfigure(config=replace(engine.config, frame_budget_ms=20.0))
    _stream(engine, range(3, 8))
    stats = engine.close()
    assert [t.index for t in traces if t.failed_closed] == [5]
    assert stats.over_budget == 1


@pytest.mark.parametrize("subject_at", [7, 8, 9, 10, 11, 12], ids=lambda i: f"subject{i}")
@pytest.mark.parametrize("swap", ["mode", "stages", "lookahead"])
def test_no_subject_pixels_reach_sinks_around_a_swap(subject_at: int, swap: str) -> None:
    """A pass-through mode; the detector sees the subject on one frame near the switch.

    Every frame within one lookahead of the subject must be frozen, on either side
    of the switch at 10, so the sink never receives those input pixels. The
    lookahead is 3 before the switch, and 1 after it in the ``lookahead`` case.
    """
    k, n = 3, 10
    radius = 1 if swap == "lookahead" and subject_at >= n else k
    frames = [_frame(i) for i in range(n + 10)]
    sink = FakeSink()
    engine, _, _ = _swap_engine(
        detectors=[FakeDetector(hits=[subject_at])], mode=FakeMode(), sinks=[sink], lookahead=k
    )
    engine.open(INFO)
    for frame in frames[:n]:
        engine.step(frame)
    args: dict[str, object] = {
        "mode": {"mode": FakeMode()},
        "stages": {"stages": []},
        "lookahead": {"config": EngineConfig(lookahead=1, cold_start_frames=0)},
    }[swap]  # type: ignore[assignment]
    engine.reconfigure(**args)  # type: ignore[arg-type]
    for frame in frames[n:]:
        engine.step(frame)
    engine.close()
    assert [m.index for m in sink.metas] == list(range(n + 10))
    for frame, image in zip(frames, sink.received):
        if abs(frame.index - subject_at) <= radius:
            assert not np.shares_memory(image, frame.image), frame.index
            assert not np.any(image == 100 + frame.index), frame.index
            assert np.all(image == BACKGROUND_VALUE), frame.index


# ---------------------------------------------------------------------------
# Timings (P3-1)
# ---------------------------------------------------------------------------


class SlowSink(FakeSink):
    """Advances ``clock`` by ``cost`` seconds per write."""

    def __init__(self, clock: FakeClock, cost: float) -> None:
        super().__init__()
        self.clock = clock
        self.cost = cost

    def write(self, image: np.ndarray, meta: FrameMeta) -> None:
        self.clock.now += self.cost
        super().write(image, meta)


def test_timings_follow_the_clock() -> None:
    clock = FakeClock()
    detect_costs = {0: 0.010, 1: 0.020, 2: 0.010}
    render_costs = {0: 0.005, 1: 0.005, 2: 0.015}
    engine = Engine(
        [SlowDetector(clock, detect_costs)],
        SlowMode(clock, render_costs),
        [],
        [SlowSink(clock, 0.002)],
        EngineConfig(lookahead=0, cold_start_frames=0),
        clock=clock,
    )
    empty = engine.timings
    assert empty.detect.mean_ms is None and empty.detect.last_ms is None
    assert empty.render.mean_ms is None and empty.write.last_ms is None
    assert empty.frames_per_second is None

    engine.open(INFO)
    engine.step(_frame(0))
    first = engine.timings
    assert first.detect.last_ms == pytest.approx(10.0)
    assert first.detect.mean_ms == pytest.approx(10.0)
    assert first.render.last_ms == pytest.approx(5.0)
    assert first.write.last_ms == pytest.approx(2.0)
    assert first.frames_per_second is None  # One frame emitted.

    engine.step(_frame(1))
    second = engine.timings
    assert second.detect.last_ms == pytest.approx(20.0)
    assert second.detect.mean_ms == pytest.approx(10.0 + 0.1 * (20.0 - 10.0))
    assert second.render.mean_ms == pytest.approx(5.0)
    # Interval between the two writes: detect 20 + render 5 + write 2 = 27 ms.
    assert second.frames_per_second == pytest.approx(1000.0 / 27.0)

    engine.step(_frame(2))
    third = engine.timings
    assert third.render.last_ms == pytest.approx(15.0)
    assert third.render.mean_ms == pytest.approx(5.0 + 0.1 * (15.0 - 5.0))
    assert third.detect.mean_ms == pytest.approx(11.0 + 0.1 * (10.0 - 11.0))
    # Second interval: detect 10 + render 15 + write 2 = 27 ms, so the mean stays 27.
    assert third.frames_per_second == pytest.approx(1000.0 / 27.0)
    assert third.write.mean_ms == pytest.approx(2.0)
    engine.close()

    # Snapshots are immutable and do not follow later frames.
    assert second.detect.last_ms == pytest.approx(20.0)
    with pytest.raises(AttributeError):
        second.frames_per_second = 1.0  # type: ignore[misc]
    with pytest.raises(AttributeError):
        second.detect.mean_ms = 1.0  # type: ignore[misc]


def test_timings_fps_is_inverse_of_mean_interval() -> None:
    clock = FakeClock()
    engine = Engine(
        [FakeDetector()],
        FakeMode(),
        [],
        [FakeSink()],
        EngineConfig(lookahead=0, cold_start_frames=0),
        clock=clock,
    )
    engine.open(INFO)
    for i, gap in enumerate([0.0, 0.010, 0.030]):
        clock.now += gap
        engine.step(_frame(i))
    mean = 0.010 + 0.1 * (0.030 - 0.010)
    assert engine.timings.frames_per_second == pytest.approx(1.0 / mean)
    engine.close()


def test_timings_skip_render_for_unrendered_frames_and_reset_on_open() -> None:
    clock = FakeClock()
    mode = SlowMode(clock, {0: 0.004, 1: 0.004})
    engine = Engine(
        [FakeDetector(fail=[1])],
        mode,
        [],
        [FakeSink()],
        EngineConfig(lookahead=0, cold_start_frames=0),
        clock=clock,
    )
    engine.open(INFO)
    engine.step(_frame(0))
    clock.now += 0.050
    engine.step(_frame(1))  # Detection fails: not rendered.
    timings = engine.timings
    assert timings.render.last_ms == pytest.approx(4.0)
    assert timings.render.mean_ms == pytest.approx(4.0)
    assert timings.detect.last_ms == pytest.approx(0.0)
    assert timings.frames_per_second == pytest.approx(1000.0 / 50.0)
    engine.close()
    engine.open(INFO)
    assert engine.timings.render.mean_ms is None
    assert engine.timings.frames_per_second is None
    engine.close()


def test_timings_with_a_still_clock_report_no_rate() -> None:
    clock = FakeClock()
    engine = Engine(
        [FakeDetector()], FakeMode(), [], [FakeSink()], EngineConfig(lookahead=0), clock=clock
    )
    engine.open(INFO)
    _stream(engine, range(3))
    timings = engine.timings
    assert timings.detect.mean_ms == 0.0
    assert timings.frames_per_second is None  # Zero mean interval: undefined.
    engine.close()


def test_timings_do_not_change_budget_decisions() -> None:
    def run(read_timings: bool) -> tuple[EngineStats, list[FrameTrace]]:
        clock = FakeClock()
        traces: list[FrameTrace] = []
        engine = Engine(
            [SlowDetector(clock, {2: 0.030, 5: 0.012})],
            SlowMode(clock, {5: 0.012, 7: 0.025}),
            [],
            [SlowSink(clock, 0.050)],  # Write time is not part of the budget.
            EngineConfig(lookahead=1, cold_start_frames=0, frame_budget_ms=20.0),
            traces.append,
            clock,
        )
        engine.open(INFO)
        for i in range(9):
            engine.step(_frame(i))
            if read_timings:
                assert engine.timings.write.last_ms is not None or i == 0
        return engine.close(), traces

    with_reads, plain = run(True), run(False)
    assert with_reads == plain
    stats, traces = plain
    assert [t.index for t in traces if t.failed_closed] == [2, 5, 7]
    assert stats.over_budget == 3


# ---------------------------------------------------------------------------
# Detection policy, result checks, refiners, output checks (Phase 4 T2)
# ---------------------------------------------------------------------------

M, P, BD, MASK, FACES = (
    DetectionKind.MOTION,
    DetectionKind.PERSONS,
    DetectionKind.BACKGROUND_DIFF,
    DetectionKind.MASK,
    DetectionKind.FACES,
)


class RegionMode:
    """Records the detections it renders with; frozen frames become the background.

    ``detect_policy`` and ``mandatory`` are set only when given, so the engine's
    defaults for an absent attribute can be tested.
    """

    name = "region"

    def __init__(
        self,
        requires: Iterable[DetectionKind],
        policy: object = None,
        mandatory: Iterable[DetectionKind] | None = None,
        output: Callable[[Frame], object] | None = None,
    ) -> None:
        self.requires = frozenset(requires)
        if policy is not None:
            self.detect_policy = policy
        if mandatory is not None:
            self.mandatory = frozenset(mandatory)
        self.output = output
        self.seen: list[tuple[int, Detections, bool]] = []
        self.mask_sums: list[int] = []

    def render(self, frame: Frame, detections: Detections, subject_present: bool) -> np.ndarray:
        self.seen.append((frame.index, detections, subject_present))
        if detections.mask is not None:
            self.mask_sums.append(int(detections.mask.sum()))
        if self.output is not None:
            return self.output(frame)  # type: ignore[return-value]
        if subject_present:
            return np.full_like(frame.image, BACKGROUND_VALUE)
        return frame.image

    def persons(self) -> list[tuple[Box, ...] | None]:
        return [d.persons for _, d, _ in self.seen]


class ResultDetector:
    """Returns ``result(frame)`` as is; records the frames it saw."""

    def __init__(
        self,
        provides: Iterable[object],
        result: Callable[[Frame], object],
        *,
        stateful: bool = False,
    ) -> None:
        self.provides = frozenset(provides)
        self.stateful = stateful
        self.result = result
        self.seen: list[int] = []
        self.resets = 0

    def detect(self, frame: Frame) -> Detections:
        self.seen.append(frame.index)
        return self.result(frame)  # type: ignore[return-value]

    def reset(self) -> None:
        self.resets += 1


def _box_detector(box: Box, *, stateful: bool = False) -> ResultDetector:
    """A persons detector that finds ``box`` on every frame."""
    return ResultDetector({P}, lambda f: Detections(persons=(box,)), stateful=stateful)


class FakeRefiner:
    """Returns ``result(frame, detections)``; records calls and resets."""

    def __init__(
        self,
        consumes: Iterable[DetectionKind],
        provides: Iterable[DetectionKind],
        result: Callable[[Frame, Detections], object] | None = None,
    ) -> None:
        self.consumes = frozenset(consumes)
        self.provides = frozenset(provides)
        self.result = result or (lambda f, d: Detections())
        self.calls: list[tuple[int, Detections]] = []
        self.resets = 0

    def refine(self, frame: Frame, detections: Detections) -> Detections:
        self.calls.append((frame.index, detections))
        return self.result(frame, detections)  # type: ignore[return-value]

    def reset(self) -> None:
        self.resets += 1


def _p_engine(
    detectors: Sequence[Detector],
    mode: Mode,
    *,
    refiners: Sequence[Refiner] = (),
    stages: Sequence[Stage] = (),
    sinks: Sequence[Sink] | None = None,
    lookahead: int = 0,
    cold_start: int = 0,
    traces: list[FrameTrace] | None = None,
) -> Engine:
    return Engine(
        detectors,
        mode,
        stages,
        list(sinks) if sinks is not None else [FakeSink()],
        EngineConfig(lookahead=lookahead, cold_start_frames=cold_start),
        None if traces is None else traces.append,
        refiners=refiners,
    )


def _p_run(
    n: int,
    detectors: Sequence[Detector],
    mode: Mode,
    **kwargs: object,
) -> tuple[EngineStats, list[FrameTrace], FakeSink]:
    traces: list[FrameTrace] = []
    sink = FakeSink()
    engine = _p_engine(detectors, mode, sinks=[sink], traces=traces, **kwargs)  # type: ignore[arg-type]
    stats = engine.run(FakeSource(n))
    return stats, traces, sink


BOX_A = Box(x=0, y=0, w=1, h=1)
BOX_B = Box(x=1, y=1, w=2, h=2)


def test_new_fakes_satisfy_protocols() -> None:
    assert isinstance(FakeRefiner({M}, {P}), Refiner)
    assert isinstance(RegionMode({M}), Mode)
    assert isinstance(RegionMode({M}, policy="all", mandatory={M}), Mode)


def test_all_policy_runs_every_stateless_detector_after_a_hit() -> None:
    a, b = _box_detector(BOX_A), _box_detector(BOX_B)
    mode = RegionMode({P}, policy="all")
    _p_run(3, [a, b], mode)
    assert a.seen == [0, 1, 2]
    assert b.seen == [0, 1, 2]
    assert mode.persons() == [(BOX_A, BOX_B)] * 3


def test_first_hit_policy_still_short_circuits() -> None:
    a, b = _box_detector(BOX_A), _box_detector(BOX_B)
    stateful = _box_detector(BOX_B, stateful=True)
    mode = RegionMode({P}, policy="first_hit")
    _p_run(3, [a, b, stateful], mode)
    assert a.seen == [0, 1, 2]
    assert b.seen == []
    assert stateful.seen == [0, 1, 2]
    assert mode.persons() == [(BOX_A, BOX_B)] * 3  # a, then the stateful one.


def test_absent_detect_policy_defaults_to_first_hit() -> None:
    a, b = _box_detector(BOX_A), _box_detector(BOX_B)
    mode = RegionMode({P})
    assert not hasattr(mode, "detect_policy")
    _p_run(3, [a, b], mode)
    assert b.seen == []
    assert mode.persons() == [(BOX_A,)] * 3


@pytest.mark.parametrize("policy", ["sometimes", "ALL", 1, ("all",)], ids=repr)
def test_invalid_detect_policy_rejected(policy: object) -> None:
    with pytest.raises(ValueError, match="detect_policy"):
        _p_engine([_box_detector(BOX_A)], RegionMode({P}, policy=policy))
    engine = _p_engine([_box_detector(BOX_A)], RegionMode({P}))
    engine.open(INFO)
    with pytest.raises(ValueError, match="detect_policy"):
        engine.reconfigure(mode=RegionMode({P}, policy=policy))
    assert engine.is_open
    engine.close()


def test_mandatory_kind_uncovered_at_construction_rejected() -> None:
    motion = FakeDetector()
    with pytest.raises(ValueError, match="mask as mandatory"):
        _p_engine([motion], RegionMode({M, MASK}, "all", mandatory={MASK}))
    # Without mandatory the same set-up is accepted: MOTION is still provided.
    assert _p_engine([motion], RegionMode({M, MASK}, "all")).detectors == (motion,)
    masker = ResultDetector({MASK}, lambda f: Detections(mask=np.zeros((H, W), dtype=bool)))
    engine = _p_engine([motion, masker], RegionMode({M, MASK}, "all", mandatory={MASK}))
    assert engine.detectors == (motion, masker)


def test_mandatory_satisfied_only_by_an_inactive_detector_rejected() -> None:
    # The persons detector provides the mandatory kind but the mode does not
    # require it, so the detector is not active.
    persons = FakeDetector(kind=DetectionKind.PERSONS)
    with pytest.raises(ValueError, match="persons as mandatory"):
        _p_engine([FakeDetector(), persons], RegionMode({M}, "all", mandatory={P}))


@pytest.mark.parametrize("policy", ["first_hit", None], ids=["explicit", "absent"])
def test_mandatory_under_first_hit_rejected(policy: str | None) -> None:
    detectors = [FakeDetector(), FakeDetector(kind=DetectionKind.PERSONS)]
    with pytest.raises(ValueError, match='detect_policy = "all"'):
        _p_engine(detectors, RegionMode({M, P}, policy, mandatory={P}))
    mode = RegionMode({M, P}, "all", mandatory={P})
    engine = _p_engine(detectors, mode)
    engine.open(INFO)
    _stream(engine, range(2))
    with pytest.raises(ValueError, match='detect_policy = "all"'):
        engine.reconfigure(mode=RegionMode({M, P}, policy, mandatory={P}))
    assert engine.mode is mode and engine.is_open
    engine.close()


def test_mandatory_persons_reach_the_mode_after_a_motion_hit() -> None:
    """The reproduced bug: the mode must never get persons=None for a mandatory kind."""
    persons = _box_detector(BOX_A)
    mode = RegionMode({M, P}, "all", mandatory={P})
    _p_run(3, [FakeDetector(hits=[0, 1, 2]), persons], mode)
    assert persons.seen == [0, 1, 2]
    assert mode.persons() == [(BOX_A,)] * 3


@pytest.mark.parametrize(
    "mandatory",
    [{"bogus"}, {"persons"}, "persons", {P, 3}],
    ids=["bogus", "plain_string_member", "string", "int_member"],
)
def test_mandatory_with_non_kind_members_rejected(mandatory: object) -> None:
    mode = RegionMode({M, P}, "all")
    mode.mandatory = mandatory  # type: ignore[attr-defined]
    detectors = [FakeDetector(), FakeDetector(kind=DetectionKind.PERSONS)]
    with pytest.raises(ValueError, match="not a DetectionKind"):
        _p_engine(detectors, mode)


def test_non_string_policy_equal_to_a_policy_name_rejected() -> None:
    class LooksLikeAll:
        def __eq__(self, other: object) -> bool:
            return True

        __hash__ = object.__hash__

    with pytest.raises(ValueError, match="detect_policy"):
        _p_engine([FakeDetector()], RegionMode({M}, LooksLikeAll()))


def test_mandatory_whose_iteration_raises_is_a_value_error() -> None:
    class Hostile:
        def __iter__(self) -> Iterator[DetectionKind]:
            raise RuntimeError("no iteration")

    mode = RegionMode({M}, "all")
    mode.mandatory = Hostile()  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="mandatory"):
        _p_engine([FakeDetector()], mode)


def test_hostile_mode_attributes_are_value_errors() -> None:
    class RaisingPolicy(RegionMode):
        @property
        def detect_policy(self) -> str:  # type: ignore[override]
            raise RuntimeError("policy broke")

    class RaisingMandatory(RegionMode):
        @property
        def mandatory(self) -> frozenset[DetectionKind]:  # type: ignore[override]
            raise RuntimeError("mandatory broke")

    class PolicyStr(str):
        def __eq__(self, other: object) -> bool:
            raise RuntimeError("eq broke")

        __hash__ = str.__hash__

    for mode in (RaisingPolicy({M}), RaisingMandatory({M}), RegionMode({M}, PolicyStr("all"))):
        with pytest.raises(ValueError):
            _p_engine([FakeDetector()], mode)


def test_detector_provides_that_cannot_be_matched_is_a_value_error() -> None:
    detector = FakeDetector()
    detector.provides = [M]  # type: ignore[assignment]  # list & frozenset: TypeError.
    with pytest.raises(ValueError, match="Detector FakeDetector: cannot match"):
        _p_engine([detector], RegionMode({M}))


def test_mandatory_kind_covered_by_a_refiner_is_accepted() -> None:
    refiner = FakeRefiner({M}, {MASK})
    mode = RegionMode({M, MASK}, policy="all", mandatory={MASK})
    assert _p_engine([FakeDetector()], mode, refiners=[refiner]).refiners == (refiner,)


def test_mandatory_kind_uncovered_at_reconfigure_changes_nothing() -> None:
    motion = FakeDetector(hits=[3])
    masker = ResultDetector({MASK}, lambda f: Detections(mask=np.zeros((H, W), dtype=bool)))
    mode = RegionMode({M, MASK}, "all", mandatory={MASK})
    sink = FakeSink()
    traces: list[FrameTrace] = []
    engine = _p_engine([motion, masker], mode, sinks=[sink], lookahead=2, traces=traces)
    config = engine.config
    engine.open(INFO)
    _stream(engine, range(4))
    written = len(sink.written)
    with pytest.raises(ValueError, match="as mandatory"):
        engine.reconfigure(detectors=[motion], stages=[FakeStage()])
    with pytest.raises(ValueError, match="as mandatory"):
        engine.reconfigure(mode=RegionMode({M, MASK, P}, "all", mandatory={P}))
    assert engine.is_open
    assert len(sink.written) == written
    assert engine.detectors == (motion, masker)
    assert engine.mode is mode and engine.stages == () and engine.config is config
    _stream(engine, range(4, 8))
    stats = engine.close()
    assert [i for i, _ in sink.written] == list(range(8))
    assert masker.seen == list(range(8))
    assert _finals(traces) == [1, 2, 3, 4, 5]
    assert stats.reconfigurations == 0


MALFORMED: dict[str, tuple[frozenset[object], object]] = {
    "not_detections": (frozenset({M}), {"motion": True}),
    "field_outside_provides": (frozenset({M}), Detections(persons=())),
    "wrong_shape_mask": (frozenset({MASK}), Detections(mask=np.zeros((H + 1, W), dtype=bool))),
}


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_malformed_detector_result_fails_the_frame_closed(
    case: str, caplog: pytest.LogCaptureFixture
) -> None:
    provides, bad = MALFORMED[case]

    def result(frame: Frame) -> object:
        return bad if frame.index == 2 else Detections()

    detector = ResultDetector(provides, result)
    mode = RegionMode({M, MASK}, policy="all")
    engine_detectors: list[Detector] = [FakeDetector(), detector]
    with caplog.at_level(logging.WARNING, logger="petrify.core.engine"):
        stats, traces, sink = _p_run(6, engine_detectors, mode, lookahead=1)
    assert detector.seen == list(range(6))
    images = dict(sink.written)
    assert sorted(images) == list(range(6))
    assert _is_black(images[2], _image(2))
    assert [t.index for t in traces if t.failed_closed] == [2]
    assert traces[2].raw
    assert _finals(traces) == [1, 2, 3]  # Neighbors frozen.
    assert 2 not in [i for i, _, _ in mode.seen]
    assert stats.failed_closed == 1 and stats.frames == 6
    assert stats.frozen + stats.passed_through + stats.failed_closed == stats.frames
    messages = [r.getMessage() for r in caplog.records]
    assert any("Frame 2 failed closed (detector ResultDetector)" in m for m in messages)


def test_detector_with_bad_provides_fails_every_frame_closed() -> None:
    # A plain string equals the str-valued enum, so the engine activates it, but
    # check_detections rejects it as an unknown kind on every frame.
    detector = ResultDetector({"motion"}, lambda f: Detections(motion=False))
    stats, traces, sink = _p_run(3, [detector], RegionMode({M}, "all"))
    assert detector.seen == [0, 1, 2]
    assert all(t.failed_closed for t in traces)
    assert all(_is_black(img, _image(i)) for i, img in sink.written)
    assert stats.failed_closed == 3 and sink.events == ["open", "close"]


def test_stateful_detector_still_sees_frame_after_malformed_result() -> None:
    for policy in ("first_hit", "all"):
        broken = ResultDetector(
            {M}, lambda f: "junk" if f.index == 1 else Detections(motion=False)
        )
        stateful = FakeDetector(stateful=True)
        stateless = FakeDetector(kind=DetectionKind.BACKGROUND_DIFF)
        _, traces, _ = _p_run(3, [broken, stateful, stateless], RegionMode({M, BD}, policy))
        assert stateful.seen == [0, 1, 2], policy
        assert stateless.seen == [0, 2], policy
        assert [t.failed_closed for t in traces] == [False, True, False], policy


def test_reused_mask_buffer_cannot_change_a_buffered_frame() -> None:
    buffer = np.zeros((H, W), dtype=bool)

    def fill(frame: Frame) -> Detections:
        buffer[:] = False
        buffer.flat[: frame.index + 1] = True  # Frame i has i + 1 mask pixels.
        return Detections(mask=buffer)

    mode = RegionMode({MASK}, policy="all")
    _p_run(6, [ResultDetector({MASK}, fill, stateful=True)], mode, lookahead=3)
    assert mode.mask_sums == [1, 2, 3, 4, 5, 6]


def test_detector_boxes_are_clipped_before_the_mode_sees_them() -> None:
    wide = Box(x=-2, y=1, w=20, h=2)
    mode = RegionMode({P})
    _p_run(1, [_box_detector(wide)], mode)
    assert mode.persons() == [(Box(x=0, y=1, w=W, h=2),)]


# Refiners ---------------------------------------------------------------------


def test_refiner_adds_detections_the_mode_sees_and_lookahead_uses() -> None:
    def track(frame: Frame, detections: Detections) -> Detections:
        return Detections(persons=(BOX_A,) if frame.index == 4 else ())

    refiner = FakeRefiner({M}, {P}, track)
    mode = RegionMode({M, P}, policy="all")
    _, traces, sink = _p_run(8, [FakeDetector()], mode, refiners=[refiner], lookahead=1)
    assert [i for i, _ in refiner.calls] == list(range(8))
    assert all(d.motion is False for _, d in refiner.calls)  # Sees the detectors' output.
    assert mode.persons()[4] == (BOX_A,)
    assert [t.index for t in traces if t.raw] == [4]
    assert _finals(traces) == [3, 4, 5]
    images = dict(sink.written)
    for i in (3, 4, 5):
        assert np.all(images[i] == BACKGROUND_VALUE)


def test_refiner_cannot_remove_detections() -> None:
    seen_by_refiner: list[tuple[Box, ...] | None] = []

    def drop_all(frame: Frame, detections: Detections) -> Detections:
        seen_by_refiner.append(detections.persons)
        return Detections(persons=())

    mode = RegionMode({P}, policy="all")
    refiner = FakeRefiner({P}, {P}, drop_all)
    _, traces, _ = _p_run(3, [_box_detector(BOX_A)], mode, refiners=[refiner])
    assert seen_by_refiner == [(BOX_A,)] * 3
    assert mode.persons() == [(BOX_A,)] * 3
    assert all(t.raw for t in traces)


def test_later_refiner_sees_earlier_refiner_output() -> None:
    first = FakeRefiner({M}, {P}, lambda f, d: Detections(persons=(BOX_A,)))
    second = FakeRefiner({P}, {FACES})
    mode = RegionMode({M, P, FACES}, policy="all")
    _p_run(2, [FakeDetector()], mode, refiners=[first, second])
    assert [d.persons for _, d in second.calls] == [(BOX_A,)] * 2


def _raise(frame: Frame, detections: Detections) -> object:
    raise RuntimeError("tracker broke")


REFINER_FAILURES: dict[str, Callable[[Frame, Detections], object]] = {
    "raises": _raise,
    "not_detections": lambda f, d: [BOX_A],
    "field_outside_provides": lambda f, d: Detections(motion=True),
    "wrong_shape_mask": lambda f, d: Detections(mask=np.zeros((H, W + 1), dtype=bool)),
}


@pytest.mark.parametrize("case", sorted(REFINER_FAILURES))
def test_refiner_failure_fails_the_frame_closed(
    case: str, caplog: pytest.LogCaptureFixture
) -> None:
    bad = REFINER_FAILURES[case]

    def result(frame: Frame, detections: Detections) -> object:
        return bad(frame, detections) if frame.index == 3 else Detections()

    broken = FakeRefiner({M}, {P, MASK}, result)
    later = FakeRefiner({M}, {P})
    mode = RegionMode({M, P, MASK}, policy="all")
    with caplog.at_level(logging.WARNING, logger="petrify.core.engine"):
        stats, traces, sink = _p_run(
            6, [FakeDetector()], mode, refiners=[broken, later], lookahead=1
        )
    assert _is_black(dict(sink.written)[3], _image(3))
    assert [t.index for t in traces if t.failed_closed] == [3]
    assert traces[3].raw
    assert _finals(traces) == [2, 3, 4]
    assert [i for i, _ in later.calls] == [0, 1, 2, 4, 5]  # Skipped on the failed frame.
    assert 3 not in [i for i, _, _ in mode.seen]
    assert stats.failed_closed == 1 and stats.frames == 6
    messages = [r.getMessage() for r in caplog.records]
    assert any("Frame 3 failed closed (refiner FakeRefiner)" in m for m in messages)


def test_refiners_do_not_run_on_a_frame_whose_detection_failed() -> None:
    refiner = FakeRefiner({M}, {P})
    _p_run(4, [FakeDetector(fail=[2])], RegionMode({M, P}, "all"), refiners=[refiner])
    assert [i for i, _ in refiner.calls] == [0, 1, 3]


@pytest.mark.parametrize("policy", ["first_hit", None], ids=["explicit", "absent"])
def test_refiner_rejected_under_first_hit(policy: str | None) -> None:
    with pytest.raises(ValueError, match="first_hit"):
        _p_engine(
            [FakeDetector()], RegionMode({M, P}, policy), refiners=[FakeRefiner({M}, {P})]
        )


def test_refiner_with_uncovered_consumes_rejected() -> None:
    with pytest.raises(ValueError, match="consumes mask"):
        _p_engine(
            [FakeDetector()], RegionMode({M, P}, "all"), refiners=[FakeRefiner({MASK}, {P})]
        )
    # A kind provided only by an inactive detector does not count.
    persons = FakeDetector(kind=DetectionKind.PERSONS)
    with pytest.raises(ValueError, match="consumes persons"):
        _p_engine(
            [FakeDetector(), persons],
            RegionMode({M, FACES}, "all"),
            refiners=[FakeRefiner({P}, {FACES})],
        )
    # A refiner may consume what an earlier refiner provides, not a later one.
    first = FakeRefiner({M}, {P})
    second = FakeRefiner({P}, {FACES})
    mode = RegionMode({M, P, FACES}, "all")
    assert _p_engine([FakeDetector()], mode, refiners=[first, second]).refiners == (first, second)
    with pytest.raises(ValueError, match="consumes persons"):
        _p_engine([FakeDetector()], mode, refiners=[second, first])


def test_refiner_inactive_when_mode_does_not_require_its_kinds() -> None:
    refiner = FakeRefiner({MASK}, {FACES})  # Uncovered consumes: ignored while inactive.
    mode = RegionMode({M})  # first_hit is fine while the refiner is inactive.
    engine = _p_engine([FakeDetector()], mode, refiners=[refiner])
    assert engine.refiners == ()
    engine.run(FakeSource(3))
    assert refiner.calls == [] and refiner.resets == 0


def test_refiner_only_provider_satisfies_requires() -> None:
    refiner = FakeRefiner(set(), {P}, lambda f, d: Detections(persons=(BOX_A,)))
    mode = RegionMode({P}, "all")
    stats, _, _ = _p_run(2, [], mode, refiners=[refiner])
    assert mode.persons() == [(BOX_A,)] * 2
    assert stats.frozen == 2


def test_refiner_reset_on_the_same_events_as_detectors() -> None:
    detector = FakeDetector(stateful=True)
    refiner = FakeRefiner({M}, {P})
    other = FakeRefiner({M}, {FACES})
    mode = RegionMode({M, P}, "all")
    engine = _p_engine([detector], mode, refiners=[refiner, other], lookahead=1)
    assert engine.refiners == (refiner,)
    assert (detector.resets, refiner.resets, other.resets) == (0, 0, 0)
    engine.open(INFO)
    assert (detector.resets, refiner.resets, other.resets) == (1, 1, 0)
    _stream(engine, range(3))
    engine.reconfigure(stages=[FakeStage()])  # Nothing new installed.
    engine.reconfigure(config=EngineConfig(lookahead=2))
    assert (detector.resets, refiner.resets, other.resets) == (1, 1, 0)
    engine.reconfigure(detectors=[detector])  # Given: reset even if unchanged.
    assert (detector.resets, refiner.resets) == (2, 1)
    engine.reconfigure(refiners=[refiner, other])  # Given: reset even if unchanged.
    assert (detector.resets, refiner.resets, other.resets) == (2, 2, 0)
    engine.reconfigure(mode=RegionMode({M, P, FACES}, "all"))  # Activates other only.
    assert engine.refiners == (refiner, other)
    assert (detector.resets, refiner.resets, other.resets) == (2, 2, 1)
    engine.close()
    engine.open(INFO)
    assert (detector.resets, refiner.resets, other.resets) == (3, 3, 2)
    engine.close()


def test_reconfigure_installs_refiners_with_a_cold_start() -> None:
    refiner = FakeRefiner({M}, {P})
    mode = RegionMode({M, P}, "all")
    traces: list[FrameTrace] = []
    engine = _p_engine([FakeDetector()], mode, lookahead=1, cold_start=2, traces=traces)
    assert engine.refiners == ()
    engine.open(INFO)
    _stream(engine, range(6))
    tail = engine.reconfigure(refiners=[refiner])
    assert engine.refiners == (refiner,)
    assert refiner.resets == 1
    _stream(engine, range(6, 12))
    engine.close()
    assert [o.trace.index for o in tail] == [5]
    assert [i for i, _ in refiner.calls] == list(range(6, 12))
    assert _finals(traces) == [0, 1, 5, 6, 7]  # Same as a detector swap.


def test_reconfigure_removing_refiners() -> None:
    refiner = FakeRefiner({M}, {P})
    engine = _p_engine([FakeDetector()], RegionMode({M, P}, "all"), refiners=[refiner])
    engine.open(INFO)
    _stream(engine, range(2))
    engine.reconfigure(refiners=[])
    _stream(engine, range(2, 4))
    engine.close()
    assert engine.refiners == ()
    assert [i for i, _ in refiner.calls] == [0, 1]


@pytest.mark.parametrize(
    "swap",
    [
        lambda: {"refiners": [FakeRefiner({MASK}, {P})]},
        lambda: {"mode": RegionMode({M, P})},
        lambda: {"refiners": [], "mode": RegionMode({M, P}, "all", mandatory={P})},
    ],
    ids=["consumes", "first_hit", "mandatory"],
)
def test_invalid_refiner_reconfigure_changes_nothing(
    swap: Callable[[], dict[str, object]],
) -> None:
    refiner = FakeRefiner({M}, {P})
    mode = RegionMode({M, P}, "all")
    sink = FakeSink()
    engine = _p_engine([FakeDetector()], mode, refiners=[refiner], sinks=[sink], lookahead=2)
    engine.open(INFO)
    _stream(engine, range(4))
    written = len(sink.written)
    with pytest.raises(ValueError):
        engine.reconfigure(**swap())  # type: ignore[arg-type]
    assert len(sink.written) == written
    assert engine.refiners == (refiner,) and engine.mode is mode
    assert refiner.resets == 1
    _stream(engine, range(4, 6))
    stats = engine.close()
    assert [i for i, _ in refiner.calls] == list(range(6))
    assert stats.reconfigurations == 0


# Output image checks ----------------------------------------------------------

BAD_OUTPUTS: dict[str, Callable[[Frame], object]] = {
    "wrong_shape": lambda f: np.zeros((H, W + 1, 3), dtype=np.uint8),
    "wrong_channels": lambda f: np.zeros((H, W), dtype=np.uint8),
    "wrong_dtype": lambda f: f.image.astype(np.float32),
    "not_an_array": lambda f: f.image.tolist(),
    "none": lambda f: None,
    "array_subclass": lambda f: np.ma.masked_array(f.image),
}


@pytest.mark.parametrize("case", sorted(BAD_OUTPUTS))
def test_mode_returning_a_bad_image_fails_closed(
    case: str, caplog: pytest.LogCaptureFixture
) -> None:
    bad = BAD_OUTPUTS[case]
    mode = RegionMode({M}, output=lambda f: bad(f) if f.index == 1 else f.image)
    stage = FakeStage()
    with caplog.at_level(logging.WARNING, logger="petrify.core.engine"):
        stats, traces, sink = _p_run(3, [FakeDetector()], mode, stages=[stage])
    images = dict(sink.written)
    assert _is_black(images[1], _image(1))
    assert type(sink.received[1]) is np.ndarray
    assert np.array_equal(images[0], _image(0) + 1)
    assert [t.failed_closed for t in traces] == [False, True, False]
    assert stage.applied == [0, 2]
    assert stats.failed_closed == 1 and stats.frames == 3
    messages = [r.getMessage() for r in caplog.records]
    assert any("Frame 1 failed closed (mode region)" in m for m in messages)


@pytest.mark.parametrize("case", ["wrong_shape", "wrong_dtype", "not_an_array"])
def test_stage_returning_a_bad_image_fails_closed(
    case: str, caplog: pytest.LogCaptureFixture
) -> None:
    bad = BAD_OUTPUTS[case]

    class BadStage:
        def apply(self, frame: Frame, image: np.ndarray) -> np.ndarray:
            return bad(frame) if frame.index == 2 else image  # type: ignore[return-value]

    later = FakeStage()
    stages: list[Stage] = [BadStage(), later]
    with caplog.at_level(logging.WARNING, logger="petrify.core.engine"):
        stats, traces, sink = _p_run(4, [FakeDetector()], RegionMode({M}), stages=stages)
    assert _is_black(dict(sink.written)[2], _image(2))
    assert [t.failed_closed for t in traces] == [False, False, True, False]
    assert later.applied == [0, 1, 3]
    assert stats.failed_closed == 1
    messages = [r.getMessage() for r in caplog.records]
    assert any("Frame 2 failed closed (stage BadStage)" in m for m in messages)


def test_stage_cannot_repair_a_bad_mode_image() -> None:
    """The mode's image is checked before the stages run, not only the final image."""

    class Cast:
        def apply(self, frame: Frame, image: np.ndarray) -> np.ndarray:
            return np.asarray(image, dtype=np.uint8)

    mode = RegionMode({M}, output=lambda f: f.image.astype(np.float32))
    stats, _, _ = _p_run(2, [FakeDetector()], mode, stages=[Cast()])
    assert stats.failed_closed == 2


# Review follow-up (T2b) -------------------------------------------------------

SHARING_OUTPUTS: dict[str, Callable[[Frame], np.ndarray]] = {
    "input_itself": lambda f: f.image,
    "full_view": lambda f: f.image[:],
    "reversed_view": lambda f: f.image[::-1],
}


@pytest.mark.parametrize("case", sorted(SHARING_OUTPUTS))
def test_mode_returning_input_memory_on_a_frozen_frame_fails_closed(
    case: str, caplog: pytest.LogCaptureFixture
) -> None:
    share = SHARING_OUTPUTS[case]
    mode = RegionMode({M}, "all", output=share)  # Broken: never hides anything.
    with caplog.at_level(logging.WARNING, logger="petrify.core.engine"):
        stats, traces, sink = _p_run(6, [FakeDetector(hits=[3])], mode, lookahead=1)
    frozen = [t.index for t in traces if t.final]
    assert frozen == [2, 3, 4]
    assert [t.index for t in traces if t.failed_closed] == frozen
    images = dict(sink.written)
    for i in range(6):
        if i in frozen:
            assert _is_black(images[i], _image(i))
            assert not np.may_share_memory(sink.received[i], _image(i))
        else:
            assert np.array_equal(images[i], share(_frame(i)))  # Pass-through is allowed.
    assert stats.failed_closed == 3 and stats.passed_through == 3
    messages = [r.getMessage() for r in caplog.records]
    assert any("Frame 2 failed closed (mode region)" in m and "share memory" in m
               for m in messages)


def test_stage_returning_input_memory_on_a_frozen_frame_fails_closed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Leak:
        def apply(self, frame: Frame, image: np.ndarray) -> np.ndarray:
            return frame.image

    later = FakeStage()
    stages: list[Stage] = [Leak(), later]
    with caplog.at_level(logging.WARNING, logger="petrify.core.engine"):
        stats, traces, sink = _p_run(4, [FakeDetector(hits=[1])], FakeMode(), stages=stages)
    assert [t.index for t in traces if t.failed_closed] == [1]
    assert _is_black(dict(sink.written)[1], _image(1))
    assert np.array_equal(dict(sink.written)[0], _image(0) + 1)
    assert later.applied == [0, 2, 3]
    assert stats.failed_closed == 1
    messages = [r.getMessage() for r in caplog.records]
    assert any("Frame 1 failed closed (stage Leak)" in m for m in messages)


def test_frozen_output_in_the_input_bounds_fails_closed_even_if_disjoint() -> None:
    """The check is the cheap bounds test, so an interleaved disjoint view is rejected."""
    base = np.full((H, 2 * W, 3), 100, dtype=np.uint8)
    frame = Frame(image=base[:, ::2], index=0, timestamp=0.0, source=INFO)
    other = base[:, 1::2]
    assert not np.shares_memory(frame.image, other)
    mode = RegionMode({M}, "all", output=lambda f: other)
    engine = _p_engine([FakeDetector(hits=[0])], mode)
    engine.open(INFO)
    (out,) = engine.step(frame)
    engine.close()
    assert out.trace.final and out.trace.failed_closed
    assert not out.image.any()


def test_frozen_frame_rendered_into_a_fresh_array_passes() -> None:
    mode = RegionMode({M}, "all", output=lambda f: f.image.copy())
    stats, _, _ = _p_run(3, [FakeDetector(hits=[1])], mode)
    assert stats.failed_closed == 0 and stats.frozen == 1


def test_reconfigure_updates_the_detect_policy() -> None:
    a, b = _box_detector(BOX_A), _box_detector(BOX_B)
    engine = _p_engine([a, b], RegionMode({P}))  # first_hit.
    engine.open(INFO)
    _stream(engine, range(2))
    engine.reconfigure(mode=RegionMode({P}, "all"))
    _stream(engine, range(2, 4))
    engine.reconfigure(mode=RegionMode({P}, "first_hit"))
    _stream(engine, range(4, 6))
    engine.close()
    assert a.seen == list(range(6))
    assert b.seen == [2, 3]


def test_refiner_cannot_satisfy_its_own_consumes() -> None:
    with pytest.raises(ValueError, match="consumes persons"):
        _p_engine([FakeDetector()], RegionMode({M, P}, "all"), refiners=[FakeRefiner({P}, {P})])


def test_refiner_sees_the_mask_a_detector_produced() -> None:
    def fill(frame: Frame) -> Detections:
        mask = np.zeros((H, W), dtype=bool)
        mask.flat[: frame.index + 1] = True
        return Detections(mask=mask)

    refiner = FakeRefiner({MASK}, {P})
    _p_run(3, [ResultDetector({MASK}, fill)], RegionMode({MASK, P}, "all"), refiners=[refiner])
    sums = [None if d.mask is None else int(d.mask.sum()) for _, d in refiner.calls]
    assert sums == [1, 2, 3]
    assert all(not d.mask.flags.writeable for _, d in refiner.calls)  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("consumes", "provides", "match"),
    [
        ({"bogus"}, {P}, "consumes holds 'bogus'"),
        ({"persons"}, {P}, "consumes holds 'persons'"),
        (None, {P}, "consumes must be a set"),
        ({M}, [P], "provides must be a set"),
        ({M}, (P,), "provides must be a set"),
        ({M}, {P, 1}, "provides holds 1"),
    ],
    ids=["bogus", "plain_string", "none", "list", "tuple", "int_member"],
)
def test_refiner_attribute_types_rejected_as_value_error(
    consumes: object, provides: object, match: str
) -> None:
    refiner = FakeRefiner({M}, {P})
    refiner.consumes = consumes  # type: ignore[assignment]
    refiner.provides = provides  # type: ignore[assignment]
    with pytest.raises(ValueError, match=match) as info:
        _p_engine([FakeDetector()], RegionMode({M, P}, "all"), refiners=[refiner])
    assert "Refiner FakeRefiner" in str(info.value)
    engine = _p_engine([FakeDetector()], RegionMode({M, P}, "all"))
    engine.open(INFO)
    with pytest.raises(ValueError, match=match):
        engine.reconfigure(refiners=[refiner])
    assert engine.refiners == () and engine.is_open
    engine.close()


def test_refiner_attribute_that_raises_is_a_value_error() -> None:
    class Hostile(FakeRefiner):
        @property
        def consumes(self) -> frozenset[DetectionKind]:  # type: ignore[override]
            raise RuntimeError("consumes broke")

        @consumes.setter
        def consumes(self, value: object) -> None:
            pass

    class Missing:
        provides = frozenset({P})

        def refine(self, frame: Frame, detections: Detections) -> Detections:
            return Detections()

        def reset(self) -> None:
            pass

    for refiner in (Hostile({M}, {P}), Missing()):
        with pytest.raises(ValueError, match="reading consumes"):
            _p_engine(
                [FakeDetector()], RegionMode({M, P}, "all"), refiners=[refiner]  # type: ignore[list-item]
            )


def test_duplicate_refiner_rejected() -> None:
    refiner = FakeRefiner({M}, {P})
    mode = RegionMode({M, P}, "all")
    with pytest.raises(ValueError, match="more than once"):
        _p_engine([FakeDetector()], mode, refiners=[refiner, refiner])
    engine = _p_engine([FakeDetector()], mode, refiners=[refiner])
    engine.open(INFO)
    with pytest.raises(ValueError, match="more than once"):
        engine.reconfigure(refiners=[refiner, FakeRefiner({M}, {P}), refiner])
    assert engine.refiners == (refiner,) and refiner.resets == 1
    _stream(engine, range(2))
    engine.close()
    assert [i for i, _ in refiner.calls] == [0, 1]
