"""Unit tests for petrify.core.engine.Engine using small fakes (no OpenCV)."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
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
    FrameTrace,
    Mode,
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
    def __init__(self, fail_write_at: int | None = None) -> None:
        self.events: list[str] = []
        self.written: list[tuple[int, np.ndarray]] = []
        self.info: SourceInfo | None = None
        self.fail_write_at = fail_write_at

    def open(self, info: SourceInfo) -> None:
        self.info = info
        self.events.append("open")

    def write(self, frame: Frame, image: np.ndarray) -> None:
        if frame.index == self.fail_write_at:
            raise OSError("sink broke")
        self.written.append((frame.index, image.copy()))

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
) -> tuple[EngineStats, list[FrameTrace], list[FakeSink]]:
    traces: list[FrameTrace] = []
    sink_list = list(sinks) if sinks is not None else [FakeSink()]
    engine = Engine(
        source or FakeSource(n),
        detectors,
        mode or FakeMode(),
        stages,
        sink_list,
        EngineConfig(lookahead=lookahead, cold_start_frames=cold_start),
        traces.append,
    )
    stats = engine.run()
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
    engine = Engine(
        FakeSource(3), [persons, motion], FakeMode(), [], [FakeSink()], EngineConfig(lookahead=0)
    )
    engine.run()
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
    assert bad.events == []
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
    engine = Engine(
        FakeSource(3), [FakeDetector()], FakeMode(), [], [sink], EngineConfig(lookahead=0), observer
    )
    with pytest.raises(ValueError, match="observer broke"):
        engine.run()
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
