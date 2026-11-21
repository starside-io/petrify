"""Tests for petrify.pipeline.run_stream with injected sources and sinks (no devices)."""

from __future__ import annotations

import dataclasses
import logging
import shutil
import signal
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, TypeVar
from unittest import mock

import cv2
import numpy as np
import pytest

import petrify.cli
import petrify.pipeline as pipeline
import petrify.sinks.file
import petrify.sources
from petrify.core import (
    BackgroundConfig,
    EncodeConfig,
    EngineConfig,
    Frame,
    FrameMeta,
    FrameTrace,
    PersonHogConfig,
    PetrifyConfig,
    RoiConfig,
    SourceInfo,
)
from petrify.pipeline import (
    NO_BACKGROUND_CHECK_WARNING,
    NO_BG_DIFF_LIVE_WARNING,
    ONLINE_BACKGROUND_WARNING,
    FfmpegNotFoundError,
    NoFramesError,
    OutputConfig,
    RunReport,
    run_file,
    run_stream,
)
from petrify.sinks import FfmpegPipeSink, PreviewSink, ThreadedSink, VirtualCamSink
from petrify.sources import FileSource, ImageFolderSource

T = TypeVar("T")

NEEDS_FFMPEG = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="needs ffmpeg and ffprobe",
)

# Frames stay at least 320x256 so a 0.5 HOG downsample never goes below the 64x128 window.
W, H = 320, 256
FPS = 10.0
SUBJECT = (0, 255, 0)  # Pure green: never present in the background texture.

ENCODE = EncodeConfig(codec="libx264", crf=23, preset="ultrafast", hwaccel=None)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class ListSink:
    """Records copies of every written image; can fail on a given write."""

    def __init__(self, *, critical: bool = True, fail_at: int | None = None,
                 on_write: Callable[[int], None] | None = None) -> None:
        self.critical = critical
        self.fail_at = fail_at
        self.on_write = on_write
        self.images: list[np.ndarray] = []
        self.metas: list[FrameMeta] = []
        self.opened = 0
        self.closed = 0
        self.aborted = 0

    def open(self, info: SourceInfo) -> None:
        self.opened += 1

    def write(self, image: np.ndarray, meta: FrameMeta) -> None:
        if self.fail_at is not None and len(self.images) == self.fail_at:
            raise RuntimeError("sink write failed")
        self.images.append(image.copy())
        self.metas.append(meta)
        if self.on_write is not None:
            self.on_write(len(self.images) - 1)

    def close(self) -> None:
        self.closed += 1

    def abort(self) -> None:
        self.aborted += 1


def _background() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(40, 120, size=(H, W, 3), dtype=np.uint8)


BACKGROUND = _background()


@dataclasses.dataclass(frozen=True)
class Scene:
    """A subject that enters, stands still, and leaves; optional ticking ROI patch."""

    total: int = 360
    enter: int = 130
    stand: int = 140
    leave: int = 230
    gone: int = 240
    tick_roi: bool = False

    def subject_x(self, i: int) -> int | None:
        if i < self.enter or i >= self.gone:
            return None
        if i < self.stand:
            return 20 + (i - self.enter) * 10
        if i < self.leave:
            return 120
        return 120 + (i - self.leave) * 20

    def image(self, i: int) -> np.ndarray:
        img = BACKGROUND.copy()
        x = self.subject_x(i)
        if x is not None:
            img[90:170, x : x + 60] = SUBJECT
        if self.tick_roi:
            img[4:12, 300:316] = (i * 37) % 256
        return img


class FakeSource:
    """In-memory source; live and not seekable by default."""

    def __init__(self, scene: Scene | None = None, *, live: bool = True,
                 interrupt_at: int | None = None, dropped: int = 0) -> None:
        self.scene = scene or Scene()
        self.info = SourceInfo(fps=FPS, width=W, height=H, frame_count=None,
                               seekable=False, live=live, uri="fake")
        self.interrupt_at = interrupt_at
        self.dropped = dropped
        self.closed = 0
        self.yielded: list[np.ndarray] = []

    def __iter__(self) -> Iterator[Frame]:
        for i in range(self.scene.total):
            if i == self.interrupt_at:
                raise KeyboardInterrupt
            img = self.scene.image(i)
            self.yielded.append(img)
            yield Frame(image=img, index=i, timestamp=i / FPS, source=self.info)

    def close(self) -> None:
        self.closed += 1


def _live_config(strategy: str, **overrides: Any) -> PetrifyConfig:
    base: dict[str, Any] = {
        "roi": RoiConfig(mode="disabled"),
        "background": BackgroundConfig(strategy=strategy, calibrate_seconds=2.0),  # type: ignore[arg-type]
        "engine": EngineConfig(lookahead=3),
        "encode": ENCODE,
    }
    base.update(overrides)
    return PetrifyConfig(**base)


def _is_black(img: np.ndarray) -> bool:
    return not img.any()


def _shows_subject(img: np.ndarray) -> bool:
    return bool(np.all(img == np.array(SUBJECT, dtype=np.uint8), axis=2).any())


def _kind(trace: FrameTrace) -> str:
    """What the engine did with a frame: black (failed closed), frozen, or pass."""
    if trace.failed_closed:
        return "black"
    return "frozen" if trace.final else "pass"


# ---------------------------------------------------------------------------
# Live path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["running", "calibrate"])
def test_live_subject_never_shown(strategy: str) -> None:
    src = FakeSource()
    sink = ListSink()
    traces: list[FrameTrace] = []
    report = run_stream(src, _live_config(strategy), OutputConfig(), sinks=[sink],
                        observer=traces.append)

    assert report.background_index is None
    assert report.interrupted is False
    assert src.closed == 1
    assert (sink.opened, sink.closed, sink.aborted) == (1, 1, 0)
    assert len(sink.images) == src.scene.total == report.stats.frames
    assert [m.index for m in sink.metas] == list(range(src.scene.total))

    kinds = [_kind(t) for t in traces]
    for k, (kind, out, inp) in enumerate(zip(kinds, sink.images, src.yielded)):
        expected = {"black": np.zeros_like(inp), "frozen": BACKGROUND, "pass": inp}[kind]
        assert np.array_equal(out, expected), (k, kind)
    assert not any(_shows_subject(o) for o in sink.images)
    # The background is not ready at the start: nothing but black until it is.
    first_ok = next(k for k, kind in enumerate(kinds) if kind != "black")
    assert first_ok > 0
    assert all(kind == "black" for kind in kinds[:first_ok])
    # Cold start frames are never passed through.
    assert "pass" not in kinds[:50]
    # Frozen while the subject is in view (when not black).
    for k in range(src.scene.enter, src.scene.gone):
        assert kinds[k] in ("frozen", "black"), (k, kinds[k])
    # Well after the subject left, frames pass through.
    assert all(kind == "pass" for kind in kinds[-30:])
    assert kinds.count("black") == report.stats.failed_closed
    assert sum(_is_black(o) for o in sink.images) == report.stats.failed_closed


def test_live_running_uses_background_recompute() -> None:
    created: list[dict[str, Any]] = []
    real = pipeline.RunningBackground

    def spy(*args: Any, **kwargs: Any) -> Any:
        created.append(kwargs)
        return real(*args, **kwargs)

    with mock.patch.object(pipeline, "RunningBackground", side_effect=spy):
        run_stream(FakeSource(Scene(total=80)), _live_config("running"), OutputConfig(),
                   sinks=[ListSink()])
        run_stream(FakeSource(Scene(total=80), live=False), _live_config("running"),
                   OutputConfig(), sinks=[ListSink()])
    assert [c["recompute_in_background"] for c in created] == [True, False]


@pytest.mark.parametrize(
    ("strategy", "bg_diff", "expected"),
    [("running", True, True), ("calibrate", True, True), ("running", False, False)],
)
def test_erase_mode_requires_differ_exactly_when_model_backed(
    strategy: str, bg_diff: bool, expected: bool
) -> None:
    seen: list[bool] = []
    real = pipeline.EraseMode

    def spy(background: Any, **kwargs: Any) -> Any:
        seen.append(kwargs["require_background_diff"])
        return real(background, **kwargs)

    overrides: dict[str, Any] = {} if bg_diff else {"background_diff": None}
    with mock.patch.object(pipeline, "EraseMode", side_effect=spy):
        run_stream(FakeSource(Scene(total=60)), _live_config(strategy, **overrides),
                   OutputConfig(), sinks=[ListSink()])
    assert seen == [expected]


class _FakeWarmupDetector:
    """Done after ``frames`` updates, then reports a fixed ROI."""

    roi_box = (296, 0, 24, 16)

    def __init__(self, fps: float, warmup_seconds: float, **kwargs: Any) -> None:
        self.args = (fps, warmup_seconds, kwargs)
        self.frames = int(warmup_seconds * fps)
        self.seen = 0
        _FakeWarmupDetector.instances.append(self)

    instances: list[_FakeWarmupDetector] = []

    @property
    def done(self) -> bool:
        return self.seen >= self.frames

    @property
    def roi(self) -> tuple[int, int, int, int] | None:
        return self.roi_box if self.done else None

    def update(self, image: np.ndarray) -> None:
        self.seen += 1


def test_live_auto_roi_pastes_nothing_before_lock() -> None:
    _FakeWarmupDetector.instances.clear()
    scene = Scene(tick_roi=True)
    src = FakeSource(scene)
    sink = ListSink()
    traces: list[FrameTrace] = []
    roi_cfg = RoiConfig(mode="auto", warmup_seconds=12.0, scan_min_freq=0.7, scan_margin=4,
                        scan_max_area_frac=0.2, scan_edge_threshold=70.0,
                        scan_edge_gap_tolerance=20, scan_edge_samples=5)
    with mock.patch.object(pipeline, "WarmupRoiDetector", _FakeWarmupDetector):
        report = run_stream(src, _live_config("calibrate", roi=roi_cfg), OutputConfig(),
                            sinks=[sink], observer=traces.append)

    (detector,) = _FakeWarmupDetector.instances
    assert detector.args == (
        FPS,
        12.0,
        {"min_freq": 0.7, "margin": 4, "max_area_frac": 0.2, "edge_threshold": 70.0,
         "edge_gap_tolerance": 20, "n_edge_samples": 5, "max_roi_area_frac": 0.15},
    )
    assert report.roi == _FakeWarmupDetector.roi_box
    x, y, w, h = _FakeWarmupDetector.roi_box
    kinds = [_kind(t) for t in traces]
    rendered = [k for k, kind in enumerate(kinds) if kind != "black"]
    lock = rendered[detector.frames - 1]  # The frame that completed the warmup.
    frozen_before = [k for k in rendered if k < lock and kinds[k] == "frozen"]
    frozen_after = [k for k in rendered if k >= lock and kinds[k] == "frozen"]
    assert frozen_before and frozen_after
    # Before the lock nothing is pasted: every frozen frame is the same background image,
    # whose ROI differs from the live frame's at least once.
    reference = sink.images[frozen_before[0]]
    for k in frozen_before:
        assert np.array_equal(sink.images[k], reference), k
    assert any(
        not np.array_equal(reference[y : y + h, x : x + w], src.yielded[k][y : y + h, x : x + w])
        for k in frozen_before
    )
    # From the lock on, the live ROI is pasted on every rendered frame.
    for k in rendered:
        if k >= lock:
            out, inp = sink.images[k], src.yielded[k]
            assert np.array_equal(out[y : y + h, x : x + w], inp[y : y + h, x : x + w]), k
    assert not any(_shows_subject(o) for o in sink.images)


def test_locked_background_on_live_source_is_rejected() -> None:
    src = FakeSource()
    with pytest.raises(ValueError, match="live sources need the running or calibrate"):
        run_stream(src, _live_config("locked"), OutputConfig(), sinks=[ListSink()])
    assert src.closed == 1


def test_no_output_is_rejected() -> None:
    src = FakeSource()
    with pytest.raises(ValueError, match="At least one output is required"):
        run_stream(src, _live_config("running"), OutputConfig())
    assert src.closed == 1
    src = FakeSource()
    with pytest.raises(ValueError, match="At least one sink"):
        run_stream(src, _live_config("running"), OutputConfig(preview=True), sinks=[])
    assert src.closed == 1


def test_output_config_rejects_device_without_virtualcam() -> None:
    with pytest.raises(ValueError, match="virtualcam_device requires"):
        OutputConfig(virtualcam_device="cam")


def test_stop_callback_ends_run_and_writes_tail() -> None:
    src = FakeSource()
    sink = ListSink()
    calls = 0

    def stop() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 60

    report = run_stream(src, _live_config("calibrate"), OutputConfig(), sinks=[sink], stop=stop)
    assert len(src.yielded) == 60
    assert len(sink.images) == report.stats.frames == 60  # The lookahead tail was written.
    assert (sink.closed, sink.aborted) == (1, 0)
    assert report.interrupted is False
    assert src.closed == 1


class _FakeGui:
    """GUI stand-in for PreviewSink: 'q' on the Nth waitKey, never opens a window."""

    def __init__(self, quit_at: int) -> None:
        self.quit_at = quit_at
        self.shown = 0

    def imshow(self, title: str, image: np.ndarray) -> None:
        self.shown += 1

    def waitKey(self, delay: int) -> int:  # noqa: N802 (mirrors the OpenCV API)
        return ord("q") if self.shown == self.quit_at else -1

    def destroyWindow(self, title: str) -> None:  # noqa: N802 (mirrors the OpenCV API)
        pass


def test_preview_stop_request_ends_run() -> None:
    src = FakeSource()
    sink = ListSink()
    preview = PreviewSink(gui=_FakeGui(quit_at=20))
    report = run_stream(src, _live_config("calibrate"), OutputConfig(), sinks=[sink, preview])
    assert preview.stop_requested
    lookahead = 3
    assert len(src.yielded) == 20 + lookahead
    assert len(sink.images) == report.stats.frames == 20 + lookahead
    assert (sink.closed, sink.aborted) == (1, 0)
    assert src.closed == 1


def _no_interrupt(call: Callable[[], T]) -> T:
    """Run ``call``; an escaped KeyboardInterrupt fails the test instead of ending pytest."""
    try:
        return call()
    except KeyboardInterrupt:
        pytest.fail("KeyboardInterrupt escaped the code under test", pytrace=False)


@pytest.fixture
def sigint_restored() -> Iterator[None]:
    """Restore the SIGINT handler after the test, even if the code under test does not."""
    previous = signal.getsignal(signal.SIGINT)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


@pytest.mark.usefixtures("sigint_restored")
def test_keyboard_interrupt_from_live_source_finishes_cleanly() -> None:
    src = FakeSource(interrupt_at=100, dropped=7)
    sink = ListSink()
    report = _no_interrupt(
        lambda: run_stream(src, _live_config("calibrate"), OutputConfig(), sinks=[sink])
    )
    assert isinstance(report, RunReport)
    assert report.interrupted is True
    assert report.dropped_frames == 7
    assert len(sink.images) == report.stats.frames == 100
    assert (sink.closed, sink.aborted) == (1, 0)
    assert src.closed == 1


@pytest.mark.usefixtures("sigint_restored")
def test_keyboard_interrupt_on_non_live_source_aborts() -> None:
    src = FakeSource(live=False, interrupt_at=100)
    sink = ListSink()
    with pytest.raises(KeyboardInterrupt):
        run_stream(src, _live_config("calibrate"), OutputConfig(), sinks=[sink])
    assert (sink.closed, sink.aborted) == (0, 1)
    assert src.closed == 1


def _default_sigint() -> bool:
    return signal.getsignal(signal.SIGINT) is signal.default_int_handler


@pytest.mark.usefixtures("sigint_restored")
@pytest.mark.skipif(not _default_sigint(), reason="needs the default SIGINT handler")
def test_sigint_inside_a_live_step_is_deferred() -> None:
    def interrupt(n: int) -> None:
        if n == 40:
            signal.raise_signal(signal.SIGINT)

    src = FakeSource()
    sink = ListSink(on_write=interrupt)
    report = _no_interrupt(
        lambda: run_stream(src, _live_config("calibrate"), OutputConfig(), sinks=[sink])
    )
    assert report.interrupted is True
    assert len(sink.images) == report.stats.frames == len(src.yielded)
    assert len(src.yielded) < src.scene.total
    assert (sink.closed, sink.aborted) == (1, 0)
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler  # Restored.


@pytest.mark.usefixtures("sigint_restored")
@pytest.mark.skipif(not _default_sigint(), reason="needs the default SIGINT handler")
def test_second_sigint_inside_a_live_step_aborts() -> None:
    def interrupt(n: int) -> None:
        if n == 40:
            signal.raise_signal(signal.SIGINT)
            signal.raise_signal(signal.SIGINT)

    src = FakeSource()
    sink = ListSink(on_write=interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_stream(src, _live_config("calibrate"), OutputConfig(), sinks=[sink])
    assert (sink.closed, sink.aborted) == (0, 1)
    assert src.closed == 1
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


@pytest.mark.usefixtures("sigint_restored")
@pytest.mark.skipif(not _default_sigint(), reason="needs the default SIGINT handler")
def test_sigint_inside_a_non_live_step_aborts() -> None:
    def interrupt(n: int) -> None:
        if n == 40:
            signal.raise_signal(signal.SIGINT)

    src = FakeSource(live=False)
    sink = ListSink(on_write=interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_stream(src, _live_config("calibrate"), OutputConfig(), sinks=[sink])
    assert (sink.closed, sink.aborted) == (0, 1)
    assert src.closed == 1


def test_failing_non_critical_sink_is_dropped() -> None:
    src = FakeSource(Scene(total=120))
    good = ListSink()
    bad = ListSink(critical=False, fail_at=10)
    report = run_stream(src, _live_config("calibrate"), OutputConfig(), sinks=[good, bad])
    assert report.stats.sink_failures == 1
    assert len(good.images) == 120
    assert (good.closed, good.aborted) == (1, 0)
    assert (bad.closed, bad.aborted) == (0, 1)
    assert src.closed == 1


def test_critical_sink_failure_propagates() -> None:
    src = FakeSource(Scene(total=120))
    other = ListSink()
    bad = ListSink(fail_at=10)
    with pytest.raises(RuntimeError, match="sink write failed"):
        run_stream(src, _live_config("calibrate"), OutputConfig(), sinks=[other, bad])
    assert (other.closed, other.aborted) == (0, 1)
    assert src.closed == 1


# ---------------------------------------------------------------------------
# Sinks built from OutputConfig
# ---------------------------------------------------------------------------


def test_d2_sinks_for_live_source_are_not_threaded(tmp_path: Path) -> None:
    src = FakeSource()
    config = _live_config("running")
    with (
        mock.patch.object(pipeline, "FfmpegPipeSink", wraps=FfmpegPipeSink) as pipe,
        mock.patch.object(pipeline, "VirtualCamSink", wraps=VirtualCamSink) as cam,
    ):
        sinks = pipeline._build_sinks(
            src, config,
            OutputConfig(file=tmp_path / "o.mp4", virtualcam=True, virtualcam_device="dev",
                         preview=True),
        )
    pipe.assert_called_once_with(tmp_path / "o.mp4", ENCODE, audio_source=None,
                                 wallclock_timestamps=True)
    cam.assert_called_once_with(device="dev")
    assert [type(s) for s in sinks] == [FfmpegPipeSink, VirtualCamSink, PreviewSink]
    assert not any(isinstance(s, ThreadedSink) for s in sinks)


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Static textured background; a green block crosses frames 60 to 79 of 100."""
    path = tmp_path_factory.mktemp("clip") / "synthetic.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (W, H))
    assert writer.isOpened()
    try:
        for i in range(100):
            frame = BACKGROUND.copy()
            if 60 <= i < 80:
                x = 20 + (i - 60) * 12
                frame[100:180, x : x + 50] = SUBJECT
            frame[4:12, 4:40] = 255 if (i // 25) % 2 else 0
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_sinks_for_file_source(clip: Path, tmp_path: Path) -> None:
    src = FileSource(clip)
    try:
        with mock.patch.object(pipeline, "FfmpegPipeSink", wraps=FfmpegPipeSink) as pipe:
            sinks = pipeline._build_sinks(src, _live_config("locked"),
                                          OutputConfig(file=tmp_path / "o.mp4", virtualcam=True))
    finally:
        src.close()
    pipe.assert_called_once_with(tmp_path / "o.mp4", ENCODE, audio_source=clip,
                                 wallclock_timestamps=False)
    assert [type(s) for s in sinks] == [FfmpegPipeSink, VirtualCamSink]


# ---------------------------------------------------------------------------
# File sources: run_stream matches run_file
# ---------------------------------------------------------------------------


class _RecordingFileSink(ListSink):
    """Replaces FileSink inside run_file: records frames, encodes nothing."""

    instances: list[_RecordingFileSink] = []

    def __init__(self, output_path: Any, audio_source: Any, encode: Any) -> None:
        super().__init__()
        _RecordingFileSink.instances.append(self)


@pytest.mark.parametrize(
    "overrides",
    [
        {"roi": RoiConfig(mode="disabled"), "background": BackgroundConfig(frame_index=3)},
        {"roi": RoiConfig(mode="auto"), "background": BackgroundConfig()},
        {
            "detector": "both",
            "person": PersonHogConfig(min_box_area=500),
            "roi": RoiConfig(mode="manual", box=(0, 0, 60, 20)),
            "background": BackgroundConfig(),
        },
        {"background_diff": None, "engine": EngineConfig(lookahead=0)},
    ],
    ids=["frame-index", "auto-roi-prescan", "both-manual-roi", "no-bg-diff-lookahead-0"],
)
def test_run_stream_matches_run_file(clip: Path, tmp_path: Path, overrides: dict[str, Any]) -> None:
    config = PetrifyConfig(encode=ENCODE, **overrides)
    _RecordingFileSink.instances.clear()
    file_traces: list[FrameTrace] = []
    with mock.patch.object(pipeline, "FileSink", _RecordingFileSink):
        file_report = run_file(clip, tmp_path / "a.mp4", config, file_traces.append)
    (expected,) = _RecordingFileSink.instances

    sink = ListSink()
    stream_traces: list[FrameTrace] = []
    stream_report = run_stream(FileSource(clip), config, OutputConfig(file=tmp_path / "b.mp4"),
                               sinks=[sink], observer=stream_traces.append)

    assert len(sink.images) == len(expected.images) == 100
    for k, (a, b) in enumerate(zip(sink.images, expected.images)):
        assert np.array_equal(a, b), k
    assert stream_traces == file_traces
    assert stream_report.stats == file_report.stats
    assert stream_report.background_index == file_report.background_index
    assert stream_report.roi == file_report.roi
    assert (stream_report.dropped_frames, stream_report.interrupted) == (0, False)
    assert not any(_shows_subject(o) for o in sink.images)


def test_file_source_with_online_background(clip: Path) -> None:
    sink = ListSink()
    report = run_stream(FileSource(clip), _live_config("calibrate"), OutputConfig(), sinks=[sink])
    assert report.background_index is None
    assert len(sink.images) == 100
    assert report.stats.failed_closed == sum(_is_black(o) for o in sink.images) > 0


# ---------------------------------------------------------------------------
# End to end with the real FFmpeg (no devices)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="needs ffmpeg and ffprobe",
)
def test_cli_run_with_real_ffmpeg(clip: Path, tmp_path: Path) -> None:
    out = tmp_path / "run_out.mp4"
    code = petrify.cli.main(
        ["run", "--source", str(clip), "--output", str(out), "--preset", "ultrafast"]
    )
    assert code == 0
    assert out.exists()
    assert not (tmp_path / "run_out.partial.mp4").exists()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames,width,height,avg_frame_rate",
         "-of", "default=nw=1", str(out)],
        check=True, capture_output=True, text=True,
    ).stdout
    fields = dict(line.split("=", 1) for line in probe.split())
    assert fields == {"width": str(W), "height": str(H), "avg_frame_rate": "25/1",
                      "nb_read_frames": "100"}


# ---------------------------------------------------------------------------
# P2-9c: live privacy defaults, data loss and robustness
# ---------------------------------------------------------------------------

# A "person" in view from the first frame, standing still, who leaves at frame 150.
PERSON_FROM_START = Scene(total=300, enter=0, stand=0, leave=150, gone=160)


def _fake_person_free(detector: Any) -> Callable[[np.ndarray], bool]:
    """Stand-in for the HOG person check: a sample is clean when no subject pixel shows."""
    return lambda image: not _shows_subject(image)


def _without_sample_check(config: PetrifyConfig) -> PetrifyConfig:
    return dataclasses.replace(
        config, background=dataclasses.replace(config.background, sample_check=False)
    )


@pytest.mark.parametrize("strategy", ["running", "calibrate"])
def test_l1_person_present_while_background_builds_is_never_shown(strategy: str) -> None:
    src = FakeSource(PERSON_FROM_START)
    sink = ListSink()
    traces: list[FrameTrace] = []
    with mock.patch.object(pipeline, "person_free", side_effect=_fake_person_free) as check:
        report = run_stream(src, _live_config(strategy), OutputConfig(), sinks=[sink],
                            observer=traces.append)
    check.assert_called_once()
    assert len(sink.images) == report.stats.frames == PERSON_FROM_START.total
    assert not any(_shows_subject(o) for o in sink.images)
    # The run recovers once the person has left: the last frames pass through.
    kinds = [_kind(t) for t in traces]
    assert all(kind == "pass" for kind in kinds[-30:])


@pytest.mark.parametrize("strategy", ["running", "calibrate"])
def test_l1_without_sample_check_the_known_leak_reproduces(strategy: str) -> None:
    # Documents what --no-background-check turns off: the person learned into the
    # background is shown on frozen frames after they leave.
    src = FakeSource(PERSON_FROM_START)
    sink = ListSink()
    with mock.patch.object(pipeline, "person_free", side_effect=_fake_person_free) as check:
        run_stream(src, _without_sample_check(_live_config(strategy)), OutputConfig(),
                   sinks=[sink])
    check.assert_not_called()
    leaked = [k for k, o in enumerate(sink.images) if _shows_subject(o)]
    assert any(k >= PERSON_FROM_START.gone for k in leaked)


@pytest.mark.parametrize("strategy", ["running", "calibrate"])
def test_l1_online_models_get_the_prescan_person_check(strategy: str) -> None:
    config = _live_config(strategy, person=PersonHogConfig(min_confidence=0.9))
    detector = object()
    sample_filter = object()
    with (
        mock.patch.object(pipeline.PersonDetector, "from_config", return_value=detector) as build,
        mock.patch.object(pipeline, "person_free", return_value=sample_filter) as check,
        mock.patch.object(pipeline, "RunningBackground") as running,
        mock.patch.object(pipeline, "CalibratedBackground") as calibrated,
    ):
        pipeline._background_for(FakeSource(), config)
    build.assert_called_once_with(config.background_scan_person_config())
    check.assert_called_once_with(detector)
    model = running if strategy == "running" else calibrated
    assert model.call_args.kwargs["sample_filter"] is sample_filter


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_l1_startup_warnings(caplog: pytest.LogCaptureFixture) -> None:
    with (
        caplog.at_level(logging.WARNING, logger="petrify.pipeline"),
        mock.patch.object(pipeline, "person_free", side_effect=_fake_person_free),
    ):
        run_stream(FakeSource(Scene(total=30)), _live_config("running"), OutputConfig(),
                   sinks=[ListSink()])
    assert _warnings(caplog).count(ONLINE_BACKGROUND_WARNING) == 1
    assert NO_BACKGROUND_CHECK_WARNING not in _warnings(caplog)
    assert "seated person" in ONLINE_BACKGROUND_WARNING
    assert "Start with the scene empty" in ONLINE_BACKGROUND_WARNING

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="petrify.pipeline"):
        run_stream(FakeSource(Scene(total=30)), _without_sample_check(_live_config("calibrate")),
                   OutputConfig(), sinks=[ListSink()])
    assert ONLINE_BACKGROUND_WARNING in _warnings(caplog)
    assert NO_BACKGROUND_CHECK_WARNING in _warnings(caplog)


def test_l1_no_online_warning_for_locked_background(
    clip: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="petrify.pipeline"):
        run_stream(FileSource(clip), PetrifyConfig(encode=ENCODE), OutputConfig(),
                   sinks=[ListSink()])
    assert ONLINE_BACKGROUND_WARNING not in caplog.text


def test_l2_warmup_detector_gets_the_area_cap() -> None:
    _FakeWarmupDetector.instances.clear()
    config = _live_config("calibrate", roi=RoiConfig(mode="auto", max_area_cap=0.05))
    with mock.patch.object(pipeline, "WarmupRoiDetector", _FakeWarmupDetector):
        pipeline._roi_stage(FakeSource(), config)
    (detector,) = _FakeWarmupDetector.instances
    assert detector.args[2]["max_roi_area_frac"] == 0.05


@pytest.mark.parametrize("live", [True, False])
def test_l4_no_bg_diff_on_live_source_warns(live: bool, caplog: pytest.LogCaptureFixture) -> None:
    config = _live_config("running", background_diff=None)
    with caplog.at_level(logging.WARNING, logger="petrify.pipeline"):
        run_stream(FakeSource(Scene(total=30), live=live), config, OutputConfig(),
                   sinks=[ListSink()])
    assert (NO_BG_DIFF_LIVE_WARNING in _warnings(caplog)) is live


class FailingSource(FakeSource):
    """A fake source whose iteration raises ``OSError`` at frame ``fail_at``."""

    def __init__(self, scene: Scene | None = None, *, fail_at: int, live: bool = True) -> None:
        super().__init__(scene, live=live)
        self.fail_at = fail_at

    def __iter__(self) -> Iterator[Frame]:
        for i in range(self.scene.total):
            if i == self.fail_at:
                raise OSError("camera unplugged")
            img = self.scene.image(i)
            self.yielded.append(img)
            yield Frame(image=img, index=i, timestamp=i / FPS, source=self.info)


class CloseFailingSink(ListSink):
    def close(self) -> None:
        super().close()
        raise OSError("close failed too")


@pytest.mark.parametrize("fail_at", [135, 200])  # While the subject is in view, and after.
def test_d1_live_source_error_keeps_the_processed_recording(
    fail_at: int, caplog: pytest.LogCaptureFixture
) -> None:
    src = FailingSource(fail_at=fail_at)
    sink = ListSink()
    traces: list[FrameTrace] = []
    with (
        caplog.at_level(logging.ERROR, logger="petrify.pipeline"),
        pytest.raises(OSError, match="camera unplugged"),
    ):
        run_stream(src, _live_config("calibrate"), OutputConfig(), sinks=[sink],
                   observer=traces.append)
    assert (sink.opened, sink.closed, sink.aborted) == (1, 1, 0)
    assert src.closed == 1
    # Every frame read before the error was written (lookahead tail included) ...
    assert len(sink.images) == len(traces) == len(src.yielded) == fail_at
    assert [m.index for m in sink.metas] == list(range(fail_at))
    # ... and each one is exactly what the mode approved: black, the background, or the
    # unchanged frame when nothing was detected. Nothing is written unprocessed.
    for k, (trace, out, inp) in enumerate(zip(traces, sink.images, src.yielded)):
        expected = {"black": np.zeros_like(inp), "frozen": BACKGROUND, "pass": inp}[_kind(trace)]
        assert np.array_equal(out, expected), k
    assert not any(_shows_subject(o) for o in sink.images)
    assert "the recording up to that point was kept" in caplog.text


def test_d1_close_failure_after_live_error_aborts_and_raises_the_original() -> None:
    src = FailingSource(fail_at=100)
    sink = CloseFailingSink()
    with pytest.raises(OSError, match="camera unplugged"):
        run_stream(src, _live_config("calibrate"), OutputConfig(), sinks=[sink])
    assert (sink.closed, sink.aborted) == (1, 1)
    assert src.closed == 1


def test_d1_non_live_source_error_still_aborts() -> None:
    src = FailingSource(fail_at=100, live=False)
    sink = ListSink()
    with pytest.raises(OSError, match="camera unplugged"):
        run_stream(src, _live_config("calibrate"), OutputConfig(), sinks=[sink])
    assert (sink.closed, sink.aborted) == (0, 1)


def test_d1_cli_exits_1_when_a_live_run_fails() -> None:
    sink = ListSink()
    with (
        mock.patch.object(petrify.sources, "open_source",
                          return_value=FailingSource(Scene(total=80), fail_at=60)),
        mock.patch.object(pipeline, "person_free", side_effect=_fake_person_free),
        mock.patch.object(pipeline, "_build_sinks", return_value=[sink]),
    ):
        assert petrify.cli.main(["run", "--source", "camera:0", "--preview"]) == 1
    assert (len(sink.images), sink.closed, sink.aborted) == (60, 1, 0)


def _write_images(folder: Path, count: int = 3) -> list[Path]:
    paths = []
    for i in range(count):
        path = folder / f"frame{i:02d}.png"
        assert cv2.imwrite(str(path), BACKGROUND)
        paths.append(path)
    return paths


def test_d3_run_stream_rejects_output_equal_to_input(clip: Path, tmp_path: Path) -> None:
    before = clip.read_bytes()
    src = FileSource(clip)
    with pytest.raises(ValueError, match="output file is the input"):
        run_stream(src, PetrifyConfig(encode=ENCODE), OutputConfig(file=clip))
    assert clip.read_bytes() == before
    # Another spelling of the same path is rejected too.
    spelled = clip.parent / "." / clip.name
    with pytest.raises(ValueError, match="output file is the input"):
        run_stream(FileSource(clip), PetrifyConfig(encode=ENCODE), OutputConfig(file=spelled))

    images = _write_images(tmp_path)
    with pytest.raises(ValueError, match="output file is the input"):
        run_stream(ImageFolderSource(tmp_path), PetrifyConfig(encode=ENCODE),
                   OutputConfig(file=images[1]))
    # Inside the folder but not one of its images: allowed.
    assert not pipeline.output_overwrites_input(ImageFolderSource(tmp_path),
                                                tmp_path / "out.mp4")


def test_d3_cli_rejects_output_equal_to_input(
    clip: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = clip.read_bytes()
    with pytest.raises(SystemExit) as excinfo:
        petrify.cli.main(["run", "--source", str(clip), "--output", str(clip)])
    assert excinfo.value.code == 2
    assert "--output is the input file" in capsys.readouterr().err
    assert clip.read_bytes() == before

    images = _write_images(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        petrify.cli.main(["run", "--source", str(tmp_path), "--output", str(images[0])])
    assert excinfo.value.code == 2
    assert "--output is one of the input files" in capsys.readouterr().err
    assert np.array_equal(cv2.imread(str(images[0])), BACKGROUND)


@NEEDS_FFMPEG
def test_d4_source_with_no_frames_is_a_clear_error(tmp_path: Path) -> None:
    out = tmp_path / "empty.mp4"
    with pytest.raises(NoFramesError, match="source delivered no frames"):
        run_stream(FakeSource(Scene(total=0)), _live_config("running"), OutputConfig(file=out))
    assert list(tmp_path.iterdir()) == []


@NEEDS_FFMPEG
def test_d4_cli_exits_1_for_a_source_with_no_frames(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    out = tmp_path / "empty.mp4"
    with mock.patch.object(petrify.sources, "open_source",
                           return_value=FakeSource(Scene(total=0))):
        code = petrify.cli.main(["run", "--source", "camera:0", "--output", str(out)])
    assert code == 1
    assert "Fatal error: The source delivered no frames; no output was written" in caplog.text
    assert "Traceback" not in caplog.text
    assert list(tmp_path.iterdir()) == []


_NO_FFMPEG = FileNotFoundError(
    "ffmpeg binary not found on PATH. Install FFmpeg: https://ffmpeg.org/download.html"
)


def test_d5_missing_ffmpeg_fails_before_processing(tmp_path: Path) -> None:
    src = FakeSource()
    with (
        mock.patch.object(petrify.sinks.file, "find_ffmpeg", side_effect=_NO_FFMPEG),
        mock.patch.object(pipeline, "_background_for") as background,
        pytest.raises(FfmpegNotFoundError, match="Install FFmpeg"),
    ):
        run_stream(src, _live_config("running"), OutputConfig(file=tmp_path / "o.mp4"))
    background.assert_not_called()
    assert src.yielded == []
    assert src.closed == 1


def test_d5_cli_exits_1_for_missing_ffmpeg_and_2_for_missing_input(
    clip: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with mock.patch.object(petrify.sinks.file, "find_ffmpeg", side_effect=_NO_FFMPEG):
        code = petrify.cli.main(["run", "--source", str(clip), "--output",
                                 str(tmp_path / "o.mp4")])
        assert code == 1
        assert "Install FFmpeg" in caplog.text
        missing = tmp_path / "missing.mp4"
        assert petrify.cli.main(["run", "--source", str(missing), "--output",
                                 str(tmp_path / "o.mp4")]) == 2
    assert list(tmp_path.iterdir()) == []
