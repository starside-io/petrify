"""Tests for petrify.pipeline.run_stream with injected sources and sinks (no devices)."""

from __future__ import annotations

import dataclasses
import shutil
import signal
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import cv2
import numpy as np
import pytest

import petrify.cli
import petrify.pipeline as pipeline
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
from petrify.pipeline import OutputConfig, RunReport, run_file, run_stream
from petrify.sinks import FfmpegPipeSink, PreviewSink, ThreadedSink, VirtualCamSink
from petrify.sources import FileSource

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
         "edge_gap_tolerance": 20, "n_edge_samples": 5},
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


def test_keyboard_interrupt_from_live_source_finishes_cleanly() -> None:
    src = FakeSource(interrupt_at=100, dropped=7)
    sink = ListSink()
    report = run_stream(src, _live_config("calibrate"), OutputConfig(), sinks=[sink])
    assert isinstance(report, RunReport)
    assert report.interrupted is True
    assert report.dropped_frames == 7
    assert len(sink.images) == report.stats.frames == 100
    assert (sink.closed, sink.aborted) == (1, 0)
    assert src.closed == 1


def test_keyboard_interrupt_on_non_live_source_aborts() -> None:
    src = FakeSource(live=False, interrupt_at=100)
    sink = ListSink()
    with pytest.raises(KeyboardInterrupt):
        run_stream(src, _live_config("calibrate"), OutputConfig(), sinks=[sink])
    assert (sink.closed, sink.aborted) == (0, 1)
    assert src.closed == 1


def _default_sigint() -> bool:
    return signal.getsignal(signal.SIGINT) is signal.default_int_handler


@pytest.mark.skipif(not _default_sigint(), reason="needs the default SIGINT handler")
def test_sigint_inside_a_live_step_is_deferred() -> None:
    def interrupt(n: int) -> None:
        if n == 40:
            signal.raise_signal(signal.SIGINT)

    src = FakeSource()
    sink = ListSink(on_write=interrupt)
    report = run_stream(src, _live_config("calibrate"), OutputConfig(), sinks=[sink])
    assert report.interrupted is True
    assert len(sink.images) == report.stats.frames == len(src.yielded)
    assert len(src.yielded) < src.scene.total
    assert (sink.closed, sink.aborted) == (1, 0)
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler  # Restored.


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


def test_sinks_for_live_source(tmp_path: Path) -> None:
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
    assert [type(s) for s in sinks] == [ThreadedSink, ThreadedSink, PreviewSink]
    assert isinstance(sinks[0].sink, FfmpegPipeSink)  # type: ignore[attr-defined]
    assert isinstance(sinks[1].sink, VirtualCamSink)  # type: ignore[attr-defined]


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
