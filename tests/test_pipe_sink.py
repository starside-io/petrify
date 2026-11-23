"""Tests for petrify.sinks.file.FfmpegPipeSink: fake Popen, plus real FFmpeg when available."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from petrify.core import EncodeConfig, FrameMeta, Sink, SourceInfo
from petrify.encoder import find_ffmpeg
from petrify.sinks import FfmpegPipeSink

W, H = 8, 6
INFO = SourceInfo(fps=25.0, width=W, height=H, frame_count=None, seekable=False, live=True)


def _meta(i: int, info: SourceInfo = INFO) -> FrameMeta:
    return FrameMeta(index=i, timestamp=i / info.fps, source=info)


def _image(i: int, h: int = H, w: int = W) -> np.ndarray:
    rng = np.random.default_rng(i)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Fake Popen
# ---------------------------------------------------------------------------


class FakeStdin:
    def __init__(self, break_after: int | None) -> None:
        self.chunks: list[bytes] = []
        self.closed = False
        self.break_after = break_after

    def write(self, data: Any) -> int:
        if self.closed:
            raise ValueError("write to closed file")
        if self.break_after is not None and len(self.chunks) >= self.break_after:
            raise BrokenPipeError(32, "Broken pipe")
        chunk = bytes(data)
        self.chunks.append(chunk)
        return len(chunk)

    def close(self) -> None:
        self.closed = True


class FakeProc:
    def __init__(
        self,
        cmd: list[str],
        stderr: Any,
        *,
        exit_code: int,
        stderr_text: str,
        break_after: int | None,
        hang: bool,
        exit_on_break: bool,
    ) -> None:
        self.cmd = cmd
        self.stdin = FakeStdin(break_after)
        self.returncode: int | None = None
        self.exit_code = exit_code
        self.hang = hang
        self.exit_on_break = exit_on_break
        self.terminated = 0
        self.killed = 0
        self.waits: list[float | None] = []
        stderr.write(stderr_text.encode())

    @property
    def output(self) -> Path:
        return Path(self.cmd[-1])

    def poll(self) -> int | None:
        if self.returncode is None and self.exit_on_break and self.stdin.break_after is not None:
            if len(self.stdin.chunks) >= self.stdin.break_after:
                self.returncode = self.exit_code
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.waits.append(timeout)
        if self.returncode is not None:
            return self.returncode
        if self.hang and not self.killed and not self.terminated:
            raise subprocess.TimeoutExpired(self.cmd, timeout or 0)
        if self.killed or self.terminated:
            self.returncode = -15 if self.terminated and not self.killed else -9
            return self.returncode
        self.returncode = self.exit_code
        if self.exit_code == 0:
            self.output.write_bytes(b"".join(self.stdin.chunks))
        else:
            self.output.write_bytes(b"partial garbage")
        return self.returncode

    def terminate(self) -> None:
        self.terminated += 1

    def kill(self) -> None:
        self.killed += 1


class FakePopen:
    def __init__(
        self,
        *,
        exit_code: int = 0,
        stderr_text: str = "",
        break_after: int | None = None,
        hang: bool = False,
        exit_on_break: bool = False,
        partial_on_start: bool = True,
        fail: BaseException | None = None,
    ) -> None:
        self.kw = dict(
            exit_code=exit_code,
            stderr_text=stderr_text,
            break_after=break_after,
            hang=hang,
            exit_on_break=exit_on_break,
        )
        self.partial_on_start = partial_on_start
        self.fail = fail
        self.procs: list[FakeProc] = []
        self.kwargs: list[dict[str, Any]] = []

    def __call__(self, cmd: list[str], **kwargs: Any) -> FakeProc:
        if self.fail is not None:
            raise self.fail
        self.kwargs.append(kwargs)
        proc = FakeProc(cmd, kwargs["stderr"], **self.kw)  # type: ignore[arg-type]
        if self.partial_on_start:
            proc.output.write_bytes(b"")
        self.procs.append(proc)
        return proc

    @property
    def proc(self) -> FakeProc:
        assert len(self.procs) == 1
        return self.procs[0]


@pytest.fixture(autouse=True)
def _fake_ffmpeg_path(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    # Fake-Popen tests get a fixed binary name; the real-FFmpeg tests use the real lookup.
    if not request.node.name.startswith("test_real_ffmpeg"):
        monkeypatch.setattr("petrify.sinks.file.find_ffmpeg", lambda: "ffmpeg")


def _sink(tmp_path: Path, popen: FakePopen, **kwargs: Any) -> FfmpegPipeSink:
    return FfmpegPipeSink(tmp_path / "out.mp4", kwargs.pop("encode", EncodeConfig()), popen=popen, **kwargs)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _cmd(tmp_path: Path, **kwargs: Any) -> list[str]:
    popen = FakePopen()
    info = kwargs.pop("info", INFO)
    sink = _sink(tmp_path, popen, **kwargs)
    sink.open(info)
    cmd = popen.proc.cmd
    sink.abort()
    return cmd


def test_command_default(tmp_path: Path) -> None:
    partial = str(tmp_path / "out.partial.mp4")
    assert _cmd(tmp_path) == [
        "ffmpeg", "-y", "-hide_banner", "-nostats",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "8x6", "-r", "25", "-i", "pipe:0",
        "-map", "0:v:0",
        "-c:v", "libx264", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p",
        partial,
    ]  # fmt: skip


def test_command_hwaccel_and_codec(tmp_path: Path) -> None:
    encode = EncodeConfig(codec="h264_videotoolbox", crf=18, preset="fast", hwaccel="videotoolbox")
    assert _cmd(tmp_path, encode=encode) == [
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-hwaccel", "videotoolbox",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "8x6", "-r", "25", "-i", "pipe:0",
        "-map", "0:v:0",
        "-c:v", "h264_videotoolbox", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p",
        str(tmp_path / "out.partial.mp4"),
    ]  # fmt: skip


def test_command_audio_source(tmp_path: Path) -> None:
    audio = tmp_path / "in.mov"
    assert _cmd(tmp_path, audio_source=audio) == [
        "ffmpeg", "-y", "-hide_banner", "-nostats",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "8x6", "-r", "25", "-i", "pipe:0",
        "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "libx264", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(tmp_path / "out.partial.mp4"),
    ]  # fmt: skip


def test_command_wallclock(tmp_path: Path) -> None:
    assert _cmd(tmp_path, wallclock_timestamps=True) == [
        "ffmpeg", "-y", "-hide_banner", "-nostats",
        "-use_wallclock_as_timestamps", "1",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "8x6", "-framerate", "25", "-i", "pipe:0",
        "-map", "0:v:0",
        "-c:v", "libx264", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-vsync", "passthrough", "-enc_time_base", "1:1000000",
        str(tmp_path / "out.partial.mp4"),
    ]  # fmt: skip


def test_command_ntsc_rate_is_exact(tmp_path: Path) -> None:
    info = SourceInfo(fps=30000 / 1001, width=W, height=H, frame_count=None, seekable=True, live=False)
    cmd = _cmd(tmp_path, info=info)
    assert cmd[cmd.index("-r") + 1] == "30000/1001"


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_fps_raises_before_starting(tmp_path: Path, fps: float) -> None:
    popen = FakePopen()
    sink = _sink(tmp_path, popen)
    info = SourceInfo(fps=fps, width=W, height=H, frame_count=None, seekable=False, live=True)
    with pytest.raises(ValueError, match="Frame rate"):
        sink.open(info)
    assert popen.procs == []
    sink.abort()


def test_stderr_goes_to_a_file_not_a_pipe(tmp_path: Path) -> None:
    popen = FakePopen()
    sink = _sink(tmp_path, popen)
    sink.open(INFO)
    kwargs = popen.kwargs[0]
    assert kwargs["stdin"] is subprocess.PIPE
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["stderr"] not in (subprocess.PIPE, None)
    assert hasattr(kwargs["stderr"], "fileno")
    sink.abort()


def test_is_a_critical_sink(tmp_path: Path) -> None:
    sink = _sink(tmp_path, FakePopen())
    assert isinstance(sink, Sink)
    assert sink.critical is True
    assert FfmpegPipeSink.critical is True


# ---------------------------------------------------------------------------
# Writing and closing
# ---------------------------------------------------------------------------


def test_bytes_written_in_order_and_moved_into_place(tmp_path: Path) -> None:
    popen = FakePopen()
    sink = _sink(tmp_path, popen)
    sink.open(INFO)
    images = [_image(i) for i in range(5)]
    for i, image in enumerate(images):
        sink.write(image, _meta(i))
    assert popen.proc.stdin.chunks == [image.tobytes() for image in images]
    sink.close()
    assert popen.proc.stdin.closed
    assert (tmp_path / "out.mp4").read_bytes() == b"".join(im.tobytes() for im in images)
    assert not sink.partial_path.exists()
    sink.close()  # Idempotent.
    sink.abort()  # No-op after close.
    assert popen.proc.terminated == 0


def test_non_contiguous_image_is_written_as_its_pixels(tmp_path: Path) -> None:
    popen = FakePopen()
    sink = _sink(tmp_path, popen)
    sink.open(INFO)
    big = _image(1, h=H, w=W * 2)
    view = big[:, ::2]
    assert not view.flags.c_contiguous
    sink.write(view, _meta(0))
    assert popen.proc.stdin.chunks == [np.ascontiguousarray(view).tobytes()]
    sink.abort()


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((H + 1, W, 3), np.uint8),
        np.zeros((H, W + 1, 3), np.uint8),
        np.zeros((H, W), np.uint8),
        np.zeros((H, W, 4), np.uint8),
        np.zeros((H, W, 3), np.float32),
        np.zeros((H, W, 3), np.uint16),
    ],
)
def test_write_validates_shape_and_dtype(tmp_path: Path, image: np.ndarray) -> None:
    popen = FakePopen()
    sink = _sink(tmp_path, popen)
    sink.open(INFO)
    with pytest.raises(ValueError, match="Expected a uint8 image"):
        sink.write(image, _meta(0))
    assert popen.proc.stdin.chunks == []
    sink.abort()


def test_write_before_open_raises(tmp_path: Path) -> None:
    sink = _sink(tmp_path, FakePopen())
    with pytest.raises(RuntimeError, match="not open"):
        sink.write(_image(0), _meta(0))


def test_double_open_raises(tmp_path: Path) -> None:
    sink = _sink(tmp_path, FakePopen())
    sink.open(INFO)
    with pytest.raises(RuntimeError, match="already open"):
        sink.open(INFO)
    sink.abort()


def test_broken_pipe_raises_oserror_with_code_and_stderr_tail(tmp_path: Path) -> None:
    lines = [f"noise line {i}" for i in range(50)] + ["Unknown encoder 'nope'"]
    popen = FakePopen(exit_code=1, stderr_text="\n".join(lines), break_after=2)
    sink = _sink(tmp_path, popen)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    sink.write(_image(1), _meta(1))
    with pytest.raises(OSError) as info:
        sink.write(_image(2), _meta(2))
    message = str(info.value)
    assert not isinstance(info.value, BrokenPipeError)
    assert isinstance(info.value.__cause__, BrokenPipeError)
    assert "exit code 1" in message
    assert "Unknown encoder 'nope'" in message
    assert "noise line 0\n" not in message  # Only the tail is quoted.
    sink.abort()
    assert not sink.partial_path.exists()
    assert not (tmp_path / "out.mp4").exists()


def test_process_already_exited_raises_on_write(tmp_path: Path) -> None:
    popen = FakePopen(exit_code=8, stderr_text="Invalid argument", break_after=1, exit_on_break=True)
    sink = _sink(tmp_path, popen)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    with pytest.raises(OSError, match=r"exit code 8\)") as info:
        sink.write(_image(1), _meta(1))
    assert "Invalid argument" in str(info.value)
    assert len(popen.proc.stdin.chunks) == 1
    sink.abort()


def test_nonzero_exit_on_close_raises_and_keeps_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "out.mp4"
    output.write_bytes(b"precious earlier recording")
    popen = FakePopen(exit_code=1, stderr_text="Conversion failed!")
    sink = _sink(tmp_path, popen)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    with pytest.raises(OSError, match="exit code 1") as info:
        sink.close()
    assert "Conversion failed!" in str(info.value)
    assert not sink.partial_path.exists()
    assert output.read_bytes() == b"precious earlier recording"
    sink.abort()  # Safe after a failed close.
    sink.close()


def test_success_replaces_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "out.mp4"
    output.write_bytes(b"old")
    popen = FakePopen()
    sink = _sink(tmp_path, popen)
    sink.open(INFO)
    sink.write(_image(3), _meta(0))
    sink.close()
    assert output.read_bytes() == _image(3).tobytes()
    assert not sink.partial_path.exists()


def test_close_timeout_kills_and_raises(tmp_path: Path) -> None:
    popen = FakePopen(hang=True, stderr_text="stuck")
    sink = _sink(tmp_path, popen)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    with pytest.raises(OSError, match="did not exit") as info:
        sink.close()
    assert "stuck" in str(info.value)
    assert popen.proc.killed == 1
    assert popen.proc.waits[0] is not None and popen.proc.waits[0] > 0
    assert not sink.partial_path.exists()
    assert not (tmp_path / "out.mp4").exists()
    sink.abort()


# ---------------------------------------------------------------------------
# Abort
# ---------------------------------------------------------------------------


def test_abort_terminates_and_removes_partial(tmp_path: Path) -> None:
    output = tmp_path / "out.mp4"
    output.write_bytes(b"keep me")
    popen = FakePopen()
    sink = _sink(tmp_path, popen)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    assert sink.partial_path.exists()
    sink.abort()
    proc = popen.proc
    assert proc.stdin.closed
    assert proc.terminated == 1
    assert proc.killed == 0
    assert not sink.partial_path.exists()
    assert output.read_bytes() == b"keep me"
    sink.abort()
    sink.close()
    assert proc.terminated == 1


def test_abort_kills_when_terminate_is_ignored(tmp_path: Path) -> None:
    popen = FakePopen(hang=True)
    sink = _sink(tmp_path, popen)
    sink.open(INFO)

    proc = popen.proc
    original_wait = proc.wait

    def stubborn_wait(timeout: float | None = None) -> int:
        if proc.terminated and not proc.killed:
            proc.waits.append(timeout)
            raise subprocess.TimeoutExpired(proc.cmd, timeout or 0)
        return original_wait(timeout)

    proc.wait = stubborn_wait  # type: ignore[method-assign]
    sink.abort()
    assert proc.terminated == 1
    assert proc.killed == 1
    assert not sink.partial_path.exists()


def test_abort_before_open_is_safe(tmp_path: Path) -> None:
    sink = _sink(tmp_path, FakePopen())
    sink.abort()
    sink.abort()
    sink.close()


def test_abort_after_failed_open_is_safe(tmp_path: Path) -> None:
    popen = FakePopen(fail=FileNotFoundError("no ffmpeg"))
    sink = _sink(tmp_path, popen)
    with pytest.raises(FileNotFoundError):
        sink.open(INFO)
    sink.abort()
    sink.abort()
    assert not sink.partial_path.exists()


def test_missing_ffmpeg_binary_fails_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing() -> str:
        raise FileNotFoundError("ffmpeg binary not found on PATH.")

    monkeypatch.setattr("petrify.sinks.file.find_ffmpeg", missing)
    popen = FakePopen()
    sink = _sink(tmp_path, popen)
    with pytest.raises(FileNotFoundError):
        sink.open(INFO)
    assert popen.procs == []
    sink.abort()


# ---------------------------------------------------------------------------
# Real FFmpeg
# ---------------------------------------------------------------------------


def _real_ffmpeg() -> str | None:
    try:
        return find_ffmpeg()
    except FileNotFoundError:
        return None


real_ffmpeg = pytest.mark.skipif(_real_ffmpeg() is None, reason="ffmpeg binary not found")

RW, RH = 320, 240
REAL_INFO = SourceInfo(fps=30.0, width=RW, height=RH, frame_count=30, seekable=True, live=False)


def _color(i: int) -> tuple[int, int, int]:
    return (40 + 5 * i, 200 - 4 * i, 120)


def _solid(i: int) -> np.ndarray:
    return np.full((RH, RW, 3), _color(i), dtype=np.uint8)


@real_ffmpeg
def test_real_ffmpeg_encodes_frames(tmp_path: Path) -> None:
    output = tmp_path / "live.mp4"
    sink = FfmpegPipeSink(output, EncodeConfig(preset="ultrafast"))
    sink.open(REAL_INFO)
    for i in range(30):
        sink.write(_solid(i), _meta(i, REAL_INFO))
    sink.close()

    assert output.is_file()
    assert not sink.partial_path.exists()
    cap = cv2.VideoCapture(str(output))
    try:
        assert cap.isOpened()
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    assert len(frames) == 30
    assert all(f.shape == (RH, RW, 3) for f in frames)
    for i in (0, 15, 29):
        mean = frames[i].reshape(-1, 3).mean(axis=0)
        assert np.allclose(mean, _color(i), atol=6), (i, mean, _color(i))


@real_ffmpeg
def test_real_ffmpeg_abort_leaves_nothing(tmp_path: Path) -> None:
    output = tmp_path / "aborted.mp4"
    sink = FfmpegPipeSink(output, EncodeConfig(preset="ultrafast"))
    sink.open(REAL_INFO)
    for i in range(10):
        sink.write(_solid(i), _meta(i, REAL_INFO))
    sink.abort()
    assert not output.exists()
    assert not sink.partial_path.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == []


@real_ffmpeg
def test_real_ffmpeg_abort_keeps_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "existing.mp4"
    original = bytes(range(256)) * 64
    output.write_bytes(original)
    sink = FfmpegPipeSink(output, EncodeConfig(preset="ultrafast"))
    sink.open(REAL_INFO)
    for i in range(10):
        sink.write(_solid(i), _meta(i, REAL_INFO))
    sink.abort()
    assert output.read_bytes() == original
    assert not sink.partial_path.exists()


@real_ffmpeg
def test_real_ffmpeg_failure_reports_stderr(tmp_path: Path) -> None:
    output = tmp_path / "bad.mp4"
    output.write_bytes(b"old")
    sink = FfmpegPipeSink(output, EncodeConfig(codec="no_such_codec_xyz"))
    sink.open(REAL_INFO)
    with pytest.raises(OSError) as info:
        for i in range(30):
            sink.write(_solid(i), _meta(i, REAL_INFO))
        sink.close()
    sink.abort()
    assert "no_such_codec_xyz" in str(info.value)
    assert "exit code" in str(info.value)
    assert output.read_bytes() == b"old"
    assert not sink.partial_path.exists()


# ---------------------------------------------------------------------------
# Review fixes (P2-9b)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SIGINT_HELPER = Path(__file__).with_name("_helper_pipe_sink_sigint.py")
WAIT_S = 60.0  # Generous bound for subprocess waits; only reached if something is broken.
SW, SH = 64, 48
SMALL_INFO = SourceInfo(fps=25.0, width=SW, height=SH, frame_count=None, seekable=False, live=True)


def _small(i: int) -> np.ndarray:
    return np.full((SH, SW, 3), (i * 9) % 256, dtype=np.uint8)


def _read_frames(path: Path) -> list[tuple[float, np.ndarray]]:
    """Every decodable frame of ``path`` with its timestamp in milliseconds."""
    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.isOpened()
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append((cap.get(cv2.CAP_PROP_POS_MSEC), frame))
    finally:
        cap.release()
    return frames


@pytest.fixture
def tracked_popen() -> Iterator[Callable[..., Any]]:
    """A real Popen that remembers its processes; none may outlive the test."""
    procs: list[subprocess.Popen[bytes]] = []

    def popen(cmd: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        proc = subprocess.Popen(cmd, **kwargs)
        procs.append(proc)
        return proc

    yield popen
    survivors = [p for p in procs if p.poll() is None]
    for proc in survivors:
        proc.kill()
        proc.wait(timeout=WAIT_S)
    assert survivors == [], "FFmpeg was still running after the test"


def test_f1_ffmpeg_runs_outside_the_terminal_process_group(tmp_path: Path) -> None:
    popen = FakePopen()
    sink = _sink(tmp_path, popen)
    sink.open(INFO)
    kwargs = popen.kwargs[0]
    if sys.platform == "win32":
        assert kwargs["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert kwargs["start_new_session"] is True
    sink.abort()


def test_f5_pipe_is_unbuffered_and_partial_writes_complete_the_frame(tmp_path: Path) -> None:
    popen = FakePopen()
    sink = _sink(tmp_path, popen)
    sink.open(INFO)
    assert popen.kwargs[0]["bufsize"] == 0  # A buffered pipe would hold back frame tails.
    stdin = popen.proc.stdin
    accepted: list[bytes] = []

    def short_write(data: Any) -> int:
        chunk = bytes(data)[:50]  # The pipe takes at most 50 bytes per call.
        accepted.append(chunk)
        return len(chunk)

    stdin.write = short_write  # type: ignore[method-assign]
    sink.write(_image(7), _meta(0))
    assert b"".join(accepted) == _image(7).tobytes()
    assert len(accepted) == -(-_image(7).nbytes // 50)
    sink.abort()


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_f2_interrupted_close_stops_ffmpeg_and_removes_partial(
    tmp_path: Path, interrupt: type[BaseException]
) -> None:
    output = tmp_path / "out.mp4"
    output.write_bytes(b"keep me")
    popen = FakePopen()
    sink = _sink(tmp_path, popen)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    proc = popen.proc
    original_wait = proc.wait

    def interrupted_wait(timeout: float | None = None) -> int:
        if not proc.terminated and not proc.killed:
            raise interrupt()  # Ctrl-C while waiting for FFmpeg to finish the tail.
        return original_wait(timeout)

    proc.wait = interrupted_wait  # type: ignore[method-assign]
    with pytest.raises(interrupt):
        sink.close()
    assert proc.terminated == 1
    assert proc.returncode is not None  # FFmpeg was stopped, not left running.
    assert not sink.partial_path.exists()
    assert output.read_bytes() == b"keep me"
    sink.abort()  # Nothing left to clean: a no-op.
    sink.close()
    assert proc.terminated == 1
    assert proc.killed == 0


@pytest.mark.parametrize("error", [OSError("cross-device link"), KeyboardInterrupt()])
def test_f2_failed_replace_removes_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    output = tmp_path / "out.mp4"
    output.write_bytes(b"keep me")
    popen = FakePopen()
    sink = _sink(tmp_path, popen)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))

    def failing_replace(src: Any, dst: Any) -> None:
        raise error

    monkeypatch.setattr("petrify.sinks.file.os.replace", failing_replace)
    with pytest.raises(type(error)):
        sink.close()
    assert not sink.partial_path.exists()
    assert output.read_bytes() == b"keep me"
    sink.abort()
    assert popen.proc.terminated == 0  # FFmpeg had already exited.


def test_f3_close_without_frames_raises_and_installs_nothing(tmp_path: Path) -> None:
    output = tmp_path / "out.mp4"
    output.write_bytes(b"precious earlier recording")
    popen = FakePopen()
    sink = _sink(tmp_path, popen)
    sink.open(INFO)
    with pytest.raises(OSError, match="No frames were written"):
        sink.close()
    proc = popen.proc
    assert proc.returncode is not None
    assert not sink.partial_path.exists()
    assert output.read_bytes() == b"precious earlier recording"
    sink.abort()
    sink.close()
    assert proc.terminated == 1


def test_f3_frame_count_restarts_on_reopen(tmp_path: Path) -> None:
    popen = FakePopen()
    sink = _sink(tmp_path, popen)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    sink.close()
    sink.open(INFO)
    with pytest.raises(OSError, match="No frames were written"):
        sink.close()
    assert (tmp_path / "out.mp4").read_bytes() == _image(0).tobytes()


def test_f4_dash_paths_are_not_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    popen = FakePopen()
    sink = FfmpegPipeSink("-y.mp4", EncodeConfig(), audio_source="-a.mov", popen=popen)
    sink.open(INFO)
    cmd = popen.proc.cmd
    assert cmd[cmd.index("pipe:0") + 2] == os.path.join(".", "-a.mov")
    assert cmd[-1] == os.path.join(".", "-y.partial.mp4")
    sink.write(_image(0), _meta(0))
    sink.close()
    assert (tmp_path / "-y.mp4").read_bytes() == _image(0).tobytes()
    assert not (tmp_path / "-y.partial.mp4").exists()


def test_f4_absolute_paths_are_passed_unchanged(tmp_path: Path) -> None:
    audio = tmp_path / "-a.mov"
    cmd = _cmd(tmp_path, audio_source=audio)
    assert cmd[cmd.index("pipe:0") + 2] == str(audio)
    assert cmd[-1] == str(tmp_path / "out.partial.mp4")


@pytest.mark.parametrize("name", ["", ".", "..", "out/", "sub/dir/", f"out{os.sep}"])
def test_f4_output_without_file_name_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="has no file name"):
        FfmpegPipeSink(name, EncodeConfig(), popen=FakePopen())


def test_f4_output_that_is_a_directory_rejected(tmp_path: Path) -> None:
    (tmp_path / "clip.mp4").mkdir()
    with pytest.raises(ValueError, match="existing directory"):
        FfmpegPipeSink(tmp_path / "clip.mp4", EncodeConfig(), popen=FakePopen())
    with pytest.raises(ValueError, match="existing directory"):
        FfmpegPipeSink(tmp_path, EncodeConfig(), popen=FakePopen())


@real_ffmpeg
@pytest.mark.skipif(os.name != "posix", reason="process groups and SIGINT are POSIX")
def test_real_ffmpeg_f1_sigint_to_the_process_group_keeps_the_recording(tmp_path: Path) -> None:
    output = tmp_path / "live.mp4"
    log, err_log = tmp_path / "helper.out", tmp_path / "helper.err"
    before, after = 10, 5
    with log.open("wb") as out, err_log.open("wb") as err:
        helper = subprocess.Popen(
            [sys.executable, str(SIGINT_HELPER), str(output), str(before), str(after)],
            stdout=out,
            stderr=err,
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            start_new_session=True,  # Its own process group: the only one this test signals.
        )

    def report() -> str:
        return f"stdout: {log.read_text()!r}\nstderr: {err_log.read_text()}"

    ffmpeg_pid: int | None = None
    try:
        deadline = time.monotonic() + WAIT_S
        while not (log.read_text().startswith("ready ") and log.read_text().endswith("\n")):
            assert helper.poll() is None, report()
            assert time.monotonic() < deadline, "helper did not get ready"
            time.sleep(0.01)
        ffmpeg_pid = int(log.read_text().split()[1])
        group = os.getpgid(helper.pid)
        assert group == helper.pid and group != os.getpgrp()
        os.killpg(group, signal.SIGINT)  # What a terminal does on Ctrl-C.
        code = helper.wait(timeout=WAIT_S)
        assert code == 0, report()
        assert log.read_text().splitlines()[-1] == "closed"
    finally:
        if helper.poll() is None:
            os.killpg(helper.pid, signal.SIGKILL)
            helper.wait(timeout=WAIT_S)
    # Existence check only (signal 0 sends nothing): close waited for FFmpeg.
    with pytest.raises(ProcessLookupError):
        os.kill(ffmpeg_pid, 0)
    assert output.is_file()
    assert not (tmp_path / "live.partial.mp4").exists()
    assert len(_read_frames(output)) == before + after


@real_ffmpeg
def test_real_ffmpeg_f3_zero_frames_keeps_existing_output(
    tmp_path: Path, tracked_popen: Callable[..., Any]
) -> None:
    output = tmp_path / "out.mp4"
    output.write_bytes(b"old")
    sink = FfmpegPipeSink(output, EncodeConfig(preset="ultrafast"), popen=tracked_popen)
    sink.open(SMALL_INFO)
    proc = sink._proc
    with pytest.raises(OSError, match="No frames were written"):
        sink.close()
    assert proc.poll() is not None
    assert output.read_bytes() == b"old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.mp4"]


@real_ffmpeg
def test_real_ffmpeg_f4_output_named_like_an_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tracked_popen: Callable[..., Any]
) -> None:
    monkeypatch.chdir(tmp_path)
    sink = FfmpegPipeSink("-y.mp4", EncodeConfig(preset="ultrafast"), popen=tracked_popen)
    sink.open(SMALL_INFO)
    try:
        for i in range(5):
            sink.write(_small(i), _meta(i, SMALL_INFO))
        sink.close()
    finally:
        sink.abort()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["-y.mp4"]
    assert len(_read_frames(tmp_path / "-y.mp4")) == 5


@real_ffmpeg
def test_real_ffmpeg_f5_wallclock_keeps_every_frame_of_a_burst(
    tmp_path: Path, tracked_popen: Callable[..., Any]
) -> None:
    output = tmp_path / "live.mp4"
    sink = FfmpegPipeSink(
        output, EncodeConfig(preset="ultrafast"), wallclock_timestamps=True, popen=tracked_popen
    )
    sink.open(SMALL_INFO)
    bursts, size, pause_s = 5, 8, 0.4
    written = 0
    first = last = 0.0
    try:
        for burst in range(bursts):
            for _ in range(size):  # Back to back: far faster than the nominal 25 fps.
                sink.write(_small(written), _meta(written, SMALL_INFO))
                last = time.monotonic()
                first = first or last
                written += 1
            if burst < bursts - 1:
                time.sleep(pause_s)
        sink.close()
    finally:
        sink.abort()
    frames = _read_frames(output)
    assert len(frames) == written == bursts * size
    stamps = [ms / 1000 for ms, _ in frames]
    # Real time is kept (loosely: FFmpeg stamps a frame when it reads it from the pipe).
    span_s, real_s = stamps[-1] - stamps[0], last - first
    assert 0.5 * real_s < span_s < 1.5 * real_s + 0.2, (span_s, real_s)
    # Each burst is stamped together, before the pause that follows it: no frame is
    # held back (in a buffered pipe) until the next write.
    for b in range(bursts - 1):
        within = stamps[b * size + size - 1] - stamps[b * size]
        gap = stamps[(b + 1) * size] - stamps[b * size + size - 1]
        assert within < gap, (b, within, gap, stamps)


# ---------------------------------------------------------------------------
# T7b: FFmpeg stopped by the same signal as the run (systemctl or docker stop)
# ---------------------------------------------------------------------------

SIGNAL_EXIT = "frame=  3\nExiting normally, received signal 15.\n"


@pytest.fixture
def short_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("petrify.sinks.file._STOP_SIGNAL_GRACE_S", 0.05)


def _readable(monkeypatch: pytest.MonkeyPatch, frames: int | None) -> list[Path]:
    """Replace the read-back of a signal-finalized file; return the paths it was asked about."""
    asked: list[Path] = []

    def count(ffmpeg_bin: str, path: Path) -> int | None:
        asked.append(path)
        return frames

    monkeypatch.setattr("petrify.sinks.file._count_readable_frames", count)
    return asked


def _stopped_ffmpeg(**kwargs: Any) -> FakePopen:
    """FFmpeg that stops reading after 3 frames and reports a stop signal (exit 255)."""
    return FakePopen(exit_code=255, stderr_text=SIGNAL_EXIT, break_after=3,
                     exit_on_break=True, **kwargs)


def test_ffmpeg_stopped_with_the_run_keeps_its_checked_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    asked = _readable(monkeypatch, 3)
    popen = _stopped_ffmpeg()
    sink = _sink(tmp_path, popen, stop_requested=lambda: True)
    sink.open(INFO)
    with caplog.at_level("WARNING", logger="petrify.sinks.file"):
        for i in range(6):
            sink.write(_image(i), _meta(i))  # Frames 3 to 5 are dropped, not an error.
        sink.close()
    output = tmp_path / "out.mp4"
    assert output.exists()  # FFmpeg's own file, moved in place.
    assert not sink.partial_path.exists()
    assert asked == [sink.partial_path]
    assert popen.proc.terminated == popen.proc.killed == 0
    messages = [r.getMessage() for r in caplog.records]
    assert any("frames processed from now on are not recorded" in m for m in messages)
    assert any(f"{output} holds 3 frames, petrify produced 6" in m for m in messages)


def test_ffmpeg_stopped_by_a_signal_at_close_keeps_its_checked_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The signal reached FFmpeg while it was finishing: it exits 255 at close.
    _readable(monkeypatch, 4)
    popen = FakePopen(exit_code=255, stderr_text=SIGNAL_EXIT)
    sink = _sink(tmp_path, popen, stop_requested=lambda: True)
    sink.open(INFO)
    for i in range(4):
        sink.write(_image(i), _meta(i))
    with caplog.at_level("WARNING", logger="petrify.sinks.file"):
        sink.close()
    assert (tmp_path / "out.mp4").exists()
    assert any("holds 4 frames, petrify produced 4" in r.getMessage() for r in caplog.records)


def test_ffmpeg_stopped_with_the_run_but_unreadable_file_fails_and_discards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _readable(monkeypatch, None)
    sink = _sink(tmp_path, _stopped_ffmpeg(), stop_requested=lambda: True)
    sink.open(INFO)
    for i in range(6):
        sink.write(_image(i), _meta(i))
    with pytest.raises(OSError, match="could not be read back"):
        sink.close()
    assert not (tmp_path / "out.mp4").exists()
    assert not sink.partial_path.exists()


def test_ffmpeg_stopped_with_the_run_and_no_frame_in_its_file_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _readable(monkeypatch, 0)
    sink = _sink(tmp_path, _stopped_ffmpeg(), stop_requested=lambda: True)
    sink.open(INFO)
    for i in range(4):
        sink.write(_image(i), _meta(i))
    with pytest.raises(OSError, match="could not be read back"):
        sink.close()
    assert not (tmp_path / "out.mp4").exists()


@pytest.mark.usefixtures("short_grace")
def test_ffmpeg_stopped_by_a_signal_without_a_run_stop_still_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FFmpeg alone was signalled: the run goes on, so this is a failure as before.
    asked = _readable(monkeypatch, 3)
    sink = _sink(tmp_path, _stopped_ffmpeg(), stop_requested=lambda: False)
    sink.open(INFO)
    for i in range(3):
        sink.write(_image(i), _meta(i))
    with pytest.raises(OSError, match="FFmpeg exited while frames"):
        sink.write(_image(3), _meta(3))
    sink.abort()
    assert asked == []
    assert not (tmp_path / "out.mp4").exists()
    assert not sink.partial_path.exists()


@pytest.mark.usefixtures("short_grace")
def test_ffmpeg_stopped_by_a_signal_at_close_without_a_run_stop_still_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = _readable(monkeypatch, 3)
    sink = _sink(tmp_path, FakePopen(exit_code=255, stderr_text=SIGNAL_EXIT),
                 stop_requested=lambda: False)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    with pytest.raises(OSError, match="FFmpeg failed"):
        sink.close()
    assert asked == []
    assert not (tmp_path / "out.mp4").exists()
    assert not sink.partial_path.exists()


def test_run_stop_arriving_within_the_grace_keeps_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FFmpeg may get the signal a moment before the run does.
    _readable(monkeypatch, 3)
    asked_at: list[float] = []
    start = time.monotonic()

    def stop_requested() -> bool:
        asked_at.append(time.monotonic())
        return time.monotonic() - start > 0.1

    sink = _sink(tmp_path, _stopped_ffmpeg(), stop_requested=stop_requested)
    sink.open(INFO)
    for i in range(5):
        sink.write(_image(i), _meta(i))
    sink.close()
    assert (tmp_path / "out.mp4").exists()
    assert len(asked_at) > 1


@pytest.mark.parametrize(
    ("exit_code", "stderr_text"),
    [
        (1, "Conversion failed!\n"),  # FFmpeg failing on its own.
        (255, "Error while encoding\n"),  # 255 without the signal message.
        (0, ""),  # Exit 0 while frames were still being written.
    ],
)
@pytest.mark.usefixtures("short_grace")
def test_ffmpeg_failing_on_its_own_still_fails_with_a_stop_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int, stderr_text: str
) -> None:
    asked = _readable(monkeypatch, 3)
    popen = FakePopen(exit_code=exit_code, stderr_text=stderr_text, break_after=3,
                      exit_on_break=True)
    sink = _sink(tmp_path, popen, stop_requested=lambda: True)
    sink.open(INFO)
    for i in range(3):
        sink.write(_image(i), _meta(i))
    with pytest.raises(OSError):
        sink.write(_image(3), _meta(3))
    sink.abort()
    assert asked == []
    assert not (tmp_path / "out.mp4").exists()
    assert not sink.partial_path.exists()


def test_ffmpeg_stopped_by_a_signal_without_stop_requested_fails_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = _readable(monkeypatch, 3)
    sink = _sink(tmp_path, _stopped_ffmpeg())  # No stop_requested: the default.
    sink.open(INFO)
    for i in range(3):
        sink.write(_image(i), _meta(i))
    with pytest.raises(OSError, match="FFmpeg exited while frames"):
        sink.write(_image(3), _meta(3))
    sink.abort()
    assert asked == []
    assert not (tmp_path / "out.mp4").exists()


def _default_sigint() -> bool:
    return signal.getsignal(signal.SIGINT) is signal.default_int_handler


@pytest.fixture
def sigint_restored() -> Iterator[None]:
    previous = signal.getsignal(signal.SIGINT)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


@pytest.mark.usefixtures("sigint_restored")
@pytest.mark.skipif(not _default_sigint(), reason="needs the default SIGINT handler")
def test_sigint_while_ffmpeg_starts_leaves_no_untracked_ffmpeg(tmp_path: Path) -> None:
    class InterruptedPopen(FakePopen):
        def __call__(self, cmd: list[str], **kwargs: Any) -> FakeProc:
            proc = super().__call__(cmd, **kwargs)
            signal.raise_signal(signal.SIGINT)  # Lands as Popen returns.
            return proc

    popen = InterruptedPopen()
    sink = _sink(tmp_path, popen)
    with pytest.raises(KeyboardInterrupt):
        sink.open(INFO)
    sink.abort()  # Possible only because the handle was stored before the interrupt.
    assert popen.proc.terminated == 1
    assert not sink.partial_path.exists()
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


@pytest.mark.usefixtures("sigint_restored")
@pytest.mark.skipif(not _default_sigint(), reason="needs the default SIGINT handler")
def test_sigint_while_ffmpeg_is_stopped_still_stops_it_and_removes_the_partial(
    tmp_path: Path,
) -> None:
    popen = FakePopen(hang=True)
    sink = _sink(tmp_path, popen)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    proc = popen.proc
    real_terminate = proc.terminate

    def terminate() -> None:
        real_terminate()
        signal.raise_signal(signal.SIGINT)  # A second Ctrl-C during the abort.

    proc.terminate = terminate  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        sink.abort()
    assert proc.terminated == 1
    assert proc.returncode is not None  # It was waited for, not left running.
    assert not sink.partial_path.exists()
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


@real_ffmpeg
def test_real_ffmpeg_reads_back_a_complete_file_and_rejects_a_truncated_one(
    tmp_path: Path,
) -> None:
    from petrify.sinks.file import _count_readable_frames  # noqa: PLC0415

    output = tmp_path / "whole.mp4"
    sink = FfmpegPipeSink(output, EncodeConfig(preset="ultrafast"))
    sink.open(REAL_INFO)
    for i in range(20):
        sink.write(_solid(i), _meta(i, REAL_INFO))
    sink.close()
    ffmpeg = find_ffmpeg()
    assert _count_readable_frames(ffmpeg, output) == 20
    cut = tmp_path / "cut.mp4"
    data = output.read_bytes()
    cut.write_bytes(data[: len(data) * 2 // 3])
    assert _count_readable_frames(ffmpeg, cut) is None


@real_ffmpeg
@pytest.mark.skipif(os.name != "posix", reason="SIGTERM delivery to FFmpeg is POSIX")
def test_real_ffmpeg_sigterm_to_ffmpeg_during_a_stopped_run_keeps_a_playable_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    live = SourceInfo(fps=30.0, width=RW, height=RH, frame_count=None, seekable=False, live=True)
    output = tmp_path / "group.mp4"
    sink = FfmpegPipeSink(output, EncodeConfig(preset="ultrafast"), wallclock_timestamps=True,
                          stop_requested=lambda: True)
    sink.open(live)
    for i in range(15):
        sink.write(_solid(i), _meta(i, live))
        time.sleep(0.01)
    os.kill(sink._proc.pid, signal.SIGTERM)  # As systemd signals every process of the unit.
    with caplog.at_level("WARNING", logger="petrify.sinks.file"):
        for i in range(15, 40):
            sink.write(_solid(i), _meta(i, live))
            time.sleep(0.01)
        sink.close()
    assert output.is_file()
    assert not sink.partial_path.exists()
    frames = _read_frames(output)
    assert 1 <= len(frames) < 40
    assert any("ends early" in r.getMessage() for r in caplog.records)
