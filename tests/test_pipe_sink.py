"""Tests for petrify.sinks.file.FfmpegPipeSink: fake Popen, plus real FFmpeg when available."""

from __future__ import annotations

import subprocess
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
