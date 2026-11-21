"""Tests for petrify.sinks.virtualcam.VirtualCamSink with a fake camera (no device is opened)."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from petrify.core import FrameMeta, Sink, SourceInfo
from petrify.sinks import VirtualCamSink

H, W = 4, 6
INFO = SourceInfo(fps=30.0, width=W, height=H, frame_count=None, seekable=False, live=True)
ROOT = Path(__file__).resolve().parents[1]


def _info(fps: float) -> SourceInfo:
    return SourceInfo(fps=fps, width=W, height=H, frame_count=None, seekable=False, live=True)


def _meta(i: int) -> FrameMeta:
    return FrameMeta(index=i, timestamp=i / INFO.fps, source=INFO)


def _image(i: int) -> np.ndarray:
    image = np.zeros((H, W, 3), dtype=np.uint8)
    image[..., 0] = 10 + i  # Blue
    image[..., 1] = 100  # Green
    image[..., 2] = 200  # Red
    return image


class FakeCamera:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.sent: list[np.ndarray] = []
        self.closes = 0
        self.device = kwargs.get("device", "Fake Virtual Camera")

    def send(self, frame: np.ndarray) -> None:
        self.sent.append(frame)

    def sleep_until_next_frame(self) -> None:
        raise AssertionError("the sink must not pace frames")

    def close(self) -> None:
        self.closes += 1


class Factory:
    """Stands in for pyvirtualcam.Camera; ``bgr`` toggles PixelFormat.BGR support."""

    def __init__(self, *, bgr: bool = True, fail: BaseException | None = None) -> None:
        members = {"RGB": "rgb-format"}
        if bgr:
            members["BGR"] = "bgr-format"
        self.PixelFormat = SimpleNamespace(**members)
        self.fail = fail
        self.cameras: list[FakeCamera] = []

    def __call__(self, **kwargs: Any) -> FakeCamera:
        if self.fail is not None:
            raise self.fail
        cam = FakeCamera(**kwargs)
        self.cameras.append(cam)
        return cam

    @property
    def cam(self) -> FakeCamera:
        assert len(self.cameras) == 1
        return self.cameras[0]


def test_satisfies_sink_and_critical_default() -> None:
    sink = VirtualCamSink(camera_factory=Factory())
    assert isinstance(sink, Sink)
    assert sink.critical is True
    assert VirtualCamSink(critical=False).critical is False


def test_constructor_arguments_bgr() -> None:
    factory = Factory()
    sink = VirtualCamSink(device="/dev/video9", camera_factory=factory)
    sink.open(INFO)
    assert factory.cam.kwargs == {
        "width": W,
        "height": H,
        "fps": 30.0,
        "fmt": "bgr-format",
        "device": "/dev/video9",
    }
    sink.close()


def test_device_omitted_when_not_given() -> None:
    factory = Factory()
    sink = VirtualCamSink(camera_factory=factory)
    sink.open(INFO)
    assert "device" not in factory.cam.kwargs
    sink.close()


@pytest.mark.parametrize(("fps", "expected"), [(30000 / 1001, 29.97), (25.0, 25.0), (0.001, 0.01)])
def test_fps_is_rounded(fps: float, expected: float) -> None:
    factory = Factory()
    sink = VirtualCamSink(camera_factory=factory)
    sink.open(_info(fps))
    assert factory.cam.kwargs["fps"] == expected
    sink.close()


@pytest.mark.parametrize("fps", [0.0, -5.0, float("nan"), float("inf")])
def test_invalid_fps_raises(fps: float) -> None:
    factory = Factory()
    sink = VirtualCamSink(camera_factory=factory)
    with pytest.raises(ValueError, match="Frame rate"):
        sink.open(_info(fps))
    assert factory.cameras == []
    sink.abort()


def test_bgr_path_sends_image_unchanged() -> None:
    factory = Factory()
    sink = VirtualCamSink(camera_factory=factory)
    sink.open(INFO)
    images = [_image(i) for i in range(3)]
    for i, image in enumerate(images):
        sink.write(image, _meta(i))
    assert all(sent is image for sent, image in zip(factory.cam.sent, images))
    sink.close()


def test_rgb_fallback_converts() -> None:
    factory = Factory(bgr=False)
    sink = VirtualCamSink(camera_factory=factory)
    sink.open(INFO)
    assert "fmt" not in factory.cam.kwargs
    image = _image(5)
    sink.write(image, _meta(0))
    (sent,) = factory.cam.sent
    assert np.array_equal(sent, image[..., ::-1])
    assert sent.flags.c_contiguous
    assert image[0, 0, 0] == 15  # Input untouched.
    sink.close()


def test_factory_without_pixel_format_uses_rgb() -> None:
    cameras: list[FakeCamera] = []

    def plain_factory(**kwargs: Any) -> FakeCamera:
        cameras.append(FakeCamera(**kwargs))
        return cameras[-1]

    sink = VirtualCamSink(camera_factory=plain_factory)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    assert "fmt" not in cameras[0].kwargs
    assert np.array_equal(cameras[0].sent[0], _image(0)[..., ::-1])
    sink.close()


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((H + 1, W, 3), np.uint8),
        np.zeros((H, W, 4), np.uint8),
        np.zeros((H, W), np.uint8),
        np.zeros((H, W, 3), np.float32),
    ],
)
def test_write_validates(image: np.ndarray) -> None:
    factory = Factory()
    sink = VirtualCamSink(camera_factory=factory)
    sink.open(INFO)
    with pytest.raises(ValueError, match="Expected a uint8 image"):
        sink.write(image, _meta(0))
    assert factory.cam.sent == []
    sink.close()


def test_write_before_open_raises() -> None:
    sink = VirtualCamSink(camera_factory=Factory())
    with pytest.raises(RuntimeError, match="not open"):
        sink.write(_image(0), _meta(0))


def test_double_open_raises() -> None:
    sink = VirtualCamSink(camera_factory=Factory())
    sink.open(INFO)
    with pytest.raises(RuntimeError, match="already open"):
        sink.open(INFO)
    sink.close()


@pytest.mark.parametrize("calls", [["close", "close"], ["close", "abort"], ["abort", "close", "abort"]])
def test_camera_closed_once(calls: list[str]) -> None:
    factory = Factory()
    sink = VirtualCamSink(camera_factory=factory)
    sink.open(INFO)
    for name in calls:
        getattr(sink, name)()
    assert factory.cam.closes == 1


@pytest.mark.parametrize("bgr", [True, False])
@pytest.mark.parametrize("calls", [["close"], ["abort"], ["close", "abort", "close"], ["abort", "close"]])
def test_v1_one_black_frame_before_the_camera_closes(bgr: bool, calls: list[str]) -> None:
    factory = Factory(bgr=bgr)
    sink = VirtualCamSink(camera_factory=factory)
    sink.open(INFO)
    sink.write(_image(3), _meta(0))
    cam = factory.cam
    order: list[str] = []
    send, close = cam.send, cam.close

    def recording_send(frame: np.ndarray) -> None:
        order.append("send")
        send(frame)

    def recording_close() -> None:
        order.append("close")
        close()

    cam.send = recording_send  # type: ignore[method-assign]
    cam.close = recording_close  # type: ignore[method-assign]
    for name in calls:
        getattr(sink, name)()
    assert order == ["send", "close"]
    assert len(cam.sent) == 2
    black = cam.sent[-1]
    assert black.shape == (H, W, 3) and black.dtype == np.uint8
    assert not black.any()


@pytest.mark.parametrize("method", ["close", "abort"])
def test_v1_black_frame_failure_is_logged_and_ignored(
    method: str, caplog: pytest.LogCaptureFixture
) -> None:
    factory = Factory()
    sink = VirtualCamSink(camera_factory=factory)
    sink.open(INFO)

    def broken_send(frame: np.ndarray) -> None:
        raise OSError("consumer gone")

    factory.cam.send = broken_send  # type: ignore[method-assign]
    with caplog.at_level(logging.DEBUG, logger="petrify.sinks.virtualcam"):
        getattr(sink, method)()
    assert factory.cam.closes == 1
    records = [r for r in caplog.records if "black frame" in r.getMessage()]
    assert [r.levelno for r in records] == [logging.DEBUG]


def test_v1_no_black_frame_without_an_open_camera() -> None:
    factory = Factory(fail=RuntimeError("device busy"))
    sink = VirtualCamSink(camera_factory=factory)
    with pytest.raises(RuntimeError):
        sink.open(INFO)
    sink.close()
    sink.abort()
    assert factory.cameras == []


def test_abort_before_open_and_after_failed_open() -> None:
    sink = VirtualCamSink(camera_factory=Factory(fail=RuntimeError("device busy")))
    sink.abort()
    sink.close()
    with pytest.raises(RuntimeError, match="device busy"):
        sink.open(INFO)
    sink.abort()
    sink.close()


def test_abort_swallows_close_errors() -> None:
    factory = Factory()
    sink = VirtualCamSink(camera_factory=factory)
    sink.open(INFO)

    def broken_close() -> None:
        raise OSError("gone")

    factory.cam.close = broken_close  # type: ignore[method-assign]
    sink.abort()
    sink.abort()


def test_missing_library_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pyvirtualcam", None)
    sink = VirtualCamSink()
    with pytest.raises(RuntimeError) as info:
        sink.open(INFO)
    message = str(info.value)
    assert "pip install petrify[virtualcam]" in message
    assert "OBS 30" in message
    assert isinstance(info.value.__cause__, ImportError)
    sink.abort()


def test_real_library_path_uses_module_pixel_format(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[FakeCamera] = []

    def camera(**kwargs: Any) -> FakeCamera:
        created.append(FakeCamera(**kwargs))
        return created[-1]

    fake_module = SimpleNamespace(Camera=camera, PixelFormat=SimpleNamespace(BGR="BGR", RGB="RGB"))
    monkeypatch.setitem(sys.modules, "pyvirtualcam", fake_module)
    sink = VirtualCamSink()
    sink.open(INFO)
    assert created[0].kwargs["fmt"] == "BGR"
    image = _image(0)
    sink.write(image, _meta(0))
    assert created[0].sent[0] is image
    sink.close()
    assert created[0].closes == 1


def test_importing_sinks_does_not_import_pyvirtualcam_or_need_a_gui() -> None:
    code = (
        "import sys\n"
        "import petrify.sinks\n"
        "from petrify.sinks import PreviewSink, VirtualCamSink, FfmpegPipeSink\n"
        "assert 'pyvirtualcam' not in sys.modules, 'pyvirtualcam imported'\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
