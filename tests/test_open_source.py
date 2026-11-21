"""Tests for petrify.sources.open_source and source_kind (no real devices or streams)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from petrify.sources import (
    CameraSource,
    FileSource,
    ImageFolderSource,
    PushSession,
    StreamSource,
    open_source,
    source_kind,
)

SECRET = "Zq9OpenSourceSecret"
TOKEN = "Tk7OpenSourceToken"
SECRET_URL = f"rtsp://admin:{SECRET}@cam.example:554/live?token={TOKEN}"


class FakeCapture:
    """Minimal ``cv2.VideoCapture`` stand-in: opened, 8x6 at 30 fps, every read fails."""

    def __init__(self, arg: Any, opened: bool = True) -> None:
        self.arg = arg
        self.opened = opened
        self.released = 0

    def isOpened(self) -> bool:  # noqa: N802 (mirrors the OpenCV API)
        return self.opened

    def get(self, prop: int) -> float:
        return {cv2.CAP_PROP_FRAME_WIDTH: 8, cv2.CAP_PROP_FRAME_HEIGHT: 6, cv2.CAP_PROP_FPS: 30}.get(
            prop, 0.0
        )

    def set(self, prop: int, value: float) -> bool:
        return True

    def read(self) -> tuple[bool, None]:
        return False, None

    def release(self) -> None:
        self.released += 1


class Factory:
    def __init__(self, opened: bool = True) -> None:
        self.opened = opened
        self.captures: list[FakeCapture] = []

    def __call__(self, arg: Any) -> FakeCapture:
        cap = FakeCapture(arg, self.opened)
        self.captures.append(cap)
        return cap


def _write_images(folder: Path, count: int = 3) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        cv2.imwrite(str(folder / f"f{i}.png"), np.full((6, 8, 3), i * 10, dtype=np.uint8))


def test_exports() -> None:
    import petrify.sources as sources

    assert {"FileSource", "ImageFolderSource", "CameraSource", "StreamSource", "PushSession",
            "open_source"} <= set(sources.__all__)
    assert sources.PushSession is PushSession


@pytest.mark.parametrize("spec", ["camera:0", "camera:3", "camera:12"])
def test_camera_spec(spec: str) -> None:
    factory = Factory()
    src = open_source(spec, capture_factory=factory)
    try:
        assert isinstance(src, CameraSource)
        assert source_kind(spec) == "camera"
        assert factory.captures[0].arg == int(spec.split(":")[1])
        assert src.info.live and not src.info.seekable
        assert src.info.uri == spec
    finally:
        src.close()


@pytest.mark.parametrize("spec", ["camera:", "camera:-1", "camera:x", "camera:1.5", "camera: 1"])
def test_malformed_camera_spec_rejected(spec: str) -> None:
    with pytest.raises(ValueError, match="camera:N"):
        source_kind(spec)
    with pytest.raises(ValueError, match="camera:N"):
        open_source(spec, capture_factory=Factory())


@pytest.mark.parametrize(
    "spec",
    [
        "rtsp://cam.example/live",
        "rtsps://cam.example/live",
        "http://cam.example/mjpeg",
        "https://cam.example/mjpeg",
        "RTSP://cam.example/live",
        SECRET_URL,
    ],
)
def test_stream_spec(spec: str) -> None:
    factory = Factory()
    src = open_source(spec, capture_factory=factory)
    try:
        assert isinstance(src, StreamSource)
        assert source_kind(spec) == "stream"
        assert factory.captures[0].arg == spec  # The full URL reaches the capture only.
        assert src.info.live
        assert SECRET not in str(src.info.uri)
        assert TOKEN not in str(src.info.uri)
    finally:
        src.close()


def test_unsupported_scheme_is_redacted() -> None:
    spec = f"ftp://admin:{SECRET}@cam.example/x?token={TOKEN}"
    for call in (lambda: source_kind(spec), lambda: open_source(spec)):
        with pytest.raises(ValueError) as excinfo:
            call()
        text = str(excinfo.value)
        assert SECRET not in text and TOKEN not in text
        assert "ftp://***@cam.example" in text
        assert "rtsp" in text


def test_stream_open_failure_is_redacted() -> None:
    with pytest.raises(OSError) as excinfo:
        open_source(SECRET_URL, capture_factory=Factory(opened=False))
    text = str(excinfo.value)
    assert SECRET not in text and TOKEN not in text
    assert "rtsp://***@cam.example:554" in text
    assert excinfo.value.__cause__ is None and excinfo.value.__context__ is None


def test_directory_spec(tmp_path: Path) -> None:
    folder = tmp_path / "frames"
    _write_images(folder)
    src = open_source(str(folder))
    assert isinstance(src, ImageFolderSource)
    assert source_kind(str(folder)) == "images"
    assert src.info.fps == 25.0
    assert src.info.frame_count == 3


def test_glob_spec_with_fps(tmp_path: Path) -> None:
    _write_images(tmp_path)
    spec = str(tmp_path / "f*.png")
    src = open_source(spec, fps=10.0)
    assert isinstance(src, ImageFolderSource)
    assert source_kind(spec) == "images"
    assert src.info.fps == 10.0
    assert not src.info.live and src.info.seekable


def test_existing_file_with_glob_characters_is_a_file(tmp_path: Path) -> None:
    path = tmp_path / "clip[1].mp4"
    path.write_bytes(b"")
    assert source_kind(str(path)) == "file"


def test_file_spec(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
    assert writer.isOpened()
    for i in range(3):
        writer.write(np.full((48, 64, 3), i * 40, dtype=np.uint8))
    writer.release()
    src = open_source(str(path))
    try:
        assert isinstance(src, FileSource)
        assert source_kind(str(path)) == "file"
        assert not src.info.live and src.info.seekable
    finally:
        src.close()


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    spec = str(tmp_path / "missing.mp4")
    assert source_kind(spec) == "file"
    with pytest.raises(FileNotFoundError, match="missing.mp4"):
        open_source(spec)


@pytest.mark.parametrize("spec", ["0", "2"])
def test_bare_integer_is_not_a_camera(spec: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    factory = Factory()
    assert source_kind(spec) == "file"
    with pytest.raises(FileNotFoundError, match=f"camera:{spec}"):
        open_source(spec, capture_factory=factory)
    assert factory.captures == []


def test_bare_integer_naming_an_existing_file_is_opened_as_a_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("0").write_bytes(b"not a video")
    factory = Factory()
    with pytest.raises(OSError, match="Cannot open video file"):
        open_source("0", capture_factory=factory)
    assert factory.captures == []


@pytest.mark.parametrize(
    "spec",
    [
        f"camera:rtsp://u:{SECRET}@h/x",
        f"camera:u:{SECRET}@h",
    ],
)
def test_c1_malformed_camera_spec_hides_credentials(spec: str) -> None:
    with pytest.raises(ValueError, match="camera:N") as excinfo:
        source_kind(spec)
    assert SECRET not in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        open_source(spec, capture_factory=Factory())
    assert SECRET not in str(excinfo.value)


def test_c1_malformed_camera_spec_without_credentials_is_shown() -> None:
    with pytest.raises(ValueError, match="'camera:x'"):
        source_kind("camera:x")


@pytest.mark.parametrize(
    "spec",
    [f"rtsp:/u:{SECRET}@h/x", f"u:{SECRET}@h:554/x", f"rtsp:/h/x/token={SECRET}://"],
)
def test_c1_mistyped_stream_spec_as_missing_file_hides_credentials(spec: str) -> None:
    assert source_kind(spec) == "file"
    with pytest.raises(FileNotFoundError) as excinfo:
        open_source(spec, capture_factory=Factory())
    assert SECRET not in str(excinfo.value)


def test_c1_image_glob_with_credentials_hides_them(tmp_path: Path) -> None:
    spec = str(tmp_path / f"u:{SECRET}@h" / "*.png")
    assert source_kind(spec) == "images"
    with pytest.raises(ValueError) as excinfo:
        open_source(spec)
    assert SECRET not in str(excinfo.value)
    assert excinfo.value.__cause__ is None and excinfo.value.__suppress_context__


def test_c1_plain_missing_path_is_shown_unchanged(tmp_path: Path) -> None:
    missing = tmp_path / "plain_missing.mp4"
    with pytest.raises(FileNotFoundError, match=f"Input file not found: {missing}"):
        open_source(str(missing))
