# Copyright 2026 petrify contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-License-Identifier: Apache-2.0

"""Tests for petrify.sinks.file.FileSink."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

import petrify.sinks.file as sink_mod
from petrify.core import EncodeConfig, Frame, Sink, SourceInfo
from petrify.sinks.file import FileSink

WIDTH, HEIGHT, FPS = 160, 256, 12.5
INFO = SourceInfo(fps=FPS, width=WIDTH, height=HEIGHT, frame_count=5, seekable=True, live=False)
ENCODE = EncodeConfig(codec="libx265", crf=19, preset="fast", hwaccel="videotoolbox")


class _EncoderSpy:
    def __init__(self, exc: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.existed: list[bool] = []
        self.sizes: list[int] = []
        self.exc = exc

    def __call__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        path = kwargs["video_path"]
        self.existed.append(os.path.exists(path))
        self.sizes.append(os.path.getsize(path))
        if self.exc is not None:
            raise self.exc


def _frame(i: int) -> Frame:
    img = np.full((HEIGHT, WIDTH, 3), 20 * i, dtype=np.uint8)
    return Frame(image=img, index=i, timestamp=i / FPS, source=INFO)


def _sink(tmp_path: Path) -> FileSink:
    return FileSink(tmp_path / "out.mp4", tmp_path / "in.mp4", ENCODE)


def test_open_write_close_encodes_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _EncoderSpy()
    monkeypatch.setattr(sink_mod, "ffmpeg_encode", spy)
    sink = _sink(tmp_path)
    sink.open(INFO)
    temp_path = sink._temp_path
    assert temp_path is not None and temp_path.endswith(".mp4")
    for i in range(5):
        f = _frame(i)
        sink.write(f, f.image)
    sink.close()

    assert len(spy.calls) == 1
    call = spy.calls[0]
    assert call == {
        "video_path": temp_path,
        "original_audio_source": tmp_path / "in.mp4",
        "output_path": tmp_path / "out.mp4",
        "codec": "libx265",
        "crf": 19,
        "preset": "fast",
        "hwaccel": "videotoolbox",
    }
    assert spy.existed == [True]
    assert spy.sizes[0] > 0
    assert not os.path.exists(temp_path)

    sink.close()
    sink.abort()
    assert len(spy.calls) == 1


def test_writer_gets_positional_fps_and_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[Any, ...]] = []
    real = cv2.VideoWriter

    def _spy_writer(*args: Any) -> cv2.VideoWriter:
        seen.append(args)
        return real(*args)

    monkeypatch.setattr(cv2, "VideoWriter", _spy_writer)
    monkeypatch.setattr(sink_mod, "ffmpeg_encode", _EncoderSpy())
    sink = _sink(tmp_path)
    sink.open(INFO)
    sink.abort()
    assert len(seen) == 1
    assert seen[0][1] == cv2.VideoWriter_fourcc(*"mp4v")
    assert seen[0][2] == FPS
    assert seen[0][3] == (WIDTH, HEIGHT)


def test_abort_removes_temp_without_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _EncoderSpy()
    monkeypatch.setattr(sink_mod, "ffmpeg_encode", spy)
    sink = _sink(tmp_path)
    sink.open(INFO)
    temp_path = sink._temp_path
    assert temp_path is not None
    f = _frame(0)
    sink.write(f, f.image)
    sink.abort()
    assert spy.calls == []
    assert not os.path.exists(temp_path)
    sink.abort()
    sink.close()
    assert spy.calls == []


def test_encoder_error_removes_temp_and_reraises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _EncoderSpy(exc=RuntimeError("ffmpeg died"))
    monkeypatch.setattr(sink_mod, "ffmpeg_encode", spy)
    sink = _sink(tmp_path)
    sink.open(INFO)
    temp_path = sink._temp_path
    assert temp_path is not None
    with pytest.raises(RuntimeError, match="ffmpeg died"):
        sink.close()
    assert spy.existed == [True]
    assert not os.path.exists(temp_path)


def test_writer_open_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[str] = []

    class _ClosedWriter:
        def __init__(self, path: str, *args: Any) -> None:
            created.append(path)

        def isOpened(self) -> bool:  # noqa: N802
            return False

    monkeypatch.setattr(cv2, "VideoWriter", _ClosedWriter)
    spy = _EncoderSpy()
    monkeypatch.setattr(sink_mod, "ffmpeg_encode", spy)
    sink = _sink(tmp_path)
    with pytest.raises(OSError, match="cv2.VideoWriter failed to open the intermediate file."):
        sink.open(INFO)
    assert len(created) == 1
    assert not os.path.exists(created[0])
    sink.close()
    sink.abort()
    assert spy.calls == []


def test_close_and_abort_without_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _EncoderSpy()
    monkeypatch.setattr(sink_mod, "ffmpeg_encode", spy)
    sink = _sink(tmp_path)
    sink.close()
    sink.abort()
    assert spy.calls == []


def test_satisfies_sink_protocol(tmp_path: Path) -> None:
    assert isinstance(_sink(tmp_path), Sink)


def test_writer_constructor_error_removes_temp_and_reraises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[str] = []

    def _raising_writer(path: str, *args: Any) -> None:
        created.append(path)
        raise cv2.error("writer constructor broke")

    monkeypatch.setattr(cv2, "VideoWriter", _raising_writer)
    spy = _EncoderSpy()
    monkeypatch.setattr(sink_mod, "ffmpeg_encode", spy)
    sink = _sink(tmp_path)
    with pytest.raises(cv2.error, match="writer constructor broke"):
        sink.open(INFO)
    assert len(created) == 1
    assert not os.path.exists(created[0])
    sink.close()
    sink.abort()
    assert spy.calls == []
