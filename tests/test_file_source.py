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

"""Tests for petrify.sources.file.FileSource."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from petrify.core import Source
from petrify.sources.file import FileSource

WIDTH, HEIGHT, FPS, N_FRAMES = 160, 256, 10.0, 12


def _write_clip(path: Path, n: int = N_FRAMES) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened()
    for i in range(n):
        img = np.full((HEIGHT, WIDTH, 3), 10 + 15 * i, dtype=np.uint8)
        cv2.rectangle(img, (10, 10 + 8 * i), (40, 30 + 8 * i), (255, 255, 255), -1)
        writer.write(img)
    writer.release()
    return path


def _plain_read(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        frames.append(img)
    cap.release()
    return frames


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    return _write_clip(tmp_path / "clip.mp4")


def test_info_matches_clip(clip: Path) -> None:
    with FileSource(clip) as src:
        info = src.info
        assert info.fps == pytest.approx(FPS)
        assert (info.width, info.height) == (WIDTH, HEIGHT)
        assert info.frame_count == N_FRAMES
        assert info.seekable is True
        assert info.live is False
        assert info.uri == str(clip)


def test_iteration_matches_sequential_read(clip: Path) -> None:
    expected = _plain_read(clip)
    assert len(expected) == N_FRAMES
    with FileSource(clip) as src:
        frames = list(src)
    assert [f.index for f in frames] == list(range(N_FRAMES))
    for f, exp in zip(frames, expected, strict=True):
        assert f.timestamp == pytest.approx(f.index / FPS)
        assert f.source is src.info
        np.testing.assert_array_equal(f.image, exp)


def test_iterating_twice_gives_same_frames(clip: Path) -> None:
    with FileSource(clip) as src:
        first = [f.image for f in src]
        second = [f.image for f in src]
    assert len(first) == len(second) == N_FRAMES
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_read_frame_and_iteration_after_it(clip: Path) -> None:
    with FileSource(clip) as src:
        frame0 = src.read_frame(0)
        assert frame0 is not None
        assert src.read_frame(7) is not None
        frames = list(src)
    assert frames[0].index == 0
    assert len(frames) == N_FRAMES
    np.testing.assert_array_equal(frame0, frames[0].image)


def test_read_frame_out_of_range_returns_none(clip: Path) -> None:
    with FileSource(clip) as src:
        assert src.read_frame(N_FRAMES + 50) is None


def test_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.mp4"
    with pytest.raises(FileNotFoundError, match=f"Input file not found: {missing}"):
        FileSource(missing)


def test_unreadable_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video at all")
    with pytest.raises(OSError, match="Cannot open video file"):
        FileSource(bad)


def test_close_idempotent_and_capture(clip: Path) -> None:
    src = FileSource(clip)
    assert isinstance(src.capture, cv2.VideoCapture)
    src.close()
    src.close()
    with pytest.raises(RuntimeError):
        _ = src.capture


def test_satisfies_source_protocol(clip: Path) -> None:
    with FileSource(clip) as src:
        assert isinstance(src, Source)
