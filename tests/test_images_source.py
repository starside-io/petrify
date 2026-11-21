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

"""Tests for petrify.sources.images.ImageFolderSource."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from petrify.core import Source
from petrify.sources.images import ImageFolderSource

WIDTH, HEIGHT, FPS = 32, 24, 10.0


def _image(seed: int, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _write(path: Path, image: np.ndarray) -> Path:
    assert cv2.imwrite(str(path), image)
    return path


@pytest.fixture
def folder(tmp_path: Path) -> tuple[Path, list[np.ndarray]]:
    images = [_image(i) for i in range(4)]
    for i, img in enumerate(images):
        _write(tmp_path / f"frame{i}.png", img)
    return tmp_path, images


def test_directory_form(folder: tuple[Path, list[np.ndarray]]) -> None:
    path, images = folder
    with ImageFolderSource(path, fps=FPS) as src:
        assert [p.name for p in src.paths] == [f"frame{i}.png" for i in range(4)]
        assert all(isinstance(p, Path) for p in src.paths)
        assert len(list(src)) == len(images)


def test_glob_form(tmp_path: Path) -> None:
    for name in ("a1.png", "a2.png", "b1.png"):
        _write(tmp_path / name, _image(0))
    with ImageFolderSource(str(tmp_path / "a*.png")) as src:
        assert [p.name for p in src.paths] == ["a1.png", "a2.png"]
        assert src.info.uri == str(tmp_path / "a*.png")


def test_glob_applies_suffix_filter(tmp_path: Path) -> None:
    _write(tmp_path / "a1.png", _image(0))
    (tmp_path / "a2.txt").write_text("not an image")
    with ImageFolderSource(str(tmp_path / "a*")) as src:
        assert [p.name for p in src.paths] == ["a1.png"]


def test_suffix_filter_ignores_other_files_and_subdirs(tmp_path: Path) -> None:
    _write(tmp_path / "f1.png", _image(0))
    (tmp_path / "notes.txt").write_text("hello")
    sub = tmp_path / "sub.png"
    sub.mkdir()
    _write(sub / "f0.png", _image(1))
    with ImageFolderSource(tmp_path) as src:
        assert [p.name for p in src.paths] == ["f1.png"]


def test_suffix_match_is_case_insensitive(tmp_path: Path) -> None:
    _write(tmp_path / "f1.PNG", _image(0))
    _write(tmp_path / "f2.Jpg", _image(1))
    _write(tmp_path / "f3.jpeg", _image(2))
    _write(tmp_path / "f4.BMP", _image(3))
    _write(tmp_path / "f5.tif", _image(4))
    with ImageFolderSource(tmp_path) as src:
        assert [p.name for p in src.paths] == ["f1.PNG", "f2.Jpg", "f3.jpeg", "f4.BMP", "f5.tif"]


def test_natural_order(tmp_path: Path) -> None:
    for name in ("f1.png", "f2.png", "f10.png", "f9.png"):
        _write(tmp_path / name, _image(0))
    with ImageFolderSource(tmp_path) as src:
        assert [p.name for p in src.paths] == ["f1.png", "f2.png", "f9.png", "f10.png"]


def test_natural_order_mixed_zero_padding(tmp_path: Path) -> None:
    for name in ("img010.png", "img2.png", "img01.png", "img1.png", "img003.png"):
        _write(tmp_path / name, _image(0))
    with ImageFolderSource(tmp_path) as src:
        # Equal numeric value (01 and 1) falls back to the plain name: "img01" < "img1".
        assert [p.name for p in src.paths] == [
            "img01.png",
            "img1.png",
            "img2.png",
            "img003.png",
            "img010.png",
        ]


def test_info_values(folder: tuple[Path, list[np.ndarray]]) -> None:
    path, images = folder
    with ImageFolderSource(path, fps=FPS) as src:
        info = src.info
        assert info.fps == FPS
        assert (info.width, info.height) == (WIDTH, HEIGHT)
        assert info.frame_count == len(images)
        assert info.seekable is True
        assert info.live is False
        assert info.uri == str(path)


def test_default_fps(folder: tuple[Path, list[np.ndarray]]) -> None:
    path, _ = folder
    assert ImageFolderSource(path).info.fps == 25.0


def test_indices_and_timestamps(folder: tuple[Path, list[np.ndarray]]) -> None:
    path, images = folder
    with ImageFolderSource(path, fps=FPS) as src:
        frames = list(src)
    assert [f.index for f in frames] == list(range(len(images)))
    for f in frames:
        assert f.timestamp == pytest.approx(f.index / FPS)
        assert f.source is src.info


def test_iterating_twice_gives_same_frames(folder: tuple[Path, list[np.ndarray]]) -> None:
    path, _ = folder
    with ImageFolderSource(path) as src:
        first = list(src)
        second = list(src)
    assert [f.index for f in first] == [f.index for f in second]
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a.image, b.image)


def test_png_round_trip_is_lossless(folder: tuple[Path, list[np.ndarray]]) -> None:
    path, images = folder
    with ImageFolderSource(path) as src:
        frames = list(src)
    for f, expected in zip(frames, images, strict=True):
        assert f.image.dtype == np.uint8
        assert f.image.shape == (HEIGHT, WIDTH, 3)
        assert np.array_equal(f.image, expected)


def test_grayscale_png_becomes_three_channels(tmp_path: Path) -> None:
    gray = _image(0)[:, :, 0]
    _write(tmp_path / "g.png", gray)
    with ImageFolderSource(tmp_path) as src:
        (frame,) = list(src)
    assert frame.image.shape == (HEIGHT, WIDTH, 3)
    assert frame.image.dtype == np.uint8
    for c in range(3):
        assert np.array_equal(frame.image[:, :, c], gray)


def test_rgba_png_becomes_three_channels(tmp_path: Path) -> None:
    bgr = _image(0)
    alpha = np.full((HEIGHT, WIDTH, 1), 128, dtype=np.uint8)
    _write(tmp_path / "a.png", np.concatenate([bgr, alpha], axis=2))
    with ImageFolderSource(tmp_path) as src:
        (frame,) = list(src)
    assert frame.image.shape == (HEIGHT, WIDTH, 3)
    assert frame.image.dtype == np.uint8


def test_sixteen_bit_png_becomes_eight_bit(tmp_path: Path) -> None:
    deep = (_image(0).astype(np.uint16) * 257).astype(np.uint16)
    _write(tmp_path / "d.png", deep)
    with ImageFolderSource(tmp_path) as src:
        (frame,) = list(src)
    assert frame.image.dtype == np.uint8
    assert frame.image.shape == (HEIGHT, WIDTH, 3)


def test_garbage_file_raises_oserror_naming_it(folder: tuple[Path, list[np.ndarray]]) -> None:
    path, _ = folder
    bad = path / "frame9.png"
    bad.write_bytes(b"this is not a png at all")
    src = ImageFolderSource(path)
    with pytest.raises(OSError, match="frame9.png"):
        list(src)


def test_truncated_file_raises_oserror_naming_it(folder: tuple[Path, list[np.ndarray]]) -> None:
    path, _ = folder
    data = (path / "frame0.png").read_bytes()
    bad = path / "frame9.png"
    bad.write_bytes(data[: len(data) // 3])
    src = ImageFolderSource(path)
    with pytest.raises(OSError, match="frame9.png"):
        list(src)


def test_undecodable_first_image_fails_at_construction(tmp_path: Path) -> None:
    (tmp_path / "f0.jpg").write_bytes(b"garbage")
    with pytest.raises(OSError, match="f0.jpg"):
        ImageFolderSource(tmp_path)


def test_size_mismatch_raises_valueerror(folder: tuple[Path, list[np.ndarray]]) -> None:
    path, _ = folder
    _write(path / "frame9.png", _image(9, width=WIDTH + 2, height=HEIGHT))
    src = ImageFolderSource(path)
    with pytest.raises(ValueError, match=r"frame9\.png.*34x24.*32x24"):
        list(src)
    with pytest.raises(ValueError, match="frame9.png"):
        src.read_frame(4)


def test_read_frame_in_and_out_of_range(folder: tuple[Path, list[np.ndarray]]) -> None:
    path, images = folder
    with ImageFolderSource(path) as src:
        for i, expected in enumerate(images):
            img = src.read_frame(i)
            assert img is not None
            assert np.array_equal(img, expected)
        assert src.read_frame(len(images)) is None
        assert src.read_frame(-1) is None


def test_read_frame_raises_on_undecodable(folder: tuple[Path, list[np.ndarray]]) -> None:
    path, _ = folder
    (path / "frame9.png").write_bytes(b"junk")
    src = ImageFolderSource(path)
    with pytest.raises(OSError, match="frame9.png"):
        src.read_frame(4)


def test_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ImageFolderSource(tmp_path / "nope")


def test_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=re.escape(str(tmp_path))):
        ImageFolderSource(tmp_path)


def test_glob_with_no_match(tmp_path: Path) -> None:
    pattern = str(tmp_path / "*.png")
    with pytest.raises(ValueError, match=r"\*\.png"):
        ImageFolderSource(pattern)


@pytest.mark.parametrize("fps", [0.0, -1.0])
def test_bad_fps(folder: tuple[Path, list[np.ndarray]], fps: float) -> None:
    path, _ = folder
    with pytest.raises(ValueError, match="fps"):
        ImageFolderSource(path, fps=fps)


def test_satisfies_source_protocol(folder: tuple[Path, list[np.ndarray]]) -> None:
    path, _ = folder
    assert isinstance(ImageFolderSource(path), Source)


def test_close_is_idempotent(folder: tuple[Path, list[np.ndarray]]) -> None:
    path, _ = folder
    src = ImageFolderSource(path)
    src.close()
    src.close()
