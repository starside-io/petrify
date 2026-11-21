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

"""Still image folder or glob frame source."""

from __future__ import annotations

import glob
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

from petrify.core.frame import Frame, SourceInfo

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})

_GLOB_CHARS = ("*", "?", "[")
_RUNS = re.compile(r"(\d+)")


def _natural_key(path: Path) -> tuple[tuple[tuple[int, int, str], ...], str]:
    """Sort key: digit runs compared as integers, other runs as plain text, then the name."""
    name = path.name
    parts = tuple(
        (0, int(run), "") if run.isdigit() else (1, 0, run) for run in _RUNS.split(name) if run
    )
    return parts, name


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


class ImageFolderSource:
    """Sequential, seekable frames from a folder or glob of still images.

    ``location`` is a directory (its direct children, not recursive) or a glob
    pattern (any string containing ``*``, ``?`` or ``[`` that is not an existing
    directory). Only files whose suffix, case-insensitive, is one of
    ``IMAGE_SUFFIXES`` are kept.

    Order is a natural sort on the file name: the name is split into digit and
    non-digit runs, digit runs compare as integers and non-digit runs as plain
    (case-sensitive) text, a digit run sorts before a text run at the same
    position, and ties (for example ``f01`` and ``f1``) are broken by the plain
    name. So ``frame2.png`` comes before ``frame10.png``. The order depends only
    on the names, never on the filesystem listing order.

    Every image is read with ``cv2.IMREAD_COLOR``: frames are always BGR uint8
    with 3 channels, grayscale is expanded, an alpha channel is discarded, and
    16-bit images are reduced to 8-bit. Nothing is ever skipped: an undecodable
    file raises ``OSError`` and a file whose size differs from the first image
    raises ``ValueError``.
    """

    def __init__(self, location: str | Path, fps: float = 25.0) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")

        text = str(location)
        as_path = Path(location)
        if as_path.is_dir():
            candidates = list(as_path.iterdir())
        elif any(ch in text for ch in _GLOB_CHARS):
            candidates = [Path(p) for p in glob.glob(text)]
        else:
            raise FileNotFoundError(f"Input directory not found: {location}")

        paths = tuple(sorted((p for p in candidates if _is_image(p)), key=_natural_key))
        if not paths:
            raise ValueError(f"No image files found at: {location}")

        self._paths = paths
        first = self._decode(paths[0])
        height, width = first.shape[:2]
        self.info = SourceInfo(
            fps=fps,
            width=width,
            height=height,
            frame_count=len(paths),
            seekable=True,
            live=False,
            uri=text,
        )
        logger.info(
            "Input: %s  |  %dx%d @ %.2f fps  |  %d images",
            text,
            width,
            height,
            fps,
            len(paths),
        )

    @property
    def paths(self) -> tuple[Path, ...]:
        """The image files in frame order."""
        return self._paths

    @staticmethod
    def _decode(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Cannot decode image file: {path}")
        return image

    def _load(self, path: Path) -> np.ndarray:
        image = self._decode(path)
        height, width = image.shape[:2]
        if (width, height) != (self.info.width, self.info.height):
            raise ValueError(
                f"Image size mismatch in {path}: got {width}x{height}, "
                f"expected {self.info.width}x{self.info.height}"
            )
        return image

    def __iter__(self) -> Iterator[Frame]:
        """Yield one frame per image from index 0; raises instead of skipping a bad file."""
        info = self.info
        for index, path in enumerate(self._paths):
            yield Frame(
                image=self._load(path), index=index, timestamp=index / info.fps, source=info
            )

    def read_frame(self, index: int) -> np.ndarray | None:
        """Return the image at ``index``, or None when out of range."""
        if not 0 <= index < len(self._paths):
            return None
        return self._load(self._paths[index])

    def close(self) -> None:
        """Nothing to release; safe to call more than once."""

    def __enter__(self) -> ImageFolderSource:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
