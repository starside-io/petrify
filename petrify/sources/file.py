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

"""Video file frame source."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

from petrify.core.frame import Frame, SourceInfo

logger = logging.getLogger(__name__)


class FileSource:
    """Sequential, seekable frames from a video file via ``cv2.VideoCapture``.

    Iteration always starts at frame 0 and yields every decodable frame in
    order, at most ``info.frame_count`` of them. ``read_frame`` gives random
    access for pre-scans; calling it while an iteration is in progress is not
    supported (it moves the shared capture position).
    """

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise OSError(f"Cannot open video file: {path}")

        self._path = path
        self._cap: cv2.VideoCapture | None = cap
        self.info = SourceInfo(
            fps=cap.get(cv2.CAP_PROP_FPS) or 25.0,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            seekable=True,
            live=False,
            uri=str(path),
        )
        logger.info(
            "Input: %s  |  %dx%d @ %.2f fps  |  ~%d frames",
            path,
            self.info.width,
            self.info.height,
            self.info.fps,
            self.info.frame_count,
        )

    @property
    def capture(self) -> cv2.VideoCapture:
        """The underlying capture, for helpers that take one (for example ROI auto-detect)."""
        if self._cap is None:
            raise RuntimeError("FileSource is closed")
        return self._cap

    def __iter__(self) -> Iterator[Frame]:
        """Yield frames from index 0, stopping at ``frame_count`` or the first failed read."""
        cap = self.capture
        info = self.info
        total = info.frame_count or 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for index in range(total):
            ret, image = cap.read()
            if not ret:
                break
            yield Frame(image=image, index=index, timestamp=index / info.fps, source=info)

    def read_frame(self, index: int) -> np.ndarray | None:
        """Return the frame at ``index``, or None when it cannot be read."""
        cap = self.capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, image = cap.read()
        if not ret:
            return None
        return image

    def close(self) -> None:
        """Release the capture; safe to call more than once."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> FileSource:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
