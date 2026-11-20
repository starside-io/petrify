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

"""Video file sink: intermediate mp4v file, then FFmpeg re-encode with original audio."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from petrify.core.config import EncodeConfig
from petrify.core.frame import Frame, SourceInfo
from petrify.encoder import ffmpeg_encode

logger = logging.getLogger(__name__)


class FileSink:
    """Write frames to a temporary mp4v file and re-encode it on ``close``.

    ``abort`` discards the temporary file without encoding. ``close`` and
    ``abort`` are idempotent and do nothing when ``open`` never succeeded.
    """

    def __init__(
        self,
        output_path: str | Path,
        audio_source: str | Path,
        encode: EncodeConfig,
    ) -> None:
        self._output_path = Path(output_path)
        self._audio_source = Path(audio_source)
        self._encode = encode
        self._writer: cv2.VideoWriter | None = None
        self._temp_path: str | None = None

    def open(self, info: SourceInfo) -> None:
        """Create the temporary file and open the intermediate writer."""
        fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        logger.info("Compositing frames -> %s", temp_path)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_path, fourcc, info.fps, (info.width, info.height))
        if not writer.isOpened():
            os.unlink(temp_path)
            raise OSError("cv2.VideoWriter failed to open the intermediate file.")

        self._writer = writer
        self._temp_path = temp_path

    def write(self, frame: Frame, image: np.ndarray) -> None:
        """Append one output image."""
        if self._writer is None:
            raise RuntimeError("FileSink is not open")
        self._writer.write(image)

    def close(self) -> None:
        """Release the writer, re-encode with FFmpeg, then remove the temporary file."""
        if self._temp_path is None:
            return
        temp_path = self._temp_path
        self._temp_path = None
        try:
            self._release_writer()
            logger.info("Re-encoding with FFmpeg -> %s", self._output_path)
            ffmpeg_encode(
                video_path=temp_path,
                original_audio_source=self._audio_source,
                output_path=self._output_path,
                codec=self._encode.codec,
                crf=self._encode.crf,
                preset=self._encode.preset,
                hwaccel=self._encode.hwaccel,
            )
        finally:
            _remove_temp(temp_path)

        logger.info("Done. Output written to: %s", self._output_path)

    def abort(self) -> None:
        """Release the writer and remove the temporary file without encoding."""
        if self._temp_path is None:
            return
        temp_path = self._temp_path
        self._temp_path = None
        try:
            self._release_writer()
        finally:
            _remove_temp(temp_path)

    def _release_writer(self) -> None:
        writer, self._writer = self._writer, None
        if writer is not None:
            writer.release()


def _remove_temp(temp_path: str) -> None:
    try:
        os.unlink(temp_path)
    except OSError as exc:
        logger.warning("Could not remove temp file %s: %s", temp_path, exc)
