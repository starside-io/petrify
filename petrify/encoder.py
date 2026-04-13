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

"""FFmpeg re-encode wrapper: takes the raw processed video and muxes original audio."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def ffmpeg_encode(
    video_path: str | Path,
    original_audio_source: str | Path,
    output_path: str | Path,
    codec: str = "libx264",
    crf: int = 23,
    preset: str = "medium",
    hwaccel: str | None = None,
) -> None:
    """Re-encode *video_path* with *codec* and mux audio from *original_audio_source*.

    The audio stream is copied as-is (``-c:a copy``).  If the original source
    has no audio track, the ``-map 1:a?`` flag makes FFmpeg silently skip it
    rather than failing.

    Parameters
    ----------
    video_path:
        Path to the intermediate raw/processed video file written by
        ``cv2.VideoWriter``.
    original_audio_source:
        Path to the *original* input video whose audio tracks should be
        preserved in the output.
    output_path:
        Destination file path.  Any existing file at this location will be
        overwritten (``-y``).
    codec:
        FFmpeg video codec name (e.g. ``libx264``, ``libx265``, ``h264_videotoolbox``).
    crf:
        Constant Rate Factor quality setting (lower = better quality / larger
        file).  Valid range depends on the codec; 0–51 for libx264/libx265.
    preset:
        FFmpeg preset string (e.g. ``ultrafast``, ``medium``, ``slow``).
        Ignored for some hardware-accelerated codecs.
    hwaccel:
        Optional FFmpeg ``-hwaccel`` value (e.g. ``cuda``, ``videotoolbox``).
        When set, ``-hwaccel <value>`` is prepended to the input arguments.

    Raises
    ------
    FileNotFoundError
        If the ``ffmpeg`` binary is not found on ``PATH``.
    subprocess.CalledProcessError
        If FFmpeg exits with a non-zero return code.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        # Fallback: check common Homebrew locations not always on PATH
        for candidate in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
            if Path(candidate).is_file():
                ffmpeg_bin = candidate
                break
    if ffmpeg_bin is None:
        raise FileNotFoundError(
            "ffmpeg binary not found on PATH. "
            "Install FFmpeg: https://ffmpeg.org/download.html"
        )

    cmd: list[str] = [ffmpeg_bin, "-y"]

    # Optional hardware acceleration (must come before the first -i).
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]

    cmd += [
        "-i", str(video_path),           # input 0: processed video (no audio)
        "-i", str(original_audio_source), # input 1: original video (for audio)
        "-map", "0:v:0",                  # take video from input 0
        "-map", "1:a?",                   # take all audio from input 1, optional
        "-c:v", codec,
        "-crf", str(crf),
        "-preset", preset,
        "-c:a", "copy",
        str(output_path),
    ]

    logger.info("FFmpeg command: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            logger.debug("FFmpeg stdout:\n%s", result.stdout)
        if result.stderr:
            # FFmpeg writes progress/info to stderr even on success.
            logger.debug("FFmpeg stderr:\n%s", result.stderr)
    except subprocess.CalledProcessError as exc:
        logger.error("FFmpeg failed (exit code %d).", exc.returncode)
        if exc.stderr:
            logger.error("FFmpeg stderr:\n%s", exc.stderr)
        raise
