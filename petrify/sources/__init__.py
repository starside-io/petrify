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

"""Frame sources, and :func:`open_source` to pick one from a command-line spec."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from petrify.core.plugin import Source
from petrify.sources.camera import CameraSource
from petrify.sources.file import FileSource, display_spec
from petrify.sources.images import ImageFolderSource
from petrify.sources.push import PushSession
from petrify.sources.stream import StreamSource, redact_url

__all__ = [
    "CameraSource",
    "FileSource",
    "ImageFolderSource",
    "PushSession",
    "SourceKind",
    "StreamSource",
    "display_spec",
    "open_source",
    "source_kind",
]

SourceKind = Literal["camera", "stream", "images", "file"]

STREAM_SCHEMES = ("rtsp", "rtsps", "http", "https")

_CAMERA = re.compile(r"camera:([0-9]+)")
_URL_SCHEME = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*)://")
_GLOB_CHARS = ("*", "?", "[")


def source_kind(spec: str) -> SourceKind:
    """Classify a source spec without opening anything.

    ``camera:N`` (N a non-negative integer) is a webcam; ``rtsp``, ``rtsps``,
    ``http`` and ``https`` URLs are network streams; an existing directory, or a
    string with glob characters that is not an existing file, is an image folder;
    anything else is a video file path. A bare integer is a file path, never a
    camera index.

    Raises ``ValueError`` for a malformed ``camera:`` spec or a URL whose scheme
    is not supported; the message holds the spec only as :func:`display_spec` shows it.
    """
    if spec.startswith("camera:"):
        if _CAMERA.fullmatch(spec) is None:
            shown = display_spec(spec)
            raise ValueError(
                f"Invalid camera source {shown if shown != spec else repr(spec)}: "
                "use camera:N with N a non-negative integer"
            )
        return "camera"
    scheme = _URL_SCHEME.match(spec.strip())
    if scheme is not None:
        if scheme.group(1).lower() in STREAM_SCHEMES:
            return "stream"
        raise ValueError(
            f"Unsupported stream URL {redact_url(spec)}: supported schemes are "
            + ", ".join(STREAM_SCHEMES)
        )
    path = Path(spec)
    if path.is_dir():
        return "images"
    if any(ch in spec for ch in _GLOB_CHARS) and not path.is_file():
        return "images"
    return "file"


def open_source(
    spec: str,
    *,
    fps: float | None = None,
    capture_factory: Callable[[Any], Any] | None = None,
    read_timeout: float | None = None,
) -> Source:
    """Open the source a spec names (see :func:`source_kind` for the forms).

    ``fps`` sets the frame rate of an image folder (default 25). ``capture_factory``
    replaces ``cv2.VideoCapture`` for the live sources (tests). ``read_timeout``
    (seconds) makes a live source raise ``OSError`` when no frame arrives for that
    long; it raises ``ValueError`` for a file or an image folder. A missing video
    file raises ``FileNotFoundError``; for a bare integer the message suggests
    ``camera:N``. No error message holds a spec with ``@`` or ``://`` other than
    as :func:`display_spec` shows it.
    """
    kind = source_kind(spec)
    live_kwargs: dict[str, Any] = {}
    if capture_factory is not None:
        live_kwargs["capture_factory"] = capture_factory
    if read_timeout is not None:
        if kind not in ("camera", "stream"):
            # A file or folder never stalls; accepting it silently would hide a mistake.
            raise ValueError("read_timeout applies only to a camera or a stream source")
        live_kwargs["read_timeout"] = read_timeout
    if kind == "camera":
        return CameraSource(int(spec.partition(":")[2]), **live_kwargs)
    if kind == "stream":
        return StreamSource(spec, **live_kwargs)
    if kind == "images":
        try:
            return ImageFolderSource(spec) if fps is None else ImageFolderSource(spec, fps=fps)
        except (OSError, ValueError) as exc:
            shown = display_spec(spec)
            if shown == spec:
                raise
            # The message may hold the spec or one of its paths: replace it whole.
            raise type(exc)(f"Cannot open image source {shown}") from None
    if spec.strip().isdigit() and not Path(spec).exists():
        raise FileNotFoundError(
            f"Input file not found: {spec} (for a webcam, use camera:{spec.strip()})"
        )
    return FileSource(spec)
