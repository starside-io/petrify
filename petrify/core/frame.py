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

"""Frame and source metadata types shared by every engine component."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SourceInfo:
    """Static properties of a frame source."""

    fps: float
    width: int
    height: int
    frame_count: int | None  # None when unknown, for example a live stream.
    seekable: bool
    live: bool
    uri: str | None = None


@dataclass(frozen=True)
class FrameMeta:
    """A frame's position and origin, without its pixels (what sinks receive)."""

    index: int  # 0-based position in the stream.
    timestamp: float  # Seconds from stream start.
    source: SourceInfo


@dataclass(frozen=True, eq=False)
class Frame:
    """One decoded frame: BGR uint8 image (H x W x 3) plus its position in the stream."""

    image: np.ndarray
    index: int  # 0-based position in the stream.
    timestamp: float  # Seconds from stream start.
    source: SourceInfo

    @property
    def meta(self) -> FrameMeta:
        """This frame's metadata, without the image."""
        return FrameMeta(index=self.index, timestamp=self.timestamp, source=self.source)
