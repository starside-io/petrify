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

"""Protocols for the pluggable engine components."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import numpy as np

from petrify.core.detections import DetectionKind, Detections
from petrify.core.frame import Frame, FrameMeta, SourceInfo


@runtime_checkable
class Detector(Protocol):
    """Produces part of a frame's detections."""

    provides: frozenset[DetectionKind]
    # True when the detector must see every frame in order to keep its internal
    # model correct (for example MOG2).
    stateful: bool

    def detect(self, frame: Frame) -> Detections:
        """Return detections with only the fields in ``provides`` set."""
        ...

    def reset(self) -> None:
        """Drop any internal state."""
        ...


@runtime_checkable
class Mode(Protocol):
    """Turns a frame and its detections into the output image."""

    name: str
    requires: frozenset[DetectionKind]

    def render(self, frame: Frame, detections: Detections, subject_present: bool) -> np.ndarray:
        """Return the output image.

        ``subject_present`` is the engine's decision after lookahead smoothing and
        cold start, not the raw detection.
        """
        ...


@runtime_checkable
class Stage(Protocol):
    """Post-processing step run after the mode on every frame."""

    def apply(self, frame: Frame, image: np.ndarray) -> np.ndarray:
        """Return the processed image."""
        ...


@runtime_checkable
class Sink(Protocol):
    """Consumes output images.

    A sink only ever receives the processed output image and the frame's metadata,
    never the input frame. ``abort`` must be safe after a failed ``open``.
    """

    # True when a failure of this sink must abort the whole run (for example a file
    # recording). A non-critical sink (for example a preview) that fails is aborted
    # and dropped while the run continues.
    critical: bool

    def open(self, info: SourceInfo) -> None: ...

    def write(self, image: np.ndarray, meta: FrameMeta) -> None: ...

    def close(self) -> None: ...

    def abort(self) -> None:
        """Discard any partial output; called instead of close when a run fails."""
        ...


@runtime_checkable
class Source(Protocol):
    """Yields frames in order."""

    info: SourceInfo

    def __iter__(self) -> Iterator[Frame]: ...

    def close(self) -> None: ...
