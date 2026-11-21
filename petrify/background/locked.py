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

"""Locked background strategy: one reference frame chosen before processing starts."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
from tqdm import tqdm

from petrify.core.config import BackgroundConfig, PersonHogConfig
from petrify.core.frame import Frame
from petrify.detectors.person_hog import PersonDetector
from petrify.sources.file import FileSource

logger = logging.getLogger(__name__)


class _PersonCheck(Protocol):
    def is_person(self, frame: np.ndarray) -> bool: ...


def find_clean_background(
    source: FileSource,
    person_detector: _PersonCheck,
    scan_step: int,
) -> tuple[np.ndarray, int]:
    """Return the latest scanned frame with no person detected, and its index.

    Visits every ``scan_step``-th frame. Falls back to frame 0 (index 0) when
    every sampled frame contains a person, and raises ``ValueError`` when frame 0
    cannot be read either.
    """
    logger.info("Auto-background scan: checking every %d frames for a clean frame...", scan_step)

    best_frame: np.ndarray | None = None
    best_idx: int = -1
    total_frames = source.info.frame_count or 0

    for idx in tqdm(
        range(0, total_frames, scan_step),
        desc="Background scan",
        unit="fr",
        dynamic_ncols=True,
        leave=False,
    ):
        frame = source.read_frame(idx)
        if frame is None:
            continue
        if not person_detector.is_person(frame):
            # Keep updating: we want the LATEST clean frame.
            best_idx = idx
            best_frame = frame.copy()

    if best_frame is None:
        logger.warning(
            "Background scan found no person-free frame, falling back to frame 0. "
            "Consider specifying --background-frame manually."
        )
        best_frame = source.read_frame(0)
        if best_frame is None:
            raise ValueError("Cannot read any frame for the background reference.")
        best_idx = 0
    else:
        logger.info(
            "Auto background: frame %d  (%.1fs into video)",
            best_idx,
            best_idx / source.info.fps,
        )

    return best_frame, best_idx


def locked_background(
    source: FileSource,
    config: BackgroundConfig,
    scan_person_config: PersonHogConfig,
) -> tuple[np.ndarray, int]:
    """Return the locked background frame and its index.

    Uses ``config.frame_index`` when set, otherwise pre-scans with a HOG person
    detector built from ``scan_person_config``. Detector errors propagate.
    """
    if config.frame_index is not None:
        bg_idx = config.frame_index
        frame = source.read_frame(bg_idx)
        if frame is None:
            raise ValueError(
                f"Cannot read background frame at index {bg_idx} "
                f"(video has ~{source.info.frame_count} frames)."
            )
        logger.info("Background reference: frame %d (manual)", bg_idx)
        return frame, bg_idx

    detector = PersonDetector.from_config(scan_person_config)
    return find_clean_background(source, detector, config.scan_step)


class LockedBackground:
    """A fixed background image exposed as a ``BackgroundModel``.

    ``current`` always returns the image, ``update`` does nothing, and ``version``
    stays 0. The caller must not mutate the image after handing it over.
    """

    def __init__(self, image: np.ndarray) -> None:
        self._image = image
        self.version = 0

    def current(self) -> np.ndarray:
        """Return the locked image."""
        return self._image

    def update(self, frame: Frame, subject_present: bool) -> None:
        """No-op: the locked background never changes."""
