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

"""Face boxes and five landmarks from the YuNet ONNX model via cv2.FaceDetectorYN."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np

from petrify.core.config import FaceYuNetConfig
from petrify.core.detections import Box, DetectionKind, Detections, Face
from petrify.core.frame import Frame
from petrify.models import FACE_YUNET_MODEL, load_model_bytes

MODEL = FACE_YUNET_MODEL
"""Pinned identity of the YuNet 2023mar face detection model."""

# One YuNet row: box (x, y, w, h), five landmark points (right eye, left eye,
# nose tip, right mouth corner, left mouth corner), then the score.
_ROW_LEN = 15
_LANDMARK_COUNT = 5

# Placeholder until the first frame sets the real size; YuNet needs a size at creation.
_INITIAL_SIZE = (320, 320)

# Smallest frame side the detector accepts. With OpenCV 4.13 a fresh net returns
# nondeterministic output for frames 32 pixels high (or less) and about 1280 wide
# (or more): absurd boxes with score 1.0, infinities, or no rows at all. A "no
# faces" answer there cannot be trusted, so such frames are refused on both axes.
MIN_FRAME_SIDE = 33


def _finite(value: float, what: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"YuNet returned a non-finite {what}: {out!r}")
    return out


def face_from_row(row: np.ndarray) -> Face:
    """Convert one YuNet output row into a :class:`Face` with five landmarks.

    The box is widened outward to whole pixels (floor of the start, ceiling of
    the end) so rounding never leaves part of the face uncovered. Clipping to
    the frame is left to :func:`~petrify.core.detections.check_detections`.

    Raises
    ------
    ValueError
        If the row does not hold 15 values, or holds a non-finite value or a
        negative box size. A broken row means the frame was not processed, so
        it must not be reported as fewer faces.
    """
    values = np.asarray(row)
    if values.shape != (_ROW_LEN,):
        raise ValueError(f"YuNet row must have shape ({_ROW_LEN},), got {values.shape}")
    x = _finite(values[0], "box x")
    y = _finite(values[1], "box y")
    w = _finite(values[2], "box width")
    h = _finite(values[3], "box height")
    if w < 0 or h < 0:
        raise ValueError(f"YuNet returned a negative box size: w={w} h={h}")
    x0, y0 = math.floor(x), math.floor(y)
    x1, y1 = math.ceil(x + w), math.ceil(y + h)
    score = _finite(values[14], "score")
    landmarks = tuple(
        (
            _finite(values[4 + 2 * i], "landmark x"),
            _finite(values[5 + 2 * i], "landmark y"),
        )
        for i in range(_LANDMARK_COUNT)
    )
    return Face(box=Box(x=x0, y=y0, w=x1 - x0, h=y1 - y0, score=score), landmarks=landmarks)


def faces_from_output(output: np.ndarray | None) -> tuple[Face, ...]:
    """Convert the ``faces`` array of ``FaceDetectorYN.detect`` into faces.

    OpenCV returns None when it finds no face; that maps to an empty tuple.

    Raises
    ------
    ValueError
        If the array is not N x 15, or any row is rejected by :func:`face_from_row`.
    """
    if output is None:
        return ()
    rows = np.asarray(output)
    if rows.ndim != 2 or rows.shape[1] != _ROW_LEN:
        raise ValueError(f"YuNet output must have shape (N, {_ROW_LEN}), got {rows.shape}")
    return tuple(face_from_row(row) for row in rows)


def _check_image(image: object) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise ValueError(f"frame image must be a numpy.ndarray, got {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"frame image must be H x W x 3 BGR, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"frame image must be uint8, got {image.dtype}")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"frame image must not be empty, got shape {image.shape}")
    if image.shape[0] < MIN_FRAME_SIDE or image.shape[1] < MIN_FRAME_SIDE:
        raise ValueError(
            f"frame image must be at least {MIN_FRAME_SIDE} pixels on each side for "
            f"face detection, got {image.shape[1]}x{image.shape[0]} (width x height)"
        )
    return image


class FaceYuNetDetector:
    """Detect faces and their five landmarks with the YuNet model.

    An instance is not thread-safe and must be used from one thread only.

    Parameters
    ----------
    model_bytes:
        Contents of the YuNet ONNX file, already verified against :data:`MODEL`
        (see :meth:`from_config`). The net is built from these bytes, so the
        file is never read again after the check.
    score_threshold, nms_threshold, top_k:
        YuNet settings, with the same ranges as :class:`FaceYuNetConfig`.

    Raises
    ------
    ValueError
        If a setting is out of range or ``model_bytes`` is not bytes.
    cv2.error
        If OpenCV cannot build a net from ``model_bytes``.
    """

    provides: ClassVar[frozenset[DetectionKind]] = frozenset(
        {DetectionKind.FACES, DetectionKind.LANDMARKS}
    )
    stateful: ClassVar[bool] = False

    def __init__(
        self,
        model_bytes: bytes,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        if not isinstance(model_bytes, (bytes, bytearray)) or not model_bytes:
            raise ValueError("model_bytes must be non-empty bytes")
        # Reuse the config's range checks so direct construction and config agree.
        FaceYuNetConfig(
            score_threshold=score_threshold, nms_threshold=nms_threshold, top_k=top_k
        )
        self._score_threshold = score_threshold
        self._nms_threshold = nms_threshold
        self._top_k = top_k
        buffer = np.frombuffer(bytes(model_bytes), dtype=np.uint8)
        self._net = cv2.FaceDetectorYN.create(
            "onnx",
            buffer,
            np.empty(0, dtype=np.uint8),
            _INITIAL_SIZE,
            score_threshold,
            nms_threshold,
            top_k,
        )
        if self._net is None:
            raise RuntimeError("cv2.FaceDetectorYN.create returned no detector")
        self._input_size: tuple[int, int] = _INITIAL_SIZE

    # Read-only: the net takes the settings at creation, so a later assignment
    # would look effective while changing nothing.
    @property
    def score_threshold(self) -> float:
        """Minimum face score passed to the net at creation."""
        return self._score_threshold

    @property
    def nms_threshold(self) -> float:
        """IoU above which overlapping boxes are suppressed, passed at creation."""
        return self._nms_threshold

    @property
    def top_k(self) -> int:
        """Candidates kept before non-maximum suppression, passed at creation."""
        return self._top_k

    @classmethod
    def from_config(
        cls, config: FaceYuNetConfig, dirs: Sequence[Path] = ()
    ) -> FaceYuNetDetector:
        """Load and verify the model, then build a detector.

        ``config.model_path`` is used when set, otherwise ``dirs`` (usually from
        :func:`petrify.models.model_dirs`) are searched in order.

        Raises
        ------
        petrify.models.ModelError
            If the file is missing, unreadable, or fails the size or SHA-256 check.
        """
        data = load_model_bytes(MODEL, path=config.model_path, dirs=dirs)
        return cls(
            data,
            score_threshold=config.score_threshold,
            nms_threshold=config.nms_threshold,
            top_k=config.top_k,
        )

    def detect(self, frame: Frame) -> Detections:
        """Return detections with only ``faces`` set, each face carrying five landmarks.

        Raises
        ------
        ValueError
            If the image is not an H x W x 3 uint8 array with both sides at
            least :data:`MIN_FRAME_SIDE`, or YuNet's output is malformed.
        cv2.error
            If OpenCV fails on the frame. Errors propagate so the engine fails
            the frame closed instead of reporting no faces.
        """
        image = _check_image(frame.image)
        height, width = image.shape[:2]
        size = (width, height)
        if size != self._input_size:
            self._net.setInputSize(size)
            self._input_size = size
        _, output = self._net.detect(image)
        return Detections(faces=faces_from_output(output))

    def reset(self) -> None:
        """No-op: the input size is re-set on each frame and is not detection state."""
