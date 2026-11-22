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

"""Person segmentation mask from the OpenCV zoo PP-HumanSeg model."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np

from petrify.core.config import MAX_DILATE_PX, SegmentConfig
from petrify.core.detections import DetectionKind, Detections
from petrify.core.frame import Frame
from petrify.models import ModelSpec, load_model_bytes

MODEL = ModelSpec(
    key="human_segmentation_pphumanseg_2023mar",
    filename="human_segmentation_pphumanseg_2023mar.onnx",
    sha256="552d8a984054e59b5d773d24b9b12022b22046ceb2bbc4c9aaeaceb36a9ddf24",
    size=6163938,
    source_url="https://github.com/opencv/opencv_zoo/tree/main/models/human_segmentation_pphumanseg",
    license="Apache-2.0",
)
"""Pinned identity of the PP-HumanSeg model file."""

INPUT_SIZE = 192
"""Side of the square model input, in pixels."""

PERSON_CHANNEL = 1
"""Output channel that holds the person probability (channel 0 is background)."""

# The model ends in a softmax. Its two channels sum to 1 up to float rounding; a
# larger deviation means the file or the runtime is not what was verified, so the
# frame is refused rather than read with the wrong meaning.
_PROB_TOLERANCE = 1e-3


def _check_image(image: object) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise ValueError(f"frame image must be a numpy.ndarray, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise ValueError(f"frame image dtype must be uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"frame image must have shape (H, W, 3), got {image.shape}")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError(f"frame image must not be empty, got shape {image.shape}")
    return image


def preprocess(image: np.ndarray) -> np.ndarray:
    """Return the model input blob for a BGR uint8 ``(H, W, 3)`` image.

    The blob has shape ``(1, 3, 192, 192)``, dtype float32, RGB channel order,
    and values normalized to ``[-1, 1]`` (scaled to ``[0, 1]``, then mean 0.5 and
    std 0.5).

    Raises
    ------
    ValueError
        If ``image`` is not a non-empty ``(H, W, 3)`` uint8 array.
    """
    image = _check_image(image)
    resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    normalized = (rgb / 255.0 - 0.5) / 0.5
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[np.newaxis])


def person_probability(output: object) -> np.ndarray:
    """Return the ``(h, w)`` float32 person probability map from a raw model output.

    ``output`` must have shape ``(1, 2, h, w)`` with finite values that are
    per-pixel probabilities over (background, person).

    Raises
    ------
    ValueError
        If the shape is wrong, a value is not finite, or the values are not
        probabilities.
    """
    if not isinstance(output, np.ndarray):
        raise ValueError(f"model output must be a numpy.ndarray, got {type(output).__name__}")
    if output.ndim != 4 or output.shape[0] != 1 or output.shape[1] != 2:
        raise ValueError(f"model output must have shape (1, 2, h, w), got {output.shape}")
    if output.shape[2] < 1 or output.shape[3] < 1:
        raise ValueError(f"model output must not be empty, got shape {output.shape}")
    if not np.issubdtype(output.dtype, np.floating):
        raise ValueError(f"model output must be floating point, got {output.dtype}")
    probs = output[0].astype(np.float32)
    if not np.isfinite(probs).all():
        raise ValueError("model output holds non-finite values")
    if probs.min() < -_PROB_TOLERANCE or probs.max() > 1.0 + _PROB_TOLERANCE:
        raise ValueError(
            f"model output is not a probability map: values span "
            f"[{float(probs.min())}, {float(probs.max())}]"
        )
    if np.abs(probs.sum(axis=0) - 1.0).max() > _PROB_TOLERANCE:
        raise ValueError("model output channels do not sum to 1")
    return np.ascontiguousarray(np.clip(probs[PERSON_CHANNEL], 0.0, 1.0))


def _overlap_rows(n_out: int, n_src: int) -> tuple[np.ndarray, np.ndarray]:
    """Return, per output index, the half-open source range its area overlaps.

    Output pixel ``o`` spans ``[o * n_src / n_out, (o + 1) * n_src / n_out)`` in
    source units. Integer arithmetic keeps the bounds exact for every size.
    """
    o = np.arange(n_out, dtype=np.int64)
    lo = (o * n_src) // n_out
    hi = -((-(o + 1) * n_src) // n_out)
    return lo, np.maximum(hi, lo + 1)


def footprint_cover(source: np.ndarray, height: int, width: int) -> np.ndarray:
    """Return the ``(height, width)`` bool mask of output pixels that overlap a True source pixel.

    Every True source pixel has its whole footprint covered: each output pixel
    whose area overlaps the source pixel's area, which is at least one output
    pixel when the map shrinks.
    """
    src_h, src_w = source.shape
    lo_y, hi_y = _overlap_rows(height, src_h)
    lo_x, hi_x = _overlap_rows(width, src_w)
    # Prefix sums turn "any True source pixel in [lo, hi)" into one subtraction.
    rows = np.zeros((src_h + 1, src_w), dtype=np.int32)
    np.cumsum(source, axis=0, out=rows[1:])
    by_row = (rows[hi_y] - rows[lo_y]) > 0
    cols = np.zeros((height, src_w + 1), dtype=np.int32)
    np.cumsum(by_row, axis=1, out=cols[:, 1:])
    return (cols[:, hi_x] - cols[:, lo_x]) > 0


def resize_mask(prob: np.ndarray, height: int, width: int, threshold: float) -> np.ndarray:
    """Threshold a probability map and resize it to a bool ``(height, width)`` mask.

    The result is the union of two masks. The interpolated map at or above
    ``threshold`` gives smooth edges when the map grows. The footprint cover of
    the source pixels at or above ``threshold`` guarantees that no pixel the
    model flagged is lost, which interpolation alone does: it averages an
    isolated pixel below the threshold, in both directions.
    """
    smooth = cv2.resize(prob, (width, height), interpolation=cv2.INTER_LINEAR) >= threshold
    return smooth | footprint_cover(prob >= threshold, height, width)


def dilate_mask(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    """Grow ``mask`` by every pixel within Euclidean distance ``dilate_px`` of it.

    A distance transform keeps the cost independent of ``dilate_px``, where a
    large elliptical kernel would cost seconds per frame. The result is a
    superset of ``mask``: a mask pixel has distance 0.
    """
    if dilate_px == 0 or not mask.any():
        return mask
    if mask.all():
        return mask
    background = np.ascontiguousarray(~mask).astype(np.uint8)
    distance = cv2.distanceTransform(background, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return mask | (distance <= dilate_px)


def postprocess(
    output: object,
    height: int,
    width: int,
    *,
    threshold: float,
    dilate_px: int,
    min_area_share: float,
) -> np.ndarray:
    """Turn a raw model output into a bool ``(height, width)`` person mask.

    A model pixel is a person when its probability is at least ``threshold``, so
    a tie counts as a person, and it is resized with :func:`resize_mask`, so
    every such pixel stays covered at frame resolution.

    ``min_area_share`` of 0 turns the area rule off and the mask is never
    dropped. Above 0, a mask that covers less than that share of the frame is
    replaced by an all-False mask (computed, no person found). That hides real
    detections when the model flags only part of a person, such as the head or
    the legs, so the frame is left uncovered. The share is measured before
    dilation, so the safety margin does not change which frames count as
    having a person.

    Raises
    ------
    ValueError
        As :func:`person_probability`, or for a non-positive frame size.
    """
    if height < 1 or width < 1:
        raise ValueError(f"frame size must be positive, got {height}x{width}")
    mask = resize_mask(person_probability(output), height, width, threshold)
    if min_area_share > 0 and np.count_nonzero(mask) < min_area_share * mask.size:
        return np.zeros((height, width), dtype=np.bool_)
    return dilate_mask(mask, dilate_px)


def _check_number(name: str, value: object, *, low_closed: bool, high_closed: bool) -> float:
    ok = (
        type(value) in (int, float)
        and math.isfinite(value)  # type: ignore[arg-type]
        and (0 <= value if low_closed else 0 < value)  # type: ignore[operator]
        and (value <= 1 if high_closed else value < 1)  # type: ignore[operator]
    )
    if not ok:
        bound = f"{'[' if low_closed else '('}0, 1{']' if high_closed else ')'}"
        raise ValueError(f"{name} must be a number in {bound}, got {value!r}")
    return float(value)  # type: ignore[arg-type]


class PersonSegmenter:
    """Person segmentation with the OpenCV zoo PP-HumanSeg ONNX model.

    Provides a bool mask at the frame's full resolution. Every failure raises,
    so the engine fails the frame closed instead of reading it as "no person".

    Parameters
    ----------
    model:
        The model file's bytes, already verified against :data:`MODEL`. They
        are parsed in memory; no path is read.
    threshold:
        Person probability, in (0, 1), from which a pixel is a person.
    dilate_px:
        Safety margin grown around the mask, from 0 to ``MAX_DILATE_PX`` pixels.
    min_area_share:
        Mask share of the frame, in [0, 1], below which the mask is dropped and
        the frame counts as having no person. 0 (the default) turns the rule
        off. Above 0 it hides real detections when the model flags only part
        of a person, such as the head or the legs, and that frame then passes
        uncovered.
    """

    provides: ClassVar[frozenset[DetectionKind]] = frozenset({DetectionKind.MASK})
    stateful: ClassVar[bool] = False

    def __init__(
        self,
        model: bytes,
        *,
        threshold: float = 0.5,
        dilate_px: int = 0,
        min_area_share: float = 0.0,
    ) -> None:
        if not isinstance(model, (bytes, bytearray)) or not model:
            raise ValueError("model must be the non-empty bytes of the model file")
        self.threshold = _check_number(
            "threshold", threshold, low_closed=False, high_closed=False
        )
        if type(dilate_px) is not int or not 0 <= dilate_px <= MAX_DILATE_PX:
            raise ValueError(
                f"dilate_px must be a whole number from 0 to {MAX_DILATE_PX}, got {dilate_px!r}"
            )
        self.dilate_px = dilate_px
        self.min_area_share = _check_number(
            "min_area_share", min_area_share, low_closed=True, high_closed=True
        )
        self._net = cv2.dnn.readNetFromONNX(np.frombuffer(bytes(model), dtype=np.uint8))

    @classmethod
    def from_config(cls, config: SegmentConfig, model_dirs: Sequence[Path]) -> PersonSegmenter:
        """Build a segmenter from ``config``, loading and verifying the model file.

        ``config.model_path``, when set, is the only file tried; otherwise
        ``model_dirs`` (see :func:`petrify.models.model_dirs`) are searched.

        Raises
        ------
        petrify.models.ModelError
            If the file is missing, unreadable, or fails the SHA-256 check.
        """
        data = load_model_bytes(MODEL, path=config.model_path, dirs=model_dirs)
        return cls(
            data,
            threshold=config.threshold,
            dilate_px=config.dilate_px,
            min_area_share=config.min_area_share,
        )

    def detect(self, frame: Frame) -> Detections:
        """Return detections with only ``mask`` set.

        Raises
        ------
        ValueError
            If the image is not ``(H, W, 3)`` uint8 or the model output is malformed.
        cv2.error
            If OpenCV fails to run the model.
        """
        image = _check_image(frame.image)
        self._net.setInput(preprocess(image))
        output = self._net.forward()
        height, width = image.shape[:2]
        mask = postprocess(
            output,
            height,
            width,
            threshold=self.threshold,
            dilate_px=self.dilate_px,
            min_area_share=self.min_area_share,
        )
        return Detections(mask=mask)

    def reset(self) -> None:
        """No-op: the segmenter keeps no per-frame state."""
