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

"""Person boxes from the NanoDet-Plus object detector (OpenCV zoo, COCO classes).

The model is loaded from bytes that :func:`petrify.models.load_model` has
already checked against :data:`MODEL`, so the file on disk is never read again
after verification. Pre and post processing are small pure functions
(:func:`letterbox`, :func:`decode`, :func:`select_persons`, :func:`map_to_frame`,
:func:`pad_boxes`, :func:`clip_to_frame`) so they can be tested without the model.

Boxes are model estimates, not exact outlines. Overlapping candidates are
merged into their union rather than dropped, and every box is padded on each
side (10% of its width and 7% of its height by default) to cover the
undershoot measured on compact poses. Extended limbs can still stick out of a
box (measured up to 26% of the box width). Recall drops for small people:
none was found below about 24 network pixels tall (about 110 px on a 1080p
frame), and wide frames shrink people further through letterboxing.

Every failure raises: a frame that could not be processed never yields "no
person", because the engine fails such a frame closed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np

from petrify.core.config import PersonDnnConfig
from petrify.core.detections import Box, DetectionKind, Detections
from petrify.core.frame import Frame
from petrify.models import PERSON_DNN_MODEL, load_model

MODEL = PERSON_DNN_MODEL

INPUT_SIZE = 416
STRIDES = (8, 16, 32)
REG_MAX = 7
NUM_CLASSES = 80
PERSON_CLASS = 0

# BGR order: the model was trained on BGR images normalized with these values.
_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
_STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)
_BINS = REG_MAX + 1
_PROJECT = np.arange(_BINS, dtype=np.float32)


class NanoDetOutputError(RuntimeError):
    """The network returned outputs that do not match the expected layout or values."""


@dataclass(frozen=True)
class LetterboxGeometry:
    """Where a frame sits inside the square network input.

    ``scale_x`` and ``scale_y`` are the actual resize ratios (input pixels per
    frame pixel) after rounding the resized size to whole pixels, so mapping
    back is exact for the resized content.
    """

    frame_w: int
    frame_h: int
    left: int
    top: int
    content_w: int
    content_h: int

    @property
    def scale_x(self) -> float:
        return self.content_w / self.frame_w

    @property
    def scale_y(self) -> float:
        return self.content_h / self.frame_h


def _check_image(image: object) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise ValueError(f"frame image must be a numpy array, got {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"frame image must have shape (H, W, 3), got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"frame image must have dtype uint8, got {image.dtype}")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError(f"frame image must not be empty, got {image.shape}")
    return image


def letterbox_geometry(frame_w: int, frame_h: int, size: int = INPUT_SIZE) -> LetterboxGeometry:
    """Return the placement of a ``frame_w`` by ``frame_h`` frame in a ``size`` square.

    The aspect ratio is kept and the content is centered. Each side of the
    content is at least one pixel, so very elongated frames still map to a
    valid input.
    """
    if frame_w < 1 or frame_h < 1:
        raise ValueError(f"frame size must be at least 1x1, got {frame_w}x{frame_h}")
    scale = min(size / frame_w, size / frame_h)
    content_w = min(size, max(1, round(frame_w * scale)))
    content_h = min(size, max(1, round(frame_h * scale)))
    return LetterboxGeometry(
        frame_w=frame_w,
        frame_h=frame_h,
        left=(size - content_w) // 2,
        top=(size - content_h) // 2,
        content_w=content_w,
        content_h=content_h,
    )


SEPARABLE_MIN_SHRINK = 3
"""Smallest shrink factor, on each axis, for which :func:`letterbox` resizes one axis at a time."""

def _separable_area(geom: LetterboxGeometry) -> bool:
    """True when :func:`letterbox` may use :func:`_area_by_axis` instead of one resize.

    The single INTER_AREA call is slow for non-integer factors (13.6 ms of the
    detector's 46 ms at 1080p); resizing one axis at a time gives the same
    bytes in 4.5 ms, saving 9.3 ms per 1080p frame (measured on every frame of
    the test clips). The faster path is used only where it was shown to give
    the same bytes and to be faster:

    - each axis shrinks at least :data:`SEPARABLE_MIN_SHRINK` times: below that
      it was not faster (640x480 took 1.7 ms against 1.5 ms);
    - neither axis shrinks by a whole factor. Each one-axis pass keeps the
      other axis at scale 1, so a whole factor on either axis sends that pass
      to OpenCV's integer area path, which sums in another way: on such sizes
      (for example 1248x484, 3339x872, 3840x13) results came out one level
      apart. With both factors fractional, both passes run OpenCV's general
      area code with the same weights in the same order as the single call,
      and the bytes matched on every size tried.
    """
    w, h = geom.frame_w, geom.frame_h
    cw, ch = geom.content_w, geom.content_h
    return (
        w >= SEPARABLE_MIN_SHRINK * cw
        and h >= SEPARABLE_MIN_SHRINK * ch
        and w % cw != 0
        and h % ch != 0
    )


def _area_by_axis(image: np.ndarray, content_w: int, content_h: int) -> np.ndarray:
    """Area resize ``image`` to ``content_w`` by ``content_h``, one axis at a time.

    Area weights are a product of a column weight and a row weight, so two
    one-axis INTER_AREA resizes compute the same weighted sums as one call.
    The intermediate is float32 so it is not rounded to whole levels between
    the two passes; ``convertScaleAbs`` then rounds and saturates once, as the
    single uint8 call does.
    """
    columns = cv2.resize(
        image.astype(np.float32), (content_w, image.shape[0]), interpolation=cv2.INTER_AREA
    )
    rows = cv2.resize(columns, (content_w, content_h), interpolation=cv2.INTER_AREA)
    return cv2.convertScaleAbs(rows)


def letterbox(image: np.ndarray, size: int = INPUT_SIZE) -> tuple[np.ndarray, LetterboxGeometry]:
    """Resize ``image`` into a black ``size`` square, keeping the aspect ratio.

    Raises
    ------
    ValueError
        If ``image`` is not a non-empty H x W x 3 uint8 array.
    """
    image = _check_image(image)
    geom = letterbox_geometry(image.shape[1], image.shape[0], size)
    if _separable_area(geom):
        resized = _area_by_axis(image, geom.content_w, geom.content_h)
    else:
        shrinking = geom.content_w < geom.frame_w or geom.content_h < geom.frame_h
        resized = cv2.resize(
            image,
            (geom.content_w, geom.content_h),
            interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR,
        )
    out = np.zeros((size, size, 3), dtype=np.uint8)
    out[geom.top : geom.top + geom.content_h, geom.left : geom.left + geom.content_w] = resized
    return out, geom


def to_blob(square: np.ndarray) -> np.ndarray:
    """Normalize a letterboxed BGR image into a 1 x 3 x H x W float32 blob."""
    normalized = (square.astype(np.float32) - _MEAN) / _STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[np.newaxis])


def _anchor_centers(stride: int, size: int = INPUT_SIZE) -> np.ndarray:
    cells = math.ceil(size / stride)
    ys, xs = np.mgrid[0:cells, 0:cells].astype(np.float32)
    # Cell centers as the OpenCV zoo NanoDet wrapper places them.
    offset = 0.5 * (stride - 1)
    return np.column_stack((xs.ravel() * stride + offset, ys.ravel() * stride + offset))


def _expected_count(stride: int, size: int = INPUT_SIZE) -> int:
    return math.ceil(size / stride) ** 2


def _pick_output(outputs: Sequence[np.ndarray], count: int, channels: int) -> np.ndarray:
    matches = [o for o in outputs if o.shape == (1, count, channels)]
    if len(matches) != 1:
        raise NanoDetOutputError(
            f"expected exactly one output of shape (1, {count}, {channels}), "
            f"found {len(matches)} among {[o.shape for o in outputs]}"
        )
    return matches[0][0]


def decode(outputs: Sequence[np.ndarray], size: int = INPUT_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Decode the six raw outputs into person boxes and person scores.

    Outputs are matched by shape, not by position: per stride one class score
    tensor ``(1, N, 80)`` and one box distribution tensor ``(1, N, 32)``. Each
    box side is the expectation of a softmax over ``REG_MAX + 1`` bins, times
    the stride, measured from the anchor center.

    Returns
    -------
    boxes, scores:
        ``(M, 4)`` float32 boxes as ``x1, y1, x2, y2`` in network input
        pixels, and ``(M,)`` float32 person scores, one per anchor.

    Raises
    ------
    NanoDetOutputError
        If the outputs do not match the expected count and shapes, hold a
        non-finite value, or a class score lies outside [0, 1].
    """
    arrays = [np.asarray(o) for o in outputs]
    expected = 2 * len(STRIDES)
    if len(arrays) != expected:
        raise NanoDetOutputError(f"expected {expected} outputs, got {len(arrays)}")
    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    for stride in STRIDES:
        count = _expected_count(stride, size)
        cls = _pick_output(arrays, count, NUM_CLASSES)
        reg = _pick_output(arrays, count, 4 * _BINS)
        if not (np.isfinite(cls).all() and np.isfinite(reg).all()):
            raise NanoDetOutputError(f"non-finite value in the stride {stride} outputs")
        if cls.min() < 0.0 or cls.max() > 1.0:
            raise NanoDetOutputError(
                f"class scores at stride {stride} lie outside [0, 1]: "
                f"min {float(cls.min())}, max {float(cls.max())}"
            )
        logits = reg.astype(np.float32).reshape(count, 4, _BINS)
        logits = logits - logits.max(axis=2, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=2, keepdims=True)
        dist = (weights @ _PROJECT) * stride
        centers = _anchor_centers(stride, size)
        all_boxes.append(
            np.column_stack(
                (
                    centers[:, 0] - dist[:, 0],
                    centers[:, 1] - dist[:, 1],
                    centers[:, 0] + dist[:, 2],
                    centers[:, 1] + dist[:, 3],
                )
            )
        )
        all_scores.append(cls[:, PERSON_CLASS].astype(np.float32))
    return (
        np.concatenate(all_boxes).astype(np.float32),
        np.concatenate(all_scores).astype(np.float32),
    )


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(box[0], others[:, 0])
    iy1 = np.maximum(box[1], others[:, 1])
    ix2 = np.minimum(box[2], others[:, 2])
    iy2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(ix2 - ix1, 0.0, None) * np.clip(iy2 - iy1, 0.0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = area + areas - inter
    out = np.zeros(len(others), dtype=np.float64)
    np.divide(inter, union, out=out, where=union > 0)
    return out


def select_persons(
    boxes: np.ndarray, scores: np.ndarray, score_threshold: float, nms_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Keep boxes with ``score >= score_threshold`` and merge overlapping ones.

    The person score is read on its own, not only when person is the top
    class: a person standing next to a car must not be dropped because the
    car scored higher at that anchor.

    Greedy non-maximum suppression runs over the candidates in score order:
    a candidate whose IoU with a kept box is above ``nms_threshold`` is not
    returned on its own. Suppression only removes duplicates, never area:
    each kept box grows to the bounding box of itself and every candidate it
    suppressed, so the returned boxes together contain every candidate that
    passed the score threshold. IoU is measured against the kept box as the
    model predicted it, not against the grown box, so growth does not cascade.

    Returns
    -------
    boxes, scores:
        The ``x1, y1, x2, y2`` boxes (grown as above) and the score of each
        kept candidate, highest score first.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    keep = scores >= score_threshold
    boxes = boxes[keep].astype(np.float64)
    scores = scores[keep]
    order = np.argsort(-scores, kind="stable")
    boxes, scores = boxes[order], scores[order]
    remaining = np.ones(len(scores), dtype=bool)
    out_boxes: list[np.ndarray] = []
    out_scores: list[float] = []
    for i in range(len(scores)):
        if not remaining[i]:
            continue
        remaining[i] = False
        later = np.flatnonzero(remaining)
        suppressed = later[_iou(boxes[i], boxes[later]) > nms_threshold]
        remaining[suppressed] = False
        group = boxes[np.concatenate(([i], suppressed))]
        out_boxes.append(
            np.array([group[:, 0].min(), group[:, 1].min(), group[:, 2].max(), group[:, 3].max()])
        )
        out_scores.append(float(scores[i]))
    if not out_boxes:
        return np.zeros((0, 4), dtype=np.float64), np.zeros((0,), dtype=np.float32)
    return np.array(out_boxes, dtype=np.float64), np.array(out_scores, dtype=np.float32)


def map_to_frame(boxes: np.ndarray, geom: LetterboxGeometry) -> np.ndarray:
    """Map ``x1, y1, x2, y2`` boxes from network input pixels to frame pixels.

    The result is not clipped: a box reaching into the letterbox padding maps
    to coordinates outside the frame. Use :func:`clip_to_frame` after padding.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    out = np.empty_like(boxes)
    out[:, 0] = (boxes[:, 0] - geom.left) / geom.scale_x
    out[:, 2] = (boxes[:, 2] - geom.left) / geom.scale_x
    out[:, 1] = (boxes[:, 1] - geom.top) / geom.scale_y
    out[:, 3] = (boxes[:, 3] - geom.top) / geom.scale_y
    return out


def pad_boxes(boxes: np.ndarray, pad_x: float, pad_y: float) -> np.ndarray:
    """Grow each ``x1, y1, x2, y2`` box by ``pad_x`` of its width and ``pad_y``
    of its height on each side."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    dx = (boxes[:, 2] - boxes[:, 0]) * pad_x
    dy = (boxes[:, 3] - boxes[:, 1]) * pad_y
    return np.column_stack((boxes[:, 0] - dx, boxes[:, 1] - dy, boxes[:, 2] + dx, boxes[:, 3] + dy))


def clip_to_frame(boxes: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
    """Clip ``x1, y1, x2, y2`` boxes to ``[0, frame_w]`` by ``[0, frame_h]``."""
    out = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    out[:, 0::2] = np.clip(out[:, 0::2], 0.0, frame_w)
    out[:, 1::2] = np.clip(out[:, 1::2], 0.0, frame_h)
    return out


def to_boxes(frame_boxes: np.ndarray, scores: np.ndarray) -> tuple[Box, ...]:
    """Convert float ``x1, y1, x2, y2`` frame boxes into integer :class:`Box` values.

    Edges are rounded outward, so the integer box always covers the float box.
    """
    out: list[Box] = []
    for (x1, y1, x2, y2), score in zip(np.asarray(frame_boxes), np.asarray(scores)):
        left, top = math.floor(x1), math.floor(y1)
        right, bottom = math.ceil(x2), math.ceil(y2)
        out.append(
            Box(
                x=left,
                y=top,
                w=max(0, right - left),
                h=max(0, bottom - top),
                score=float(score),
            )
        )
    return tuple(out)


def _check_threshold(name: str, value: object, *, high_closed: bool) -> float:
    ok = (
        type(value) in (int, float)
        and math.isfinite(value)  # type: ignore[arg-type]
        and (0 < value <= 1 if high_closed else 0 < value < 1)  # type: ignore[operator]
    )
    if not ok:
        bound = "(0, 1]" if high_closed else "(0, 1)"
        raise ValueError(f"{name} must be a number in {bound}, got {value!r}")
    return float(value)  # type: ignore[arg-type]


def _check_pad(name: str, value: object) -> float:
    ok = (
        type(value) in (int, float)
        and math.isfinite(value)  # type: ignore[arg-type]
        and 0 <= value <= 1  # type: ignore[operator]
    )
    if not ok:
        raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")
    return float(value)  # type: ignore[arg-type]


class PersonDnnDetector:
    """Person boxes from NanoDet-Plus (416 x 416, COCO class 0).

    Parameters
    ----------
    model:
        The ONNX file's bytes, already verified against :data:`MODEL`. Use
        :meth:`from_config` to find and verify them.
    score_threshold:
        Minimum person score in (0, 1) for a box to be kept.
    nms_threshold:
        IoU in (0, 1] above which a lower scored box is merged into a higher
        scored one (see :func:`select_persons`).
    pad_x, pad_y:
        Fractions in [0, 1] of the box width and height added on each side of
        every box (see :meth:`detect_boxes`).
    """

    provides: ClassVar[frozenset[DetectionKind]] = frozenset({DetectionKind.PERSONS})
    stateful: ClassVar[bool] = False

    def __init__(
        self,
        model: bytes,
        *,
        score_threshold: float = 0.4,
        nms_threshold: float = 0.5,
        pad_x: float = 0.10,
        pad_y: float = 0.07,
    ) -> None:
        if not isinstance(model, bytes) or not model:
            raise ValueError("model must be the non-empty bytes of the ONNX file")
        self.score_threshold = _check_threshold(
            "score_threshold", score_threshold, high_closed=False
        )
        self.nms_threshold = _check_threshold("nms_threshold", nms_threshold, high_closed=True)
        self.pad_x = _check_pad("pad_x", pad_x)
        self.pad_y = _check_pad("pad_y", pad_y)
        self._net = cv2.dnn.readNetFromONNX(np.frombuffer(model, dtype=np.uint8))
        self._output_names = tuple(self._net.getUnconnectedOutLayersNames())
        self._model_path: Path | None = None

    @property
    def model_path(self) -> Path | None:
        """The model file this detector was built from, or None when built from bytes.

        Set by :meth:`from_config` from the path the verified bytes were read
        from, so a caller can report it without reading the file again.
        """
        return self._model_path

    @classmethod
    def from_config(
        cls, config: PersonDnnConfig, model_dirs: Sequence[Path] = ()
    ) -> PersonDnnDetector:
        """Build a detector from a :class:`PersonDnnConfig`.

        ``config.model_path`` is tried alone when set; otherwise ``model_dirs``
        (see :func:`petrify.models.model_dirs`) is searched in order. The box
        padding comes from ``config.pad_x`` and ``config.pad_y``.

        Raises
        ------
        petrify.models.ModelError
            If the model file is missing, unreadable, or fails its size or
            SHA-256 check.
        """
        loaded = load_model(MODEL, path=config.model_path, dirs=model_dirs)
        detector = cls(
            loaded.data,
            score_threshold=config.score_threshold,
            nms_threshold=config.nms_threshold,
            pad_x=config.pad_x,
            pad_y=config.pad_y,
        )
        detector._model_path = loaded.path
        return detector

    def detect(self, frame: Frame) -> Detections:
        """Return detections with only ``persons`` set (see :meth:`detect_boxes`)."""
        return Detections(persons=self.detect_boxes(frame.image))

    def reset(self) -> None:
        """No-op: the detector keeps no per-frame state."""

    def detect_boxes(self, image: np.ndarray) -> tuple[Box, ...]:
        """Return person boxes in frame pixel coordinates, highest score first.

        Boxes are model estimates, not exact outlines. Overlapping candidates
        are merged into the union of their boxes (:func:`select_persons`);
        then each box grows by ``pad_x`` of its width and ``pad_y`` of its
        height on each side, is clipped to the frame, and is rounded outward.
        The default pad covers the undershoot measured on compact poses (up
        to 8.5% of the box width per side and 6.2% of the height at the
        feet). Extended limbs can still stick out of the box: an outstretched
        arm was measured up to 26% of the box width beyond it.

        Recall drops for small people: none was found below about 24 network
        pixels tall, which is about 110 px on a 1080p frame, and wide frames
        shrink people further through letterboxing.

        Raises
        ------
        ValueError
            If ``image`` is not a non-empty H x W x 3 uint8 array.
        NanoDetOutputError
            If the network output is malformed or not finite.
        cv2.error
            If OpenCV fails during inference.
        """
        square, geom = letterbox(image)
        self._net.setInput(to_blob(square))
        outputs = self._net.forward(self._output_names)
        boxes, scores = decode(outputs)
        boxes, scores = select_persons(boxes, scores, self.score_threshold, self.nms_threshold)
        padded = pad_boxes(map_to_frame(boxes, geom), self.pad_x, self.pad_y)
        return to_boxes(clip_to_frame(padded, geom.frame_w, geom.frame_h), scores)
