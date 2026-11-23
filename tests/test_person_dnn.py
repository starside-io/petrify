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

"""Tests for the NanoDet person box detector."""

from __future__ import annotations

import math
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from petrify.core import (
    Box,
    DetectionKind,
    Detections,
    Detector,
    Frame,
    PersonDnnConfig,
    SourceInfo,
    check_detections,
)
from petrify.detectors import person_dnn as pd
from petrify.detectors.person_dnn import (
    INPUT_SIZE,
    MODEL,
    NanoDetOutputError,
    PersonDnnDetector,
    clip_to_frame,
    decode,
    letterbox,
    letterbox_geometry,
    map_to_frame,
    pad_boxes,
    select_persons,
    to_boxes,
)
from petrify.models import (
    MODEL_DIR_ENV,
    LoadedModel,
    ModelIntegrityError,
    ModelNotFoundError,
    load_model_bytes,
)

STRIDES = (8, 16, 32)
FIELDS = ("motion", "persons", "background_diff", "faces", "mask")


def _frame(image: np.ndarray, index: int = 0) -> Frame:
    h, w = image.shape[:2]
    info = SourceInfo(fps=25.0, width=w, height=h, frame_count=None, seekable=False, live=False)
    return Frame(image=image, index=index, timestamp=index / 25.0, source=info)


def _count(stride: int) -> int:
    return math.ceil(INPUT_SIZE / stride) ** 2


def _outputs(fill_reg: float = 0.0) -> list[np.ndarray]:
    """Return all-zero class scores and constant box logits in the model's output order."""
    outs: list[np.ndarray] = []
    for stride in STRIDES:
        n = _count(stride)
        outs.append(np.zeros((1, n, 80), dtype=np.float32))
        outs.append(np.full((1, n, 32), fill_reg, dtype=np.float32))
    return outs


def _set_anchor(
    outs: list[np.ndarray],
    stride: int,
    row: int,
    col: int,
    sides: tuple[int, int, int, int],
    scores: dict[int, float],
) -> None:
    """Make one anchor predict ``sides`` bins (left, top, right, bottom) with ``scores``."""
    level = STRIDES.index(stride)
    cells = math.ceil(INPUT_SIZE / stride)
    i = row * cells + col
    cls, reg = outs[2 * level], outs[2 * level + 1]
    for class_id, score in scores.items():
        cls[0, i, class_id] = score
    logits = np.full((4, 8), -50.0, dtype=np.float32)
    for side, bin_index in enumerate(sides):
        logits[side, bin_index] = 50.0
    reg[0, i] = logits.reshape(-1)


def _center(stride: int, row: int, col: int) -> tuple[float, float]:
    offset = 0.5 * (stride - 1)
    return col * stride + offset, row * stride + offset


class _FakeNet:
    def __init__(self, outputs: list[np.ndarray] | None = None, error: bool = False) -> None:
        self.outputs = outputs
        self.error = error
        self.inputs: list[np.ndarray] = []

    def setInput(self, blob: np.ndarray) -> None:  # noqa: N802 - mirrors cv2.dnn.Net
        self.inputs.append(blob)

    def forward(self, names: tuple[str, ...]) -> list[np.ndarray]:
        if self.error:
            raise cv2.error("simulated OpenCV failure")
        assert self.outputs is not None
        return self.outputs

    def getUnconnectedOutLayersNames(self) -> tuple[str, ...]:  # noqa: N802
        return ("a", "b", "c", "d", "e", "f")


def _fake_detector(
    monkeypatch: pytest.MonkeyPatch, net: _FakeNet, **kwargs: float
) -> PersonDnnDetector:
    """Build a detector through its constructor with ``net`` in place of the ONNX network.

    Padding defaults to 0 here so the geometry tests see the model's own box;
    pass ``pad_x`` and ``pad_y`` to test padding.
    """
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda buffer: net)
    kwargs.setdefault("pad_x", 0.0)
    kwargs.setdefault("pad_y", 0.0)
    return PersonDnnDetector(b"fake model", **kwargs)  # type: ignore[arg-type]


def _two_anchor_outputs(
    second_col: int, second_score: float, first_score: float = 0.9
) -> list[np.ndarray]:
    """Two stride 16 anchors on row 13 with 64 x 64 boxes; one column apart gives IoU 0.6."""
    outs = _outputs()
    _set_anchor(outs, 16, row=13, col=10, sides=(2, 2, 2, 2), scores={0: first_score})
    _set_anchor(outs, 16, row=13, col=second_col, sides=(2, 2, 2, 2), scores={0: second_score})
    return outs


def _contains(outer: Box, x1: float, y1: float, x2: float, y2: float) -> bool:
    return outer.x <= x1 and outer.y <= y1 and outer.x + outer.w >= x2 and outer.y + outer.h >= y2


# ---------------------------------------------------------------------------
# Model spec
# ---------------------------------------------------------------------------


def test_model_spec_values() -> None:
    assert MODEL.filename == "object_detection_nanodet_2022nov.onnx"
    assert MODEL.sha256 == "4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186"
    assert MODEL.size == 3800954
    assert MODEL.key == "object_detection_nanodet_2022nov"
    assert MODEL.source_url.startswith("https://")
    assert MODEL.license


# ---------------------------------------------------------------------------
# Letterbox geometry and inverse mapping
# ---------------------------------------------------------------------------


def test_geometry_landscape() -> None:
    g = letterbox_geometry(640, 480)
    assert (g.content_w, g.content_h, g.left, g.top) == (416, 312, 0, 52)


def test_geometry_portrait() -> None:
    g = letterbox_geometry(480, 640)
    assert (g.content_w, g.content_h, g.left, g.top) == (312, 416, 52, 0)


def test_geometry_square_upscale() -> None:
    g = letterbox_geometry(1, 1)
    assert (g.content_w, g.content_h, g.left, g.top) == (416, 416, 0, 0)


@pytest.mark.parametrize("w, h", [(1, 10000), (10000, 1), (1, 2000), (3, 5000)])
def test_geometry_elongated_keeps_one_pixel(w: int, h: int) -> None:
    g = letterbox_geometry(w, h)
    assert 1 <= g.content_w <= INPUT_SIZE and 1 <= g.content_h <= INPUT_SIZE
    assert max(g.content_w, g.content_h) == INPUT_SIZE
    assert g.left + g.content_w <= INPUT_SIZE and g.top + g.content_h <= INPUT_SIZE


@pytest.mark.parametrize("w, h", [(0, 10), (10, 0), (-1, 5)])
def test_geometry_rejects_empty(w: int, h: int) -> None:
    with pytest.raises(ValueError):
        letterbox_geometry(w, h)


@pytest.mark.parametrize(
    "w, h", [(640, 480), (1920, 1080), (480, 640), (416, 416), (1, 1), (8, 8), (7, 3000), (3000, 7)]
)
def test_letterbox_image_places_content_and_black_padding(w: int, h: int) -> None:
    image = np.full((h, w, 3), 200, dtype=np.uint8)
    square, g = letterbox(image)
    assert square.shape == (INPUT_SIZE, INPUT_SIZE, 3) and square.dtype == np.uint8
    content = square[g.top : g.top + g.content_h, g.left : g.left + g.content_w]
    assert (content == 200).all()
    mask = np.ones(square.shape[:2], dtype=bool)
    mask[g.top : g.top + g.content_h, g.left : g.left + g.content_w] = False
    assert (square[mask] == 0).all()


def _reference_resize(image: np.ndarray) -> np.ndarray:
    """The single resize call letterbox made before the one-axis-at-a-time path."""
    g = letterbox_geometry(image.shape[1], image.shape[0])
    shrinking = g.content_w < g.frame_w or g.content_h < g.frame_h
    return cv2.resize(
        image,
        (g.content_w, g.content_h),
        interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR,
    )


def _test_images(w: int, h: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    ys, xs = np.indices((h, w))
    checker = (((ys + xs) % 2) * 255).astype(np.uint8)
    return [
        rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
        (rng.integers(0, 2, (h, w, 3)) * 255).astype(np.uint8),
        np.full((h, w, 3), 137, dtype=np.uint8),
        np.repeat(checker[:, :, np.newaxis], 3, axis=2),
    ]


# Sizes where resizing one axis at a time gave values one level off the single
# call: each has a whole shrink factor on one axis, so one pass took OpenCV's
# integer area path. They must take the single call.
_SEPARABLE_MISMATCH_SIZES = [
    (3840, 13), (3840, 5), (5, 3840), (13, 3840), (1248, 484), (1248, 1066),
    (1000, 1248), (322, 1248), (3339, 872), (3749, 387), (4159, 1260), (774, 2494),
]


@pytest.mark.parametrize(
    "w, h",
    [
        (1920, 1080), (3840, 2160), (1280, 720), (640, 480), (2560, 1440), (1919, 1081),
        (1931, 1087), (2003, 1123), (1249, 937), (1248, 936), (417, 300), (415, 300),
        (417, 417), (415, 415), (6000, 300), (300, 6000), (5000, 20), (20, 5000),
        (4000, 331), (7, 3000), *_SEPARABLE_MISMATCH_SIZES,
    ],
)
def test_letterbox_content_is_byte_identical_to_one_resize(w: int, h: int) -> None:
    # The one-axis-at-a-time path is a speedup only: the network must see the
    # same bytes as before, so detections cannot change.
    for image in _test_images(w, h, seed=w * 10007 + h):
        square, g = letterbox(image)
        content = square[g.top : g.top + g.content_h, g.left : g.left + g.content_w]
        assert np.array_equal(content, _reference_resize(image))


@pytest.mark.parametrize("w, h", [(1920, 1080), (3840, 2160), (1280, 720), (1931, 1087)])
def test_large_frames_use_the_separable_path(w: int, h: int) -> None:
    assert pd._separable_area(letterbox_geometry(w, h))


@pytest.mark.parametrize(
    "w, h",
    [
        (640, 480),  # shrinks less than 3 times: the single call is as fast
        (1000, 750),
        (1248, 936),  # an integer scale on both axes: OpenCV's own fast path
        (415, 300),  # no shrink at all
        (416, 416),
        *_SEPARABLE_MISMATCH_SIZES,  # a whole shrink factor on one axis
    ],
)
def test_other_frames_take_the_single_resize(
    monkeypatch: pytest.MonkeyPatch, w: int, h: int
) -> None:
    assert not pd._separable_area(letterbox_geometry(w, h))

    def refuse(*args: object) -> np.ndarray:
        raise AssertionError("the separable path must not run for this size")

    monkeypatch.setattr(pd, "_area_by_axis", refuse)
    image = np.full((h, w, 3), 90, dtype=np.uint8)
    square, g = letterbox(image)
    content = square[g.top : g.top + g.content_h, g.left : g.left + g.content_w]
    assert np.array_equal(content, _reference_resize(image))


@pytest.mark.parametrize(
    "w, h", [(640, 480), (1920, 1080), (480, 640), (1, 1), (8, 8), (5, 3000), (3000, 5)]
)
def test_map_to_frame_round_trip(w: int, h: int) -> None:
    g = letterbox_geometry(w, h)
    rng = np.random.default_rng(w * 7 + h)
    x = np.sort(rng.uniform(0, w, (20, 2)), axis=1)
    y = np.sort(rng.uniform(0, h, (20, 2)), axis=1)
    frame_boxes = np.column_stack((x[:, 0], y[:, 0], x[:, 1], y[:, 1]))
    net_boxes = np.column_stack(
        (
            frame_boxes[:, 0] * g.scale_x + g.left,
            frame_boxes[:, 1] * g.scale_y + g.top,
            frame_boxes[:, 2] * g.scale_x + g.left,
            frame_boxes[:, 3] * g.scale_y + g.top,
        )
    )
    np.testing.assert_allclose(map_to_frame(net_boxes, g), frame_boxes, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize(
    "w, h, scale_x, scale_y, left, top",
    [
        (5, 3000, 0.2, 416 / 3000, 207, 0),
        (3000, 5, 416 / 3000, 0.2, 0, 207),
        (640, 480, 0.65, 0.65, 0, 52),
        (1920, 1080, 416 / 1920, 234 / 1080, 0, 91),
        # 999 * 416 / 2000 = 207.79 rounds to 208 content pixels, not 207.
        (999, 2000, 208 / 999, 416 / 2000, 104, 0),
    ],
)
def test_geometry_literal_scales(
    w: int, h: int, scale_x: float, scale_y: float, left: int, top: int
) -> None:
    g = letterbox_geometry(w, h)
    assert g.scale_x == pytest.approx(scale_x)
    assert g.scale_y == pytest.approx(scale_y)
    assert (g.left, g.top) == (left, top)


@pytest.mark.parametrize(
    "w, h, net_box, frame_box",
    [
        (5, 3000, [207.0, 0.0, 208.0, 416.0], [0.0, 0.0, 5.0, 3000.0]),
        (5, 3000, [207.5, 104.0, 208.0, 312.0], [2.5, 750.0, 5.0, 2250.0]),
        (3000, 5, [104.0, 207.5, 312.0, 208.0], [750.0, 2.5, 2250.0, 5.0]),
        (640, 480, [0.0, 52.0, 416.0, 364.0], [0.0, 0.0, 640.0, 480.0]),
        (1920, 1080, [104.0, 91.0, 208.0, 208.0], [480.0, 0.0, 960.0, 540.0]),
    ],
)
def test_map_to_frame_literal_boxes(
    w: int, h: int, net_box: list[float], frame_box: list[float]
) -> None:
    out = map_to_frame(np.array([net_box]), letterbox_geometry(w, h))
    np.testing.assert_allclose(out, [frame_box], atol=1e-9)


def test_map_to_frame_does_not_clip() -> None:
    g = letterbox_geometry(640, 480)  # top padding 52 px
    out = map_to_frame(np.array([[-10.0, 0.0, 500.0, 416.0]]), g)
    np.testing.assert_allclose(out, [[-10 / 0.65, -80.0, 500 / 0.65, 560.0]])


def test_clip_to_frame() -> None:
    out = clip_to_frame(np.array([[-15.4, -80.0, 769.2, 560.0], [10, 20, 30, 40]]), 640, 480)
    np.testing.assert_allclose(out, [[0.0, 0.0, 640.0, 480.0], [10, 20, 30, 40]])


def test_pad_boxes_arithmetic() -> None:
    out = pad_boxes(np.array([[10.0, 20.0, 110.0, 220.0], [0.0, 0.0, 0.0, 0.0]]), 0.1, 0.07)
    np.testing.assert_allclose(out, [[0.0, 6.0, 120.0, 234.0], [0.0, 0.0, 0.0, 0.0]])


def test_pad_zero_is_identity() -> None:
    boxes = np.array([[10.5, 20.25, 110.0, 220.75]])
    np.testing.assert_array_equal(pad_boxes(boxes, 0.0, 0.0), boxes)


def test_pad_one_triples_the_size() -> None:
    out = pad_boxes(np.array([[100.0, 100.0, 110.0, 120.0]]), 1.0, 1.0)
    np.testing.assert_allclose(out, [[90.0, 80.0, 120.0, 140.0]])


def test_to_boxes_rounds_outward() -> None:
    boxes = to_boxes(np.array([[1.2, 2.8, 10.1, 20.0]]), np.array([0.7], dtype=np.float32))
    assert boxes == (Box(x=1, y=2, w=10, h=18, score=pytest.approx(0.7)),)
    assert all(type(v) is int for v in (boxes[0].x, boxes[0].y, boxes[0].w, boxes[0].h))
    assert type(boxes[0].score) is float


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


def test_decode_known_box() -> None:
    outs = _outputs()
    _set_anchor(outs, 16, row=10, col=5, sides=(2, 1, 3, 4), scores={0: 0.9})
    boxes, scores = decode(outs)
    assert boxes.shape == (sum(_count(s) for s in STRIDES), 4)
    i = _count(8) + 10 * 26 + 5
    cx, cy = _center(16, 10, 5)
    np.testing.assert_allclose(boxes[i], [cx - 32, cy - 16, cx + 48, cy + 64], atol=1e-3)
    assert scores[i] == pytest.approx(0.9)
    assert np.count_nonzero(scores) == 1


def test_decode_uniform_logits_give_mean_distance() -> None:
    boxes, _ = decode(_outputs(fill_reg=0.0))
    cx, cy = _center(32, 0, 0)
    last = len(boxes) - _count(32)
    np.testing.assert_allclose(boxes[last], [cx - 112, cy - 112, cx + 112, cy + 112], atol=1e-3)


def test_decode_is_independent_of_output_order() -> None:
    outs = _outputs()
    _set_anchor(outs, 8, row=3, col=4, sides=(1, 1, 1, 1), scores={0: 0.8})
    a = decode(outs)
    b = decode(list(reversed(outs)))
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_decode_reads_only_the_person_column() -> None:
    outs = _outputs()
    _set_anchor(outs, 32, row=1, col=1, sides=(1, 1, 1, 1), scores={2: 0.95})
    _, scores = decode(outs)
    assert scores.max() == 0.0


def test_decode_handles_large_logits_without_overflow() -> None:
    outs = _outputs(fill_reg=1e4)
    boxes, _ = decode(outs)
    assert np.isfinite(boxes).all()


@pytest.mark.parametrize("drop", [0, 5])
def test_decode_rejects_missing_output(drop: int) -> None:
    outs = _outputs()
    del outs[drop]
    with pytest.raises(NanoDetOutputError):
        decode(outs)


def test_decode_rejects_extra_output() -> None:
    with pytest.raises(NanoDetOutputError):
        decode([*_outputs(), np.zeros((1, 10, 80), dtype=np.float32)])


@pytest.mark.parametrize(
    "index, shape",
    [(0, (1, 2704, 81)), (1, (1, 2704, 31)), (2, (1, 675, 80)), (4, (169, 80)), (5, (2, 169, 32))],
)
def test_decode_rejects_wrong_shapes(index: int, shape: tuple[int, ...]) -> None:
    outs = _outputs()
    outs[index] = np.zeros(shape, dtype=np.float32)
    with pytest.raises(NanoDetOutputError):
        decode(outs)


def test_decode_rejects_duplicate_shape() -> None:
    outs = _outputs()
    outs[2] = np.zeros((1, _count(8), 80), dtype=np.float32)  # two stride 8 score maps
    with pytest.raises(NanoDetOutputError):
        decode(outs)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("index", [0, 3, 5])
def test_decode_rejects_non_finite(bad: float, index: int) -> None:
    outs = _outputs()
    outs[index][0, 7, 3] = bad
    with pytest.raises(NanoDetOutputError, match="non-finite"):
        decode(outs)


@pytest.mark.parametrize("bad", [-0.01, 1.5])
def test_decode_rejects_scores_outside_unit_range(bad: float) -> None:
    outs = _outputs()
    outs[2][0, 0, 0] = bad
    with pytest.raises(NanoDetOutputError, match=r"\[0, 1\]"):
        decode(outs)


# ---------------------------------------------------------------------------
# Person selection and NMS
# ---------------------------------------------------------------------------


def test_select_persons_threshold_is_inclusive_and_filters() -> None:
    boxes = np.array([[0, 0, 10, 10], [100, 100, 120, 130], [200, 0, 210, 5]], dtype=np.float32)
    scores = np.array([0.4, 0.39, 0.9], dtype=np.float32)
    kept, kept_scores = select_persons(boxes, scores, 0.4, 0.5)
    np.testing.assert_allclose(kept_scores, [0.9, 0.4])
    np.testing.assert_array_equal(kept, boxes[[2, 0]])


def test_select_persons_merges_overlap_into_union() -> None:
    boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [300, 300, 350, 350]], dtype=np.float32)
    scores = np.array([0.6, 0.8, 0.5], dtype=np.float32)
    kept, kept_scores = select_persons(boxes, scores, 0.4, 0.5)
    np.testing.assert_allclose(kept_scores, [0.8, 0.5])
    np.testing.assert_array_equal(kept, [[0, 0, 105, 105], [300, 300, 350, 350]])


def test_select_persons_suppressed_box_that_sticks_out_stays_covered() -> None:
    # IoU 100 / 120 = 0.83: the lower scored box is suppressed but reaches 20 px further right.
    boxes = np.array([[0, 0, 100, 200], [0, 0, 120, 200]], dtype=np.float32)
    scores = np.array([0.9, 0.5], dtype=np.float32)
    kept, kept_scores = select_persons(boxes, scores, 0.4, 0.5)
    np.testing.assert_array_equal(kept, [[0, 0, 120, 200]])
    np.testing.assert_allclose(kept_scores, [0.9])


def test_select_persons_growth_does_not_cascade() -> None:
    # At nms 0.1: B overlaps A (IoU 0.25) and is merged. C touches A only at an edge
    # (IoU 0) but overlaps the grown box (IoU 0.3), so it must still stand on its own.
    boxes = np.array([[0, 0, 100, 100], [60, 0, 160, 100], [100, 0, 200, 100]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    kept, _ = select_persons(boxes, scores, 0.4, 0.1)
    np.testing.assert_array_equal(kept, [[0, 0, 160, 100], [100, 0, 200, 100]])


def test_select_persons_two_people_case_from_review() -> None:
    # Candidates measured on two overlapping people in a 1920 x 1080 frame. The third
    # (score 0.542) is the only one that reaches person B's right edge; plain NMS at
    # 0.5 dropped it and left 13% of B uncovered.
    boxes = np.array(
        [
            [502.4, 144.9, 681.7, 598.0],
            [505.0, 150.5, 717.2, 593.1],
            [542.3, 158.5, 788.6, 591.4],
            [501.6, 152.9, 687.4, 597.5],
            [507.9, 157.6, 739.4, 594.8],
            [538.9, 166.1, 791.8, 591.5],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.576, 0.59, 0.542, 0.426, 0.573, 0.441], dtype=np.float32)
    kept, _ = select_persons(boxes, scores, 0.4, 0.5)
    for candidate in boxes:
        assert any(
            k[0] <= candidate[0]
            and k[1] <= candidate[1]
            and k[2] >= candidate[2]
            and k[3] >= candidate[3]
            for k in kept
        )
    np.testing.assert_allclose(kept[0], [501.6, 144.9, 791.8, 598.0], atol=1e-3)


@pytest.mark.parametrize("seed", range(40))
def test_select_persons_union_contains_every_candidate(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(0, 80))
    xy = rng.uniform(0, 400, (n, 2))
    wh = rng.uniform(0, 120, (n, 2))
    boxes = np.column_stack((xy, xy + wh)).astype(np.float32)
    scores = rng.uniform(0, 1, n).astype(np.float32)
    threshold = float(rng.uniform(0.05, 0.9))
    nms = float(rng.uniform(0.05, 1.0))
    kept, kept_scores = select_persons(boxes, scores, threshold, nms)
    assert len(kept) == len(kept_scores) <= int((scores >= threshold).sum())
    assert (np.diff(kept_scores) <= 0).all()
    for candidate in boxes[scores >= threshold]:
        assert any(
            k[0] <= candidate[0]
            and k[1] <= candidate[1]
            and k[2] >= candidate[2]
            and k[3] >= candidate[3]
            for k in kept
        ), candidate


def test_select_persons_keeps_overlap_below_nms_threshold() -> None:
    # IoU of these two boxes is 1/3.
    boxes = np.array([[0, 0, 100, 100], [50, 0, 150, 100]], dtype=np.float32)
    scores = np.array([0.6, 0.8], dtype=np.float32)
    kept, _ = select_persons(boxes, scores, 0.4, 0.5)
    assert len(kept) == 2
    kept, _ = select_persons(boxes, scores, 0.4, 0.3)
    assert len(kept) == 1


def test_select_persons_iou_equal_to_threshold_is_not_merged() -> None:
    # IoU is exactly 5000 / 10000 = 0.5; only an IoU above the threshold merges.
    boxes = np.array([[0, 0, 100, 100], [0, 0, 100, 50]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    kept, _ = select_persons(boxes, scores, 0.4, 0.5)
    np.testing.assert_array_equal(kept, boxes)


def test_select_persons_empty() -> None:
    kept, kept_scores = select_persons(
        np.zeros((5, 4), dtype=np.float32), np.zeros(5, dtype=np.float32), 0.4, 0.5
    )
    assert kept.shape == (0, 4) and kept_scores.shape == (0,)


# ---------------------------------------------------------------------------
# Detector without the model (fake network)
# ---------------------------------------------------------------------------


def test_fake_network_box_lands_in_frame_coordinates(monkeypatch: pytest.MonkeyPatch) -> None:
    outs = _outputs()
    # Stride 16 anchor at row 13, col 13; center (215.5, 215.5) in input pixels.
    _set_anchor(outs, 16, row=13, col=13, sides=(2, 4, 2, 4), scores={0: 0.9, 2: 0.95})
    det = _fake_detector(monkeypatch, _FakeNet(outs))
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = det.detect(_frame(image))
    checked = check_detections(result, det.provides, image.shape)
    assert checked.persons is not None and len(checked.persons) == 1
    box = checked.persons[0]
    scale = 640 / 416
    x1, y1 = (215.5 - 32) * scale, (215.5 - 64 - 52) * scale
    x2, y2 = (215.5 + 32) * scale, (215.5 + 64 - 52) * scale
    assert box.x == math.floor(x1) and box.y == math.floor(y1)
    assert box.x + box.w == math.ceil(x2) and box.y + box.h == math.ceil(y2)
    assert box.score == pytest.approx(0.9)


def test_fake_network_default_pad(monkeypatch: pytest.MonkeyPatch) -> None:
    outs = _outputs()
    _set_anchor(outs, 16, row=13, col=13, sides=(2, 4, 2, 4), scores={0: 0.9})
    net = _FakeNet(outs)
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda buffer: net)
    det = PersonDnnDetector(b"fake model")
    assert (det.pad_x, det.pad_y) == (0.10, 0.07)
    (box,) = det.detect_boxes(np.zeros((480, 640, 3), dtype=np.uint8))
    # Unpadded frame box: x 282.31 to 380.77, y 153.08 to 350.0 (98.46 x 196.92).
    scale = 640 / 416
    x1, x2 = (215.5 - 32) * scale, (215.5 + 32) * scale
    y1, y2 = (215.5 - 64 - 52) * scale, (215.5 + 64 - 52) * scale
    dx, dy = 0.10 * (x2 - x1), 0.07 * (y2 - y1)
    assert (box.x, box.y) == (math.floor(x1 - dx), math.floor(y1 - dy))
    assert (box.x + box.w, box.y + box.h) == (math.ceil(x2 + dx), math.ceil(y2 + dy))
    assert (box.x, box.y, box.x + box.w, box.y + box.h) == (272, 139, 391, 364)


def test_fake_network_pad_is_clipped_at_frame_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    outs = _outputs()
    # Stride 16 anchor at row 4, col 1: center (23.5, 71.5); 64 x 64 box reaching past
    # the left edge and into the 52 px top padding of a 640 x 480 frame.
    _set_anchor(outs, 16, row=4, col=1, sides=(2, 2, 2, 2), scores={0: 0.9})
    det = _fake_detector(monkeypatch, _FakeNet(outs), pad_x=0.10, pad_y=0.07)
    (box,) = det.detect_boxes(np.zeros((480, 640, 3), dtype=np.uint8))
    scale = 640 / 416
    x1, x2 = (23.5 - 32) * scale, (23.5 + 32) * scale
    y1, y2 = (71.5 - 32 - 52) * scale, (71.5 + 32 - 52) * scale
    dx, dy = 0.10 * (x2 - x1), 0.07 * (y2 - y1)  # padding uses the unclipped size
    assert (box.x, box.y) == (0, 0)
    assert (box.w, box.h) == (math.ceil(x2 + dx), math.ceil(y2 + dy)) == (96, 87)


def test_fake_network_pad_clipped_at_far_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    outs = _outputs()
    # Stride 32 anchor at the bottom right cell: the box spills over both far edges.
    _set_anchor(outs, 32, row=12, col=12, sides=(1, 1, 1, 1), scores={0: 0.9})
    det = _fake_detector(monkeypatch, _FakeNet(outs), pad_x=0.5, pad_y=0.5)
    (box,) = det.detect_boxes(np.zeros((416, 416, 3), dtype=np.uint8))
    # Center 399.5; box 367.5 to 431.5; pad 32 each side.
    assert (box.x, box.y, box.x + box.w, box.y + box.h) == (335, 335, 416, 416)


def test_fake_network_score_threshold_from_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.zeros((416, 416, 3), dtype=np.uint8)
    outs = _two_anchor_outputs(second_col=20, second_score=0.45)
    low = _fake_detector(monkeypatch, _FakeNet(outs))
    assert len(low.detect_boxes(image)) == 2
    high = _fake_detector(monkeypatch, _FakeNet(outs), score_threshold=0.5)
    assert len(high.detect_boxes(image)) == 1


def test_fake_network_nms_threshold_from_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.zeros((416, 416, 3), dtype=np.uint8)
    outs = _two_anchor_outputs(second_col=11, second_score=0.8)  # IoU 0.6
    merged = _fake_detector(monkeypatch, _FakeNet(outs))
    (box,) = merged.detect_boxes(image)
    assert (box.x, box.x + box.w) == (135, 216)  # union of x 135.5 to 199.5 and 151.5 to 215.5
    separate = _fake_detector(monkeypatch, _FakeNet(outs), nms_threshold=0.7)
    assert len(separate.detect_boxes(image)) == 2


def test_fake_network_thresholds_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.zeros((416, 416, 3), dtype=np.uint8)
    monkeypatch.setattr(
        pd, "load_model", lambda spec, path=None, dirs=(): LoadedModel(b"fake", Path("m"))
    )
    outs_score = _two_anchor_outputs(second_col=20, second_score=0.45)
    outs_nms = _two_anchor_outputs(second_col=11, second_score=0.8)

    def build(config: PersonDnnConfig, outs: list[np.ndarray]) -> PersonDnnDetector:
        net = _FakeNet(outs)
        monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda buffer: net)
        return PersonDnnDetector.from_config(config, [])

    assert len(build(PersonDnnConfig(), outs_score).detect_boxes(image)) == 2
    assert len(build(PersonDnnConfig(score_threshold=0.5), outs_score).detect_boxes(image)) == 1
    assert len(build(PersonDnnConfig(), outs_nms).detect_boxes(image)) == 1
    assert len(build(PersonDnnConfig(nms_threshold=0.7), outs_nms).detect_boxes(image)) == 2
    det = build(PersonDnnConfig(), outs_nms)
    assert (det.pad_x, det.pad_y) == (0.10, 0.07)


def test_fake_network_blob_is_normalized_bgr(monkeypatch: pytest.MonkeyPatch) -> None:
    net = _FakeNet(_outputs())
    det = _fake_detector(monkeypatch, net)
    image = np.zeros((416, 416, 3), dtype=np.uint8)
    image[..., 0] = 255  # blue channel only
    det.detect_boxes(image)
    blob = net.inputs[0]
    assert blob.shape == (1, 3, 416, 416) and blob.dtype == np.float32
    np.testing.assert_allclose(
        blob[0, :, 0, 0], [(255 - 103.53) / 57.375, -116.28 / 57.12, -123.675 / 58.395], rtol=1e-5
    )


def test_fake_network_nothing_found_is_empty_tuple(monkeypatch: pytest.MonkeyPatch) -> None:
    det = _fake_detector(monkeypatch, _FakeNet(_outputs()))
    result = det.detect(_frame(np.zeros((480, 640, 3), dtype=np.uint8)))
    assert result.persons == ()
    assert {f for f in FIELDS if getattr(result, f) is not None} == {"persons"}


def test_fake_network_nan_output_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    outs = _outputs()
    outs[1][0, 0, 0] = np.nan
    det = _fake_detector(monkeypatch, _FakeNet(outs))
    with pytest.raises(NanoDetOutputError):
        det.detect(_frame(np.zeros((480, 640, 3), dtype=np.uint8)))


def test_fake_network_opencv_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    det = _fake_detector(monkeypatch, _FakeNet(error=True))
    with pytest.raises(cv2.error):
        det.detect(_frame(np.zeros((480, 640, 3), dtype=np.uint8)))


# ---------------------------------------------------------------------------
# Input and constructor validation
# ---------------------------------------------------------------------------

BAD_IMAGES = [
    pytest.param(np.zeros((48, 64), dtype=np.uint8), id="grayscale"),
    pytest.param(np.zeros((48, 64, 4), dtype=np.uint8), id="bgra"),
    pytest.param(np.zeros((48, 64, 1), dtype=np.uint8), id="one-channel"),
    pytest.param(np.zeros((48, 64, 3), dtype=np.float32), id="float32"),
    pytest.param(np.zeros((48, 64, 3), dtype=np.uint16), id="uint16"),
    pytest.param(np.zeros((0, 64, 3), dtype=np.uint8), id="zero-height"),
    pytest.param(np.zeros((48, 0, 3), dtype=np.uint8), id="zero-width"),
    pytest.param([[[0, 0, 0]]], id="list"),
]


@pytest.mark.parametrize("image", BAD_IMAGES)
def test_letterbox_rejects_bad_images(image: object) -> None:
    with pytest.raises(ValueError):
        letterbox(image)  # type: ignore[arg-type]


@pytest.mark.parametrize("image", BAD_IMAGES)
def test_detector_rejects_bad_images_before_inference(
    monkeypatch: pytest.MonkeyPatch, image: object
) -> None:
    net = _FakeNet(_outputs())
    det = _fake_detector(monkeypatch, net)
    with pytest.raises(ValueError):
        det.detect_boxes(image)  # type: ignore[arg-type]
    assert net.inputs == []


@pytest.mark.parametrize("model", [b"", "not bytes", bytearray(b"x"), None])
def test_constructor_rejects_non_bytes(model: object) -> None:
    with pytest.raises(ValueError):
        PersonDnnDetector(model)  # type: ignore[arg-type]


def test_constructor_rejects_garbage_bytes() -> None:
    with pytest.raises(cv2.error):
        PersonDnnDetector(b"this is not an onnx file")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"score_threshold": 0.0},
        {"score_threshold": 1.0},
        {"score_threshold": float("nan")},
        {"score_threshold": True},
        {"score_threshold": "0.4"},
        {"nms_threshold": 0.0},
        {"nms_threshold": 1.01},
        {"nms_threshold": float("inf")},
    ],
)
def test_constructor_rejects_bad_thresholds(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PersonDnnDetector(b"x", **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["pad_x", "pad_y"])
@pytest.mark.parametrize(
    "value", [-0.01, 1.01, float("nan"), float("inf"), True, "0.1", None, np.float64(0.1)]
)
def test_constructor_rejects_bad_pads(name: str, value: object) -> None:
    with pytest.raises(ValueError, match=name):
        PersonDnnDetector(b"x", **{name: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, 0.0, 0.5, 1, 1.0])
def test_constructor_accepts_pad_bounds(monkeypatch: pytest.MonkeyPatch, value: float) -> None:
    det = _fake_detector(monkeypatch, _FakeNet(_outputs()), pad_x=value, pad_y=value)
    assert det.pad_x == det.pad_y == float(value)


def test_class_attributes() -> None:
    assert PersonDnnDetector.provides == frozenset({DetectionKind.PERSONS})
    assert PersonDnnDetector.stateful is False


def test_from_config_missing_model_raises(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError):
        PersonDnnDetector.from_config(PersonDnnConfig(), [tmp_path])


def test_from_config_rejects_wrong_file(tmp_path: Path) -> None:
    (tmp_path / MODEL.filename).write_bytes(b"\0" * MODEL.size)
    with pytest.raises(ModelIntegrityError):
        PersonDnnDetector.from_config(PersonDnnConfig(), [tmp_path])


def test_from_config_explicit_path_wins(tmp_path: Path) -> None:
    missing = tmp_path / "elsewhere.onnx"
    with pytest.raises(ModelNotFoundError) as info:
        PersonDnnDetector.from_config(PersonDnnConfig(model_path=str(missing)), [tmp_path])
    assert str(missing) in str(info.value)


def test_module_has_no_import_time_network() -> None:
    assert not any(isinstance(v, cv2.dnn.Net) for v in vars(pd).values())


# ---------------------------------------------------------------------------
# Real model
# ---------------------------------------------------------------------------


def _model_dir() -> Path | None:
    value = os.environ.get(MODEL_DIR_ENV, "")
    if not value:
        return None
    directory = Path(value).expanduser()
    return directory if (directory / MODEL.filename).is_file() else None


needs_model = pytest.mark.skipif(
    _model_dir() is None,
    reason=f"{MODEL.filename} not found: set {MODEL_DIR_ENV} to the directory that holds it",
)


@pytest.fixture(scope="module")
def real_detector() -> PersonDnnDetector:
    directory = _model_dir()
    assert directory is not None
    return PersonDnnDetector.from_config(PersonDnnConfig(), [directory])


@needs_model
def test_real_model_loads_and_conforms(real_detector: PersonDnnDetector) -> None:
    assert isinstance(real_detector, Detector)
    assert real_detector.score_threshold == pytest.approx(0.4)
    real_detector.reset()


@needs_model
def test_real_model_outputs_match_expected_layout(real_detector: PersonDnnDetector) -> None:
    square, _ = letterbox(np.full((480, 640, 3), 90, dtype=np.uint8))
    real_detector._net.setInput(pd.to_blob(square))
    outs = real_detector._net.forward(real_detector._output_names)
    assert sorted(o.shape for o in outs) == sorted(
        [(1, _count(s), c) for s in STRIDES for c in (80, 32)]
    )
    boxes, scores = decode(outs)
    assert np.isfinite(boxes).all() and (0 <= scores).all() and (scores <= 1).all()


@needs_model
@pytest.mark.parametrize("value", [0, 128, 255])
def test_real_model_blank_frame_has_no_person(real_detector: PersonDnnDetector, value: int) -> None:
    image = np.full((480, 640, 3), value, dtype=np.uint8)
    assert real_detector.detect(_frame(image)).persons == ()


@needs_model
@pytest.mark.parametrize(
    "w, h", [(640, 480), (1920, 1080), (1, 1), (8, 8), (1, 3000), (3000, 1), (13, 900)]
)
def test_real_model_output_passes_check(real_detector: PersonDnnDetector, w: int, h: int) -> None:
    image = np.random.default_rng(w + h).integers(0, 256, (h, w, 3), dtype=np.uint8)
    result = real_detector.detect(_frame(image))
    checked = check_detections(result, real_detector.provides, image.shape)
    assert isinstance(checked, Detections) and checked.persons is not None
    assert checked.persons == result.persons  # boxes are already clipped to the frame


@needs_model
def test_real_model_frame_size_change_between_calls(real_detector: PersonDnnDetector) -> None:
    rng = np.random.default_rng(3)
    images = [
        rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        for w, h in [(640, 480), (320, 900), (1920, 1080), (8, 8)]
    ]
    images.append(images[0])
    results = []
    for i, image in enumerate(images):
        result = real_detector.detect(_frame(image, index=i))
        check_detections(result, real_detector.provides, image.shape)
        results.append(result.persons)
    # The same frame after other sizes gives the same answer: no state leaks between calls.
    assert results[0] == results[-1]


@needs_model
@pytest.mark.parametrize("image", BAD_IMAGES)
def test_real_model_rejects_bad_images(real_detector: PersonDnnDetector, image: object) -> None:
    with pytest.raises(ValueError):
        real_detector.detect_boxes(image)  # type: ignore[arg-type]


@needs_model
def test_real_model_uses_the_verified_bytes(tmp_path: Path) -> None:
    directory = _model_dir()
    assert directory is not None
    data = load_model_bytes(MODEL, dirs=[directory])
    copy = tmp_path / MODEL.filename
    copy.write_bytes(data)
    det = PersonDnnDetector.from_config(PersonDnnConfig(), [tmp_path])
    copy.unlink()  # the detector must not need the file after construction
    assert det.detect(_frame(np.zeros((48, 64, 3), dtype=np.uint8))).persons == ()
    assert det.model_path == copy


# ---------------------------------------------------------------------------
# Real model on a real person (needs the model and the demo clip)
# ---------------------------------------------------------------------------

DEMO_CLIP = Path(__file__).resolve().parent.parent / "data" / "camera.mp4"
PERSON_FRAME = 350
# Hand-checked on frame 350: head, torso, and legs down to the shoes of the man at
# the door, leaving out the outstretched right arm. x1, y1, x2, y2 in frame pixels.
PERSON_CORE = (1300, 190, 1430, 710)
# Top-scoring raw box on that frame, measured independently in review: x, y, w, h.
PERSON_MEASURED = (1261, 175, 197, 551)

needs_clip = pytest.mark.skipif(
    not DEMO_CLIP.is_file(), reason=f"demo clip not available: {DEMO_CLIP}"
)


@pytest.fixture(scope="module")
def person_frame() -> np.ndarray:
    cap = cv2.VideoCapture(str(DEMO_CLIP))
    try:
        # Read sequentially: seeking in an mp4 can land on a nearby frame.
        for _ in range(PERSON_FRAME + 1):
            ok, image = cap.read()
            assert ok, f"could not read frame {PERSON_FRAME} of {DEMO_CLIP}"
    finally:
        cap.release()
    assert image.shape == (1080, 1920, 3)
    return image


@needs_model
@needs_clip
def test_real_person_is_covered(real_detector: PersonDnnDetector, person_frame: np.ndarray) -> None:
    boxes = real_detector.detect_boxes(person_frame)
    assert any(_contains(b, *PERSON_CORE) for b in boxes), boxes
    check_detections(Detections(persons=boxes), real_detector.provides, person_frame.shape)


@needs_model
@needs_clip
def test_real_person_raw_box_matches_measurement(person_frame: np.ndarray) -> None:
    directory = _model_dir()
    assert directory is not None
    det = PersonDnnDetector(load_model_bytes(MODEL, dirs=[directory]), pad_x=0.0, pad_y=0.0)
    square, geom = letterbox(person_frame)
    det._net.setInput(pd.to_blob(square))
    boxes, scores = decode(det._net.forward(det._output_names))
    top = map_to_frame(boxes[int(np.argmax(scores))], geom)[0]
    x, y, w, h = PERSON_MEASURED
    # A few pixels of tolerance: a wrong anchor offset shifts the box by about 16 px.
    np.testing.assert_allclose(top, [x, y, x + w, y + h], atol=4)
    # Without padding the returned box is the union of overlapping candidates, so it
    # contains the top candidate and stays close to it.
    (box,) = det.detect_boxes(person_frame)
    assert _contains(box, *top)
    np.testing.assert_allclose([box.x, box.y, box.x + box.w, box.y + box.h], top, atol=12)
