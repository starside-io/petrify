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

"""Person segmentation detector: pure pre and postprocessing, and the real model."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from petrify.core import DetectionKind, Detections, Detector, Frame, SegmentConfig, SourceInfo
from petrify.core.config import MAX_DILATE_PX
from petrify.core.detections import check_detections
from petrify.detectors import segment
from petrify.detectors.segment import (
    INPUT_SIZE,
    MODEL,
    PERSON_CHANNEL,
    PersonSegmenter,
    dilate_mask,
    footprint_cover,
    person_probability,
    postprocess,
    preprocess,
    resize_mask,
)
from petrify.models import (
    MODEL_DIR_ENV,
    LoadedModel,
    ModelIntegrityError,
    ModelNotFoundError,
    load_model_bytes,
)

PROVIDES = frozenset({DetectionKind.MASK})


def _frame(image: np.ndarray, index: int = 0) -> Frame:
    h, w = image.shape[:2]
    info = SourceInfo(fps=25.0, width=w, height=h, frame_count=None, seekable=False, live=False)
    return Frame(image=image, index=index, timestamp=index / 25.0, source=info)


def _output(person: np.ndarray) -> np.ndarray:
    """Build a (1, 2, h, w) softmax-like output from a person probability map."""
    person = np.asarray(person, dtype=np.float32)
    return np.stack([1.0 - person, person])[np.newaxis].astype(np.float32)


def _post(
    output: np.ndarray,
    height: int,
    width: int,
    *,
    threshold: float = 0.5,
    dilate_px: int = 0,
    min_area_share: float = 0.0,
) -> np.ndarray:
    return postprocess(
        output,
        height,
        width,
        threshold=threshold,
        dilate_px=dilate_px,
        min_area_share=min_area_share,
    )


def _disc_count(radius: int) -> int:
    """Lattice points within Euclidean distance ``radius`` of the origin."""
    r = np.arange(-radius, radius + 1)
    return int(np.count_nonzero(r[:, None] ** 2 + r[None, :] ** 2 <= radius**2))


# ---------------------------------------------------------------------------
# Model spec
# ---------------------------------------------------------------------------


def test_model_spec_values() -> None:
    assert MODEL.filename == "human_segmentation_pphumanseg_2023mar.onnx"
    assert MODEL.sha256 == "552d8a984054e59b5d773d24b9b12022b22046ceb2bbc4c9aaeaceb36a9ddf24"
    assert MODEL.size == 6163938
    assert MODEL.key == "human_segmentation_pphumanseg_2023mar"
    assert MODEL.source_url.startswith("https://")
    assert MODEL.license


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, 1, 3), (8, 8, 3), (480, 640, 3), (3, 900, 3), (700, 2, 3)])
def test_preprocess_shape_dtype_range(shape: tuple[int, int, int]) -> None:
    img = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    blob = preprocess(img)
    assert blob.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
    assert blob.dtype == np.float32
    assert blob.flags.c_contiguous
    assert blob.min() >= -1.0 and blob.max() <= 1.0


def _quadrant_image(height: int, width: int) -> np.ndarray:
    """Black top-left, red right band, green bottom band (BGR)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, width // 2 :] = (0, 0, 255)
    img[height // 2 :, :] = (0, 255, 0)
    return img


@pytest.mark.parametrize("height,width", [(100, 400), (400, 100), (192, 192), (37, 501)])
def test_preprocess_uses_the_whole_frame(height: int, width: int) -> None:
    blob = preprocess(_quadrant_image(height, width))[0]
    third = INPUT_SIZE // 3
    # Top-left stays black in every channel.
    np.testing.assert_allclose(blob[:, :third, :third], -1.0)
    # Top-right is red: R high, G and B low.
    np.testing.assert_allclose(blob[0, :third, -third:], 1.0)
    np.testing.assert_allclose(blob[1, :third, -third:], -1.0)
    # The bottom band is green across the full width.
    np.testing.assert_allclose(blob[1, -third:, :], 1.0)
    np.testing.assert_allclose(blob[0, -third:, :], -1.0)


def test_detect_feeds_the_whole_frame_to_the_network(fake_net: _FakeNet) -> None:
    img = _quadrant_image(120, 360)
    PersonSegmenter(b"model").detect(_frame(img))
    assert len(fake_net.inputs) == 1
    np.testing.assert_array_equal(fake_net.inputs[0], preprocess(img))
    assert fake_net.inputs[0][0, 0, :10, -10:].min() == 1.0
    assert fake_net.inputs[0][0, 1, -10:, :].min() == 1.0


def test_preprocess_is_rgb_and_normalized() -> None:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[..., 0] = 255  # BGR blue
    blob = preprocess(img)
    np.testing.assert_allclose(blob[0, 0], -1.0)  # R
    np.testing.assert_allclose(blob[0, 1], -1.0)  # G
    np.testing.assert_allclose(blob[0, 2], 1.0)  # B


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 1), dtype=np.uint8),
        np.zeros((10, 10, 4), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
        np.zeros((10, 10, 3), dtype=np.uint16),
        np.zeros((0, 10, 3), dtype=np.uint8),
        np.zeros((10, 0, 3), dtype=np.uint8),
        np.zeros((2, 10, 10, 3), dtype=np.uint8),
        [[[0, 0, 0]]],
    ],
)
def test_preprocess_rejects_bad_images(bad: object) -> None:
    with pytest.raises(ValueError):
        preprocess(bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# person_probability
# ---------------------------------------------------------------------------


def test_person_probability_takes_person_channel() -> None:
    person = np.zeros((4, 6), dtype=np.float32)
    person[:, :3] = 0.9
    prob = person_probability(_output(person))
    assert PERSON_CHANNEL == 1
    assert prob.shape == (4, 6)
    assert prob.dtype == np.float32
    np.testing.assert_allclose(prob, person)


@pytest.mark.parametrize(
    "bad",
    [
        np.full((2, 4, 4), 0.5, dtype=np.float32),  # no batch axis
        np.full((2, 2, 4, 4), 0.5, dtype=np.float32),  # batch of 2
        np.full((1, 1, 4, 4), 0.5, dtype=np.float32),  # one channel
        np.full((1, 3, 4, 4), 1 / 3, dtype=np.float32),  # three channels
        np.full((1, 2, 0, 4), 0.5, dtype=np.float32),  # empty
        np.full((1, 2, 4, 4, 1), 0.5, dtype=np.float32),  # extra axis
        np.ones((1, 2, 4, 4), dtype=np.uint8),  # integer
        [[[[0.5]], [[0.5]]]],  # not an array
    ],
)
def test_person_probability_rejects_malformed_shapes(bad: object) -> None:
    with pytest.raises(ValueError):
        person_probability(bad)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_person_probability_rejects_non_finite(value: float) -> None:
    out = _output(np.full((4, 4), 0.5))
    out[0, 1, 2, 2] = value
    with pytest.raises(ValueError, match="non-finite"):
        person_probability(out)


def test_person_probability_rejects_logits() -> None:
    logits = np.zeros((1, 2, 4, 4), dtype=np.float32)
    logits[0, 1] = 3.0
    logits[0, 0] = -3.0
    with pytest.raises(ValueError, match="probability"):
        person_probability(logits)


def test_person_probability_rejects_channels_not_summing_to_one() -> None:
    out = np.full((1, 2, 4, 4), 0.2, dtype=np.float32)
    with pytest.raises(ValueError, match="sum to 1"):
        person_probability(out)


# ---------------------------------------------------------------------------
# postprocess
# ---------------------------------------------------------------------------


def test_threshold_boundary_counts_as_person() -> None:
    at = _post(_output(np.full((8, 8), 0.5)), 8, 8, threshold=0.5)
    assert at.all()
    below = _post(_output(np.full((8, 8), np.nextafter(np.float32(0.5), np.float32(0)))), 8, 8)
    assert not below.any()


def test_threshold_is_applied() -> None:
    person = np.full((8, 8), 0.3, dtype=np.float32)
    assert not _post(_output(person), 8, 8, threshold=0.5).any()
    assert _post(_output(person), 8, 8, threshold=0.25).all()


def test_channel_choice_follows_person_channel() -> None:
    person = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    person[:, : INPUT_SIZE // 2] = 1.0
    mask = _post(_output(person), INPUT_SIZE, INPUT_SIZE)
    assert mask[:, : INPUT_SIZE // 2].all()
    assert not mask[:, INPUT_SIZE // 2 :].any()


@pytest.mark.parametrize("height,width", [(100, 300), (300, 100), (480, 640), (1080, 1920)])
def test_resize_to_non_square_frames(height: int, width: int) -> None:
    person = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    person[:, : INPUT_SIZE // 2] = 1.0  # left half
    person[: INPUT_SIZE // 4, :] = 1.0  # top quarter
    mask = _post(_output(person), height, width)
    assert mask.shape == (height, width)
    assert mask.dtype == np.bool_
    margin_w, margin_h = max(2, width // 50), max(2, height // 50)
    assert mask[:, : width // 2 - margin_w].all()
    assert mask[: height // 4 - margin_h, :].all()
    assert not mask[height // 4 + margin_h :, width // 2 + margin_w :].any()


def _footprint(s: int, n_src: int, n_out: int) -> tuple[int, int]:
    """Half-open range of output pixels whose area overlaps source pixel ``s``.

    Output ``o`` overlaps ``s`` when ``o * n_src < (s + 1) * n_out`` and
    ``(o + 1) * n_src > s * n_out``; solved for ``o`` in integers.
    """
    lo = (s * n_out) // n_src
    hi = -((-(s + 1) * n_out) // n_src)
    return lo, hi


def _assert_footprints_covered(person: np.ndarray, mask: np.ndarray, threshold: float) -> None:
    src_h, src_w = person.shape
    height, width = mask.shape
    for y, x in zip(*np.nonzero(person >= threshold), strict=True):
        y0, y1 = _footprint(int(y), src_h, height)
        x0, x1 = _footprint(int(x), src_w, width)
        assert y1 > y0 and x1 > x0
        assert mask[y0:y1, x0:x1].all(), (
            f"source ({y}, {x}) p={person[y, x]} footprint rows {y0}:{y1} cols {x0}:{x1} "
            f"not covered in a {height}x{width} frame"
        )


@pytest.mark.parametrize(
    "height,width,y,x,value",
    [
        (1, 1, 191, 191, 1.0),
        (2, 2, 95, 95, 1.0),
        (100, 100, 95, 95, 0.75),
        (300, 300, 20, 20, 0.9),
        (193, 193, 96, 96, 0.9),
    ],
)
def test_isolated_person_pixel_is_never_lost(
    height: int, width: int, y: int, x: int, value: float
) -> None:
    person = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    person[y, x] = value
    mask = _post(_output(person), height, width, threshold=0.5, min_area_share=1e-9)
    assert mask.any()
    _assert_footprints_covered(person, mask, 0.5)


def test_every_flagged_pixel_footprint_is_covered_at_random_sizes() -> None:
    rng = np.random.default_rng(7)
    sizes = [(1, 1), (1, 700), (700, 1), (191, 193), (192, 192), (384, 96), (2, 3)]
    sizes += [(int(rng.integers(1, 701)), int(rng.integers(1, 701))) for _ in range(250)]
    corners = [(0, 0), (0, INPUT_SIZE - 1), (INPUT_SIZE - 1, 0), (INPUT_SIZE - 1, INPUT_SIZE - 1)]
    for height, width in sizes:
        person = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        count = int(rng.integers(1, 6))
        for _ in range(count):
            person[rng.integers(0, INPUT_SIZE), rng.integers(0, INPUT_SIZE)] = rng.uniform(0.5, 1)
        edge = int(rng.integers(0, INPUT_SIZE))
        person[rng.choice([0, INPUT_SIZE - 1]), edge] = rng.uniform(0.5, 1)
        person[edge, rng.choice([0, INPUT_SIZE - 1])] = rng.uniform(0.5, 1)
        cy, cx = corners[int(rng.integers(0, 4))]
        person[cy, cx] = rng.uniform(0.5, 1)
        # Just below the threshold: allowed to vanish, must not break anything.
        person[rng.integers(0, INPUT_SIZE), rng.integers(0, INPUT_SIZE)] = 0.49
        mask = _post(_output(person), height, width, threshold=0.5)
        assert mask.shape == (height, width)
        _assert_footprints_covered(person, mask, 0.5)


def test_footprint_cover_matches_brute_force() -> None:
    rng = np.random.default_rng(8)
    for _ in range(40):
        src_h, src_w = int(rng.integers(1, 30)), int(rng.integers(1, 30))
        height, width = int(rng.integers(1, 80)), int(rng.integers(1, 80))
        source = rng.random((src_h, src_w)) > 0.85
        expected = np.zeros((height, width), dtype=np.bool_)
        for y, x in zip(*np.nonzero(source), strict=True):
            y0, y1 = _footprint(int(y), src_h, height)
            x0, x1 = _footprint(int(x), src_w, width)
            expected[y0:y1, x0:x1] = True
        np.testing.assert_array_equal(footprint_cover(source, height, width), expected)


def test_upscale_follows_the_interpolated_contour() -> None:
    # One 0.9 pixel on a 0.3 background, grown 10x. Its footprint is 10 output
    # pixels per axis; the bilinear map stays at or above 0.5 for 2/3 of a
    # source pixel from its center, so the mask reaches past the footprint.
    prob = np.full((5, 5), 0.3, dtype=np.float32)
    prob[2, 2] = 0.9
    mask = resize_mask(prob, 50, 50, 0.5)
    row = np.nonzero(mask[25])[0]
    assert row.min() < 20 and row.max() > 29
    assert row.min() >= 17 and row.max() <= 32
    assert not mask[0].any() and not mask[:, 0].any()


def test_resize_mask_identity_size() -> None:
    prob = np.random.default_rng(1).random((5, 7), dtype=np.float32)
    np.testing.assert_array_equal(resize_mask(prob, 5, 7, 0.5), prob >= 0.5)


@pytest.mark.parametrize("radius", [1, 3, 10])
def test_dilation_grows_a_point_to_a_disc(radius: int) -> None:
    mask = np.zeros((64, 64), dtype=np.bool_)
    mask[32, 32] = True
    grown = dilate_mask(mask, radius)
    assert np.count_nonzero(grown) == _disc_count(radius)
    ys, xs = np.nonzero(grown)
    assert (ys.min(), ys.max(), xs.min(), xs.max()) == (
        32 - radius,
        32 + radius,
        32 - radius,
        32 + radius,
    )


def test_dilation_matches_brute_force_discs() -> None:
    # Sparse masks and radii up to 80: approximate distance masks (3x3, 5x5)
    # only drift from the true Euclidean disc away from the seed pixels.
    rng = np.random.default_rng(5)
    for trial in range(40):
        h, w = int(rng.integers(1, 170)), int(rng.integers(1, 170))
        if trial % 2:
            mask = rng.random((h, w)) > rng.uniform(0.9, 0.995)
        else:
            mask = np.zeros((h, w), dtype=np.bool_)
            for _ in range(int(rng.integers(1, 4))):
                mask[rng.integers(0, h), rng.integers(0, w)] = True
        radius = int(rng.integers(1, 81))
        ys, xs = np.mgrid[0:h, 0:w]
        expected = np.zeros((h, w), dtype=np.bool_)
        for y, x in zip(*np.nonzero(mask), strict=True):
            expected |= (ys - y) ** 2 + (xs - x) ** 2 <= radius**2
        np.testing.assert_array_equal(dilate_mask(mask, radius), expected)


def test_dilation_zero_is_identity_and_never_shrinks() -> None:
    rng = np.random.default_rng(2)
    for _ in range(5):
        mask = rng.random((40, 60)) > 0.9
        np.testing.assert_array_equal(dilate_mask(mask, 0), mask)
        for radius in (1, 4, MAX_DILATE_PX):
            grown = dilate_mask(mask, radius)
            assert grown.dtype == np.bool_
            assert grown.shape == mask.shape
            assert not (mask & ~grown).any()


def test_dilation_of_empty_and_full_masks() -> None:
    empty = np.zeros((5, 5), dtype=np.bool_)
    full = np.ones((5, 5), dtype=np.bool_)
    assert not dilate_mask(empty, 3).any()
    assert dilate_mask(full, 3).all()
    tiny = np.ones((1, 1), dtype=np.bool_)
    assert dilate_mask(tiny, MAX_DILATE_PX).all()


def test_postprocess_dilation_grows_by_the_expected_amount() -> None:
    person = np.zeros((100, 100), dtype=np.float32)
    person[40:60, 40:60] = 1.0
    base = _post(_output(person), 100, 100)
    grown = _post(_output(person), 100, 100, dilate_px=5)
    assert np.count_nonzero(base) == 400
    assert not (base & ~grown).any()
    ys, xs = np.nonzero(grown)
    assert (ys.min(), ys.max(), xs.min(), xs.max()) == (35, 64, 35, 64)
    # Corners grow as quarter discs, not squares: the square, four 20x5 side
    # strips, and the disc's points off both axes.
    assert not grown[35, 35]
    assert np.count_nonzero(grown) == 400 + 4 * 20 * 5 + (_disc_count(5) - (4 * 5 + 1))


def _area_output(count: int, side: int = 100) -> np.ndarray:
    person = np.zeros(side * side, dtype=np.float32)
    person[:count] = 1.0
    return _output(person.reshape(side, side))


def test_min_area_share_at_boundary_is_a_person() -> None:
    mask = _post(_area_output(2500), 100, 100, min_area_share=0.25)
    assert np.count_nonzero(mask) == 2500


def test_min_area_share_below_boundary_is_all_false() -> None:
    mask = _post(_area_output(2499), 100, 100, min_area_share=0.25)
    assert mask is not None
    assert mask.shape == (100, 100)
    assert mask.dtype == np.bool_
    assert not mask.any()
    assert not Detections(mask=mask).subject_present


def test_min_area_share_zero_never_drops_the_mask() -> None:
    person = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    person[10, 10] = 0.9
    mask = _post(_output(person), 1080, 1920, min_area_share=0.0)
    assert mask.any()
    assert np.count_nonzero(mask) < 0.0001 * mask.size
    empty = _post(_output(np.zeros((INPUT_SIZE, INPUT_SIZE))), 40, 30, min_area_share=0.0)
    assert empty.shape == (40, 30) and not empty.any()


def test_min_area_share_is_off_by_default(fake_net: _FakeNet) -> None:
    assert SegmentConfig().min_area_share == 0.0
    assert PersonSegmenter(b"model").min_area_share == 0.0
    person = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    person[100, 100] = 0.9
    fake_net.output = _output(person)
    mask = PersonSegmenter(b"model").detect(_frame(np.zeros((480, 640, 3), np.uint8))).mask
    assert mask.any()


def test_min_area_share_is_measured_before_dilation() -> None:
    mask = _post(_area_output(2499), 100, 100, min_area_share=0.25, dilate_px=10)
    assert not mask.any()


@pytest.mark.parametrize("height,width", [(0, 5), (5, 0)])
def test_postprocess_rejects_empty_frame_size(height: int, width: int) -> None:
    with pytest.raises(ValueError):
        _post(_output(np.zeros((4, 4))), height, width)


def test_postprocess_mask_passes_check_detections() -> None:
    person = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    person[50:150, 60:120] = 0.8
    for shape in [(480, 640, 3), (1, 1, 3), (7, 333, 3)]:
        mask = _post(_output(person), shape[0], shape[1], dilate_px=3)
        checked = check_detections(Detections(mask=mask), PROVIDES, shape)
        assert checked.mask is not None
        assert checked.mask.shape == shape[:2]


# ---------------------------------------------------------------------------
# PersonSegmenter without the model (stand-in network)
# ---------------------------------------------------------------------------


class _FakeNet:
    def __init__(self, output: object = None, error: Exception | None = None) -> None:
        self.output = output
        self.error = error
        self.inputs: list[np.ndarray] = []

    def setInput(self, blob: np.ndarray) -> None:  # noqa: N802 (OpenCV name)
        self.inputs.append(blob)

    def forward(self) -> object:
        if self.error is not None:
            raise self.error
        return self.output


@pytest.fixture
def fake_net(monkeypatch: pytest.MonkeyPatch) -> _FakeNet:
    net = _FakeNet(output=_output(np.zeros((INPUT_SIZE, INPUT_SIZE))))
    monkeypatch.setattr(segment.cv2.dnn, "readNetFromONNX", lambda buf: net)
    return net


def test_protocol_members(fake_net: _FakeNet) -> None:
    det = PersonSegmenter(b"model")
    assert isinstance(det, Detector)
    assert det.provides == PROVIDES
    assert det.stateful is False
    det.reset()
    out = det.detect(_frame(np.zeros((48, 64, 3), dtype=np.uint8)))
    assert out.mask is not None and not out.mask.any()
    assert (out.motion, out.persons, out.background_diff, out.faces) == (None,) * 4
    check_detections(out, det.provides, (48, 64, 3))
    assert fake_net.inputs[0].shape == (1, 3, INPUT_SIZE, INPUT_SIZE)


def test_detect_uses_settings(fake_net: _FakeNet) -> None:
    person = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    person[90:100, 90:100] = 0.6
    fake_net.output = _output(person)
    image = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    assert PersonSegmenter(b"m", threshold=0.5).detect(_frame(image)).mask.sum() == 100
    assert not PersonSegmenter(b"m", threshold=0.7).detect(_frame(image)).mask.any()
    assert not PersonSegmenter(b"m", min_area_share=0.01).detect(_frame(image)).mask.any()
    grown = PersonSegmenter(b"m", dilate_px=2).detect(_frame(image)).mask
    assert grown.sum() > 100


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 4), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
        np.zeros((0, 0, 3), dtype=np.uint8),
    ],
)
def test_detect_rejects_bad_frames(fake_net: _FakeNet, bad: np.ndarray) -> None:
    det = PersonSegmenter(b"model")
    with pytest.raises(ValueError):
        det.detect(_frame(bad))
    assert fake_net.inputs == []


@pytest.mark.parametrize(
    "output",
    [
        np.zeros((1, 2, 4), dtype=np.float32),
        np.zeros((1, 3, 192, 192), dtype=np.float32),
        np.full((1, 2, 192, 192), np.nan, dtype=np.float32),
        np.full((1, 2, 192, 192), 4.0, dtype=np.float32),
        None,
    ],
)
def test_detect_raises_on_bad_model_output(fake_net: _FakeNet, output: object) -> None:
    fake_net.output = output
    with pytest.raises(ValueError):
        PersonSegmenter(b"model").detect(_frame(np.zeros((20, 20, 3), dtype=np.uint8)))


def test_detect_propagates_opencv_errors(fake_net: _FakeNet) -> None:
    fake_net.error = cv2.error("boom")
    with pytest.raises(cv2.error):
        PersonSegmenter(b"model").detect(_frame(np.zeros((20, 20, 3), dtype=np.uint8)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 0},
        {"threshold": 1},
        {"threshold": float("nan")},
        {"threshold": True},
        {"threshold": "0.5"},
        {"dilate_px": -1},
        {"dilate_px": MAX_DILATE_PX + 1},
        {"dilate_px": 1.0},
        {"dilate_px": True},
        {"min_area_share": -0.01},
        {"min_area_share": True},
        {"min_area_share": 1.5},
        {"min_area_share": float("inf")},
    ],
)
def test_constructor_rejects_bad_settings(fake_net: _FakeNet, kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PersonSegmenter(b"model", **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("model", [b"", None, "path.onnx"])
def test_constructor_rejects_non_bytes_model(fake_net: _FakeNet, model: object) -> None:
    with pytest.raises(ValueError):
        PersonSegmenter(model)  # type: ignore[arg-type]


def test_constructor_accepts_boundary_settings(fake_net: _FakeNet) -> None:
    det = PersonSegmenter(b"model", dilate_px=MAX_DILATE_PX, min_area_share=1)
    assert det.dilate_px == MAX_DILATE_PX
    assert det.min_area_share == 1.0
    assert PersonSegmenter(b"model", min_area_share=0).min_area_share == 0.0


def test_from_config_forwards_every_setting(
    fake_net: _FakeNet, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def fake_load(spec: object, *, path: object = None, dirs: object = ()) -> LoadedModel:
        seen.update(spec=spec, path=path, dirs=dirs)
        return LoadedModel(b"verified", Path("/a/m.onnx"))

    monkeypatch.setattr(segment, "load_model", fake_load)
    cfg = SegmentConfig(threshold=0.37, dilate_px=9, min_area_share=0.125, model_path="m.onnx")
    det = PersonSegmenter.from_config(cfg, [Path("/a")])
    assert (det.threshold, det.dilate_px, det.min_area_share) == (0.37, 9, 0.125)
    assert seen == {"spec": MODEL, "path": "m.onnx", "dirs": [Path("/a")]}
    assert det.model_path == Path("/a/m.onnx")


def test_model_path_is_none_when_built_from_bytes_and_read_only(fake_net: _FakeNet) -> None:
    det = PersonSegmenter(b"verified")
    assert det.model_path is None
    with pytest.raises(AttributeError):
        det.model_path = Path("x")  # type: ignore[misc]


def test_garbage_model_bytes_raise_opencv_error() -> None:
    with pytest.raises(cv2.error):
        PersonSegmenter(b"this is not an onnx file")


def test_from_config_refuses_wrong_file(tmp_path: Path) -> None:
    bad = tmp_path / MODEL.filename
    bad.write_bytes(b"x" * 10)
    with pytest.raises(ModelIntegrityError):
        PersonSegmenter.from_config(SegmentConfig(), [tmp_path])
    with pytest.raises(ModelIntegrityError):
        PersonSegmenter.from_config(SegmentConfig(model_path=str(bad)), [])


def test_from_config_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError):
        PersonSegmenter.from_config(SegmentConfig(), [tmp_path / "nowhere"])


# ---------------------------------------------------------------------------
# Real model
# ---------------------------------------------------------------------------


def _model_dir() -> Path:
    env = os.environ.get(MODEL_DIR_ENV, "")
    if not env:
        pytest.skip(f"{MODEL_DIR_ENV} is not set, so the segmentation model cannot be found")
    path = Path(env).expanduser() / MODEL.filename
    if not path.exists():
        pytest.skip(f"segmentation model not found at {path}")
    return path.parent


@pytest.fixture(scope="module")
def real() -> PersonSegmenter:
    return PersonSegmenter.from_config(SegmentConfig(), [_model_dir()])


def test_real_model_loads_through_both_lookups() -> None:
    directory = _model_dir()
    data = load_model_bytes(MODEL, dirs=[directory])
    assert len(data) == MODEL.size
    det = PersonSegmenter.from_config(
        SegmentConfig(model_path=str(directory / MODEL.filename)), []
    )
    assert isinstance(det, Detector)


def test_real_model_output_is_a_probability_map(real: PersonSegmenter) -> None:
    real._net.setInput(preprocess(np.full((480, 640, 3), 90, dtype=np.uint8)))
    out = real._net.forward()
    assert out.shape == (1, 2, INPUT_SIZE, INPUT_SIZE)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0].sum(axis=0), 1.0, atol=1e-4)
    person_probability(out)


def test_real_model_black_frame_has_no_person(real: PersonSegmenter) -> None:
    out = check_detections(
        real.detect(_frame(np.zeros((480, 640, 3), dtype=np.uint8))), PROVIDES, (480, 640, 3)
    )
    assert out.mask is not None
    assert not out.mask.any()
    assert not out.subject_present


def test_real_model_white_frame_is_valid(real: PersonSegmenter) -> None:
    # Observed: the model marks a few percent of a plain white frame as person.
    # That over-covers, which is the safe direction, so only validity is checked.
    out = check_detections(
        real.detect(_frame(np.full((480, 640, 3), 255, dtype=np.uint8))),
        PROVIDES,
        (480, 640, 3),
    )
    assert out.mask is not None


@pytest.mark.parametrize(
    "shape", [(1, 1, 3), (8, 8, 3), (480, 640, 3), (1080, 1920, 3), (2, 900, 3), (900, 2, 3)]
)
def test_real_model_output_passes_check(real: PersonSegmenter, shape: tuple[int, ...]) -> None:
    img = np.random.default_rng(3).integers(0, 256, shape, dtype=np.uint8)
    out = check_detections(real.detect(_frame(img)), PROVIDES, shape)
    assert out.mask is not None
    assert out.mask.shape == shape[:2]


def test_real_model_frame_size_change_between_calls(real: PersonSegmenter) -> None:
    rng = np.random.default_rng(4)
    for i, shape in enumerate([(480, 640, 3), (240, 320, 3), (720, 1280, 3), (480, 640, 3)]):
        img = rng.integers(0, 256, shape, dtype=np.uint8)
        out = check_detections(real.detect(_frame(img, i)), PROVIDES, shape)
        assert out.mask.shape == shape[:2]


def test_real_model_dilation_never_shrinks(real: PersonSegmenter) -> None:
    img = np.full((240, 320, 3), 255, dtype=np.uint8)
    img[60:200, 120:200] = 30
    base = real.detect(_frame(img)).mask
    grown = PersonSegmenter.from_config(SegmentConfig(dilate_px=8), [_model_dir()]).detect(
        _frame(img)
    ).mask
    assert not (base & ~grown).any()


@pytest.mark.parametrize(
    "bad",
    [np.zeros((10, 10), dtype=np.uint8), np.zeros((10, 10, 3), dtype=np.float64)],
)
def test_real_model_rejects_bad_frames(real: PersonSegmenter, bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        real.detect(_frame(bad))
