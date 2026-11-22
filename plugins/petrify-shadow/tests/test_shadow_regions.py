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

"""Region geometry: padding, clipping, mask crop and dilation, orphan person boxes."""

from __future__ import annotations

import dataclasses
import math
import random
from fractions import Fraction

import numpy as np
import pytest
from petrify_shadow.regions import (
    FACE_ZONE,
    HEAD_ZONE,
    HELD_GROWTH_CAP,
    MIN_PAD_PX,
    ORPHAN_HEAD_MASK_SHARE,
    ORPHAN_MASK_SHARE,
    PERSON_PAD,
    PreparedMask,
    Rect,
    dilate_px,
    held_growth,
    orphan_persons,
    pad_face,
    pad_person,
    prepare_mask,
)

from petrify.core.detections import Box, Face

H, W = 1080, 1920


def _contains(rect: Rect, x0: int, y0: int, x1: int, y1: int) -> bool:
    return rect.x0 <= x0 and rect.y0 <= y0 and rect.x1 >= x1 and rect.y1 >= y1


def _in_frame(rect: Rect, h: int, w: int) -> bool:
    return 0 <= rect.x0 < rect.x1 <= w and 0 <= rect.y0 < rect.y1 <= h


def _box_in_frame(box: Box, h: int, w: int) -> tuple[int, int, int, int]:
    """The box intersected with the frame, as half-open (x0, y0, x1, y1)."""
    return (
        min(max(box.x, 0), w),
        min(max(box.y, 0), h),
        min(max(box.x + box.w, 0), w),
        min(max(box.y + box.h, 0), h),
    )


# --- Rect -------------------------------------------------------------------


def test_rect_is_frozen_and_half_open() -> None:
    rect = Rect(2, 3, 7, 11)
    assert (rect.width, rect.height, rect.area, rect.is_empty) == (5, 8, 40, False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rect.x0 = 0  # type: ignore[misc]
    image = np.zeros((20, 20), np.uint8)
    image[rect.y0 : rect.y1, rect.x0 : rect.x1] = 1
    assert int(image.sum()) == rect.area


def test_rect_inverted_or_flat_is_empty() -> None:
    assert Rect(5, 5, 5, 9).is_empty
    assert Rect(5, 5, 9, 5).is_empty
    assert Rect(9, 9, 5, 5).width == 0
    assert Rect(9, 9, 5, 5).area == 0


def test_built_rects_are_clipped_python_ints() -> None:
    rng = random.Random(7)
    for _ in range(500):
        h, w = rng.randint(1, 60), rng.randint(1, 60)
        box = Box(rng.randint(-20, 70), rng.randint(-20, 70), rng.randint(0, 40), rng.randint(0, 40))
        for rect in (pad_face(box, rng.random(), h, w), pad_person(box, h, w)):
            if rect is not None:
                assert _in_frame(rect, h, w)
                assert all(type(v) is int for v in dataclasses.astuple(rect))


# --- pad_face ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("box", "padding", "expected"),
    [
        # side 12, top 22.5, bottom 7.5: the half pixels round outward.
        (Box(100, 100, 40, 50), 0.3, Rect(88, 77, 152, 158)),
        # padding 0: only the 2 px minimum on every side.
        (Box(100, 100, 40, 50), 0.0, Rect(98, 98, 142, 152)),
        # padding 1: side 40, top 75, bottom 25.
        (Box(100, 100, 40, 50), 1.0, Rect(60, 25, 180, 175)),
        # side 1.2, top 1.8, bottom 0.6 are all raised to 2.
        (Box(10, 10, 4, 4), 0.3, Rect(8, 8, 16, 16)),
        # side 1.5 -> 2, top 2.25 stays fractional and rounds out to 3, bottom 0.75 -> 2.
        (Box(10, 10, 5, 5), 0.3, Rect(8, 7, 17, 17)),
        # side 2.1 rounds out on both sides, top 4.5, bottom 1.5 -> 2.
        (Box(10, 10, 7, 10), 0.3, Rect(7, 5, 20, 22)),
        # 0.1 * 30 is exactly 3 on the decimal, so no extra pixel from float error.
        (Box(100, 100, 30, 40), 0.1, Rect(97, 94, 133, 142)),
        # An integer padding is accepted.
        (Box(100, 100, 40, 50), 1, Rect(60, 25, 180, 175)),
    ],
)
def test_pad_face_literal(box: Box, padding: float, expected: Rect) -> None:
    assert pad_face(box, padding, H, W) == expected


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        (Box(0, 0, 40, 50), Rect(0, 0, 52, 58)),  # top left corner
        (Box(60, 50, 40, 50), Rect(48, 27, 100, 100)),  # bottom right corner
        (Box(0, 40, 40, 50), Rect(0, 17, 52, 98)),  # left edge
        (Box(60, 40, 40, 50), Rect(48, 17, 100, 98)),  # right edge
        (Box(30, 0, 40, 50), Rect(18, 0, 82, 58)),  # top edge
        (Box(30, 50, 40, 50), Rect(18, 27, 82, 100)),  # bottom edge
        (Box(60, 0, 40, 50), Rect(48, 0, 100, 58)),  # top right corner
        (Box(0, 50, 40, 50), Rect(0, 27, 52, 100)),  # bottom left corner
    ],
)
def test_pad_face_clips_at_each_edge_and_corner(box: Box, expected: Rect) -> None:
    assert pad_face(box, 0.3, 100, 100) == expected


def test_pad_face_zero_area_and_off_frame() -> None:
    assert pad_face(Box(10, 10, 0, 20), 0.3, H, W) is None
    assert pad_face(Box(10, 10, 20, 0), 0.3, H, W) is None
    # check_detections puts a wholly off-frame box on the edge with zero area.
    assert pad_face(Box(W, 10, 0, 20), 0.3, H, W) is None
    # Unclipped input far outside: nothing reaches the frame.
    assert pad_face(Box(200, 200, 10, 10), 0.0, 100, 100) is None
    assert pad_face(Box(-10, 10, 8, 10), 0.0, 100, 100) is None
    # Partly off frame: the padding that reaches into the frame is kept.
    assert pad_face(Box(-10, 10, 9, 10), 0.0, 100, 100) == Rect(0, 8, 1, 22)


def test_pad_face_tiny_frames() -> None:
    assert pad_face(Box(0, 0, 1, 1), 0.3, 1, 1) == Rect(0, 0, 1, 1)
    assert pad_face(Box(0, 0, 1, 1), 1.0, 1, 1) == Rect(0, 0, 1, 1)
    assert pad_face(Box(0, 0, 1, 1), 0.3, 1, 3) == Rect(0, 0, 3, 1)
    assert pad_face(Box(0, 0, 0, 0), 0.3, 1, 1) is None
    assert pad_face(Box(0, 0, 1, 1), 0.3, 0, 0) is None


@pytest.mark.parametrize("padding", [-0.1, math.nan, math.inf, True, "0.3", None])
def test_pad_face_rejects_bad_padding(padding: object) -> None:
    with pytest.raises(ValueError, match="padding"):
        pad_face(Box(10, 10, 10, 10), padding, H, W)  # type: ignore[arg-type]


def test_pad_face_rejects_negative_box_size() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        pad_face(Box(10, 10, -1, 10), 0.3, H, W)


def test_pad_face_contains_box_property() -> None:
    """The padded rect contains the box in the frame plus the exact padding, clipped."""
    rng = random.Random(1)
    for _ in range(3000):
        h, w = rng.randint(1, 80), rng.randint(1, 80)
        box = Box(rng.randint(-30, 90), rng.randint(-30, 90), rng.randint(0, 50), rng.randint(0, 50))
        padding = rng.choice([0.0, 1.0, round(rng.random(), 3), rng.random()])
        rect = pad_face(box, padding, h, w)
        bx0, by0, bx1, by1 = _box_in_frame(box, h, w)
        if box.w == 0 or box.h == 0:
            assert rect is None
            continue
        if rect is None:
            # Allowed only when even the exact padded rect misses the frame.
            p = Fraction(repr(padding))
            side = max(Fraction(MIN_PAD_PX), p * box.w)
            assert (
                box.x + box.w + side <= 0
                or box.x - side >= w
                or box.y + box.h + max(Fraction(MIN_PAD_PX), p * box.h / 2) <= 0
                or box.y - max(Fraction(MIN_PAD_PX), Fraction(3, 2) * p * box.h) >= h
            )
            continue
        assert _in_frame(rect, h, w)
        if bx1 > bx0 and by1 > by0:
            assert _contains(rect, bx0, by0, bx1, by1)
        # Each side reaches at least the exact padding, clipped.
        p = Fraction(repr(padding))
        side = max(Fraction(MIN_PAD_PX), p * box.w)
        top = max(Fraction(MIN_PAD_PX), Fraction(3, 2) * p * box.h)
        bottom = max(Fraction(MIN_PAD_PX), p * box.h / 2)
        assert rect.x0 <= max(box.x - side, 0)
        assert rect.x1 >= min(box.x + box.w + side, w)
        assert rect.y0 <= max(box.y - top, 0)
        assert rect.y1 >= min(box.y + box.h + bottom, h)
        # Outward rounding adds less than one pixel per side.
        assert rect.x0 > box.x - side - 1
        assert rect.x1 < box.x + box.w + side + 1


# --- held growth ------------------------------------------------------------


def test_held_growth_linear_and_capped() -> None:
    assert held_growth(0) == 1
    assert held_growth(1) == Fraction(11, 10)
    assert held_growth(3) == Fraction(13, 10)
    assert held_growth(9) == Fraction(19, 10)
    assert held_growth(10) == HELD_GROWTH_CAP == 2
    assert held_growth(1000) == 2
    with pytest.raises(ValueError):
        held_growth(-1)
    with pytest.raises(ValueError):
        held_growth(1.5)  # type: ignore[arg-type]


def test_pad_face_grow_literal() -> None:
    box = Box(100, 100, 40, 50)
    # Unclipped padded rect: x 88 to 152, y 77.5 to 157.5.
    assert pad_face(box, 0.3, H, W, grow=1) == Rect(88, 77, 152, 158)
    # 1.1: half width 35.2 around 120, half height 44 around 117.5.
    assert pad_face(box, 0.3, H, W, grow=held_growth(1)) == Rect(84, 73, 156, 162)
    # 2: half width 64, half height 80.
    assert pad_face(box, 0.3, H, W, grow=held_growth(10)) == Rect(56, 37, 184, 198)
    assert pad_face(box, 0.3, H, W, grow=held_growth(50)) == Rect(56, 37, 184, 198)
    assert pad_face(box, 0.3, H, W, grow=2.0) == Rect(56, 37, 184, 198)


def test_pad_face_grow_clips_and_grows_before_clipping() -> None:
    box = Box(0, 0, 40, 50)
    grown = pad_face(box, 0.3, 200, 200, grow=2)
    # Unclipped x -12 to 52 and y -22.5 to 57.5, doubled around (20, 17.5).
    assert grown == Rect(0, 0, 84, 98)
    # Growing the clipped rect Rect(0, 0, 52, 58) around its own center would
    # reach only x 78 and y 87: growth before clipping covers more.
    assert grown.x1 > 78 and grown.y1 > 87
    assert pad_face(Box(0, 0, 40, 50), 0.3, 60, 60, grow=2) == Rect(0, 0, 60, 60)


def test_pad_face_grow_rejects_shrinking() -> None:
    with pytest.raises(ValueError, match="grow"):
        pad_face(Box(10, 10, 10, 10), 0.3, H, W, grow=0.9)


def test_pad_face_grow_contains_ungrown_property() -> None:
    rng = random.Random(3)
    for _ in range(2000):
        h, w = rng.randint(1, 80), rng.randint(1, 80)
        box = Box(rng.randint(-20, 80), rng.randint(-20, 80), rng.randint(1, 40), rng.randint(1, 40))
        padding = round(rng.random(), 2)
        base = pad_face(box, padding, h, w)
        previous = base
        for held in range(0, 13):
            grown = pad_face(box, padding, h, w, grow=held_growth(held))
            if previous is not None:
                assert grown is not None
                assert _contains(grown, previous.x0, previous.y0, previous.x1, previous.y1)
            previous = grown


# --- pad_person -------------------------------------------------------------


def test_person_pad_constant() -> None:
    assert PERSON_PAD == Fraction(5, 100)


@pytest.mark.parametrize(
    ("box", "h", "w", "expected"),
    [
        (Box(100, 100, 200, 400), H, W, Rect(90, 80, 310, 520)),
        (Box(10, 10, 20, 20), H, W, Rect(8, 8, 32, 32)),  # 1 px raised to 2
        (Box(10, 10, 50, 30), H, W, Rect(7, 8, 63, 42)),  # 2.5 rounds outward
        (Box(0, 0, 100, 100), 100, 100, Rect(0, 0, 100, 100)),
        (Box(0, 0, 50, 50), 100, 100, Rect(0, 0, 53, 53)),  # top left corner
        (Box(50, 50, 50, 50), 100, 100, Rect(47, 47, 100, 100)),  # bottom right
        (Box(0, 0, 1, 1), 1, 1, Rect(0, 0, 1, 1)),  # 1x1 frame
    ],
)
def test_pad_person_literal(box: Box, h: int, w: int, expected: Rect) -> None:
    assert pad_person(box, h, w) == expected


def test_pad_person_zero_area_and_off_frame() -> None:
    assert pad_person(Box(10, 10, 0, 10), H, W) is None
    assert pad_person(Box(10, 10, 10, 0), H, W) is None
    assert pad_person(Box(300, 10, 10, 10), 100, 100) is None
    assert pad_person(Box(-12, 10, 10, 10), 100, 100) is None
    # Partly off frame: the padding that reaches into the frame is kept.
    assert pad_person(Box(-11, 10, 10, 10), 100, 100) == Rect(0, 8, 1, 22)
    with pytest.raises(ValueError):
        pad_person(Box(10, 10, 10, -1), H, W)


def test_pad_person_contains_box_property() -> None:
    rng = random.Random(5)
    for _ in range(3000):
        h, w = rng.randint(1, 80), rng.randint(1, 80)
        box = Box(rng.randint(-30, 90), rng.randint(-30, 90), rng.randint(1, 60), rng.randint(1, 60))
        rect = pad_person(box, h, w)
        bx0, by0, bx1, by1 = _box_in_frame(box, h, w)
        if bx1 > bx0 and by1 > by0:
            assert rect is not None
            assert _in_frame(rect, h, w)
            assert _contains(rect, bx0, by0, bx1, by1)
            side = max(Fraction(MIN_PAD_PX), PERSON_PAD * box.w)
            vert = max(Fraction(MIN_PAD_PX), PERSON_PAD * box.h)
            assert rect.x0 <= max(box.x - side, 0)
            assert rect.x1 >= min(box.x + box.w + side, w)
            assert rect.y0 <= max(box.y - vert, 0)
            assert rect.y1 >= min(box.y + box.h + vert, h)


# --- dilate_px --------------------------------------------------------------


@pytest.mark.parametrize(
    ("h", "dilate", "expected"),
    [
        (480, 0.01, 5),
        (1080, 0.01, 11),
        (500, 0.01, 5),  # exactly 5 on the decimal, no extra pixel from float error
        (100, 0.01, 1),
        (1080, 0.0, 0),
        (1080, 0, 0),
        (0, 0.01, 0),
        (1, 0.001, 1),  # any positive dilation gives at least one pixel
        (720, 0.025, 18),
    ],
)
def test_dilate_px(h: int, dilate: float, expected: int) -> None:
    assert dilate_px(h, dilate) == expected


@pytest.mark.parametrize("dilate", [-0.01, math.nan, math.inf])
def test_dilate_px_rejects_bad_values(dilate: float) -> None:
    with pytest.raises(ValueError):
        dilate_px(1080, dilate)


# --- integer box coordinates ------------------------------------------------


@pytest.mark.parametrize(
    "box",
    [
        Box(100, 100, 40.9, 50),  # type: ignore[arg-type]
        Box(100, 100, 40, 50.5),  # type: ignore[arg-type]
        Box(100.5, 100, 40, 50),  # type: ignore[arg-type]
        Box(100, np.float64(100.0), 40, 50),  # type: ignore[arg-type]
        Box(100, 100, True, 50),  # type: ignore[arg-type]
    ],
)
def test_non_integral_box_raises(box: Box) -> None:
    """A float size would be truncated inward (40.9 to 40), so it raises instead."""
    with pytest.raises(ValueError, match="integers"):
        pad_face(box, 0.3, H, W)
    with pytest.raises(ValueError, match="integers"):
        pad_person(box, H, W)
    with pytest.raises(ValueError, match="integers"):
        orphan_persons((box,), (), None, "face", H, W)
    with pytest.raises(ValueError, match="integers"):
        orphan_persons((Box(0, 0, 10, 10),), (Face(box=box),), None, "face", H, W)
    with pytest.raises(ValueError, match="integers"):
        orphan_persons((box,), (), np.zeros((H, W), bool), "mask", H, W)


def test_numpy_integer_box_accepted() -> None:
    box = Box(np.int64(100), np.int32(100), np.int64(40), np.uint16(50))  # type: ignore[arg-type]
    rect = pad_face(box, 0.3, H, W)
    assert rect == Rect(88, 77, 152, 158)
    assert all(type(v) is int for v in dataclasses.astuple(rect))
    assert pad_person(box, H, W) == pad_person(Box(100, 100, 40, 50), H, W)


# --- prepare_mask -----------------------------------------------------------


def _within_distance(mask: np.ndarray, r: int) -> np.ndarray:
    """Every pixel whose exact integer squared distance to a mask pixel is at most r * r."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    yy, xx = np.mgrid[0:h, 0:w]
    out = np.zeros((h, w), bool)
    # One mask pixel at a time keeps memory small; the arithmetic is int64, exact.
    for y, x in zip(ys.tolist(), xs.tolist()):
        out |= (yy - y) ** 2 + (xx - x) ** 2 <= r * r
    return out


def _full(pm: PreparedMask, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, bool)
    rect = pm.rect
    out[rect.y0 : rect.y1, rect.x0 : rect.x1] = pm.crop
    return out


def _readonly(mask: np.ndarray) -> np.ndarray:
    mask = np.ascontiguousarray(mask, dtype=bool)
    mask.setflags(write=False)
    return mask


def _disk(r: int) -> np.ndarray:
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return xx * xx + yy * yy <= r * r


def test_prepare_mask_none_and_empty() -> None:
    assert prepare_mask(None, 5) is None
    assert prepare_mask(_readonly(np.zeros((30, 40), bool)), 5) is None
    assert prepare_mask(_readonly(np.zeros((30, 40), bool)), 0) is None


def test_prepare_mask_single_pixel() -> None:
    mask = np.zeros((40, 50), bool)
    mask[20, 25] = True
    pm = prepare_mask(_readonly(mask), 3)
    assert pm is not None
    assert pm.rect == Rect(22, 17, 29, 24)
    assert pm.crop.shape == (7, 7)
    assert pm.crop.dtype == np.bool_
    np.testing.assert_array_equal(pm.crop, _disk(3))
    pm0 = prepare_mask(_readonly(mask), 0)
    assert pm0 is not None
    assert pm0.rect == Rect(25, 20, 26, 21)
    np.testing.assert_array_equal(pm0.crop, np.ones((1, 1), bool))


@pytest.mark.parametrize("r", [1, 2, 5, 13, 25, 60])
def test_distance_exactly_r_is_kept_and_next_is_not(r: int) -> None:
    mask = np.zeros((2 * r + 9, 2 * r + 9), bool)
    c = r + 4
    mask[c, c] = True
    pm = prepare_mask(_readonly(mask), r)
    assert pm is not None
    full = _full(pm, mask.shape)
    assert full[c, c + r] and full[c - r, c]  # distance exactly r
    assert not full[c + 1, c + r]  # distance sqrt(r*r + 1)
    if r == 5:
        assert full[c + 3, c + 4]  # 3-4-5 triangle, distance exactly 5
        assert not full[c + 4, c + 4]


@pytest.mark.parametrize(
    ("y", "x", "expected"),
    [
        (0, 25, Rect(21, 0, 30, 5)),  # top border
        (39, 25, Rect(21, 35, 30, 40)),  # bottom border
        (20, 0, Rect(0, 16, 5, 25)),  # left border
        (20, 49, Rect(45, 16, 50, 25)),  # right border
        (0, 0, Rect(0, 0, 5, 5)),  # corner
        (39, 49, Rect(45, 35, 50, 40)),  # opposite corner
    ],
)
def test_prepare_mask_touching_each_border(y: int, x: int, expected: Rect) -> None:
    mask = np.zeros((40, 50), bool)
    mask[y, x] = True
    pm = prepare_mask(_readonly(mask), 4)
    assert pm is not None
    assert pm.rect == expected
    np.testing.assert_array_equal(_full(pm, mask.shape), _within_distance(mask, 4))


def test_prepare_mask_r0_is_exact_crop() -> None:
    rng = np.random.default_rng(11)
    mask = rng.random((30, 45)) < 0.05
    mask[:5] = False
    pm = prepare_mask(_readonly(mask), 0)
    assert pm is not None
    ys, xs = np.nonzero(mask)
    assert pm.rect == Rect(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    np.testing.assert_array_equal(_full(pm, mask.shape), mask)


def test_prepare_mask_matches_exact_distance() -> None:
    """The crop is exactly the pixels within distance r, and the rect loses none of them."""
    rng = np.random.default_rng(2026)
    shapes = [(1, 1), (1, 9), (9, 1), (17, 31), (31, 17), (24, 60), (48, 40)]
    for trial in range(120):
        h, w = shapes[trial % len(shapes)]
        density = rng.choice([0.002, 0.02, 0.2, 0.7])
        mask = rng.random((h, w)) < density
        if trial % 7 == 0:
            mask[:] = False
            mask[rng.integers(h), rng.integers(w)] = True
        r = int(rng.choice([0, 1, 2, 3, 5, 7, 8, 13, 29, 40, 100]))
        pm = prepare_mask(_readonly(mask), r)
        if not mask.any():
            assert pm is None
            continue
        assert pm is not None
        assert pm.crop.shape == (pm.rect.height, pm.rect.width)
        assert _in_frame(pm.rect, h, w)
        expected = _within_distance(mask, r)
        np.testing.assert_array_equal(_full(pm, (h, w)), expected)
        inside = np.zeros((h, w), bool)
        inside[pm.rect.y0 : pm.rect.y1, pm.rect.x0 : pm.rect.x1] = True
        assert not (expected & ~inside).any()
        assert not (mask & ~_full(pm, (h, w))).any()


@pytest.mark.parametrize("r", [300, 1000, 2047, 2048, 4095])
def test_prepare_mask_exact_at_large_radii(r: int) -> None:
    """Up to r 4095 the float32 threshold neither loses distance r nor admits the next one."""
    h, w = 5, r + 40
    mask = np.zeros((h, w), bool)
    mask[0, 3] = True
    pm = prepare_mask(_readonly(mask), r)
    assert pm is not None
    full = _full(pm, (h, w))
    np.testing.assert_array_equal(full, _within_distance(mask, r))
    assert full[0, 3 + r]  # distance exactly r
    assert not full[1, 3 + r]  # distance sqrt(r*r + 1)


def test_prepare_mask_rect_is_bounding_rect_grown() -> None:
    mask = np.zeros((100, 80), bool)
    mask[30:61, 20:41] = True
    pm = prepare_mask(_readonly(mask), 6)
    assert pm is not None
    assert pm.rect == Rect(14, 24, 47, 67)
    big = prepare_mask(_readonly(mask), 500)
    assert big is not None
    assert big.rect == Rect(0, 0, 80, 100)
    assert big.crop.all()


def test_prepare_mask_absurd_radius_is_bounded_by_the_frame() -> None:
    mask = np.zeros((30, 40), bool)
    mask[3, 5] = True
    pm = prepare_mask(_readonly(mask), 10**15)
    assert pm is not None
    assert pm.rect == Rect(0, 0, 40, 30)
    assert pm.crop.shape == (30, 40)
    assert pm.crop.all()
    # The far corner is at distance sqrt(26*26 + 34*34), about 42.8.
    near = prepare_mask(_readonly(mask), 42)
    assert near is not None
    assert near.rect == Rect(0, 0, 40, 30)
    assert not near.crop[29, 39]
    assert near.crop[29, 37]


def test_prepare_mask_does_not_alias_or_mutate() -> None:
    mask = np.zeros((30, 40), bool)
    mask[10:20, 10:20] = True
    source = _readonly(mask)
    before = source.copy()
    for r in (0, 3, 1000):
        pm = prepare_mask(source, r)
        assert pm is not None
        assert not np.shares_memory(pm.crop, source)
        assert not pm.crop.flags.writeable
        np.testing.assert_array_equal(source, before)
    with pytest.raises(dataclasses.FrozenInstanceError):
        pm.rect = Rect(0, 0, 1, 1)  # type: ignore[misc]


def test_prepare_mask_non_contiguous_input() -> None:
    base = np.zeros((40, 60), bool)
    base[10:15, 20:24] = True
    view = base[:, ::2]  # strided columns
    view.setflags(write=False)
    pm = prepare_mask(view, 2)
    assert pm is not None
    np.testing.assert_array_equal(_full(pm, view.shape), _within_distance(np.array(view), 2))


@pytest.mark.parametrize(
    "mask",
    [np.zeros((4, 4), np.uint8), np.zeros((4, 4, 1), bool), [[True]]],
)
def test_prepare_mask_rejects_bad_masks(mask: object) -> None:
    with pytest.raises(ValueError):
        prepare_mask(mask, 1)  # type: ignore[arg-type]


def test_prepare_mask_rejects_negative_radius() -> None:
    with pytest.raises(ValueError):
        prepare_mask(_readonly(np.ones((3, 3), bool)), -1)


# --- orphan_persons ---------------------------------------------------------


def _face(x: int, y: int, w: int, h: int) -> Face:
    return Face(box=Box(x, y, w, h))


def test_orphan_constants() -> None:
    assert ORPHAN_MASK_SHARE == Fraction(30, 100)
    assert ORPHAN_HEAD_MASK_SHARE == Fraction(30, 100)
    assert FACE_ZONE == Fraction(1, 3)
    assert HEAD_ZONE == Fraction(1, 4)


def test_orphans_need_person_boxes() -> None:
    faces = (_face(10, 10, 10, 10),)
    for primary in ("face", "mask", "bodybox"):
        assert orphan_persons(None, faces, None, primary, 100, 100) == ()
        assert orphan_persons((), faces, None, primary, 100, 100) == ()


def test_orphans_face_primary() -> None:
    person = Box(100, 100, 200, 400)
    padded = pad_person(person, H, W)
    # A face center inside the top third: not an orphan.
    assert orphan_persons((person,), (_face(150, 120, 40, 50),), None, "face", H, W) == ()
    # No faces at all, or None: an orphan.
    assert orphan_persons((person,), (), None, "face", H, W) == (padded,)
    assert orphan_persons((person,), None, None, "face", H, W) == (padded,)
    # A face whose box overlaps the person but whose center is outside: an orphan.
    assert orphan_persons((person,), (_face(80, 80, 30, 30),), None, "face", H, W) == (padded,)
    # The mask is ignored for the face primary.
    full = np.ones((H, W), bool)
    assert orphan_persons((person,), (), full, "face", H, W) == (padded,)


@pytest.mark.parametrize(
    "face",
    [
        _face(90, 150, 20, 20),  # center x 100, on the left edge
        _face(290, 150, 20, 20),  # center x 300, on the right edge
        _face(150, 90, 20, 20),  # center y 100, on the top edge
        _face(90, 90, 20, 20),  # center on the top left corner
        _face(150, 190, 20, 20),  # center y 200, on the lower line of the face zone
    ],
)
def test_face_center_on_edge_does_not_clear(face: Face) -> None:
    """A center on an edge does not count, so the person box stays an orphan and is filled."""
    person = Box(100, 100, 200, 300)  # face zone: y 100 to 200
    assert orphan_persons((person,), (face,), None, "face", H, W) == (pad_person(person, H, W),)


def test_face_center_half_pixel_inside() -> None:
    person = Box(100, 100, 200, 300)
    # Box 91 to 110 has center 100.5, just inside the left edge.
    assert orphan_persons((person,), (_face(91, 150, 19, 20),), None, "face", H, W) == ()
    # Box 290 to 309 has center 299.5, just inside the right edge.
    assert orphan_persons((person,), (_face(290, 150, 19, 20),), None, "face", H, W) == ()
    # Box 189 to 210 has center 199.5, just above the face zone line at 200.
    assert orphan_persons((person,), (_face(150, 189, 20, 21),), None, "face", H, W) == ()


def test_face_zone_line_with_fractional_third() -> None:
    person = Box(0, 0, 100, 100)  # face zone ends at 33.33
    assert orphan_persons((person,), (_face(40, 23, 20, 20),), None, "face", H, W) == ()
    assert orphan_persons((person,), (_face(40, 24, 20, 20),), None, "face", H, W) == (
        pad_person(person, H, W),
    )


def test_face_low_in_the_box_does_not_clear() -> None:
    big = Box(50, 50, 300, 400)
    padded = pad_person(big, 480, 640)
    # A 3x3 false positive at belt height, and a hand near the feet.
    assert orphan_persons((big,), (_face(200, 300, 3, 3),), None, "face", 480, 640) == (padded,)
    assert orphan_persons((big,), (_face(200, 430, 20, 20),), None, "face", 480, 640) == (padded,)


def test_one_face_clears_one_of_two_overlapping_persons() -> None:
    a, b = Box(100, 100, 100, 300), Box(160, 120, 100, 300)
    # Center (182, 140) lies in the face zone of both; equal areas, so A (first) wins.
    face = _face(170, 125, 24, 30)
    assert orphan_persons((a, b), (face,), None, "face", 480, 640) == (pad_person(b, 480, 640),)
    assert orphan_persons((b, a), (face,), None, "face", 480, 640) == (pad_person(a, 480, 640),)


def test_one_face_clears_only_the_smallest_box() -> None:
    outer, inner = Box(0, 0, 400, 400), Box(100, 100, 50, 100)
    face = _face(110, 105, 20, 20)  # center (120, 115), in both face zones
    assert orphan_persons((outer, inner), (face,), None, "face", 480, 640) == (
        pad_person(outer, 480, 640),
    )
    assert orphan_persons((inner, outer), (face,), None, "face", 480, 640) == (
        pad_person(outer, 480, 640),
    )


def test_two_faces_two_persons() -> None:
    a, b = Box(100, 100, 100, 300), Box(160, 120, 100, 300)
    in_both = _face(170, 125, 24, 30)
    only_b = _face(220, 150, 24, 30)  # center (232, 165), only in B
    assert orphan_persons((a, b), (in_both, only_b), None, "face", 480, 640) == ()
    # Two faces that both pick A leave B an orphan: covering more on doubt.
    assert orphan_persons((a, b), (in_both, in_both), None, "face", 480, 640) == (
        pad_person(b, 480, 640),
    )


def test_two_persons_one_face() -> None:
    left, right = Box(0, 0, 100, 300), Box(200, 0, 100, 300)
    faces = (_face(30, 20, 40, 40),)
    assert orphan_persons((left, right), faces, None, "face", H, W) == (pad_person(right, H, W),)


def test_face_outside_every_person() -> None:
    persons = (Box(0, 0, 100, 300), Box(200, 0, 100, 300))
    faces = (_face(500, 500, 40, 40),)
    assert orphan_persons(persons, faces, None, "face", H, W) == tuple(
        pad_person(p, H, W) for p in persons
    )


def test_zero_area_face_does_not_clear_a_person() -> None:
    person = Box(100, 100, 200, 400)
    faces = (_face(150, 150, 0, 40), _face(150, 150, 40, 0))
    assert orphan_persons((person,), faces, None, "face", H, W) == (pad_person(person, H, W),)


def test_zero_area_persons_are_dropped() -> None:
    persons = (Box(100, 100, 0, 400), Box(100, 100, 200, 0), Box(W, 0, 0, 50))
    for primary in ("face", "mask"):
        assert orphan_persons(persons, (), None, primary, H, W) == ()


def test_orphans_mask_primary_threshold() -> None:
    person = Box(10, 10, 10, 10)  # area 100, head zone 3 rows (2.5 rounded up)
    padded = pad_person(person, 100, 100)
    mask = np.zeros((100, 100), bool)
    mask[10:13, 10:20] = True  # exactly 30 pixels, a share of 0.30
    mask.setflags(write=False)
    assert orphan_persons((person,), None, mask, "mask", 100, 100) == ()
    below = mask.copy()
    below[12, 19] = False  # 29 pixels, just below the share
    assert orphan_persons((person,), None, below, "mask", 100, 100) == (padded,)


def test_orphans_mask_head_zone_threshold() -> None:
    person = Box(0, 0, 10, 20)  # area 200, head zone rows 0 to 5, area 50
    padded = pad_person(person, 100, 100)
    mask = np.zeros((100, 100), bool)
    mask[5:20, 0:10] = True  # the body, 75 percent of the box
    mask[0, 0:10] = True
    mask[1, 0:5] = True  # 15 head pixels, exactly 0.30 of the head zone
    assert orphan_persons((person,), None, mask, "mask", 100, 100) == ()
    mask[1, 4] = False  # 14 head pixels, just below
    assert orphan_persons((person,), None, mask, "mask", 100, 100) == (padded,)


def test_orphans_mask_on_lower_body_only() -> None:
    big = Box(50, 50, 300, 400)
    mask = np.zeros((480, 640), bool)
    mask[300:450, 50:350] = True  # 37.5 percent of the box, nothing in the head zone
    assert orphan_persons((big,), None, mask, "mask", 480, 640) == (pad_person(big, 480, 640),)


def test_orphans_mask_counts_only_the_raw_box() -> None:
    person = Box(10, 10, 10, 10)
    mask = np.zeros((100, 100), bool)
    # Mask pixels in the padding ring around the box do not count.
    mask[8:22, 8:22] = True
    mask[10:20, 10:20] = False
    assert orphan_persons((person,), None, mask, "mask", 100, 100) == (
        pad_person(person, 100, 100),
    )


@pytest.mark.parametrize(
    ("person", "rows", "cols"),
    [
        (Box(90, 10, 20, 10), slice(10, 15), slice(90, 100)),  # past the right edge
        (Box(10, 90, 10, 20), slice(90, 100), slice(10, 15)),  # past the bottom edge
    ],
)
def test_orphans_mask_off_frame_part_counts_as_no_mask(
    person: Box, rows: slice, cols: slice
) -> None:
    """50 mask pixels: half the in-frame area, but only a quarter of the box's own area."""
    mask = np.zeros((100, 100), bool)
    mask[rows, cols] = True
    assert int(mask.sum()) == 50
    assert orphan_persons((person,), None, mask, "mask", 100, 100) == (
        pad_person(person, 100, 100),
    )


def test_orphans_mask_none_counts_as_zero() -> None:
    persons = (Box(10, 10, 10, 10), Box(50, 50, 20, 20))
    assert orphan_persons(persons, None, None, "mask", 100, 100) == tuple(
        pad_person(p, 100, 100) for p in persons
    )


def test_orphans_mask_primary_ignores_faces() -> None:
    person = Box(10, 10, 10, 10)
    mask = np.zeros((100, 100), bool)
    assert orphan_persons((person,), (_face(12, 12, 4, 4),), mask, "mask", 100, 100) == (
        pad_person(person, 100, 100),
    )


def test_orphans_mask_known_limit_other_persons_silhouette() -> None:
    """Documented limit: the mask has no identity, so A's silhouette can vouch for B."""
    a, b = Box(100, 100, 100, 300), Box(160, 120, 100, 300)
    mask = np.zeros((480, 640), bool)
    mask[100:400, 100:200] = True  # only A is segmented
    assert orphan_persons((a, b), None, mask, "mask", 480, 640) == ()


def test_orphans_bodybox_primary() -> None:
    persons = (Box(10, 10, 10, 10),)
    assert orphan_persons(persons, (), None, "bodybox", 100, 100) == ()


def test_orphans_reject_bad_input() -> None:
    with pytest.raises(ValueError, match="primary"):
        orphan_persons((Box(0, 0, 5, 5),), (), None, "faces", 10, 10)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        orphan_persons((Box(0, 0, 5, 5),), (), np.zeros((5, 5), bool), "mask", 10, 10)
    with pytest.raises(ValueError):
        orphan_persons((Box(0, 0, -5, 5),), (), None, "face", 10, 10)
