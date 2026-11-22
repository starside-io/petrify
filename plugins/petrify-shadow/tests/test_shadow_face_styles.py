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

"""Face styles: region bounds, per-style properties, irreversibility, eyebar geometry."""

from __future__ import annotations

import math
import time

import cv2
import numpy as np
import pytest
from petrify_shadow.regions import Rect, pad_face
from petrify_shadow.styles import face as face_module
from petrify_shadow.styles.face import (
    GAUSSIAN_SIGMA_SHARE,
    PIXELATE_BLOCKS,
    apply_eyebar,
    apply_fill,
    apply_gaussian,
    apply_noise,
    apply_pixelate,
    eyebar_polygon,
)

from petrify.core.detections import Box, Face

STRENGTHS = ("high", "medium", "low")
COLOR = (17, 200, 93)

GAUSSIAN_MIN_MAD = 40.0
"""Smallest mean absolute difference a gaussian cover must cause on uniform random input.

Uniform values in [1, 255] lie about 64 on average from their mean, so a strong
blur gives close to 64; 40 leaves room for the smallest rects.
"""

GAUSSIAN_MIN_HF_REDUCTION = 50.0
"""Factor by which the gaussian cover must reduce high-frequency energy (rects of 4 px and up)."""


def _frame(h: int, w: int, seed: int = 0) -> np.ndarray:
    """Random uint8 frame in [1, 255], read-only so a stray write to it raises."""
    img = np.random.default_rng(seed).integers(1, 256, (h, w, 3), dtype=np.uint8)
    img.flags.writeable = False
    return img


def _structured(h: int, w: int, seed: int = 0) -> np.ndarray:
    """Frame with detail at several scales, so stronger covers change it measurably more."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w, 3), np.float64)
    for cell in (2, 6, 20, 60):
        small = rng.uniform(-1.0, 1.0, (max(1, h // cell), max(1, w // cell), 3))
        acc += cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    img = np.clip(128 + 30 * acc, 1, 255).astype(np.uint8)
    img.flags.writeable = False
    return img


def _rect_mask(shape: tuple[int, ...], x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros(shape[:2], bool)
    mask[max(y0, 0) : max(y1, 0), max(x0, 0) : max(x1, 0)] = True
    return mask


def _assert_outside_unchanged(before: np.ndarray, after: np.ndarray, region: np.ndarray) -> None:
    assert np.array_equal(before[~region], after[~region])


def _hf_energy(a: np.ndarray) -> float:
    f = a.astype(np.float64)
    return float((np.diff(f, axis=0) ** 2).mean() + (np.diff(f, axis=1) ** 2).mean())


def _distinct_colors(a: np.ndarray) -> int:
    packed = a[..., 0].astype(np.int32) | (a[..., 1].astype(np.int32) << 8) | (a[..., 2].astype(np.int32) << 16)
    return len(np.unique(packed))


LADDER = sorted({base * 2**k for base in (4, 6) for k in range(12)})
"""Block size ladder written out independently of the module: 4, 6, 8, 12, 16, 24, ..."""


def _oracle_block(side: int, strength: str) -> int:
    need = max(4, math.ceil(side / PIXELATE_BLOCKS[strength]))
    return next(v for v in LADDER if v >= need)


def _pixelate_grid(rect: Rect, strength: str, h: int, w: int) -> tuple[int, int, int, int, int]:
    """Independent oracle: block size and the grown, frame-clipped rect."""
    block = _oracle_block(max(rect.width, rect.height), strength)
    gx0 = (rect.x0 // block) * block
    gy0 = (rect.y0 // block) * block
    gx1 = min(math.ceil(rect.x1 / block) * block, w)
    gy1 = min(math.ceil(rect.y1 / block) * block, h)
    return block, gx0, gy0, gx1, gy1


# (frame height, frame width, rect): frame edges, corners, the whole frame,
# non-square rects and frames, tiny rects.
CASES = [
    (120, 160, Rect(40, 30, 100, 90)),
    (120, 160, Rect(0, 0, 37, 53)),
    (120, 160, Rect(113, 71, 160, 120)),
    (120, 160, Rect(0, 50, 160, 70)),
    (120, 160, Rect(0, 0, 160, 120)),
    (97, 131, Rect(130, 96, 131, 97)),
    (97, 131, Rect(5, 90, 49, 97)),
    (97, 131, Rect(61, 0, 66, 40)),
    (7, 5, Rect(1, 2, 4, 6)),
    (1, 1, Rect(0, 0, 1, 1)),
    (480, 640, Rect(250, 100, 400, 295)),
]
CASE_IDS = [f"{h}x{w}-{r.x0},{r.y0},{r.x1},{r.y1}" for h, w, r in CASES]


def _render(style: str, out: np.ndarray, rect: Rect, strength: str = "high", seed: int = 3) -> None:
    if style == "gaussian":
        apply_gaussian(out, rect, strength)
    elif style == "pixelate":
        apply_pixelate(out, rect, strength)
    elif style == "noise":
        apply_noise(out, rect, np.random.default_rng(seed))
    elif style == "facebox":
        apply_fill(out, rect, COLOR)
    elif style == "eyebar":
        apply_eyebar(out, Face(Box(rect.x0, rect.y0, rect.width, rect.height)), rect, COLOR)
    else:
        raise AssertionError(style)


ALL_STYLES = ("gaussian", "pixelate", "noise", "facebox", "eyebar")


# Argument validation


def _bad_outs() -> list[np.ndarray]:
    ro = np.zeros((10, 10, 3), np.uint8)
    ro.flags.writeable = False
    base = np.zeros((10, 10, 3), np.uint8)
    return [
        ro,
        np.zeros((10, 10), np.uint8),
        np.zeros((10, 10, 4), np.uint8),
        np.zeros((10, 10, 3), np.float32),
        base[:, ::-1],
        base[:, :, ::-1],
        base[::-1],
        base[:, 2:8],
        base[::2],
    ]


@pytest.mark.parametrize("style", ALL_STYLES)
@pytest.mark.parametrize("index", range(9))
def test_bad_out_is_rejected(style: str, index: int) -> None:
    out = _bad_outs()[index]
    with pytest.raises(ValueError):
        _render(style, out, Rect(1, 1, 5, 5))


@pytest.mark.parametrize("style", ALL_STYLES)
def test_non_array_out_is_rejected(style: str) -> None:
    with pytest.raises(ValueError):
        _render(style, [[[0, 0, 0]]], Rect(0, 0, 1, 1))  # type: ignore[arg-type]


def test_bad_strength_color_and_rng_are_rejected() -> None:
    out = np.zeros((10, 10, 3), np.uint8)
    with pytest.raises(ValueError):
        apply_gaussian(out, Rect(0, 0, 5, 5), "max")
    with pytest.raises(ValueError):
        apply_pixelate(out, Rect(0, 0, 5, 5), "max")
    for color in (
        (0, 0), (0, 0, 256), (0, -1, 0), (0.5, 0, 0), (True, 0, 0), (np.bool_(True), 0, 0),
        b"abc", bytearray(b"abc"), "abc", None, 7,
    ):
        with pytest.raises(ValueError):
            apply_fill(out, Rect(0, 0, 5, 5), color)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        apply_noise(out, Rect(0, 0, 5, 5), None)  # type: ignore[arg-type]
    assert not out.any()


@pytest.mark.parametrize("style", ALL_STYLES)
@pytest.mark.parametrize(
    "rect",
    [Rect(10, 10, 20.9, 20), Rect(10.0, 10, 20, 20), Rect(10, 10, 20, np.float64(20)), Rect(True, 0, 5, 5)],  # type: ignore[arg-type]
)
def test_non_integral_rect_bounds_are_rejected(style: str, rect: Rect) -> None:
    # Truncating a float bound would leave its last column or row uncovered.
    out = _frame(30, 30).copy()
    before = out.copy()
    with pytest.raises(ValueError):
        _render(style, out, rect)
    assert np.array_equal(out, before)


def test_numpy_integer_rect_bounds_are_accepted() -> None:
    out = _frame(30, 30).copy()
    apply_fill(out, Rect(np.int64(2), np.int32(3), np.uint16(9), np.int64(12)), COLOR)
    assert (out[3:12, 2:9] == COLOR).all()


@pytest.mark.parametrize("style", ALL_STYLES)
@pytest.mark.parametrize("rect", [Rect(4, 4, 4, 9), Rect(3, 3, 3, 3), Rect(9, 2, 5, 6)])
def test_empty_rect_changes_nothing(style: str, rect: Rect) -> None:
    img = _frame(12, 14)
    out = img.copy()
    _render(style, out, rect)
    assert np.array_equal(out, img)


@pytest.mark.parametrize("style", ALL_STYLES)
def test_rect_past_the_frame_does_not_wrap(style: str) -> None:
    # A rect starting at negative coordinates must be clipped, not read as numpy
    # negative indices that would land on the opposite edge of the frame.
    img = _frame(40, 50)
    out = img.copy()
    _render(style, out, Rect(-5, -5, 10, 10))
    # The clipped rect is (0, 0, 10, 10); pixelate grows it to its 4 px grid.
    limit = 12 if style == "pixelate" else 10
    region = _rect_mask(out.shape, 0, 0, limit, limit)
    _assert_outside_unchanged(img, out, region)


@pytest.mark.parametrize("style", ALL_STYLES)
def test_contiguous_row_slice_is_drawn_in_place(style: str) -> None:
    # A slice of whole rows is C-contiguous: OpenCV and numpy both write through it.
    img = _frame(60, 80)
    big = img.copy()
    view = big[10:50]
    assert view.flags.c_contiguous
    _render(style, view, Rect(5, 5, 30, 30))
    assert not np.array_equal(big, img)
    region = _rect_mask(big.shape, 0, 10, 80, 50)
    _assert_outside_unchanged(img, big, region)


# facebox


@pytest.mark.parametrize(("h", "w", "rect"), CASES, ids=CASE_IDS)
def test_fill(h: int, w: int, rect: Rect) -> None:
    img = _frame(h, w)
    out = img.copy()
    apply_fill(out, rect, COLOR)
    region = _rect_mask(out.shape, rect.x0, rect.y0, rect.x1, rect.y1)
    assert (out[region] == COLOR).all()
    _assert_outside_unchanged(img, out, region)
    assert not np.may_share_memory(out, img)


# noise


@pytest.mark.parametrize(("h", "w", "rect"), CASES, ids=CASE_IDS)
def test_noise_replaces_with_generator_output(h: int, w: int, rect: Rect) -> None:
    img = _frame(h, w)
    out = img.copy()
    apply_noise(out, rect, np.random.default_rng(11))
    expected = np.random.default_rng(11).integers(0, 256, (rect.height, rect.width, 3), dtype=np.uint8)
    assert np.array_equal(out[rect.y0 : rect.y1, rect.x0 : rect.x1], expected)
    region = _rect_mask(out.shape, rect.x0, rect.y0, rect.x1, rect.y1)
    _assert_outside_unchanged(img, out, region)


def test_noise_differs_between_calls_on_one_generator() -> None:
    img = _frame(100, 120)
    rect = Rect(10, 10, 90, 84)
    rng = np.random.default_rng(5)
    first = img.copy()
    second = img.copy()
    apply_noise(first, rect, rng)
    apply_noise(second, rect, rng)
    a = first[rect.y0 : rect.y1, rect.x0 : rect.x1]
    b = second[rect.y0 : rect.y1, rect.x0 : rect.x1]
    assert a.shape[0] >= 64 and a.shape[1] >= 64
    assert (a == b).mean() < 0.05


def test_noise_never_creates_or_seeds_a_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(9)
    out = _frame(30, 30).copy()

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("apply_noise must use the generator it is given")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(np.random, "seed", forbidden)
    apply_noise(out, Rect(2, 2, 20, 20), rng)


# Irreversibility


def _differ_inside(h: int, w: int, rect: Rect) -> tuple[np.ndarray, np.ndarray]:
    a = _frame(h, w, seed=1)
    b = a.copy()
    b[rect.y0 : rect.y1, rect.x0 : rect.x1] = _frame(h, w, seed=2)[rect.y0 : rect.y1, rect.x0 : rect.x1]
    assert not np.array_equal(a, b)
    return a, b


@pytest.mark.parametrize("style", ["noise", "facebox"])
@pytest.mark.parametrize(("h", "w", "rect"), CASES, ids=CASE_IDS)
def test_cover_carries_no_information_about_the_input(style: str, h: int, w: int, rect: Rect) -> None:
    a, b = _differ_inside(h, w, rect)
    out_a, out_b = a.copy(), b.copy()
    _render(style, out_a, rect, seed=21)
    _render(style, out_b, rect, seed=21)
    sl = (slice(rect.y0, rect.y1), slice(rect.x0, rect.x1))
    assert np.array_equal(out_a[sl], out_b[sl])


@pytest.mark.parametrize("style", ["gaussian", "pixelate"])
@pytest.mark.parametrize("strength", STRENGTHS)
def test_gaussian_and_pixelate_depend_on_the_input(style: str, strength: str) -> None:
    # Documents the limit: these covers are a function of the face pixels, so they
    # make no irreversibility claim (the factory's weakened() says so).
    rect = Rect(30, 20, 130, 140)
    a, b = _differ_inside(160, 160, rect)
    out_a, out_b = a.copy(), b.copy()
    _render(style, out_a, rect, strength)
    _render(style, out_b, rect, strength)
    sl = (slice(rect.y0, rect.y1), slice(rect.x0, rect.x1))
    assert not np.array_equal(out_a[sl], out_b[sl])




# gaussian


@pytest.mark.parametrize("strength", STRENGTHS)
@pytest.mark.parametrize(("h", "w", "rect"), CASES, ids=CASE_IDS)
def test_gaussian(h: int, w: int, rect: Rect, strength: str) -> None:
    img = _frame(h, w)
    out = img.copy()
    apply_gaussian(out, rect, strength)
    region = _rect_mask(out.shape, rect.x0, rect.y0, rect.x1, rect.y1)
    _assert_outside_unchanged(img, out, region)
    if rect.width >= 4 and rect.height >= 4:
        before = img[rect.y0 : rect.y1, rect.x0 : rect.x1]
        after = out[rect.y0 : rect.y1, rect.x0 : rect.x1]
        assert np.abs(before.astype(np.int16) - after).mean() > GAUSSIAN_MIN_MAD
        assert _hf_energy(before) > GAUSSIAN_MIN_HF_REDUCTION * _hf_energy(after)


def _edge_spread(profile: np.ndarray) -> float:
    """Standard deviation of the derivative of a rising edge profile (its line spread)."""
    d = np.clip(np.diff(profile.astype(np.float64)), 0.0, None)
    x = np.arange(len(d)) + 0.5
    mean = float((d * x).sum() / d.sum())
    return math.sqrt(float((d * (x - mean) ** 2).sum() / d.sum()))


def _exact_step_profile(n: int, edge: int, sigma: float) -> np.ndarray:
    """An exact gaussian of ``sigma`` applied to a 0 to 1 step between pixels ``edge - 1`` and ``edge``.

    BORDER_REPLICATE extends both sides of the step to infinity, so the exact result
    inside the window is the normal CDF at each pixel center.
    """
    x = np.arange(n) - (edge - 0.5)
    return np.array([0.5 * (1.0 + math.erf(v / (sigma * math.sqrt(2.0)))) for v in x])


GAUSSIAN_SIDES = sorted({*range(10, 651, 16), 11, 13, 650})


@pytest.mark.parametrize("strength", STRENGTHS)
def test_gaussian_effective_sigma_is_at_least_the_target(strength: str) -> None:
    # The edge spread is measured inside the rect, where the window truncates the
    # tails: an exact gaussian of the target sigma itself measures only 0.74 to 0.94
    # of sigma for "high". So the cover is compared with an exact gaussian of the
    # target sigma, measured the same way on the same window: at or above it means
    # the effective sigma is at least the target.
    worst = math.inf
    for side in GAUSSIAN_SIDES:
        target = GAUSSIAN_SIGMA_SHARE[strength] * side
        for vertical, offset in ((True, 0), (True, -2), (False, 3)):
            if vertical:
                h, w = side, side
                edge = w // 2 + offset
                img = np.zeros((h, w, 3), np.uint8)
                img[:, edge:] = 255
            else:
                # Non-square rect, longer side vertical, horizontal edge.
                h, w = side, max(1, int(side / 1.3))
                edge = h // 2 + offset
                img = np.zeros((h, w, 3), np.uint8)
                img[edge:, :] = 255
            apply_gaussian(img, Rect(0, 0, w, h), strength)
            profile = img[:, w // 2, 1] if not vertical else img[h // 2, :, 1]
            got = _edge_spread(profile / 255.0)
            ref = _edge_spread(_exact_step_profile(len(profile), edge, target))
            worst = min(worst, got / ref)
    assert worst >= 1.0, f"effective sigma {worst:.3f} of an exact gaussian of the target"


def test_gaussian_reads_its_source_from_out() -> None:
    # A rect already covered by a flat fill stays that fill: the blur sees the
    # cover in out, not the original frame pixels.
    img = _frame(80, 80)
    out = img.copy()
    rect = Rect(10, 10, 70, 60)
    apply_fill(out, rect, COLOR)
    apply_gaussian(out, rect, "high")
    assert (out[rect.y0 : rect.y1, rect.x0 : rect.x1] == COLOR).all()


# pixelate


@pytest.mark.parametrize("strength", STRENGTHS)
@pytest.mark.parametrize(("h", "w", "rect"), CASES, ids=CASE_IDS)
def test_pixelate(h: int, w: int, rect: Rect, strength: str) -> None:
    img = _frame(h, w)
    out = img.copy()
    apply_pixelate(out, rect, strength)
    block, gx0, gy0, gx1, gy1 = _pixelate_grid(rect, strength, h, w)
    assert gx0 <= rect.x0 and gy0 <= rect.y0 and gx1 >= rect.x1 and gy1 >= rect.y1
    region = _rect_mask(out.shape, gx0, gy0, gx1, gy1)
    _assert_outside_unchanged(img, out, region)

    blocks = PIXELATE_BLOCKS[strength]
    nbx = math.ceil((gx1 - gx0) / block)
    nby = math.ceil((gy1 - gy0) / block)
    for by in range(gy0, gy1, block):
        for bx in range(gx0, gx1, block):
            cell = out[by : min(by + block, gy1), bx : min(bx + block, gx1)].reshape(-1, 3)
            assert (cell == cell[0]).all()
            if by + block <= gy1 and bx + block <= gx1:
                # A whole block holds the rounded mean of its own input pixels.
                mean = img[by : by + block, bx : bx + block].reshape(-1, 3).mean(axis=0)
                assert np.abs(cell[0].astype(np.float64) - mean).max() <= 1.0
    distinct = _distinct_colors(out[gy0:gy1, gx0:gx1])
    assert distinct <= nbx * nby <= (blocks + 1) ** 2


@pytest.mark.parametrize("strength", STRENGTHS)
@pytest.mark.parametrize("side", range(1, 21))
@pytest.mark.parametrize("origin", [(0, 0), (5, 7)])
def test_pixelate_changes_a_small_rect(strength: str, side: int, origin: tuple[int, int]) -> None:
    # Without a minimum block size, a rect of at most ``blocks`` pixels would get 1
    # pixel blocks and come out unchanged (a face clipped at a frame corner).
    img = _frame(40, 40, seed=side)
    out = img.copy()
    x0, y0 = origin
    rect = Rect(x0, y0, x0 + side, y0 + side)
    apply_pixelate(out, rect, strength)
    sl = (slice(rect.y0, rect.y1), slice(rect.x0, rect.x1))
    assert not np.array_equal(out[sl], img[sl])
    block, gx0, gy0, gx1, gy1 = _pixelate_grid(rect, strength, 40, 40)
    assert block >= 4
    for by in range(gy0, gy1, block):
        for bx in range(gx0, gx1, block):
            cell = out[by : by + block, bx : bx + block].reshape(-1, 3)
            assert (cell == cell[0]).all()


def test_pixelate_block_follows_the_ladder() -> None:
    for strength in STRENGTHS:
        for side in range(1, 2200):
            block = face_module._pixelate_block(side, strength)
            assert block == _oracle_block(side, strength)
            raw = math.ceil(side / PIXELATE_BLOCKS[strength])
            assert block >= max(4, raw)
            # Each ladder step is at most 1.5 times the previous one.
            if raw > 4:
                assert block < 1.5 * raw


def test_upscale_blocks_is_exact_for_block_49() -> None:
    # cv2.resize with INTER_NEAREST gives the first pixel of each 49 px block the
    # previous block's value. The ladder never produces 49, so this pins the helper
    # itself against a switch back to cv2.resize.
    small = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    big = face_module._upscale_blocks(small, 49)
    ys, xs = np.mgrid[0 : 4 * 49, 0 : 5 * 49]
    assert big.shape == (4 * 49, 5 * 49, 3)
    assert np.array_equal(big, small[ys // 49, xs // 49])


@pytest.mark.parametrize("strength", STRENGTHS)
def test_pixelate_edge_blocks_use_their_own_pixels(strength: str) -> None:
    # Blocks cut by the frame edge are completed by replicating edge pixels. On a
    # flat frame every block, cut or not, must keep the flat color; zero padding
    # would darken the cut blocks.
    img = np.full((50, 50, 3), 200, np.uint8)
    out = img.copy()
    apply_pixelate(out, Rect(31, 33, 50, 50), strength)
    block = _oracle_block(19, strength)
    assert 50 % block != 0
    assert np.array_equal(out, img)


def test_pixelate_grid_is_anchored_to_the_frame() -> None:
    # Two rects of the same size a few pixels apart share the grid, so the blocks
    # they both fully cover come out identical.
    img = _structured(200, 200)
    a, b = img.copy(), img.copy()
    apply_pixelate(a, Rect(40, 40, 100, 100), "high")
    apply_pixelate(b, Rect(43, 42, 103, 102), "high")
    assert np.array_equal(a[50:100, 50:100], b[50:100, 50:100])


def _jittered_sides(face_w: int, frames: int, rng: np.random.Generator) -> list[int]:
    """Longer side of the padded rect for boxes jittered by up to 2 px on each value."""
    sides = []
    for _ in range(frames):
        j = rng.integers(-2, 3, 4)
        box = Box(500 + int(j[0]), 300 + int(j[1]), face_w + int(j[2]), int(face_w * 1.3) + int(j[3]))
        r = pad_face(box, 0.3, 1080, 1920)
        sides.append(max(r.width, r.height))
    return sides


def _change_share(values: list[int]) -> float:
    a = np.array(values)
    return float(np.mean(a[1:] != a[:-1]))


JITTER_MAX_CHANGE = 0.10
"""Largest share of consecutive frames whose block size may differ, away from a ladder threshold."""

JITTER_MAX_MEAN_CHANGE = 0.05
"""Largest share averaged over every face width from 24 to 400 px.

The jitter moves the raw block size by at most one unit, and a ladder step spans
a third to a half of its value, so only widths whose raw size sits next to a
threshold change at all; measured, about 2.6 percent, against 16 to 41 percent
before the ladder.
"""


@pytest.mark.parametrize("strength", STRENGTHS)
@pytest.mark.parametrize("face_w", [32, 64, 150, 400])
def test_pixelate_block_size_is_stable_under_jitter(face_w: int, strength: str) -> None:
    sides = _jittered_sides(face_w, 200, np.random.default_rng(face_w))
    blocks = [face_module._pixelate_block(s, strength) for s in sides]
    near_threshold = _oracle_block(min(sides), strength) != _oracle_block(max(sides), strength)
    if near_threshold:
        # The jitter range straddles a ladder threshold, so the size may flip; it
        # still only takes the two neighbouring ladder values.
        assert len(set(blocks)) <= 2
    else:
        assert _change_share(blocks) <= JITTER_MAX_CHANGE
        assert len(set(blocks)) == 1


@pytest.mark.parametrize("strength", STRENGTHS)
def test_pixelate_block_size_changes_rarely_across_face_widths(strength: str) -> None:
    rng = np.random.default_rng(0)
    shares = []
    for face_w in range(24, 401, 2):
        sides = _jittered_sides(face_w, 40, rng)
        shares.append(_change_share([face_module._pixelate_block(s, strength) for s in sides]))
    assert float(np.mean(shares)) <= JITTER_MAX_MEAN_CHANGE


# Strength ordering


@pytest.mark.parametrize("style", ["gaussian", "pixelate"])
@pytest.mark.parametrize("side", [150, 400])
def test_stronger_changes_more(style: str, side: int) -> None:
    tall = int(side * 1.3)
    img = _structured(tall + 40, side + 40, seed=4)
    rect = Rect(20, 20, 20 + side, 20 + tall)
    change = {}
    for strength in STRENGTHS:
        out = img.copy()
        _render(style, out, rect, strength)
        # Measured inside the input rect only: the pixelate grown rect differs per
        # strength, and counting it would reward a larger block grid, not a stronger cover.
        sl = (slice(rect.y0, rect.y1), slice(rect.x0, rect.x1))
        change[strength] = float(np.abs(out[sl].astype(np.int16) - img[sl]).mean())
    assert change["high"] > change["medium"] > change["low"] > 0


# eyebar


def _eyebar_mask(shape: tuple[int, ...], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, polygon, 1, lineType=cv2.LINE_8)
    return mask.astype(bool)


def _corners(polygon: np.ndarray | None) -> list[tuple[int, int]]:
    assert polygon is not None
    assert polygon.dtype == np.int32 and polygon.shape == (4, 2)
    return sorted(map(tuple, polygon.tolist()))


def test_eyebar_polygon_horizontal_is_literal() -> None:
    rect = Rect(20, 20, 80, 100)
    eyes = ((40.0, 50.0), (60.0, 50.0), (50.0, 60.0), (42.0, 70.0), (58.0, 70.0))
    # d = 20, box width 40: length max(44, 40) = 44, thickness max(3, 14, 10) = 14.
    assert _corners(eyebar_polygon(eyes, rect, 40)) == [(28, 43), (28, 57), (72, 43), (72, 57)]


def test_eyebar_polygon_profile_face_uses_the_box_floors() -> None:
    # A profile face: d = 16 is about 0.15 of the 104 px box. Length max(35.2, 104)
    # = 104, thickness max(3, 11.2, 26) = 26, centered on (103, 60).
    rect = Rect(40, 20, 170, 180)
    eyes = ((95.0, 60.0), (111.0, 60.0))
    assert _corners(eyebar_polygon(eyes, rect, 104)) == [(51, 47), (51, 73), (155, 47), (155, 73)]


def test_eyebar_corners_are_rounded_not_truncated() -> None:
    rect = Rect(20, 20, 80, 100)
    eyes = ((40.6, 50.6), (60.6, 50.6))
    # Exact corners 28.6, 72.6, 43.6, 57.6: rounding gives 29, 73, 44, 58.
    assert _corners(eyebar_polygon(eyes, rect, 40)) == [(29, 44), (29, 58), (73, 44), (73, 58)]


def test_eyebar_has_a_minimum_thickness() -> None:
    # d = 2.5 and a box 8 px wide: 0.7 * d and 0.25 * 8 are both under 3 px, so
    # the 3 px floor sets the thickness.
    rect = Rect(10, 10, 30, 30)
    polygon = eyebar_polygon(((19.0, 20.0), (21.5, 20.0)), rect, 8)
    assert polygon is not None
    assert int(polygon[:, 1].max() - polygon[:, 1].min()) >= 3


def _eyes_at(angle_deg: float, center: tuple[float, float], d: float) -> tuple[tuple[float, float], ...]:
    a = math.radians(angle_deg)
    hx, hy = math.cos(a) * d / 2, math.sin(a) * d / 2
    cx, cy = center
    return ((cx - hx, cy - hy), (cx + hx, cy + hy), (cx, cy + d / 2), (cx - hx, cy + d), (cx + hx, cy + d))


@pytest.mark.parametrize("angle", [30.0, -45.0])
@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("box_w", [40, 80])
def test_eyebar_follows_the_head(angle: float, swap: bool, box_w: int) -> None:
    img = _frame(160, 160)
    rect = Rect(40, 30, 120, 130)
    landmarks = _eyes_at(angle, (80.0, 70.0), 30.0)
    if swap:
        landmarks = (landmarks[1], landmarks[0], *landmarks[2:])
    polygon = eyebar_polygon(landmarks, rect, box_w)
    assert polygon is not None
    contour = polygon.reshape(-1, 1, 2).astype(np.float32)
    for eye in landmarks[:2]:
        assert cv2.pointPolygonTest(contour, (float(eye[0]), float(eye[1])), False) > 0
    # The long edge is parallel to the eye line (modulo 180 degrees).
    edges = [polygon[(i + 1) % 4] - polygon[i] for i in range(4)]
    longest = max(edges, key=lambda e: float(np.hypot(*e)))
    edge_angle = math.degrees(math.atan2(float(longest[1]), float(longest[0])))
    assert abs(((edge_angle - angle) + 90) % 180 - 90) < 3
    assert float(np.hypot(*longest)) >= max(2.2 * 30.0, box_w) - 1.5

    out = img.copy()
    face = Face(Box(60, 40, box_w, box_w), landmarks)
    apply_eyebar(out, face, rect, COLOR)
    mask = _eyebar_mask(out.shape, polygon)
    assert (out[mask] == COLOR).all()
    _assert_outside_unchanged(img, out, mask)
    for x, y in landmarks[:2]:
        assert tuple(out[round(y), round(x)]) == COLOR


def test_eyebar_swapped_eyes_give_the_same_bar() -> None:
    rect = Rect(40, 30, 120, 130)
    landmarks = _eyes_at(30.0, (80.0, 70.0), 30.0)
    swapped = (landmarks[1], landmarks[0], *landmarks[2:])
    a = eyebar_polygon(landmarks, rect, 50)
    b = eyebar_polygon(swapped, rect, 50)
    assert a is not None and b is not None
    assert np.array_equal(_eyebar_mask((200, 200), a), _eyebar_mask((200, 200), b))


FALLBACKS = {
    "none": None,
    "one_point": ((60.0, 50.0),),
    "close_eyes": ((60.0, 50.0), (61.0, 51.0)),
    "far_eyes": ((10.0, 50.0), (110.0, 50.0)),  # d = 100 > 1.5 * 60
    "midpoint_left": ((10.0, 50.0), (30.0, 50.0)),
    "midpoint_below": ((50.0, 115.0), (70.0, 115.0)),
    "midpoint_on_right_edge": ((80.0, 50.0), (100.0, 50.0)),
    "nan": ((float("nan"), 50.0), (70.0, 50.0)),
}


@pytest.mark.parametrize("name", list(FALLBACKS))
def test_eyebar_fallback_fills_the_rect(name: str) -> None:
    img = _frame(130, 140)
    rect = Rect(30, 20, 90, 100)
    landmarks = FALLBACKS[name]
    assert eyebar_polygon(landmarks, rect, 40) is None
    out = img.copy()
    apply_eyebar(out, Face(Box(40, 30, 40, 50), landmarks), rect, COLOR)
    expected = img.copy()
    apply_fill(expected, rect, COLOR)
    assert np.array_equal(out, expected)


def test_eyebar_may_extend_past_the_rect_and_the_frame() -> None:
    # Face in the top-left corner with one eye off the frame: the bar is clipped by
    # OpenCV, still covers the visible eye, and nothing outside the bar changes.
    img = _frame(60, 80)
    rect = Rect(0, 0, 30, 40)
    landmarks = ((-6.0, 3.0), (14.0, 5.0))
    polygon = eyebar_polygon(landmarks, rect, 20)
    assert polygon is not None
    assert polygon[:, 0].min() < 0 and polygon[:, 1].min() < 0
    out = img.copy()
    apply_eyebar(out, Face(Box(0, 0, 20, 30), landmarks), rect, COLOR)
    mask = _eyebar_mask(out.shape, polygon)
    assert (out[mask] == COLOR).all()
    _assert_outside_unchanged(img, out, mask)
    assert tuple(out[5, 14]) == COLOR


def test_eyebar_with_landmarks_far_off_frame_falls_back() -> None:
    img = _frame(50, 50)
    rect = Rect(10, 10, 40, 40)
    out = img.copy()
    apply_eyebar(out, Face(Box(10, 10, 30, 30), ((-1e9, -1e9), (1e9, 1e9))), rect, COLOR)
    expected = img.copy()
    apply_fill(expected, rect, COLOR)
    assert np.array_equal(out, expected)


# Timing (printed, not asserted)


def _ms(run, out: np.ndarray, repeats: int = 30) -> float:  # type: ignore[no-untyped-def]
    run(out)
    start = time.perf_counter()
    for _ in range(repeats):
        run(out)
    return (time.perf_counter() - start) / repeats * 1000


def test_print_timings(capsys: pytest.CaptureFixture[str]) -> None:
    img = _structured(1080, 1920)
    rng = np.random.default_rng(0)
    lines = []
    for side in (150, 400):
        rect = Rect(600, 300, 600 + side, 300 + int(side * 1.3))
        d = 0.36 * side
        landmarks = ((600 + side / 2 - d / 2, 300 + side * 0.5), (600 + side / 2 + d / 2, 300 + side * 0.5))
        face = Face(Box(rect.x0, rect.y0, rect.width, rect.height), landmarks)
        runs = {
            "gaussian": lambda out, r=rect: apply_gaussian(out, r, "high"),
            "pixelate": lambda out, r=rect: apply_pixelate(out, r, "high"),
            "noise": lambda out, r=rect: apply_noise(out, r, rng),
            "facebox": lambda out, r=rect: apply_fill(out, r, COLOR),
            "eyebar": lambda out, r=rect, f=face: apply_eyebar(out, f, r, COLOR),
        }
        for name, run in runs.items():
            lines.append(f"{name:9s} {side} px face: {_ms(run, img.copy()):.3f} ms")
    # Padded rects of a 150 px face (242x310), laid out over the frame.
    for count in (1, 10, 50):
        rects = [
            Rect(x, y, min(x + 242, 1920), min(y + 310, 1080))
            for i in range(count)
            for x, y in [(30 + (i % 7) * 260, 20 + (i // 7) * 120)]
        ]

        def run_fill(out: np.ndarray, rs: list[Rect] = rects) -> None:
            for r in rs:
                apply_fill(out, r, COLOR)

        lines.append(f"facebox   {count:2d} faces 242x310: {_ms(run_fill, img.copy()):.3f} ms")
    with capsys.disabled():
        print("\nface style timings (1080p frame, high strength):")
        for line in lines:
            print("  " + line)
