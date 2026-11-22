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

"""Body styles: silhouette, outline and bodybox fills, on synthetic frames."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
from petrify_shadow.options import COLORS
from petrify_shadow.regions import PreparedMask, Rect, dilate_px, prepare_mask
from petrify_shadow.styles.body import (
    apply_bodybox,
    apply_outline,
    apply_silhouette,
    outline_thickness,
)

BLACK = COLORS["black"]
SIZES = [(1, 1), (2, 3), (7, 5), (48, 64), (90, 120), (97, 203)]


def _frame(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    # Values start at 1, so a black fill is always a visible change.
    return rng.integers(1, 256, (h, w, 3), dtype=np.uint8)


def _blob(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    for _ in range(int(rng.integers(1, 4))):
        center = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        axes = (int(rng.integers(0, max(1, w // 3) + 1)), int(rng.integers(0, max(1, h // 3) + 1)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
    return mask.astype(bool)


def _edge_masks(h: int, w: int) -> dict[str, np.ndarray]:
    """Masks touching each frame edge, a single pixel, and the full frame."""
    masks: dict[str, np.ndarray] = {}
    for name, sl in {
        "left": np.s_[h // 4 : h - h // 4, : max(1, w // 3)],
        "right": np.s_[h // 4 : h - h // 4, w - max(1, w // 3) :],
        "top": np.s_[: max(1, h // 3), w // 4 : w - w // 4],
        "bottom": np.s_[h - max(1, h // 3) :, w // 4 : w - w // 4],
    }.items():
        m = np.zeros((h, w), bool)
        m[sl] = True
        masks[name] = m
    for name, (y, x) in {
        "pixel_center": (h // 2, w // 2),
        "pixel_corner": (0, 0),
        "pixel_far_corner": (h - 1, w - 1),
    }.items():
        m = np.zeros((h, w), bool)
        m[y, x] = True
        masks[name] = m
    masks["full"] = np.ones((h, w), bool)
    return masks


def _all_masks(rng: np.random.Generator, h: int, w: int) -> list[np.ndarray]:
    masks = [m for m in _edge_masks(h, w).values() if m.any()]
    masks += [b for b in (_blob(rng, h, w) for _ in range(3)) if b.any()]
    return masks


def _full(pm: PreparedMask, h: int, w: int) -> np.ndarray:
    """The prepared crop placed back into a full frame."""
    out = np.zeros((h, w), bool)
    r = pm.rect
    out[r.y0 : r.y1, r.x0 : r.x1] = pm.crop
    return out


def _shift_any(mask: np.ndarray, offsets: list[tuple[int, int]]) -> np.ndarray:
    """True where some ``mask[y + dy, x + dx]`` is True, with False past the frame."""
    h, w = mask.shape
    pad = max([abs(o) for off in offsets for o in off] + [0])
    padded = np.zeros((h + 2 * pad, w + 2 * pad), bool)
    padded[pad : pad + h, pad : pad + w] = mask
    out = np.zeros((h, w), bool)
    for dy, dx in offsets:
        out |= padded[pad + dy : pad + dy + h, pad + dx : pad + dx + w]
    return out


def _ref_erode(mask: np.ndarray, t: int) -> np.ndarray:
    """Erosion by the size ``2t+1`` ellipse with everything past the frame outside.

    Written as "not dilated complement" so it does not share code with the style.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * t + 1, 2 * t + 1))
    offsets = [(int(y) - t, int(x) - t) for y, x in zip(*np.nonzero(kernel))]
    h, w = mask.shape
    pad = t
    outside = np.ones((h + 2 * pad, w + 2 * pad), bool)
    outside[pad : pad + h, pad : pad + w] = ~mask
    hit = np.zeros((h, w), bool)
    for dy, dx in offsets:
        hit |= outside[pad + dy : pad + dy + h, pad + dx : pad + dx + w]
    return mask & ~hit


def _within(mask: np.ndarray, r: int) -> np.ndarray:
    """Every pixel within Euclidean distance ``r`` of a mask pixel."""
    offsets = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dy * dy + dx * dx <= r * r]
    return _shift_any(mask, offsets)


def _snapshot(pm: PreparedMask) -> np.ndarray:
    return pm.crop.copy()


def _assert_crop_intact(pm: PreparedMask, before: np.ndarray) -> None:
    assert np.array_equal(pm.crop, before)
    assert not pm.crop.flags.writeable


# Silhouette


@pytest.mark.parametrize("h,w", SIZES)
@pytest.mark.parametrize("r", [0, 1, 3])
@pytest.mark.parametrize("color_name", list(COLORS))
def test_silhouette_fills_crop_and_nothing_else(h: int, w: int, r: int, color_name: str) -> None:
    rng = np.random.default_rng(h * 1000 + w * 10 + r)
    color = COLORS[color_name]
    for mask in _all_masks(rng, h, w):
        img = _frame(rng, h, w)
        # Make sure the frame already holds the fill color in places, so a pixel
        # that happens to equal it is not mistaken for an untouched one or vice versa.
        img[rng.random((h, w)) < 0.2] = color
        pm = prepare_mask(mask, r)
        assert pm is not None
        before = _snapshot(pm)
        out = img.copy()
        apply_silhouette(out, pm, color)
        covered = _full(pm, h, w)
        assert (out[covered] == np.array(color, np.uint8)).all()
        assert np.array_equal(out[~covered], img[~covered])
        _assert_crop_intact(pm, before)


def test_silhouette_1080p() -> None:
    rng = np.random.default_rng(1080)
    h, w = 1080, 1920
    img = _frame(rng, h, w)
    mask = _blob(rng, h, w)
    mask[500:1080, 0:300] = True  # Touches the left and bottom edges.
    pm = prepare_mask(mask, dilate_px(h, 0.01))
    assert pm is not None
    out = img.copy()
    apply_silhouette(out, pm, COLORS["gray"])
    covered = _full(pm, h, w)
    assert (out[covered] == 128).all()
    assert np.array_equal(out[~covered], img[~covered])


@pytest.mark.parametrize("h,w", [(1, 1), (3, 2), (31, 17), (64, 48), (80, 133)])
@pytest.mark.parametrize("r", [0, 1, 2, 4, 6])
def test_silhouette_covers_original_mask_and_radius(h: int, w: int, r: int) -> None:
    rng = np.random.default_rng(7 * h + 13 * w + r)
    for mask in _all_masks(rng, h, w):
        img = _frame(rng, h, w)
        out = img.copy()
        pm = prepare_mask(mask, r)
        assert pm is not None
        apply_silhouette(out, pm, BLACK)
        must = _within(mask, r)
        assert must[mask].all()
        # The frame holds no zero, so a zero pixel is a pixel the fill reached.
        assert (out[must] == 0).all()


@pytest.mark.parametrize("h,w", [(5, 5), (48, 64), (97, 203)])
def test_silhouette_is_irreversible(h: int, w: int) -> None:
    rng = np.random.default_rng(h + w)
    for mask in _all_masks(rng, h, w):
        pm = prepare_mask(mask, 2)
        assert pm is not None
        covered = _full(pm, h, w)
        a = _frame(rng, h, w)
        b = a.copy()
        b[covered] = _frame(rng, h, w)[covered]
        out_a, out_b = a.copy(), b.copy()
        apply_silhouette(out_a, pm, COLORS["red"])
        apply_silhouette(out_b, pm, COLORS["red"])
        assert np.array_equal(out_a, out_b)


def test_noncontiguous_out_is_rejected() -> None:
    rng = np.random.default_rng(3)
    big = _frame(rng, 40, 60)
    out = big[::2, ::2]  # A writable strided view, still HxWx3.
    before = big.copy()
    pm = _pm()
    for call in _calls(out, pm, BLACK):
        with pytest.raises(ValueError, match="contiguous"):
            call()
    with pytest.raises(ValueError, match="contiguous"):
        apply_bodybox(out, [Rect(0, 0, 2, 2)], BLACK)
    assert np.array_equal(big, before)


def test_fill_that_opencv_does_not_do_in_place_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # If OpenCV ever hands back a new array, the cover would be lost silently.
    from petrify_shadow.styles import body

    real = body.cv2.copyTo
    monkeypatch.setattr(body.cv2, "copyTo", lambda src, mask, dst: real(src, mask, dst.copy()))
    out = np.ones((20, 30, 3), np.uint8)
    with pytest.raises(RuntimeError):
        apply_silhouette(out, _pm(), BLACK)
    with pytest.raises(RuntimeError):
        apply_outline(out, _pm(), BLACK, 2)


# Outline


def _check_outline(img: np.ndarray, pm: PreparedMask, t: int, color: tuple[int, int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    before = _snapshot(pm)
    out = img.copy()
    apply_outline(out, pm, color, t)
    full = _full(pm, h, w)
    band = full & ~_ref_erode(full, t)
    assert (out[band] == np.array(color, np.uint8)).all()
    assert np.array_equal(out[~band], img[~band])
    _assert_crop_intact(pm, before)
    return band


@pytest.mark.parametrize("h,w", SIZES)
@pytest.mark.parametrize("t", [1, 2, 3, 5])
@pytest.mark.parametrize("r", [0, 2])
def test_outline_is_mask_minus_erosion(h: int, w: int, t: int, r: int) -> None:
    rng = np.random.default_rng(h * w + 31 * t + r)
    for mask in _all_masks(rng, h, w):
        img = _frame(rng, h, w)
        pm = prepare_mask(mask, r)
        assert pm is not None
        band = _check_outline(img, pm, t, BLACK)
        # Black on a frame without zeros: the changed pixels are exactly the band,
        # so neither the interior nor the exterior was touched.
        out = img.copy()
        apply_outline(out, pm, BLACK, t)
        assert np.array_equal((out != img).any(axis=2), band)


def test_outline_thicker_than_mask_fills_whole_mask() -> None:
    rng = np.random.default_rng(11)
    h, w = 60, 80
    for mask in _all_masks(rng, h, w):
        pm = prepare_mask(mask, 1)
        assert pm is not None
        img = _frame(rng, h, w)
        out = img.copy()
        apply_outline(out, pm, BLACK, 200)
        full = _full(pm, h, w)
        assert (out[full] == 0).all()
        assert np.array_equal(out[~full], img[~full])


def test_outline_runs_along_frame_edges_for_full_mask() -> None:
    rng = np.random.default_rng(5)
    h, w, t = 30, 41, 3
    img = _frame(rng, h, w)
    pm = prepare_mask(np.ones((h, w), bool), 2)
    assert pm is not None
    out = img.copy()
    apply_outline(out, pm, BLACK, t)
    ring = np.ones((h, w), bool)
    ring[t : h - t, t : w - t] = False
    assert np.array_equal((out == 0).all(axis=2), ring)
    assert np.array_equal(out[~ring], img[~ring])


@pytest.mark.parametrize("edge", ["left", "right", "top", "bottom"])
def test_outline_is_closed_along_each_frame_edge(edge: str) -> None:
    rng = np.random.default_rng(17)
    h, w, t = 50, 70, 2
    mask = _edge_masks(h, w)[edge]
    pm = prepare_mask(mask, 0)
    assert pm is not None
    img = _frame(rng, h, w)
    out = img.copy()
    apply_outline(out, pm, BLACK, t)
    drawn = (out == 0).all(axis=2)
    line = {
        "left": np.s_[:, 0],
        "right": np.s_[:, w - 1],
        "top": np.s_[0, :],
        "bottom": np.s_[h - 1, :],
    }[edge]
    # Every mask pixel on the frame edge is part of the outline.
    assert mask[line].any()
    assert np.array_equal(drawn[line], mask[line])


def test_outline_single_pixel_and_tiny_frames() -> None:
    rng = np.random.default_rng(2)
    for h, w in [(1, 1), (1, 5), (4, 1)]:
        img = _frame(rng, h, w)
        mask = np.zeros((h, w), bool)
        mask[0, 0] = True
        pm = prepare_mask(mask, 0)
        assert pm is not None
        _check_outline(img, pm, 1, COLORS["white"])


@pytest.mark.parametrize(
    "h,expected",
    [(0, 2), (1, 2), (10, 2), (240, 2), (480, 2), (500, 2), (625, 3), (720, 3), (1080, 4), (1125, 5), (2160, 9)],
)
def test_outline_thickness_literals(h: int, expected: int) -> None:
    assert outline_thickness(h) == expected


@pytest.mark.parametrize("h", [-1, 1.5, True, "480", None])
def test_outline_thickness_rejects_bad_heights(h: object) -> None:
    with pytest.raises(ValueError):
        outline_thickness(h)  # type: ignore[arg-type]


# Bodybox


def test_bodybox_fills_rects_including_overlaps_and_edges() -> None:
    rng = np.random.default_rng(23)
    for h, w in [(1, 1), (7, 5), (48, 64), (97, 203)]:
        img = _frame(rng, h, w)
        rects = [
            Rect(0, 0, w, 1),
            Rect(w - 1, 0, w, h),
            Rect(0, h - 1, 1, h),
            Rect(w // 4, h // 4, w // 2 + 1, h // 2 + 1),
            Rect(w // 3, h // 3, w - w // 5, h - h // 5),
            Rect(w // 2, h // 2, w // 2 + 1, h // 2 + 1),
        ]
        out = img.copy()
        apply_bodybox(out, rects, COLORS["green"])
        covered = np.zeros((h, w), bool)
        for r in rects:
            covered[r.y0 : r.y1, r.x0 : r.x1] = True
        assert (out[covered] == np.array(COLORS["green"], np.uint8)).all()
        assert np.array_equal(out[~covered], img[~covered])


def test_bodybox_empty_list_and_empty_rect_do_nothing() -> None:
    rng = np.random.default_rng(29)
    img = _frame(rng, 20, 30)
    out = img.copy()
    apply_bodybox(out, [], BLACK)
    apply_bodybox(out, (), BLACK)
    apply_bodybox(out, [Rect(5, 5, 5, 9), Rect(3, 4, 8, 4)], BLACK)
    assert np.array_equal(out, img)


def test_bodybox_is_irreversible() -> None:
    rng = np.random.default_rng(31)
    h, w = 60, 90
    rects = [Rect(3, 4, 30, 50), Rect(20, 10, 90, 25)]
    covered = np.zeros((h, w), bool)
    for r in rects:
        covered[r.y0 : r.y1, r.x0 : r.x1] = True
    a = _frame(rng, h, w)
    b = a.copy()
    b[covered] = _frame(rng, h, w)[covered]
    out_a, out_b = a.copy(), b.copy()
    apply_bodybox(out_a, rects, COLORS["blue"])
    apply_bodybox(out_b, rects, COLORS["blue"])
    assert np.array_equal(out_a, out_b)


def test_bodybox_bad_rect_leaves_out_unchanged() -> None:
    rng = np.random.default_rng(37)
    img = _frame(rng, 20, 30)
    out = img.copy()
    with pytest.raises(ValueError):
        apply_bodybox(out, [Rect(0, 0, 10, 10), Rect(-1, 0, 5, 5)], BLACK)
    assert np.array_equal(out, img)


# Input validation


def _pm(h: int = 20, w: int = 30) -> PreparedMask:
    mask = np.zeros((h, w), bool)
    mask[5:10, 8:14] = True
    pm = prepare_mask(mask, 1)
    assert pm is not None
    return pm


def _calls(out: object, pm: object, color: object) -> list:
    return [
        lambda: apply_silhouette(out, pm, color),  # type: ignore[arg-type]
        lambda: apply_outline(out, pm, color, 2),  # type: ignore[arg-type]
    ]


@pytest.mark.parametrize(
    "out",
    [
        np.zeros((20, 30), np.uint8),
        np.zeros((20, 30, 4), np.uint8),
        np.zeros((20, 30, 3), np.float32),
        np.zeros((20, 30, 3), np.uint16),
        [[[0, 0, 0]]],
        None,
    ],
)
def test_bad_out_is_rejected(out: object) -> None:
    pm = _pm()
    for call in _calls(out, pm, BLACK):
        with pytest.raises(ValueError):
            call()
    with pytest.raises(ValueError):
        apply_bodybox(out, [], BLACK)  # type: ignore[arg-type]


def test_read_only_out_is_rejected() -> None:
    out = np.zeros((20, 30, 3), np.uint8)
    out.setflags(write=False)
    pm = _pm()
    for call in _calls(out, pm, BLACK):
        with pytest.raises(ValueError, match="writable"):
            call()
    with pytest.raises(ValueError, match="writable"):
        apply_bodybox(out, [Rect(0, 0, 1, 1)], BLACK)


@pytest.mark.parametrize(
    "color", [(0, 0), (0, 0, 0, 0), (0, 0, 256), (0, -1, 0), (0.0, 0, 0), (True, 0, 0), "red", 5, None]
)
def test_bad_color_is_rejected(color: object) -> None:
    out = np.zeros((20, 30, 3), np.uint8)
    pm = _pm()
    for call in _calls(out, pm, color):
        with pytest.raises(ValueError):
            call()
    with pytest.raises(ValueError):
        apply_bodybox(out, [Rect(0, 0, 1, 1)], color)  # type: ignore[arg-type]
    assert not out.any()


def test_numpy_color_values_are_accepted() -> None:
    out = np.ones((20, 30, 3), np.uint8)
    apply_bodybox(out, [Rect(0, 0, 2, 2)], tuple(np.array([0, 128, 255], np.uint8)))  # type: ignore[arg-type]
    assert out[0, 0].tolist() == [0, 128, 255]


def test_crop_that_does_not_match_its_rect_is_rejected() -> None:
    out = np.zeros((20, 30, 3), np.uint8)
    crop = np.ones((4, 5), bool)
    crop.setflags(write=False)
    for pm in [
        PreparedMask(Rect(2, 2, 8, 6), crop),  # Rect is 4 by 6, crop 4 by 5.
        PreparedMask(Rect(2, 2, 7, 7), crop),  # Rect is 5 by 5.
        PreparedMask(Rect(2, 2, 7, 6), crop.astype(np.uint8)),  # Wrong dtype.
        PreparedMask(Rect(2, 2, 7, 6), np.ones((4, 5, 1), bool)),  # Wrong ndim.
    ]:
        for call in _calls(out, pm, BLACK):
            with pytest.raises(ValueError):
                call()
    assert not out.any()


@pytest.mark.parametrize(
    "rect", [Rect(-1, 0, 4, 5), Rect(0, -2, 5, 3), Rect(26, 0, 31, 5), Rect(0, 16, 5, 21), Rect(5, 5, 3, 8)]
)
def test_rect_outside_frame_is_rejected(rect: Rect) -> None:
    out = np.zeros((20, 30, 3), np.uint8)
    crop = np.ones((max(0, rect.y1 - rect.y0), max(0, rect.x1 - rect.x0)), bool)
    pm = PreparedMask(rect, crop)
    for call in _calls(out, pm, BLACK):
        with pytest.raises(ValueError):
            call()
    with pytest.raises(ValueError):
        apply_bodybox(out, [rect], BLACK)
    assert not out.any()


def test_non_rect_and_non_prepared_mask_are_rejected() -> None:
    out = np.zeros((20, 30, 3), np.uint8)
    for call in _calls(out, (Rect(0, 0, 1, 1), np.ones((1, 1), bool)), BLACK):
        with pytest.raises(ValueError):
            call()
    with pytest.raises(ValueError):
        apply_bodybox(out, [(0, 0, 1, 1)], BLACK)  # type: ignore[list-item]
    with pytest.raises(ValueError):
        apply_bodybox(out, [Rect(0.0, 0, 1, 1)], BLACK)  # type: ignore[arg-type]


@pytest.mark.parametrize("t", [0, -1, 1.5, True, None])
def test_bad_thickness_is_rejected(t: object) -> None:
    out = np.zeros((20, 30, 3), np.uint8)
    with pytest.raises(ValueError):
        apply_outline(out, _pm(), BLACK, t)  # type: ignore[arg-type]
    assert not out.any()


def test_writable_crop_is_not_modified() -> None:
    # A crop someone left writable must still come back unchanged.
    pm0 = _pm()
    crop = pm0.crop.copy()
    pm = PreparedMask(pm0.rect, crop)
    before = crop.copy()
    out = np.zeros((20, 30, 3), np.uint8)
    apply_silhouette(out, pm, BLACK)
    apply_outline(out, pm, COLORS["white"], 3)
    assert np.array_equal(crop, before)


# Timing (printed, not asserted)


def _median_ms(fn, runs: int = 30) -> float:
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


@pytest.mark.parametrize("h,w", [(480, 640), (1080, 1920)])
def test_timing(h: int, w: int, capsys: pytest.CaptureFixture[str]) -> None:
    rng = np.random.default_rng(41)
    img = _frame(rng, h, w)
    mask = np.zeros((h, w), np.uint8)
    # A standing person: an ellipse about a quarter of the width and most of the height.
    cv2.ellipse(mask, (w // 2, h // 2), (w // 8, int(h * 0.42)), 0, 0, 360, 1, -1)
    mask_b = mask.astype(bool)
    r = dilate_px(h, 0.01)
    pm = prepare_mask(mask_b, r)
    assert pm is not None
    t = outline_thickness(h)
    out = img.copy()
    box = [Rect(w // 2 - w // 8, h // 2 - int(h * 0.42), w // 2 + w // 8, h // 2 + int(h * 0.42))]
    results = {
        "prepare_mask": _median_ms(lambda: prepare_mask(mask_b, r)),
        "silhouette": _median_ms(lambda: apply_silhouette(out, pm, BLACK)),
        "outline": _median_ms(lambda: apply_outline(out, pm, BLACK, t)),
        "bodybox": _median_ms(lambda: apply_bodybox(out, box, BLACK)),
    }
    with capsys.disabled():
        line = ", ".join(f"{k} {v:.3f} ms" for k, v in results.items())
        print(f"\nbody styles {w}x{h} (r={r}, t={t}): {line}")
