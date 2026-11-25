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

"""Region geometry for the shadow mode.

Every function here decides which pixels get covered, so every rounding,
clipping and threshold decision errs toward covering more:

- Padding and growth are computed exactly, in integers over a common
  denominator, then rounded outward once: the left and top edges down, the right
  and bottom edges up.
- Rectangles are clipped to the frame only after rounding, so clipping never
  removes an in-frame pixel that the padding reached.
- A face center on the edge of a person box, or on the lower edge of its face
  zone, does not count as inside it, so that person box stays an orphan and is
  filled (with ``orphan`` ``box`` or ``head``).

Option values (``padding``, ``dilate``) are decimals a person typed. They are read
as the shortest decimal that round-trips to the float, so ``0.3`` means exactly
3/10: binary representation error then cannot move an edge by a pixel in either
direction, and results match a hand computation on the decimal value.

Box coordinates must be integers (Python or numpy), as :func:`check_detections`
guarantees. A float coordinate raises instead of being truncated, since
truncating a right or bottom edge would move it inward.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Literal

import cv2
import numpy as np

from petrify.core.detections import Box, Face

MIN_PAD_PX = 2
"""Smallest padding in pixels on each side, so a tiny box still gets a margin."""

PERSON_PAD = Fraction(5, 100)
"""Padding of a person box on each side, as a share of the box width or height."""

FACE_ZONE = Fraction(1, 3)
"""A face clears a full standing body box only when its center lies in this top share of its height.

Tuned on CCTV footage, where person boxes hold whole standing bodies. It does
not apply to a box that is not a full standing body (see :func:`_is_standing`).
"""

ORPHAN_MASK_SHARE = Fraction(30, 100)
"""A person box whose own area holds less than this share of mask pixels is an orphan."""

HEAD_ZONE = Fraction(1, 4)
"""The head zone of a person box: this top share of its height, rounded up to whole rows."""

ORPHAN_HEAD_MASK_SHARE = Fraction(30, 100)
"""A person box whose head zone holds less than this share of mask pixels is an orphan."""

HELD_GROWTH_PER_FRAME = Fraction(25, 100)
"""Growth of a held box's padded rect per held frame, as a share of its size.

Measured on both demo clips: with 25 percent, 98.2 to 99.5 percent of real
person boxes one to three frames later lie fully inside the grown cover, against
90.5 to 96.9 percent with 10 percent.
"""

HELD_GROWTH_CAP = Fraction(3)
"""Largest growth factor of a held box's padded rect."""

HELD_REGION_FRAMES: int = math.ceil((HELD_GROWTH_CAP - 1) / HELD_GROWTH_PER_FRAME)
"""Held count at which a held box reaches the growth cap (8 with the constants above).

A held box with a count below this still counts as a region. From this count on
its cover stops growing while the subject may keep moving, so it no longer
proves the subject is covered: it is still drawn, but it is not trusted.
"""

HEAD_SHARE = Fraction(1, 2)
"""The head rect of a person box reaches this top share of the box height, rounded up.

Evidence, measured on the two demo clips only, not a guarantee:

- On orphan boxes, the boxes this rect is drawn on: across 400 frames (two
  ranges of 200 frames, one per clip), no real head or face was left visible. The tightest pose was arms
  raised over the head, where the face sat at 0.40 of the box height with
  about 6 percent of the box height to spare. Bending, sitting low
  and two people in one box were not in the footage and remain a risk.
- On boxes a face had cleared, the padded face cover reached at most 0.44 of
  the box height (729 of 729 and 79 of 79 inside the head rect). Those boxes
  have the face center in the top third by construction (:data:`FACE_ZONE`), so
  this figure does not show where the head sits in an orphan box.
"""

HEAD_MIN_HEIGHT_PX = 150
"""Smallest person box height, in pixels, that :func:`head_rect` accepts.

On smaller boxes the head sits lower in the box (the segmenter head band
reached 0.66 of the box height under 150 px on camera2.mp4), so they keep the
whole-box fill.
"""

HEAD_MIN_ASPECT = Fraction(7, 5)
"""Smallest height over width of a person box that :func:`head_rect` accepts.

A wider box is more likely a seated or bending person, a person cut at the chest
by the frame edge, or two people side by side, whose head need not lie in the
top half.
"""

STANDING_MIN_ASPECT = HEAD_MIN_ASPECT
"""Smallest height over width of a person box that counts as a full standing body.

The same 7/5 as :data:`HEAD_MIN_ASPECT`: a wider box is more likely a person cut
off by a frame edge, seated or bending, or two people side by side, and there the
face need not lie in the top third.
"""

BOTTOM_TOUCH_PX = 2
"""A person box whose bottom edge lies this many rows or fewer above the frame bottom touches it.

HOG and tracker-held boxes can end 1 or 2 px short of the last row of a body cut
off by the frame, so ``y + h_box >= h - BOTTOM_TOUCH_PX`` counts as cut off.
"""

LOOSE_FACE_ZONE = Fraction(1, 2)
"""A face clears a box that is not a full standing body only with its center in this top share."""

LOOSE_MIN_FACE_SHARE = Fraction(1, 6)
"""A face clears a box that is not a full standing body only when at least this share of its width.

A small face inside a large cut-off box is more likely someone behind that
person than that person's own face, and must not clear the box.
"""

Primary = Literal["face", "mask", "bodybox"]
PRIMARIES: frozenset[str] = frozenset({"face", "mask", "bodybox"})

OrphanReason = Literal["face", "box", "head"]
"""Why a person box is an orphan, as :func:`orphan_reasons` reports it.

- ``"face"``: face primary, no face cleared the box.
- ``"box"``: mask primary, the box's own area holds less than
  :data:`ORPHAN_MASK_SHARE` mask pixels (the segmenter missed the person).
- ``"head"``: mask primary, the box share is met but its head zone holds less
  than :data:`ORPHAN_HEAD_MASK_SHARE`.
"""


@dataclass(frozen=True)
class Rect:
    """A pixel rectangle, half-open: columns ``x0`` to ``x1 - 1``, rows ``y0`` to ``y1 - 1``.

    Half-open bounds map directly onto numpy slices, ``image[y0:y1, x0:x1]``.
    """

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def height(self) -> int:
        return max(0, self.y1 - self.y0)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def is_empty(self) -> bool:
        return self.area == 0


@dataclass(frozen=True)
class PreparedMask:
    """A subject mask cropped to where it can matter, and dilated.

    ``crop`` is a read-only bool array of shape ``(rect.height, rect.width)``: the
    mask dilated by the radius given to :func:`prepare_mask`, restricted to
    ``rect``. Every True pixel of the dilated full-frame mask lies inside ``rect``,
    so nothing outside it needs covering.
    """

    rect: Rect
    crop: np.ndarray


def _is_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


@lru_cache(maxsize=64)
def _parse_decimal(value: float) -> Fraction:
    # Cached because a frame pads every face with the same option value, and the
    # string round trip is the costly part of the exact arithmetic.
    return Fraction(repr(value))


def _decimal(value: object, name: str) -> Fraction:
    """Return a non-negative finite option value as an exact fraction."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real number, got {type(value).__name__}")
    if isinstance(value, (int, np.integer)):
        out = Fraction(int(value))
    else:
        as_float = float(value)
        if not math.isfinite(as_float):
            raise ValueError(f"{name} must be finite, got {as_float!r}")
        out = _parse_decimal(as_float)
    if out < 0:
        raise ValueError(f"{name} must not be negative, got {value!r}")
    return out


def _count(value: object, name: str) -> int:
    """Return a non-negative integer argument as a Python int."""
    if not _is_int(value):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    out = int(value)  # type: ignore[call-overload]
    if out < 0:
        raise ValueError(f"{name} must not be negative, got {out}")
    return out


def _box(box: Box) -> tuple[int, int, int, int]:
    """Return ``(x, y, w, h)`` of ``box`` as Python ints.

    A non-integral or negative value is malformed input. Raising lets the engine
    fail closed (a black frame) instead of guessing which pixels were meant.
    """
    x, y, bw, bh = box.x, box.y, box.w, box.h
    if not (_is_int(x) and _is_int(y) and _is_int(bw) and _is_int(bh)):
        raise ValueError(
            f"box coordinates must be integers, got x={x!r} y={y!r} w={bw!r} h={bh!r}"
        )
    if bw < 0 or bh < 0:
        raise ValueError(f"box size must be non-negative, got w={bw} h={bh}")
    return int(x), int(y), int(bw), int(bh)


def _outward(
    left: int, top: int, right: int, bottom: int, den: int, h: int, w: int
) -> Rect | None:
    """Round the edges ``left / den`` and so on outward, then clip to an ``h`` by ``w`` frame."""
    x0 = min(max(left // den, 0), w)
    y0 = min(max(top // den, 0), h)
    x1 = min(max(-(-right // den), 0), w)
    y1 = min(max(-(-bottom // den), 0), h)
    if x1 <= x0 or y1 <= y0:
        return None
    return Rect(x0, y0, x1, y1)


def held_count(box: Box) -> int:
    """Return ``box.held`` as a Python int, or raise ValueError.

    :func:`~petrify.core.detections.check_detections` already guarantees a
    non-negative int on engine results; checking again keeps a hand-built box
    with a bool, float or negative count from being drawn as fresh: it raises,
    and the engine fails the frame closed.
    """
    return _count(box.held, "held")


def held_growth(held_frames: int) -> Fraction:
    """Return the growth factor for a box held for ``held_frames`` frames.

    ``held_frames`` is the box's :attr:`~petrify.core.detections.Box.held`: the
    frames it has been carried without a matching detection.

    The factor grows by :data:`HELD_GROWTH_PER_FRAME` per held frame, linearly,
    and stops at :data:`HELD_GROWTH_CAP`, which it reaches at
    :data:`HELD_REGION_FRAMES`. Zero held frames gives 1 (no growth). Pass the
    result as ``grow`` to :func:`pad_face` or :func:`pad_person`.
    """
    n = _count(held_frames, "held_frames")
    return min(1 + n * HELD_GROWTH_PER_FRAME, HELD_GROWTH_CAP)


def pad_face(
    box: Box, padding: float, h: int, w: int, *, grow: float | Fraction = 1
) -> Rect | None:
    """Return the padded cover rect of a face box in an ``h`` by ``w`` frame.

    With ``p = padding`` and the box ``w_box`` by ``h_box``: left and right grow by
    ``p * w_box``, the top by ``1.5 * p * h_box`` (hair), the bottom by
    ``0.5 * p * h_box`` (chin), each by at least :data:`MIN_PAD_PX`. With ``grow``
    above 1 (a held face, see :func:`held_growth`) the padded rect is then scaled
    by ``grow`` around its own center. Edges are rounded outward once, at the end,
    and the result is clipped to the frame.

    Growth is applied before clipping, on the unclipped padded rect: growing an
    already clipped rect around its shifted center would cover less on the side
    away from the frame edge.

    Returns None for a zero-area box, or when nothing of the result lies in the
    frame. Raises ValueError for a non-integral coordinate, a negative box size,
    a negative or non-finite padding, or a ``grow`` below 1.
    """
    x, y, bw, bh = _box(box)
    frame_h, frame_w = _count(h, "h"), _count(w, "w")
    p = _decimal(padding, "padding")
    g = _growth(grow)
    if bw == 0 or bh == 0:
        return None
    # Every edge is an integer count of 1/den pixels, den = 2 * denominator(p):
    # p * w_box is 2 * n * w_box units, 1.5 * p * h_box is 3 * n * h_box units and
    # 0.5 * p * h_box is n * h_box units, so no rounding happens before _outward.
    n, d = p.numerator, p.denominator
    den = 2 * d
    floor_units = MIN_PAD_PX * den
    side = max(floor_units, 2 * n * bw)
    top = max(floor_units, 3 * n * bh)
    bottom = max(floor_units, n * bh)
    left, right = x * den - side, (x + bw) * den + side
    upper, lower = y * den - top, (y + bh) * den + bottom
    return _outward(*_scaled(left, upper, right, lower, den, g), frame_h, frame_w)


def _growth(grow: float | Fraction) -> Fraction:
    """Return ``grow`` as an exact fraction of at least 1, or raise ValueError."""
    g = grow if isinstance(grow, Fraction) else _decimal(grow, "grow")
    if g < 1:
        raise ValueError(f"grow must be at least 1, got {grow!r}")
    return g


def _scaled(
    left: int, upper: int, right: int, lower: int, den: int, g: Fraction
) -> tuple[int, int, int, int, int]:
    """Scale the rect with edges ``left / den`` and so on by ``g`` around its center.

    The new edges are (L + R) / 2 -+ (R - L) * g / 2, which over the denominator
    2 * gb * den stays an exact integer, so no rounding happens before
    :func:`_outward`.
    """
    if g == 1:
        return left, upper, right, lower, den
    ga, gb = g.numerator, g.denominator
    sum_x, span_x = (left + right) * gb, (right - left) * ga
    sum_y, span_y = (upper + lower) * gb, (lower - upper) * ga
    return sum_x - span_x, sum_y - span_y, sum_x + span_x, sum_y + span_y, den * 2 * gb


def pad_person(box: Box, h: int, w: int, *, grow: float | Fraction = 1) -> Rect | None:
    """Return a person box padded by :data:`PERSON_PAD` per side, in an ``h`` by ``w`` frame.

    Each side grows by at least :data:`MIN_PAD_PX`. With ``grow`` above 1 (a
    held person box, see :func:`held_growth`) the padded rect is then scaled by
    ``grow`` around its own center, before clipping, for the same reason as in
    :func:`pad_face`. Edges are rounded outward once and the result is clipped
    to the frame. Returns None for a zero-area box, or when nothing of the result
    lies in the frame. Raises ValueError for a non-integral coordinate, a
    negative box size, or a ``grow`` below 1.
    """
    x, y, bw, bh = _box(box)
    frame_h, frame_w = _count(h, "h"), _count(w, "w")
    g = _growth(grow)
    if bw == 0 or bh == 0:
        return None
    n, den = PERSON_PAD.numerator, PERSON_PAD.denominator
    side = max(MIN_PAD_PX * den, n * bw)
    vert = max(MIN_PAD_PX * den, n * bh)
    edges = _scaled(
        x * den - side, y * den - vert, (x + bw) * den + side, (y + bh) * den + vert, den, g
    )
    return _outward(*edges, frame_h, frame_w)


def head_rect(box: Box, h: int, w: int) -> Rect | None:
    """Return the head rect of a person box in an ``h`` by ``w`` frame, or None.

    The rect has the left, right and top edges of :func:`pad_person` (ungrown)
    and its bottom edge at ``y + ceil(h_box * HEAD_SHARE)``: the top half of the
    box, rounded up to whole rows, then clipped to the frame.
    Full padded width, because a face can sit anywhere across a box that touches
    a frame edge or holds two people.

    It applies only to a box that is fresh (``held`` 0), at least
    :data:`HEAD_MIN_HEIGHT_PX` tall, and at least :data:`HEAD_MIN_ASPECT` times as
    tall as wide (compared in integers, so 7/5 is exact). A held box no longer
    proves where the person is, so it keeps its grown whole-box cover.

    Returns None when the box is not eligible, has zero area, or when nothing of
    the head rect lies in the frame; the caller then fills the whole padded box,
    which covers more. Raises ValueError for a non-integral coordinate, a
    negative box size or frame size, or a ``held`` that is not a non-negative
    integer.
    """
    _, _, bw, bh = _box(box)
    frame_h, frame_w = _count(h, "h"), _count(w, "w")
    held = held_count(box)
    if held != 0 or bh < HEAD_MIN_HEIGHT_PX:
        return None
    if bh * HEAD_MIN_ASPECT.denominator < bw * HEAD_MIN_ASPECT.numerator:
        return None
    return _top_share(box, frame_h, frame_w)


def _top_share(box: Box, h: int, w: int) -> Rect | None:
    """Return the rect of :func:`head_rect` without its eligibility test.

    Kept apart so that the measurement tool ``tools/head_sheet.py`` can show what
    the rect would hold on the boxes the rule refuses, with this same geometry.
    """
    _, y, _, bh = _box(box)
    padded = pad_person(box, h, w)
    if padded is None:
        return None
    n, d = HEAD_SHARE.numerator, HEAD_SHARE.denominator
    # Rounded up, so the rect holds at least the top share of the box; then
    # clipped, as every edge in this module is clipped only after rounding.
    bottom = min(max(y + -(-bh * n // d), 0), h)
    if bottom <= padded.y0:
        return None
    return Rect(padded.x0, padded.y0, padded.x1, bottom)


def dilate_px(h: int, dilate: float) -> int:
    """Return the silhouette dilation radius in pixels, ``ceil(dilate * h)``.

    ``h`` is the frame height, so the margin scales with the resolution. A
    ``dilate`` of 0 gives 0. Raises ValueError for a negative or non-finite value.
    """
    return math.ceil(_decimal(dilate, "dilate") * _count(h, "h"))


def prepare_mask(mask: np.ndarray | None, r: int) -> PreparedMask | None:
    """Crop ``mask`` to its bounding rect grown by ``r``, and dilate the crop by ``r``.

    Returns None for a None mask or one with no True pixel. Otherwise ``rect`` is
    the bounding rect of the True pixels grown by ``r`` on each side and clipped
    to the frame, and ``crop`` is the Euclidean dilation of the mask by ``r``
    inside ``rect``. ``r`` 0 means no dilation.

    Guarantee: ``crop`` is True on every mask pixel and on every in-frame pixel
    whose Euclidean distance to the nearest mask pixel is at most ``r``, and
    every True pixel of the full-frame dilation lies inside ``rect``. For ``r``
    up to 4095 it is True on no other pixel. For a larger ``r`` float32 cannot
    tell ``r`` from the next possible pixel distance ``sqrt(r*r + 1)``, so a few
    pixels just past ``r`` may also be True, which only covers more.

    Why the threshold loses no pixel at distance exactly ``r``: the distance
    between two pixels is ``sqrt(k)`` for an integer ``k``, and OpenCV's precise
    L2 transform is exact, so a pixel with ``k <= r*r`` gets at most
    ``float32(r) == r``. The test is ``d <= T`` with
    ``T = float32(sqrt(r*r + 0.5))``, deliberately above ``r``: round to nearest
    is monotonic and ``r`` is representable, so ``T >= r`` for every ``r``. For
    every ``r`` up to 4095 (checked exhaustively), ``float32(sqrt(r*r + 1)) > T``,
    so the next distance stays out. The margin also absorbs any transform error
    smaller than about ``0.25 / r``.

    Only the crop is transformed, never the full frame, and the window never
    grows past the frame, so an absurd ``r`` costs no more than the frame. The
    result is a new read-only array and never aliases ``mask``, which is left
    unchanged (it may be read-only). Raises ValueError for a mask that is not a
    2-d bool array or a negative ``r``.
    """
    radius = _count(r, "r")
    if mask is None:
        return None
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.ndim != 2:
        raise ValueError("mask must be a 2-d numpy array of dtype bool")
    as_u8 = mask.view(np.uint8)
    if not as_u8.flags.c_contiguous:
        as_u8 = np.ascontiguousarray(as_u8)
    # boundingRect reads the array once in C and gives (0, 0, 0, 0) for no True pixel.
    bx, by, bw, bh = cv2.boundingRect(as_u8)
    if bw == 0 or bh == 0:
        return None
    frame_h, frame_w = mask.shape
    # No two frame pixels are farther apart than h + w, so a larger reach adds
    # nothing; capping it keeps the arithmetic and the window bounded by the frame.
    reach = min(radius, frame_h + frame_w)
    x0, y0 = max(bx - reach, 0), max(by - reach, 0)
    x1, y1 = min(bx + bw + reach, frame_w), min(by + bh + reach, frame_h)
    window = as_u8[y0:y1, x0:x1]
    if radius == 0:
        crop = window.astype(np.bool_)
    elif reach * reach >= (frame_h - 1) ** 2 + (frame_w - 1) ** 2:
        # Every frame pixel is within r of any mask pixel.
        crop = np.ones(window.shape, np.bool_)
    else:
        # The window holds every mask pixel, so distances measured inside it equal
        # full-frame distances: a nearest mask pixel is never outside the window.
        distance = cv2.distanceTransform(1 - window, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        crop = distance <= np.float32(math.sqrt(radius * radius + 0.5))
    crop.setflags(write=False)
    return PreparedMask(rect=Rect(int(x0), int(y0), int(x1), int(y1)), crop=crop)


def _has_area(box: Box) -> bool:
    _, _, bw, bh = _box(box)
    return bw > 0 and bh > 0


def _is_standing(box: Box, h: int) -> bool:
    """True when ``box`` may hold a full standing body in a frame ``h`` rows tall.

    That is when its raw bottom edge ``y + h_box`` (not clipped to the frame)
    lies more than :data:`BOTTOM_TOUCH_PX` rows above the bottom of the frame
    (``y + h_box < h - BOTTOM_TOUCH_PX``), and its height over width is at least
    :data:`STANDING_MIN_ASPECT` (compared in integers, so 7/5 is exact). A box
    that touches the bottom of the frame is cut off there, as the upper body of
    a person in front of a webcam is. Raises ValueError for a malformed box or a
    negative ``h``.
    """
    _, y, bw, bh = _box(box)
    frame_h = _count(h, "h")
    if y + bh >= frame_h - BOTTOM_TOUCH_PX:
        return False
    n, d = STANDING_MIN_ASPECT.numerator, STANDING_MIN_ASPECT.denominator
    return bh * d >= bw * n


def _clear_rank(cx2: int, cy2: int, fw: int, box: Box, h: int) -> int | None:
    """Return how the face centered at (cx2 / 2, cy2 / 2), ``fw`` wide, may clear ``box``.

    None when it may not. The center must lie strictly inside the box and
    strictly above the lower edge of a zone at the top of it: the top
    :data:`FACE_ZONE` of its height when the box may hold a full standing body
    (:func:`_is_standing`). In a box that is not a full standing body (cut off by
    the bottom of the frame, or wider), the zone is the top
    :data:`LOOSE_FACE_ZONE` and the face must also be at least
    :data:`LOOSE_MIN_FACE_SHARE` of the box width. The rank is 0 when the center
    also lies in the top :data:`FACE_ZONE` (the stricter fit) and 1 otherwise, so
    a face prefers a box it fits under the stricter rule. Doubled coordinates
    keep the half-pixel center of an odd-sized box exact. A point on any edge
    does not clear: the person box then stays an orphan and is filled.
    """
    x, y, bw, bh = _box(box)
    if not (2 * x < cx2 < 2 * (x + bw) and 2 * y < cy2):
        return None
    n, d = FACE_ZONE.numerator, FACE_ZONE.denominator
    # cy2 / 2 < y + bh * n / d, multiplied through by 2 * d.
    in_third = cy2 * d < 2 * (y * d + bh * n)
    if _is_standing(box, h):
        return 0 if in_third else None
    sn, sd = LOOSE_MIN_FACE_SHARE.numerator, LOOSE_MIN_FACE_SHARE.denominator
    if fw * sd < bw * sn:
        return None
    n, d = LOOSE_FACE_ZONE.numerator, LOOSE_FACE_ZONE.denominator
    if cy2 * d >= 2 * (y * d + bh * n):
        return None
    return 0 if in_third else 1


def _below_share(mask: np.ndarray, x: int, y: int, bw: int, bh: int, share: Fraction) -> bool:
    """True when the mask pixels in the box ``(x, y, bw, bh)`` are less than ``share`` of its area.

    The share is measured against the box's own area, so any part of the box
    outside the frame counts as holding no mask pixels.
    """
    frame_h, frame_w = mask.shape
    x0, x1 = min(max(x, 0), frame_w), min(max(x + bw, 0), frame_w)
    y0, y1 = min(max(y, 0), frame_h), min(max(y + bh, 0), frame_h)
    count = int(np.count_nonzero(mask[y0:y1, x0:x1]))
    return count * share.denominator < share.numerator * bw * bh


def _mask_orphan(mask: np.ndarray, box: Box) -> OrphanReason | None:
    """Return which mask share ``box`` fails, ``"box"`` first, or None when it meets both.

    The whole-box share is tested first, so a box failing both is reported as
    ``"box"``: the segmenter missed that person and the whole box must be filled.
    """
    x, y, bw, bh = _box(box)
    if _below_share(mask, x, y, bw, bh, ORPHAN_MASK_SHARE):
        return "box"
    n, d = HEAD_ZONE.numerator, HEAD_ZONE.denominator
    head_rows = -(-bh * n // d)  # rounded up, so the zone holds at least the top share
    if _below_share(mask, x, y, bw, head_rows, ORPHAN_HEAD_MASK_SHARE):
        return "head"
    return None


def orphan_indices(
    persons: Sequence[Box] | None,
    faces: Sequence[Face] | None,
    mask: np.ndarray | None,
    primary: Primary,
    h: int,
    w: int,
) -> tuple[int, ...]:
    """Return the positions in ``persons`` of the boxes the primary styles do not cover.

    Only runs when ``persons`` is not None; zero-area person boxes are never
    orphans. By ``primary``:

    - ``"face"``: each face with a non-zero area clears at most one person box
      that qualifies. A box that may hold a full standing body
      (:func:`_is_standing`) qualifies when the face-box center lies strictly
      inside its top :data:`FACE_ZONE`. A box that touches the bottom of the
      frame (within :data:`BOTTOM_TOUCH_PX`), or whose height over width is below
      :data:`STANDING_MIN_ASPECT`, qualifies when the center lies strictly inside
      its top :data:`LOOSE_FACE_ZONE` and the face is at least
      :data:`LOOSE_MIN_FACE_SHARE` of the box width. Of the boxes that qualify,
      the face clears one where its center lies in the top :data:`FACE_ZONE`
      before one where it does not, then the smallest area, then the first in
      input order. Every box no face cleared is an orphan. So one face cannot
      clear two overlapping or nested persons, a face found at belt height or
      lower clears nothing, and a small face of someone behind a large cut-off
      box does not clear it. The caller decides which faces may clear a box: the
      shadow mode passes only the faces whose box has ``held`` 0.
    - ``"mask"``: a person box is an orphan when its own area holds less than
      :data:`ORPHAN_MASK_SHARE` mask pixels, or when its head zone (the top
      :data:`HEAD_ZONE` of its height, whole rows rounded up) holds less than
      :data:`ORPHAN_HEAD_MASK_SHARE`. Shares are measured against the raw box
      area, and parts of the box outside the frame count as no mask. A None mask
      counts as no pixels.
    - ``"bodybox"``: none, the body style covers every person box itself.

    Left uncovered by design:

    - ``persons`` None: no person detector ran, so no orphan can be found.
    - Mask primary: the mask carries no identity, so another person's
      silhouette that fills the head zone and the box share keeps an unsegmented
      person from being an orphan.
    - Face primary: a false face detection, or the face of someone else, that
      qualifies for a person box whose real face was missed still clears that
      box: in the top third of a standing body box, or in the top half of a box
      cut off by the bottom of the frame or wider than a standing body when it
      is at least a sixth of the box width.

    Indices are returned in increasing order. ``h`` and ``w`` are the frame
    size. Raises ValueError for an unknown ``primary``, a mask whose shape is not
    ``(h, w)``, or a malformed box.
    """
    return tuple(i for i, _ in orphan_reasons(persons, faces, mask, primary, h, w))


def orphan_reasons(
    persons: Sequence[Box] | None,
    faces: Sequence[Face] | None,
    mask: np.ndarray | None,
    primary: Primary,
    h: int,
    w: int,
) -> tuple[tuple[int, OrphanReason], ...]:
    """Return ``(index, reason)`` for each orphan :func:`orphan_indices` names, in its order.

    The indices are exactly those of :func:`orphan_indices`. The reason (see
    :data:`OrphanReason`) is ``"face"`` for every face-primary orphan; for a
    mask-primary orphan it is ``"box"`` when the whole-box share fails (whether
    or not the head-zone share also fails) and ``"head"`` when only the head-zone
    share fails. A None mask fails the whole-box share. Raises ValueError like
    :func:`orphan_indices`.
    """
    if primary not in PRIMARIES:
        raise ValueError(f"primary must be one of {sorted(PRIMARIES)}, got {primary!r}")
    frame_h, frame_w = _count(h, "h"), _count(w, "w")
    if persons is None or primary == "bodybox":
        return ()
    boxes = [(i, box) for i, box in enumerate(persons) if _has_area(box)]
    if primary == "face":
        cleared: set[int] = set()
        for face in faces or ():
            if not _has_area(face.box):
                continue
            fx, fy, fw, fh = _box(face.box)
            cx2, cy2 = 2 * fx + fw, 2 * fy + fh
            best: tuple[tuple[int, int], int] | None = None
            for i, box in boxes:
                rank = _clear_rank(cx2, cy2, fw, box, frame_h)
                if rank is None:
                    continue
                key = (rank, box.w * box.h)
                if best is None or key < best[0]:
                    best = (key, i)
            if best is not None:
                cleared.add(best[1])
        return tuple((i, "face") for i, _ in boxes if i not in cleared)
    if mask is not None and mask.shape != (frame_h, frame_w):
        raise ValueError(
            f"mask shape {mask.shape} does not match the frame {(frame_h, frame_w)}"
        )
    out: list[tuple[int, OrphanReason]] = []
    for i, box in boxes:
        reason: OrphanReason | None = "box" if mask is None else _mask_orphan(mask, box)
        if reason is not None:
            out.append((i, reason))
    return tuple(out)


def orphan_persons(
    persons: Sequence[Box] | None,
    faces: Sequence[Face] | None,
    mask: np.ndarray | None,
    primary: Primary,
    h: int,
    w: int,
) -> tuple[Rect, ...]:
    """Return the padded rects of the person boxes the primary styles do not cover.

    The orphans are the boxes :func:`orphan_indices` names, in the order of
    ``persons``, each padded by :func:`pad_person` with
    ``grow=held_growth(box.held)``. Raises ValueError like
    :func:`orphan_indices`, and for a box whose ``held`` is not a non-negative
    integer.
    """
    indices = orphan_indices(persons, faces, mask, primary, h, w)
    if persons is None:
        return ()
    rects = (
        pad_person(persons[i], h, w, grow=held_growth(held_count(persons[i]))) for i in indices
    )
    return tuple(rect for rect in rects if rect is not None)
