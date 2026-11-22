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

"""Body styles: silhouette (black or colored), outline, and filled person boxes.

Every function writes only into ``out``, a writable ``HxWx3`` uint8 array the
caller copied from the frame, and only inside the region it documents. Inputs
that do not fit together (a wrong shape, a rect outside the frame, a crop of the
wrong size) raise ValueError instead of being clipped or guessed at: the engine
then fails closed and emits a black frame.

Fills replace pixels with a constant and never blend, so the covered pixels of
the output carry no information about the input.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import cv2
import numpy as np

from petrify_shadow.regions import PreparedMask, Rect

OUTLINE_SHARE_PER_MILLE = 4
"""Outline thickness as a share of the frame height, in thousandths (0.004)."""

MIN_OUTLINE_PX = 2
"""Thinnest outline in pixels, so it stays visible on small frames."""


def outline_thickness(h: int) -> int:
    """Return the outline thickness for a frame ``h`` pixels high.

    This is ``max(2, round(0.004 * h))``, computed in integers with halves
    rounded up. Integer math keeps ``0.004 * h`` exact (the float is not), and
    rounding halves up avoids Python's round-half-to-even, which would give 2 for
    ``h = 625`` where a hand computation gives 3. Raises ValueError for a
    negative or non-integer ``h``.
    """
    if isinstance(h, (bool, np.bool_)) or not isinstance(h, (int, np.integer)):
        raise ValueError(f"h must be an integer, got {type(h).__name__}")
    height = int(h)
    if height < 0:
        raise ValueError(f"h must not be negative, got {height}")
    return max(MIN_OUTLINE_PX, (OUTLINE_SHARE_PER_MILLE * height + 500) // 1000)


def _check_out(out: object) -> tuple[int, int]:
    """Return the frame size of a valid ``out``, or raise ValueError."""
    if not isinstance(out, np.ndarray):
        raise ValueError(f"out must be a numpy array, got {type(out).__name__}")
    if out.dtype != np.uint8 or out.ndim != 3 or out.shape[2] != 3:
        raise ValueError(
            f"out must be an HxWx3 uint8 array, got shape {out.shape} dtype {out.dtype}"
        )
    if not out.flags.writeable:
        raise ValueError("out must be writable: pass a copy of the frame, not the frame")
    if not out.flags.c_contiguous:
        # OpenCV draws in place only into arrays it can wrap without a copy. A
        # copy would take the fill and leave ``out`` showing the person.
        raise ValueError("out must be C-contiguous: pass a copy of the frame")
    return int(out.shape[0]), int(out.shape[1])


def _check_color(color: object) -> np.ndarray:
    """Return ``color`` as a 3-element uint8 array, or raise ValueError."""
    try:
        values = tuple(color)  # type: ignore[call-overload]
    except TypeError:
        raise ValueError(f"color must be three integers, got {color!r}") from None
    if len(values) != 3 or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
        for v in values
    ):
        raise ValueError(f"color must be three integers, got {color!r}")
    if any(not 0 <= int(v) <= 255 for v in values):
        raise ValueError(f"color values must be in 0 to 255, got {color!r}")
    return np.array([int(v) for v in values], dtype=np.uint8)


def _check_rect(rect: object, h: int, w: int) -> Rect:
    """Return ``rect`` if it lies inside an ``h`` by ``w`` frame, or raise ValueError.

    A negative bound would wrap around in a numpy slice and paint the wrong side
    of the frame, and a bound past the frame would be clipped silently, so both
    are refused rather than corrected.
    """
    if not isinstance(rect, Rect):
        raise ValueError(f"expected a Rect, got {type(rect).__name__}")
    bounds = (rect.x0, rect.y0, rect.x1, rect.y1)
    if any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in bounds
    ):
        raise ValueError(f"rect bounds must be integers, got {rect!r}")
    if not (0 <= rect.x0 <= rect.x1 <= w and 0 <= rect.y0 <= rect.y1 <= h):
        raise ValueError(f"rect {rect!r} does not lie inside the frame {(h, w)}")
    return rect


def _check_prepared(pm: object, h: int, w: int) -> PreparedMask:
    """Return ``pm`` if its crop fits its rect and its rect fits the frame."""
    if not isinstance(pm, PreparedMask):
        raise ValueError(f"expected a PreparedMask, got {type(pm).__name__}")
    rect = _check_rect(pm.rect, h, w)
    crop = pm.crop
    if not isinstance(crop, np.ndarray) or crop.dtype != np.bool_ or crop.ndim != 2:
        raise ValueError("crop must be a 2-d numpy array of dtype bool")
    if crop.shape != (rect.height, rect.width):
        raise ValueError(
            f"crop shape {crop.shape} does not match its rect {(rect.height, rect.width)}"
        )
    return pm


def _require_in_place(result: object, target: np.ndarray) -> None:
    # OpenCV returns a new array instead of raising when it cannot write into the
    # one it was given. The fill would then be lost and the pixels left visible,
    # so that case raises and the engine fails closed.
    if result is not target:
        raise RuntimeError("OpenCV did not draw in place; refusing to return an uncovered frame")


def _solid(h: int, w: int, color: np.ndarray) -> np.ndarray:
    """Return a new ``h`` by ``w`` image of one color.

    Built from one constant plane per channel: numpy broadcasting a 3-value
    color across an image measured about 35 times slower on a person-sized rect.
    """
    return cv2.merge([np.full((h, w), int(v), np.uint8) for v in color])


def _fill_where(out: np.ndarray, rect: Rect, where: np.ndarray, color: np.ndarray) -> None:
    """Set every pixel of ``out`` inside ``rect`` where ``where`` is True to ``color``.

    Only the rect is visited, never the full frame. ``cv2.copyTo`` is used
    because numpy's masked copy and boolean indexing measured 30 to 80 times
    slower on a person-sized crop.
    """
    region = out[rect.y0 : rect.y1, rect.x0 : rect.x1]
    mask = where.view(np.uint8)
    if not mask.flags.c_contiguous:
        mask = np.ascontiguousarray(mask)
    source = _solid(rect.height, rect.width, color)
    _require_in_place(cv2.copyTo(source, mask, region), region)


def apply_silhouette(out: np.ndarray, pm: PreparedMask, color: tuple[int, int, int]) -> None:
    """Fill every True pixel of ``pm.crop`` with ``color``, in place in ``out``.

    The crop is placed at ``pm.rect``. Pixels outside the rect, and pixels inside
    it where the crop is False, are left untouched. The caller passes black for
    the ``silhouette`` style and the ``color`` option for ``silhouette_color``.
    ``pm.crop`` is only read. Raises ValueError for an ``out`` that is not a
    writable HxWx3 uint8 array, a rect outside the frame, a crop whose shape is
    not the rect's, or a color that is not three integers in 0 to 255.
    """
    h, w = _check_out(out)
    fill = _check_color(color)
    prepared = _check_prepared(pm, h, w)
    rect = prepared.rect
    if rect.is_empty:
        return
    _fill_where(out, rect, prepared.crop, fill)


@lru_cache(maxsize=32)
def _erosion_kernel(thickness: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * thickness + 1, 2 * thickness + 1)
    )
    # Shared between calls through the cache, so nothing may write to it.
    kernel.setflags(write=False)
    return kernel


def apply_outline(
    out: np.ndarray, pm: PreparedMask, color: tuple[int, int, int], thickness: int
) -> None:
    """Draw the inner edge of ``pm.crop`` in ``color``, ``thickness`` pixels wide.

    The drawn pixels are exactly the crop minus its erosion by OpenCV's
    elliptical structuring element of size ``2 * thickness + 1``: a mask pixel is
    drawn when some pixel within that element around it is outside the mask.
    Nothing is drawn inside the eroded mask or outside the crop. A thickness
    larger than the mask erodes it away completely, so the whole mask is drawn.

    Edges: everything outside the crop counts as outside the mask. Where the
    crop border lies inside the frame this is exact, since the dilated mask is
    False everywhere outside its rect. Where the crop border is the frame edge,
    the area past the frame also counts as outside, so the outline runs along
    the frame edge and is always a closed band around the visible mask. That
    draws more pixels than leaving the band open, and a mask that fills the frame
    gets a band of ``thickness`` pixels along all four frame edges.

    Raises ValueError like :func:`apply_silhouette`, and for a thickness that is
    not a positive integer.
    """
    h, w = _check_out(out)
    fill = _check_color(color)
    if isinstance(thickness, (bool, np.bool_)) or not isinstance(
        thickness, (int, np.integer)
    ):
        raise ValueError(f"thickness must be an integer, got {type(thickness).__name__}")
    t = int(thickness)
    if t < 1:
        raise ValueError(f"thickness must be at least 1, got {t}")
    prepared = _check_prepared(pm, h, w)
    rect = prepared.rect
    if rect.is_empty:
        return
    crop_u8 = prepared.crop.view(np.uint8)
    if not crop_u8.flags.c_contiguous:
        # OpenCV needs a contiguous input; the copy leaves the read-only crop alone.
        crop_u8 = np.ascontiguousarray(crop_u8)
    # erode's default border value counts the area past the edge as inside the
    # mask, which would leave the band open at every crop border. Zero makes it
    # count as outside, which is the behavior documented above.
    eroded = cv2.erode(
        crop_u8, _erosion_kernel(t), borderType=cv2.BORDER_CONSTANT, borderValue=0
    )
    # Both are 0 or 1, so "greater" is exactly "in the mask and not in the erosion".
    band = np.greater(crop_u8, eroded)
    _fill_where(out, rect, band, fill)


def apply_bodybox(
    out: np.ndarray, rects: Sequence[Rect], color: tuple[int, int, int]
) -> None:
    """Fill each rect in ``rects`` with ``color``, in place in ``out``.

    Overlapping rects are fine, since every fill writes the same color. An empty
    sequence does nothing. Every rect is checked before any is filled, so a bad
    rect raises ValueError with ``out`` still unchanged.
    """
    h, w = _check_out(out)
    fill = _check_color(color)
    checked = [_check_rect(rect, h, w) for rect in rects]
    for rect in checked:
        if rect.is_empty:
            continue
        # Copying a solid image is a plain row copy; assigning the 3-value color
        # directly broadcasts it per pixel and is much slower.
        out[rect.y0 : rect.y1, rect.x0 : rect.x1] = _solid(rect.height, rect.width, fill)
