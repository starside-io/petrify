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

"""Face covers: gaussian, pixelate, noise, facebox and eyebar.

Every function draws into ``out``, an ``HxWx3`` uint8 array the caller copied
from the frame, and reads its source pixels from ``out`` too. Reading from
``out`` rather than from the original frame means a later style can never bring
original pixels back over an earlier cover.

Each function writes only inside its documented region: the rect for gaussian,
noise and facebox, the rect grown outward to the block grid for pixelate, and
the bar polygon (or the rect, on fallback) for eyebar. Rect bounds must be
integers; a rect that strays past the frame is clipped to it, so a negative bound
cannot wrap around through numpy indexing.

``out`` must be C-contiguous. OpenCV draws in place only into arrays it can wrap
without a copy; for any other layout it would draw into a temporary copy and the
cover would be lost, so such an ``out`` raises ValueError before anything is drawn.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

from petrify.core.detections import Face
from petrify_shadow.regions import Rect

GAUSSIAN_SIGMA_SHARE: dict[str, float] = {"high": 0.20, "medium": 0.10, "low": 0.05}
"""Target blur sigma as a share of the rect's longer side, per strength."""

GAUSSIAN_WORK_SIGMA = 2.0
"""Sigma, in downscaled pixels, that sets the downscale factor (``target / 2``).

Blurring a crop downscaled by ``target / 2`` approximates a blur of the full crop
with the target sigma, at a small fraction of the cost of a large kernel.
"""

GAUSSIAN_MARGIN = 1.1
"""Factor applied to the blur run on the downscaled crop.

The downscale, the small-image blur and the linear upscale each lose a little
spread at the rect border (the upscale clamps to the edge pixels). Without the
margin the edge spread of a step measured up to 2.5 percent below an exact
gaussian of the target sigma on the same crop; with it, it is at or above.
"""

GAUSSIAN_KERNEL_RADIUS = 4.0
"""Kernel radius in sigmas. OpenCV's default for uint8 is 3, which drops about 3 percent of the variance."""

PIXELATE_BLOCKS: dict[str, int] = {"high": 6, "medium": 10, "low": 16}
"""Number of blocks across the rect's longer side, per strength, before quantization."""

PIXELATE_MIN_BLOCK = 4
"""Smallest block size in pixels.

Without it a rect of at most ``blocks`` pixels gets 1 pixel blocks and is left
unchanged, which happens to a face clipped at a frame corner.
"""

PIXELATE_LADDER_BASES = (4, 6)
"""Block sizes are ``base * 2**k``: 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, ...

Each step is 1.5 or 1.33 times the previous one, so the block size changes only
when the box size crosses one of these rare thresholds, not on every few pixels
of detector jitter.
"""

EYEBAR_LENGTH = 2.2
"""Bar length as a multiple of the eye distance."""

EYEBAR_THICKNESS = 0.7
"""Bar thickness as a multiple of the eye distance."""

EYEBAR_MIN_THICKNESS = 3.0
"""Smallest bar thickness in pixels, so a small face still gets a visible bar."""

EYEBAR_MIN_LENGTH_SHARE = 1.0
"""Smallest bar length as a share of the face box width.

On a profile face the eye distance shrinks to 0.14 to 0.19 of the box width, and
a bar sized from it alone covers about a tenth of the face.
"""

EYEBAR_MIN_THICKNESS_SHARE = 0.25
"""Smallest bar thickness as a share of the face box width, for the same reason."""

EYEBAR_MIN_EYE_DISTANCE = 2.0
"""Below this eye distance in pixels the landmarks are too coarse to place a bar."""

EYEBAR_MAX_EYE_DISTANCE_SHARE = 1.5
"""Above this eye distance, as a share of the rect width, the landmarks are implausible."""


def _is_int(v: object) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def _check_out(out: object) -> None:
    """Reject an ``out`` that cannot be drawn into safely."""
    if not isinstance(out, np.ndarray):
        raise ValueError(f"out must be a numpy array, got {type(out).__name__}")
    if out.ndim != 3 or out.shape[2] != 3 or out.dtype != np.uint8:
        raise ValueError(f"out must be an HxWx3 uint8 array, got shape {out.shape} dtype {out.dtype}")
    if not out.flags.writeable:
        raise ValueError("out must be writable: pass a copy of the frame, not the frame")
    if not out.flags.c_contiguous:
        raise ValueError("out must be C-contiguous: pass a copy of the frame")


def _clip(out: np.ndarray, rect: Rect) -> tuple[int, int, int, int] | None:
    """Return ``rect`` clipped to ``out`` as ``(x0, y0, x1, y1)``, or None when empty.

    Float bounds are refused rather than truncated: truncating ``x1 = 20.9`` to 20
    would leave column 20 uncovered.
    """
    if not isinstance(rect, Rect):
        raise ValueError(f"expected a Rect, got {type(rect).__name__}")
    bounds = (rect.x0, rect.y0, rect.x1, rect.y1)
    if not all(_is_int(v) for v in bounds):
        raise ValueError(f"rect bounds must be integers, got {rect!r}")
    h, w = out.shape[:2]
    x0 = min(max(int(rect.x0), 0), w)
    y0 = min(max(int(rect.y0), 0), h)
    x1 = min(max(int(rect.x1), x0), w)
    y1 = min(max(int(rect.y1), y0), h)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _check_color(color: object) -> tuple[int, int, int]:
    # bytes and str iterate to ints or characters and would slip through a
    # generic sequence check.
    if isinstance(color, (bytes, bytearray, memoryview, str)):
        raise ValueError(f"color must be three integers, got {color!r}")
    try:
        values = tuple(color)  # type: ignore[call-overload]
    except TypeError:
        raise ValueError(f"color must be three integers, got {color!r}") from None
    if len(values) != 3 or not all(_is_int(v) and 0 <= v <= 255 for v in values):
        raise ValueError(f"color must be three integers in [0, 255], got {color!r}")
    return int(values[0]), int(values[1]), int(values[2])


def apply_gaussian(out: np.ndarray, rect: Rect, strength: str) -> None:
    """Blur the pixels of ``rect`` with an effective sigma of at least ``share * longer side``.

    The blur is computed as downscale (INTER_AREA), a small gaussian with
    BORDER_REPLICATE, then upscale (INTER_LINEAR). The replicated border keeps
    pixels from outside the rect out of the result, and the small kernel keeps the
    cost near a millisecond for a large face. The small blur runs
    :data:`GAUSSIAN_MARGIN` above the nominal value so the losses of the
    approximation never leave the result weaker than an exact gaussian of the
    target sigma on the same crop.
    """
    _check_out(out)
    if strength not in GAUSSIAN_SIGMA_SHARE:
        raise ValueError(f"unknown strength {strength!r}")
    clipped = _clip(out, rect)
    if clipped is None:
        return
    x0, y0, x1, y1 = clipped
    w, h = x1 - x0, y1 - y0
    sigma = GAUSSIAN_SIGMA_SHARE[strength] * max(w, h)
    factor = max(1.0, sigma / GAUSSIAN_WORK_SIGMA)
    # Rounding the small size down makes each small pixel span at least
    # ``factor`` pixels, so the effective sigma errs toward more blur.
    sw = max(1, int(w / factor))
    sh = max(1, int(h / factor))
    small_sigma = GAUSSIAN_WORK_SIGMA * GAUSSIAN_MARGIN
    radius = math.ceil(GAUSSIAN_KERNEL_RADIUS * small_sigma)
    crop = out[y0:y1, x0:x1]
    small = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(
        small, (2 * radius + 1, 2 * radius + 1), sigmaX=small_sigma, sigmaY=small_sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    out[y0:y1, x0:x1] = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _pixelate_block(side: int, strength: str) -> int:
    """Return the block size for a rect whose longer side is ``side``.

    The raw size ``ceil(side / blocks)`` is raised to at least
    :data:`PIXELATE_MIN_BLOCK`, then rounded up to the next ladder value, which
    errs toward coarser blocks.
    """
    need = max(PIXELATE_MIN_BLOCK, math.ceil(side / PIXELATE_BLOCKS[strength]))
    k = 0
    while True:
        for base in PIXELATE_LADDER_BASES:
            size = base << k
            if size >= need:
                return size
        k += 1


def _upscale_blocks(small: np.ndarray, block: int) -> np.ndarray:
    """Repeat every pixel of ``small`` into a ``block`` by ``block`` square.

    This is the exact nearest-neighbour upscale. cv2.resize with INTER_NEAREST
    computes the source index with a rounded ``1 / block`` and, for some block
    sizes (49, for example), gives the first pixel of a block the previous block's
    color, which would break the frame-anchored grid.
    """
    return np.repeat(np.repeat(small, block, axis=0), block, axis=1)


def apply_pixelate(out: np.ndarray, rect: Rect, strength: str) -> None:
    """Replace the pixels of ``rect`` with blocks of their mean color.

    The block size comes from :func:`_pixelate_block`: about ``longer side /
    blocks``, at least 4 pixels, rounded up to a coarse ladder. The grid is
    anchored to frame coordinates (block edges at multiples of the block size) and
    the rect is grown outward to whole blocks, then clipped to the frame, so the
    cover never shrinks below the rect. A rect smaller than a block becomes one to
    four constant blocks, depending on where the grid lines fall.

    What is stable across frames: while the block size stays on the same ladder
    step, the grid lines stay where they are in the frame, so a few pixels of
    detector jitter do not resample the face on a shifted grid. What is not: the
    block size changes when the box size crosses a ladder threshold, the grown
    rect gains or loses a row or column of blocks as the box crosses a grid line,
    and a face that moves relative to the fixed grid is sampled at a different
    phase on every frame. Averaging many frames of a moving face therefore still
    recovers detail with this or any pixelation; use noise or facebox for a cover
    that cannot be undone.
    """
    _check_out(out)
    if strength not in PIXELATE_BLOCKS:
        raise ValueError(f"unknown strength {strength!r}")
    clipped = _clip(out, rect)
    if clipped is None:
        return
    x0, y0, x1, y1 = clipped
    frame_h, frame_w = out.shape[:2]
    block = _pixelate_block(max(x1 - x0, y1 - y0), strength)
    gx0 = (x0 // block) * block
    gy0 = (y0 // block) * block
    gx1 = -(-x1 // block) * block
    gy1 = -(-y1 // block) * block
    cx1 = min(gx1, frame_w)
    cy1 = min(gy1, frame_h)
    region = out[gy0:cy1, gx0:cx1]
    # Blocks cut by the frame edge are completed by replicating the edge pixels,
    # so every block has whole-block size and the downscale factor is an exact
    # integer: each small pixel is then the mean of exactly one grid block, made
    # of that block's own pixels (zero padding would darken edge blocks).
    padded = cv2.copyMakeBorder(region, 0, gy1 - cy1, 0, gx1 - cx1, cv2.BORDER_REPLICATE)
    nbx = (gx1 - gx0) // block
    nby = (gy1 - gy0) // block
    small = cv2.resize(padded, (nbx, nby), interpolation=cv2.INTER_AREA)
    big = _upscale_blocks(small, block)
    out[gy0:cy1, gx0:cx1] = big[: cy1 - gy0, : cx1 - gx0]


def apply_noise(out: np.ndarray, rect: Rect, rng: np.random.Generator) -> None:
    """Replace the pixels of ``rect`` with uniform random values from ``rng``.

    The pixels are replaced, never added to: the output then carries no
    information about the input, whatever the seed. The generator is the
    caller's; this function never creates or seeds one.
    """
    _check_out(out)
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator, got {type(rng).__name__}")
    clipped = _clip(out, rect)
    if clipped is None:
        return
    x0, y0, x1, y1 = clipped
    out[y0:y1, x0:x1] = rng.integers(0, 256, (y1 - y0, x1 - x0, 3), dtype=np.uint8)


def apply_fill(out: np.ndarray, rect: Rect, color: Sequence[int]) -> None:
    """Fill ``rect`` with the solid BGR ``color``.

    cv2.rectangle fills about 20 times faster than broadcasting a tuple through a
    numpy slice. It draws in place because ``out`` is checked to be C-contiguous.
    """
    _check_out(out)
    bgr = _check_color(color)
    clipped = _clip(out, rect)
    if clipped is None:
        return
    x0, y0, x1, y1 = clipped
    # cv2.rectangle takes inclusive corners.
    cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), bgr, thickness=cv2.FILLED, lineType=cv2.LINE_8)


def eyebar_polygon(
    landmarks: Sequence[Sequence[float]] | None, rect: Rect, face_width: float
) -> np.ndarray | None:
    """Return the four corners of the eye bar as an int32 ``(4, 2)`` array, or None.

    The bar is centered on the midpoint of the first two landmarks (the eyes) and
    rotated by ``atan2(dy, dx)``, with ``d`` the eye distance. Its length is
    ``max(2.2 * d, 1.0 * face_width)`` and its thickness ``max(3, 0.7 * d, 0.25 *
    face_width)``, where ``face_width`` is the width of the detected face box (not
    the padded rect), so a profile face with close-set eyes still gets a bar
    sized to the face. Swapping the two eyes turns the angle by 180 degrees, which
    gives the same bar. Corners are rounded to the nearest pixel.

    None means the landmarks cannot place a bar and the caller must fill the whole
    rect: no landmarks, fewer than two, a non-finite value, ``d < 2``, ``d`` above
    1.5 times the rect width, or a midpoint outside the rect. The corners may lie
    outside the frame; OpenCV clips the fill.
    """
    if landmarks is None or len(landmarks) < 2:
        return None
    try:
        (rx, ry), (lx, ly) = (
            (float(landmarks[0][0]), float(landmarks[0][1])),
            (float(landmarks[1][0]), float(landmarks[1][1])),
        )
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(v) for v in (rx, ry, lx, ly)):
        return None
    dx, dy = lx - rx, ly - ry
    d = math.hypot(dx, dy)
    if d < EYEBAR_MIN_EYE_DISTANCE or d > EYEBAR_MAX_EYE_DISTANCE_SHARE * rect.width:
        return None
    cx, cy = (rx + lx) / 2.0, (ry + ly) / 2.0
    if not (rect.x0 <= cx < rect.x1 and rect.y0 <= cy < rect.y1):
        return None
    fw = float(face_width) if math.isfinite(face_width) and face_width > 0 else 0.0
    angle = math.degrees(math.atan2(dy, dx))
    length = max(EYEBAR_LENGTH * d, EYEBAR_MIN_LENGTH_SHARE * fw)
    thickness = max(EYEBAR_MIN_THICKNESS, EYEBAR_THICKNESS * d, EYEBAR_MIN_THICKNESS_SHARE * fw)
    corners = cv2.boxPoints(((cx, cy), (length, thickness), angle))
    return np.rint(corners).astype(np.int32)


def apply_eyebar(out: np.ndarray, face: Face, rect: Rect, color: Sequence[int]) -> None:
    """Draw a solid bar over the eyes of ``face``, or fill ``rect`` when it cannot.

    The bar is filled with LINE_8 (no antialiasing), so its edge pixels are fully
    covered rather than blended with the eyes below.
    """
    _check_out(out)
    bgr = _check_color(color)
    _clip(out, rect)
    polygon = eyebar_polygon(face.landmarks, rect, face.box.w)
    if polygon is None:
        apply_fill(out, rect, bgr)
        return
    cv2.fillConvexPoly(out, polygon, bgr, lineType=cv2.LINE_8)
