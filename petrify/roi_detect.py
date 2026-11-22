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

"""Automatic timestamp ROI detection via frame-differencing + edge analysis.

Algorithm: two passes
----------------------
**Pass 1: frequency anchor** (finds the Y strip and which corner):

1. Sample ``n_samples`` pairs of frames, each pair spaced ~1 second apart.
2. Build a per-pixel *change-frequency map*: fraction of pairs where the pixel
   changed by more than ``diff_threshold`` grey levels.
3. Threshold at ``min_freq`` to isolate consistently-changing pixels.
4. Morphologically close the mask to merge nearby glyphs into blobs.
5. Reject blobs by area (too large = person/vehicle; too small = noise).
6. Score surviving candidates by change frequency + proximity to a corner
   (security-camera timestamps always live in a corner).
7. The winner is the *anchor*: it accurately locates the **seconds** digits.

Why only seconds?  Only the seconds digits tick every 1 s; minutes change
every 60 s and hours every 3600 s, far below ``min_freq`` in a 60-pair scan.
The anchor gives us the correct Y band and which horizontal edge the timestamp
is pressed against.

**Pass 2: edge-based X extension** (recovers the full timestamp width):

Given the anchor's Y band and corner side, extract that horizontal strip from
a sample frame and run a Laplacian edge detector.  Every text character (even
the static hours and camera-name label that never change) produces strong
spatial edges at the glyph boundaries.  Scanning column-by-column we find the
outermost column with above-threshold edge energy, giving us the full text
width in one cheap operation.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure steps (no capture access)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnchorPlan:
    """Result of Pass 1: the anchor blob and every decision derived from it.

    ``side`` says how the X extent is found: ``"left"`` and ``"right"`` need the
    edge scan of the ``[y_band_start, y_band_end)`` strip (see :func:`edge_far_x`),
    ``"center"`` only pads the anchor.
    """

    frame_w: int
    frame_h: int
    anchor: tuple[int, int, int, int]
    """``(x, y, w, h)`` of the winning blob (the seconds digits)."""
    score: float
    cx_rel: float
    cy_rel: float
    y_final: int
    h_final: int
    y_band_start: int
    y_band_end: int
    side: Literal["left", "right", "center"]


def find_anchor(
    freq_map: np.ndarray,
    *,
    min_freq: float = 0.6,
    margin: int = 8,
    min_area: int = 50,
    max_area_frac: float = 0.10,
    close_kw: int = 30,
    close_kh: int = 15,
    snap_h_thresh: float = 0.45,
    snap_v_thresh: float = 0.35,
) -> AnchorPlan | None:
    """Pass 1 on a change-frequency map: pick the anchor and derive the band and side.

    ``freq_map`` is float32, ``H x W``, each value the fraction (0..1) of frame
    pairs in which that pixel changed. Parameters mean the same as in
    :func:`auto_detect_roi`. Returns ``None`` (after logging the same warning as
    :func:`auto_detect_roi`) when no candidate survives.
    """
    frame_h, frame_w = freq_map.shape[:2]
    frame_diag = (frame_w ** 2 + frame_h ** 2) ** 0.5
    max_area = frame_h * frame_w * max_area_frac

    # Threshold and morphologically close to merge text glyphs.
    consistent = (freq_map >= min_freq).astype(np.uint8) * 255
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kw, close_kh))
    consistent = cv2.morphologyEx(consistent, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(
        consistent, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        logger.warning(
            "auto_detect_roi: no consistently-changing region found, ROI disabled."
        )
        return None

    candidates: list[tuple[float, int, int, int, int]] = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w / 2.0, y + h / 2.0

        corner_dist = min(
            (cx ** 2 + cy ** 2) ** 0.5,
            ((frame_w - cx) ** 2 + cy ** 2) ** 0.5,
            (cx ** 2 + (frame_h - cy) ** 2) ** 0.5,
            ((frame_w - cx) ** 2 + (frame_h - cy) ** 2) ** 0.5,
        ) / frame_diag

        mean_freq = float(np.mean(freq_map[y: y + h, x: x + w]))
        score = mean_freq - 0.3 * corner_dist
        candidates.append((score, x, y, w, h))

    if not candidates:
        logger.warning(
            "auto_detect_roi: all candidates rejected by area/size filters, "
            "ROI disabled.  Try raising --roi-scan-max-area-frac."
        )
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    score, ax, ay, aw, ah = candidates[0]   # anchor bbox

    cx_rel = (ax + aw / 2.0) / frame_w
    cy_rel = (ay + ah / 2.0) / frame_h

    logger.info(
        "Pass 1 anchor: x=%d y=%d w=%d h=%d  freq=%.2f  cx_rel=%.2f cy_rel=%.2f",
        ax, ay, aw, ah, score, cx_rel, cy_rel,
    )

    # Determine Y band with generous margin for multi-line timestamps.
    y_expand = max(margin, ah // 2)

    if cy_rel <= snap_v_thresh:
        # Timestamp near top: snap to top, use edge band from y=0.
        y_final = 0
        y_band_start = 0
        y_band_end = min(frame_h, ay + ah + y_expand * 2)
        h_final = y_band_end
    elif cy_rel >= 1.0 - snap_v_thresh:
        # Timestamp near bottom: snap to bottom.
        y_band_start = max(0, ay - y_expand * 2)
        y_band_end = frame_h
        y_final = y_band_start
        h_final = frame_h - y_final
    else:
        y_band_start = max(0, ay - y_expand)
        y_band_end = min(frame_h, ay + ah + y_expand)
        y_final = y_band_start
        h_final = y_band_end - y_band_start

    side: Literal["left", "right", "center"]
    if cx_rel <= snap_h_thresh:
        side = "left"
    elif cx_rel >= 1.0 - snap_h_thresh:
        side = "right"
    else:
        side = "center"

    return AnchorPlan(
        frame_w=frame_w,
        frame_h=frame_h,
        anchor=(ax, ay, aw, ah),
        score=score,
        cx_rel=cx_rel,
        cy_rel=cy_rel,
        y_final=y_final,
        h_final=h_final,
        y_band_start=y_band_start,
        y_band_end=y_band_end,
        side=side,
    )


def band_edge_magnitude(gray: np.ndarray, y_start: int, y_end: int) -> np.ndarray:
    """Per-pixel Laplacian magnitude (``CV_64F``) of the ``[y_start, y_end)`` strip.

    The Laplacian is taken on the strip itself, as Pass 2 always has, so the
    strip's first and last rows use OpenCV's border handling rather than the
    rows outside the strip.
    """
    strip = gray[y_start:y_end, :]
    lap = cv2.Laplacian(strip, cv2.CV_64F)
    return np.abs(lap)


def edge_far_x(
    col_energy: np.ndarray,
    *,
    from_left: bool,
    edge_threshold: float = 80.0,
    gap_tolerance: int = 30,
    frame_w: int | None = None,
) -> int:
    """Gap-tolerant column scan of Pass 2: the far X of the text block.

    ``col_energy[x]`` is the edge energy of column ``x`` (the per-column maximum
    of the accumulated Laplacian magnitude). Columns are scanned from the near
    edge; the scan stops at the first run of more than ``gap_tolerance`` cold
    columns after a hot one. Returns the last hot column, or the far frame edge
    (``frame_w - 1`` from the left, ``0`` from the right) when no column is hot.
    ``frame_w`` defaults to ``len(col_energy)``.
    """
    if frame_w is None:
        frame_w = len(col_energy)

    # Gap-based scan: advance until the gap after the text block exceeds gap_tolerance.
    xs = range(frame_w) if from_left else range(frame_w - 1, -1, -1)
    last_good = -1
    gap_count = 0

    for x in xs:
        if col_energy[x] > edge_threshold:
            last_good = x
            gap_count = 0
        elif last_good >= 0:
            gap_count += 1
            if gap_count > gap_tolerance:
                break   # first real gap after the text block: stop here

    if last_good < 0:
        logger.debug("_edge_x_extent: no hot columns found, falling back.")
        return (frame_w - 1) if from_left else 0

    logger.debug("_edge_x_extent: far_x=%d  (from_left=%s)", last_good, from_left)
    return last_good


def assemble_roi(
    plan: AnchorPlan, far_x: int | None, margin: int = 8
) -> tuple[int, int, int, int]:
    """Final ``(x, y, w, h)`` from the Pass 1 plan and the Pass 2 far X.

    ``far_x`` is required when ``plan.side`` is ``"left"`` or ``"right"`` and
    ignored for ``"center"``.
    """
    frame_w = plan.frame_w
    if plan.side == "left":
        if far_x is None:
            raise ValueError("far_x is required for a left-snapped anchor")
        x_final = 0
        w_final = min(frame_w, far_x + margin)
    elif plan.side == "right":
        if far_x is None:
            raise ValueError("far_x is required for a right-snapped anchor")
        x_final = max(0, far_x - margin)
        w_final = frame_w - x_final
    else:
        # Anchor is in the centre: no snap, just pad.
        ax, _, aw, _ = plan.anchor
        x_final = max(0, ax - margin)
        w_final = min(frame_w - x_final, aw + 2 * margin)
    return (x_final, plan.y_final, w_final, plan.h_final)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _edge_x_extent(
    cap: cv2.VideoCapture,
    total_frames: int,
    y_start: int,
    y_end: int,
    from_left: bool,
    frame_w: int,
    edge_threshold: float = 80.0,
    gap_tolerance: int = 30,
    n_edge_samples: int = 3,
) -> int:
    """Return the outermost X coordinate of the text block in a horizontal strip.

    Algorithm
    ---------
    1. Sample ``n_edge_samples`` frames; compute per-pixel Laplacian magnitude;
       take the per-column **maximum** across frames.  Using max (not sum) means
       a single frame with a stationary scene feature can raise a column, but
       that is corrected by the gap-based scan below.
    2. Scan columns from the near edge toward the far edge.  A column is "hot"
       if its max Laplacian energy exceeds ``edge_threshold``.
    3. Stop scanning at the **first gap wider than** ``gap_tolerance`` pixels.
       Within a timestamp, inter-character gaps are typically 5 to 25 px; the
       empty space after the last character is ≥ 40 px, so 30 px is a reliable
       separator.  Isolated hot scene-content columns past that gap are ignored.

    Parameters
    ----------
    cap:
        Open video capture; position is restored to 0 on return.
    total_frames:
        Total frame count.
    y_start, y_end:
        Vertical slice (full-frame pixel coordinates) to analyse.
    from_left:
        ``True`` → scan left→right, return last hot column (timestamp on left).
        ``False`` → scan right→left, return last hot column (timestamp on right).
    frame_w:
        Frame width in pixels (fallback when nothing is found).
    edge_threshold:
        Per-column Laplacian magnitude above which a column is considered to
        contain a text edge.  80 separates the typical sharp text glyph
        boundaries (>90) from diffuse background texture (<70).
    gap_tolerance:
        Maximum number of consecutive below-threshold columns allowed inside
        the text block.  30 px bridges inter-character spaces without letting
        the scan jump over the post-text gap.
    n_edge_samples:
        Number of frames whose edge maps are combined (per-pixel max).

    Returns
    -------
    int
        Pixel x of the far edge of the text block, or a frame-edge fallback.
    """
    step = max(1, total_frames // n_edge_samples)
    indices = list(range(0, total_frames, step))[:n_edge_samples]

    edge_acc: np.ndarray | None = None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mag = band_edge_magnitude(gray, y_start, y_end)

        if edge_acc is None:
            edge_acc = mag
        else:
            edge_acc = np.maximum(edge_acc, mag)   # per-pixel max across frames

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if edge_acc is None:
        logger.debug("_edge_x_extent: no frames read, falling back.")
        return (frame_w - 1) if from_left else 0

    # Per-column max energy.
    col_energy = edge_acc.max(axis=0)

    return edge_far_x(
        col_energy,
        from_left=from_left,
        edge_threshold=edge_threshold,
        gap_tolerance=gap_tolerance,
        frame_w=frame_w,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def auto_detect_roi(
    cap: cv2.VideoCapture,
    total_frames: int,
    fps: float,
    n_samples: int = 60,
    min_freq: float = 0.6,
    margin: int = 8,
    min_area: int = 50,
    max_area_frac: float = 0.10,
    diff_threshold: int = 20,
    close_kw: int = 30,
    close_kh: int = 15,
    snap_h_thresh: float = 0.45,
    snap_v_thresh: float = 0.35,
    edge_threshold: float = 80.0,
    edge_gap_tolerance: int = 30,
    n_edge_samples: int = 3,
) -> tuple[int, int, int, int] | None:
    """Detect the burned-in timestamp ROI (two-pass: frequency anchor + edge width).

    Parameters
    ----------
    cap:
        Open ``cv2.VideoCapture``.  Position is restored to 0 on return.
    total_frames:
        Total frame count of the video.
    fps:
        Frames per second; used to set the ~1-second inter-pair gap.
    n_samples:
        Number of frame pairs to diff for Pass 1.  60 is enough for most clips.
    min_freq:
        Fraction of pairs in which a pixel must change to be considered part
        of the timestamp anchor (seconds digits).  0.6 works when the subject
        occupies up to ~60% of the clip.
    margin:
        Pixel padding added around the final bounding box.
    min_area:
        Minimum contour area (px²) to qualify as a timestamp candidate.
    max_area_frac:
        Maximum candidate area as a fraction of total frame area.  Blobs
        larger than this are assumed to be motion regions, not a timestamp.
    diff_threshold:
        Grayscale difference (0 to 255) required to mark a pixel as changed.
    close_kw, close_kh:
        Morphological-close kernel size for Pass 1.  Merges adjacent glyphs.
    snap_h_thresh:
        If the anchor's horizontal centre is within this fraction of the
        frame width from the left or right edge, snap to that edge and use
        edge detection to find the far X boundary.  Default: 0.45.
    snap_v_thresh:
        Same for the vertical direction.  Default: 0.35.
    edge_threshold:
        Laplacian column energy threshold for Pass 2.  80 reliably separates
        sharp text glyph edges from diffuse background texture.
    edge_gap_tolerance:
        Maximum inter-character gap (px) allowed without stopping the scan
        in Pass 2.  30 bridges typical letter spacing without jumping over
        the blank region after the last character.  Default: 30.
    n_edge_samples:
        Number of frames combined (per-pixel max) for the edge map.  Default: 3.

    Returns
    -------
    tuple[int, int, int, int] or None
        ``(x, y, w, h)`` bounding box of the detected ROI, or ``None``
        if no suitable region was found.  A ``--roi`` hint is logged so the
        user can hard-code the value on future runs to skip the scan.
    """
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # ------------------------------------------------------------------
    # Pass 1: build frequency map, find seconds anchor
    # ------------------------------------------------------------------
    pair_gap = max(1, round(fps))   # ~1-second gap so the clock ticks

    usable = total_frames - pair_gap
    if usable <= 0:
        logger.warning("auto_detect_roi: video too short to sample, ROI disabled.")
        return None

    step = max(1, usable // n_samples)
    sample_indices = list(range(0, usable, step))[:n_samples]

    logger.info(
        "Auto ROI scan: diffing %d frame pairs (gap=%d fr) across %d frames...",
        len(sample_indices), pair_gap, total_frames,
    )

    accumulator = np.zeros((frame_h, frame_w), dtype=np.float32)
    valid_pairs = 0

    for idx in tqdm(
        sample_indices,
        desc="ROI scan",
        unit="pair",
        dynamic_ncols=True,
        leave=False,
    ):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret1, f1 = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx + pair_gap)
        ret2, f2 = cap.read()

        if not ret1 or not ret2:
            continue

        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(g1, g2)
        _, binary = cv2.threshold(diff, diff_threshold, 1, cv2.THRESH_BINARY)
        accumulator += binary.astype(np.float32)
        valid_pairs += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if valid_pairs == 0:
        logger.warning("auto_detect_roi: no valid frame pairs read, ROI disabled.")
        return None

    freq_map = accumulator / valid_pairs

    plan = find_anchor(
        freq_map,
        min_freq=min_freq,
        margin=margin,
        min_area=min_area,
        max_area_frac=max_area_frac,
        close_kw=close_kw,
        close_kh=close_kh,
        snap_h_thresh=snap_h_thresh,
        snap_v_thresh=snap_v_thresh,
    )
    if plan is None:
        return None

    # Pass 2: edge-based X extension. Laplacian edges in the detected Y strip give
    # the full text width (static characters like hours and camera labels too).
    far_x: int | None = None
    if plan.side != "center":
        far_x = _edge_x_extent(
            cap, total_frames, plan.y_band_start, plan.y_band_end,
            from_left=plan.side == "left", frame_w=frame_w,
            edge_threshold=edge_threshold,
            gap_tolerance=edge_gap_tolerance,
            n_edge_samples=n_edge_samples,
        )
    x_final, y_final, w_final, h_final = assemble_roi(plan, far_x, margin)

    logger.info(
        "Auto ROI detected: x=%d y=%d w=%d h=%d",
        x_final, y_final, w_final, h_final,
    )
    logger.info(
        "  → To skip auto-detection next time: --roi %d %d %d %d",
        x_final, y_final, w_final, h_final,
    )

    return (x_final, y_final, w_final, h_final)


# ---------------------------------------------------------------------------
# Streaming detector
# ---------------------------------------------------------------------------

class WarmupRoiDetector:
    """Detect the timestamp ROI from the first ``warmup_seconds`` of a stream, then lock.

    For sources that cannot be scanned in advance. Feed every frame in order with
    :meth:`update`; after ``ceil(warmup_seconds * fps)`` frames the ROI is computed
    with the same pure steps as :func:`auto_detect_roi` (:func:`find_anchor`,
    :func:`edge_far_x`, :func:`assemble_roi`), :attr:`done` becomes True and the
    buffers are freed. Later calls to :meth:`update` are no-ops.

    Pass 1 pairs frame ``i`` with frame ``i - pair_gap`` (``pair_gap = max(1,
    round(fps))``, about one second, as in the file version) for every
    ``i >= pair_gap`` in the window: every available pair is used, where the file
    version subsamples ``n_samples`` pairs across the whole clip. Only the last
    ``pair_gap`` grayscale frames, the change accumulator and the pair count are
    kept, so memory does not grow with the warmup length.

    Pass 2 uses ``n_edge_samples`` frames of the window chosen with the file
    version's own formula applied to the window length (``step = max(1, n //
    n_edge_samples)``, indices ``0, step, 2 * step, ...``). Their grayscale images
    are kept (``n_edge_samples * H * W`` bytes) and the Laplacian of the anchor's
    Y band is computed at the end with :func:`band_edge_magnitude`, the same
    function :func:`_edge_x_extent` uses. For identical frames this is therefore
    exactly the file version's per-band computation. A running per-pixel maximum
    of the full-frame Laplacian would not be: OpenCV reflects the band's first
    and last rows when the Laplacian is taken on the band, but uses the real
    neighbour rows when it is taken on the full frame, so those two rows differ.
    Keeping the few grayscale frames is also smaller than one ``CV_64F`` map.

    ``max_roi_area_frac`` caps the final ROI, after snapping and edge extension:
    a box larger than that fraction of the frame is discarded (:attr:`roi` stays
    None, with one warning) because it is too large to be a timestamp. On a scene
    with no burned-in timestamp, such as a webcam, a moving person can qualify as
    the changing anchor and grow into a large region that would otherwise be
    pasted live on every frozen frame. ``max_area_frac`` limits only the anchor
    blob, not the final box. The file version :func:`auto_detect_roi` has no such
    cap. Remaining limitation: a region under the cap is trusted and pasted live,
    so a person moving through it is visible there.

    Parameters other than ``fps``, ``warmup_seconds`` and ``max_roi_area_frac``
    mean the same as in :func:`auto_detect_roi`.
    """

    def __init__(
        self,
        fps: float,
        warmup_seconds: float = 10.0,
        *,
        min_freq: float = 0.6,
        margin: int = 8,
        min_area: int = 50,
        max_area_frac: float = 0.10,
        diff_threshold: int = 20,
        close_kw: int = 30,
        close_kh: int = 15,
        snap_h_thresh: float = 0.45,
        snap_v_thresh: float = 0.35,
        edge_threshold: float = 80.0,
        edge_gap_tolerance: int = 30,
        n_edge_samples: int = 3,
        max_roi_area_frac: float = 0.15,
    ) -> None:
        if not fps > 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        if not warmup_seconds > 0:
            raise ValueError(f"warmup_seconds must be > 0, got {warmup_seconds}")
        if n_edge_samples < 1:
            raise ValueError(f"n_edge_samples must be >= 1, got {n_edge_samples}")
        if not 0 < max_roi_area_frac <= 1:
            raise ValueError(f"max_roi_area_frac must be in (0, 1], got {max_roi_area_frac}")
        self._max_roi_area_frac = max_roi_area_frac
        self._min_freq = min_freq
        self._margin = margin
        self._min_area = min_area
        self._max_area_frac = max_area_frac
        self._diff_threshold = diff_threshold
        self._close_kw = close_kw
        self._close_kh = close_kh
        self._snap_h_thresh = snap_h_thresh
        self._snap_v_thresh = snap_v_thresh
        self._edge_threshold = edge_threshold
        self._edge_gap_tolerance = edge_gap_tolerance

        self._pair_gap = max(1, round(fps))
        self._warmup_frames = math.ceil(warmup_seconds * fps)
        step = max(1, self._warmup_frames // n_edge_samples)
        self._edge_indices = frozenset(
            list(range(0, self._warmup_frames, step))[:n_edge_samples]
        )

        self._grays: deque[np.ndarray] = deque(maxlen=self._pair_gap)
        self._edge_grays: list[np.ndarray] = []
        self._accumulator: np.ndarray | None = None
        self._pairs = 0
        self._seen = 0
        self._shape: tuple[int, ...] | None = None
        self._done = False
        self._roi: tuple[int, int, int, int] | None = None

    @property
    def done(self) -> bool:
        """True once the warmup window has been consumed."""
        return self._done

    @property
    def roi(self) -> tuple[int, int, int, int] | None:
        """``(x, y, w, h)``; None until :attr:`done`, and None afterwards if nothing was found."""
        return self._roi

    @property
    def pair_gap(self) -> int:
        """Distance in frames between the two frames of a pair."""
        return self._pair_gap

    @property
    def warmup_frames(self) -> int:
        """Number of frames consumed before the ROI locks."""
        return self._warmup_frames

    @property
    def frames_seen(self) -> int:
        """Frames consumed so far (stops at :attr:`warmup_frames`)."""
        return self._seen

    @property
    def buffered_frames(self) -> int:
        """Grayscale frames held for pairing (never more than :attr:`pair_gap`)."""
        return len(self._grays)

    def update(self, image: np.ndarray) -> None:
        """Consume the next frame (BGR uint8, H x W x 3). No-op once :attr:`done`.

        Raises ``ValueError`` if the frame's shape differs from the first frame's.
        """
        if self._done:
            return
        if self._shape is None:
            self._shape = image.shape
        elif image.shape != self._shape:
            raise ValueError(
                f"WarmupRoiDetector: frame shape {image.shape} differs from the "
                f"first frame's {self._shape}"
            )

        index = self._seen
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if index >= self._pair_gap:
            # The deque holds frames index - pair_gap .. index - 1; the oldest pairs with this one.
            diff = cv2.absdiff(self._grays[0], gray)
            _, binary = cv2.threshold(diff, self._diff_threshold, 1, cv2.THRESH_BINARY)
            if self._accumulator is None:
                self._accumulator = np.zeros(gray.shape[:2], dtype=np.float32)
            self._accumulator += binary.astype(np.float32)
            self._pairs += 1
        self._grays.append(gray)
        if index in self._edge_indices:
            self._edge_grays.append(gray)

        self._seen += 1
        if self._seen >= self._warmup_frames:
            try:
                self._roi = self._compute()
            finally:
                self._release()
                self._done = True

    def _compute(self) -> tuple[int, int, int, int] | None:
        """Run the pure steps on the accumulated state."""
        if self._accumulator is None or self._pairs == 0:
            logger.warning(
                "WarmupRoiDetector: warmup window of %d frames is too short to form "
                "a frame pair (gap=%d fr); ROI disabled.",
                self._warmup_frames,
                self._pair_gap,
            )
            return None

        freq_map = self._accumulator / self._pairs
        plan = find_anchor(
            freq_map,
            min_freq=self._min_freq,
            margin=self._margin,
            min_area=self._min_area,
            max_area_frac=self._max_area_frac,
            close_kw=self._close_kw,
            close_kh=self._close_kh,
            snap_h_thresh=self._snap_h_thresh,
            snap_v_thresh=self._snap_v_thresh,
        )
        if plan is None:
            return None

        far_x: int | None = None
        if plan.side != "center":
            edge_acc: np.ndarray | None = None
            for gray in self._edge_grays:
                mag = band_edge_magnitude(gray, plan.y_band_start, plan.y_band_end)
                edge_acc = mag if edge_acc is None else np.maximum(edge_acc, mag)
            assert edge_acc is not None  # Frame 0 is always an edge sample.
            far_x = edge_far_x(
                edge_acc.max(axis=0),
                from_left=plan.side == "left",
                edge_threshold=self._edge_threshold,
                gap_tolerance=self._edge_gap_tolerance,
                frame_w=plan.frame_w,
            )
        roi = assemble_roi(plan, far_x, self._margin)
        _, _, w, h = roi
        share = (w * h) / (plan.frame_w * plan.frame_h)
        if share > self._max_roi_area_frac:
            logger.warning(
                "WarmupRoiDetector: detected ROI %s covers %.1f%% of the frame, more than "
                "the %.1f%% cap; it is too large to be a timestamp, so the ROI overlay "
                "is disabled.",
                roi,
                share * 100,
                self._max_roi_area_frac * 100,
            )
            return None
        return roi

    def _release(self) -> None:
        """Free every buffer held during warmup."""
        self._grays.clear()
        self._edge_grays = []
        self._accumulator = None
