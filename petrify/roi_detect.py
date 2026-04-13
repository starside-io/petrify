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

Algorithm — two passes
----------------------
**Pass 1 — frequency anchor** (finds the Y strip and which corner):

1. Sample ``n_samples`` pairs of frames, each pair spaced ~1 second apart.
2. Build a per-pixel *change-frequency map*: fraction of pairs where the pixel
   changed by more than ``diff_threshold`` grey levels.
3. Threshold at ``min_freq`` to isolate consistently-changing pixels.
4. Morphologically close the mask to merge nearby glyphs into blobs.
5. Reject blobs by area (too large = person/vehicle; too small = noise).
6. Score surviving candidates by change frequency + proximity to a corner
   (security-camera timestamps always live in a corner).
7. The winner is the *anchor* — it accurately locates the **seconds** digits.

Why only seconds?  Only the seconds digits tick every 1 s; minutes change
every 60 s and hours every 3600 s — far below ``min_freq`` in a 60-pair scan.
The anchor gives us the correct Y band and which horizontal edge the timestamp
is pressed against.

**Pass 2 — edge-based X extension** (recovers the full timestamp width):

Given the anchor's Y band and corner side, extract that horizontal strip from
a sample frame and run a Laplacian edge detector.  Every text character — even
the static hours and camera-name label that never change — produces strong
spatial edges at the glyph boundaries.  Scanning column-by-column we find the
outermost column with above-threshold edge energy, giving us the full text
width in one cheap operation.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


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
       Within a timestamp, inter-character gaps are typically 5–25 px; the
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
        strip = gray[y_start:y_end, :]
        lap = cv2.Laplacian(strip, cv2.CV_64F)
        mag = np.abs(lap)

        if edge_acc is None:
            edge_acc = mag
        else:
            edge_acc = np.maximum(edge_acc, mag)   # per-pixel max across frames

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if edge_acc is None:
        logger.debug("_edge_x_extent: no frames read — falling back.")
        return (frame_w - 1) if from_left else 0

    # Per-column max energy.
    col_energy = edge_acc.max(axis=0)

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
                break   # first real gap after the text block — stop here

    if last_good < 0:
        logger.debug("_edge_x_extent: no hot columns found — falling back.")
        return (frame_w - 1) if from_left else 0

    logger.debug("_edge_x_extent: far_x=%d  (from_left=%s)", last_good, from_left)
    return last_good


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
        Grayscale difference (0–255) required to mark a pixel as changed.
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
    frame_diag = (frame_w ** 2 + frame_h ** 2) ** 0.5
    max_area = frame_h * frame_w * max_area_frac

    # ------------------------------------------------------------------
    # Pass 1 — build frequency map, find seconds anchor
    # ------------------------------------------------------------------
    pair_gap = max(1, round(fps))   # ~1-second gap so the clock ticks

    usable = total_frames - pair_gap
    if usable <= 0:
        logger.warning("auto_detect_roi: video too short to sample — ROI disabled.")
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
        logger.warning("auto_detect_roi: no valid frame pairs read — ROI disabled.")
        return None

    freq_map = accumulator / valid_pairs

    # Threshold and morphologically close to merge text glyphs.
    consistent = (freq_map >= min_freq).astype(np.uint8) * 255
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kw, close_kh))
    consistent = cv2.morphologyEx(consistent, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(
        consistent, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        logger.warning(
            "auto_detect_roi: no consistently-changing region found — ROI disabled."
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
            "auto_detect_roi: all candidates rejected by area/size filters — "
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

    # ------------------------------------------------------------------
    # Pass 2 — edge-based X extension
    # Use Laplacian edges in the detected Y strip to find the full text
    # width (captures static characters like hours and camera labels too).
    # ------------------------------------------------------------------

    # Determine Y band with generous margin for multi-line timestamps.
    y_expand = max(margin, ah // 2)

    if cy_rel <= snap_v_thresh:
        # Timestamp near top → snap to top, use edge band from y=0.
        y_final = 0
        y_band_start = 0
        y_band_end = min(frame_h, ay + ah + y_expand * 2)
        h_final = y_band_end
    elif cy_rel >= 1.0 - snap_v_thresh:
        # Timestamp near bottom → snap to bottom.
        y_band_start = max(0, ay - y_expand * 2)
        y_band_end = frame_h
        y_final = y_band_start
        h_final = frame_h - y_final
    else:
        y_band_start = max(0, ay - y_expand)
        y_band_end = min(frame_h, ay + ah + y_expand)
        y_final = y_band_start
        h_final = y_band_end - y_band_start

    # Determine horizontal extent via edge detection in the Y band.
    if cx_rel <= snap_h_thresh:
        from_left = True
        # Snap to left edge; use edge detector to find right boundary.
        far_x = _edge_x_extent(
            cap, total_frames, y_band_start, y_band_end,
            from_left=True, frame_w=frame_w,
            edge_threshold=edge_threshold,
            gap_tolerance=edge_gap_tolerance,
            n_edge_samples=n_edge_samples,
        )
        x_final = 0
        w_final = min(frame_w, far_x + margin)
    elif cx_rel >= 1.0 - snap_h_thresh:
        from_left = False
        # Snap to right edge; use edge detector to find left boundary.
        far_x = _edge_x_extent(
            cap, total_frames, y_band_start, y_band_end,
            from_left=False, frame_w=frame_w,
            edge_threshold=edge_threshold,
            gap_tolerance=edge_gap_tolerance,
            n_edge_samples=n_edge_samples,
        )
        x_final = max(0, far_x - margin)
        w_final = frame_w - x_final
    else:
        # Anchor is in the centre — no snap, just pad.
        x_final = max(0, ax - margin)
        w_final = min(frame_w - x_final, aw + 2 * margin)

    logger.info(
        "Auto ROI detected: x=%d y=%d w=%d h=%d",
        x_final, y_final, w_final, h_final,
    )
    logger.info(
        "  → To skip auto-detection next time: --roi %d %d %d %d",
        x_final, y_final, w_final, h_final,
    )

    return (x_final, y_final, w_final, h_final)
