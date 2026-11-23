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

"""Measure where heads sit inside person boxes, to check a head-from-body rule.

Runs YuNet, the DNN person detector and the segmenter on the frames of a clip
and prints three tables:

1. For each person box a face clears (the shadow plugin's own rule, through
   :func:`petrify_shadow.regions.orphan_indices`): the padded face rect of
   :func:`petrify_shadow.regions.pad_face` as shares of the person box, face
   height over box height, box height over width, and the count of boxes that
   touch a frame edge.
2. The segmenter head band: 12 percent of the box height starting at the top
   row of the mask inside the box, for boxes whose mask share is at least 30
   percent and whose height over width is at least 1.4, split by box height.
3. The proposed head rule: the head rect is the width of the padded person box
   (:func:`petrify_shadow.regions.pad_person`), from its top edge down to
   ``y + ceil(h / 2)``. It applies only to boxes at least 150 px tall with
   height over width at least 1.4; any other box keeps the whole-box fill. The
   table gives the share of padded face rects lying fully inside the head rect,
   every violation with its frame, and the share of orphan boxes that the rule
   accepts.

It also writes two contact sheets into the output directory,
``<clip>_head_orphans.png`` and ``<clip>_head_cleared.png``. Each tile shows one
person box (green), its head rect (yellow) when the rule accepts the box, and
the padded face rect (red) when a face cleared it. Violations come first on the
cleared sheet.

Example::

    PYTHONPATH=. python plugins/petrify-shadow/tools/head_sheet.py \\
        data/camera.mp4 models /tmp/sheets --step 2

The sheets show frames of the clip: write them outside the repository.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from petrify_shadow.options import DEFAULT_PADDING  # noqa: E402
from petrify_shadow.regions import Rect, orphan_indices, pad_face, pad_person  # noqa: E402

from petrify.core.config import FaceYuNetConfig, PersonDnnConfig, SegmentConfig  # noqa: E402
from petrify.core.detections import Box, Face  # noqa: E402
from petrify.detectors.face_yunet import FaceYuNetDetector  # noqa: E402
from petrify.detectors.person_dnn import PersonDnnDetector  # noqa: E402
from petrify.detectors.segment import PersonSegmenter  # noqa: E402
from petrify.sources import FileSource  # noqa: E402

# The rule under test, as proposed in the phase 6 design.
HEAD_MIN_HEIGHT_PX = 150
HEAD_MIN_ASPECT_NUM, HEAD_MIN_ASPECT_DEN = 7, 5  # height / width at least 1.4

# The segmenter head proxy.
BAND_SHARE = 0.12
MIN_MASK_SHARE = 0.30

PERCENTILES = (("min", 0), ("p5", 5), ("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))

TILE_W, TILE_H = 160, 240
COLUMNS = 10
BOX_COLOR = (0, 200, 0)
HEAD_COLOR = (0, 220, 255)
FACE_COLOR = (0, 0, 255)


@dataclass(frozen=True)
class Seen:
    """One person box on one frame, with the padded face rect that cleared it, if any.

    ``top`` is the top-half rect of the box whether or not the rule accepts the
    box, so containment can also be measured on the boxes it refuses.
    """

    frame: int
    box: Box
    face_rect: Rect | None
    face_h: int
    top: Rect | None
    eligible: bool
    edge: bool

    @property
    def head(self) -> Rect | None:
        """The head rect the rule would fill, or None for the whole-box fill."""
        return self.top if self.eligible else None

    @property
    def inside(self) -> bool:
        r, top = self.face_rect, self.top
        if r is None or top is None:
            return False
        return r.x0 >= top.x0 and r.y0 >= top.y0 and r.x1 <= top.x1 and r.y1 <= top.y1


def eligible(box: Box) -> bool:
    """True when the proposed rule accepts ``box``; integer arithmetic, so 1.4 is exact."""
    return box.h >= HEAD_MIN_HEIGHT_PX and box.h * HEAD_MIN_ASPECT_DEN >= box.w * HEAD_MIN_ASPECT_NUM


def top_half(box: Box, h: int, w: int) -> Rect | None:
    """Return the top-half rect of ``box`` in an ``h`` by ``w`` frame.

    The left, right and top edges are those of the padded person box. The
    bottom edge is ``y + ceil(h / 2)``, rounded outward and clipped after
    rounding like every edge in the plugin's regions.
    """
    padded = pad_person(box, h, w)
    if padded is None:
        return None
    bottom = min(max(box.y + -(-box.h // 2), 0), h)
    if bottom <= padded.y0:
        return None
    return Rect(padded.x0, padded.y0, padded.x1, bottom)


def _touches_edge(box: Box, h: int, w: int) -> bool:
    return box.x <= 0 or box.y <= 0 or box.x + box.w >= w or box.y + box.h >= h


def _clearing(persons: tuple[Box, ...], face: Face, h: int, w: int) -> int | None:
    """Return the index of the person box ``face`` clears alone, by the plugin's own rule."""
    everyone = set(orphan_indices(persons, (), None, "face", h, w))
    left = set(orphan_indices(persons, (face,), None, "face", h, w))
    cleared = everyone - left
    return cleared.pop() if cleared else None


def _band_bottom(mask: np.ndarray, box: Box) -> float | None:
    """Return the bottom of the mask head band as a share of the box height.

    None when the box holds less than :data:`MIN_MASK_SHARE` mask pixels of its
    own area (parts outside the frame count as no mask, as in the plugin).
    """
    frame_h, frame_w = mask.shape
    x0, x1 = min(max(box.x, 0), frame_w), min(max(box.x + box.w, 0), frame_w)
    y0, y1 = min(max(box.y, 0), frame_h), min(max(box.y + box.h, 0), frame_h)
    sub = mask[y0:y1, x0:x1]
    if np.count_nonzero(sub) < MIN_MASK_SHARE * box.w * box.h:
        return None
    rows = np.flatnonzero(sub.any(axis=1))
    top = (y0 - box.y) + int(rows[0])
    band = max(1, round(BAND_SHARE * box.h))
    return (top + band) / box.h


def _pct(values: list[float]) -> str:
    if not values:
        return "n=0"
    a = np.asarray(values, dtype=np.float64)
    parts = [f"{name} {np.percentile(a, p):.3f}" for name, p in PERCENTILES]
    return f"n={a.size}  " + "  ".join(parts)


def _share(part: int, whole: int) -> str:
    return f"{part}/{whole} ({100 * part / whole:.1f}%)" if whole else f"{part}/0"


def _tile(image: np.ndarray, seen: Seen, label: str) -> np.ndarray:
    h, w = image.shape[:2]
    box = seen.box
    margin = max(4, box.w // 5, box.h // 10)
    x0, y0 = max(box.x - margin, 0), max(box.y - margin, 0)
    x1, y1 = min(box.x + box.w + margin, w), min(box.y + box.h + margin, h)
    crop = image[y0:y1, x0:x1].copy()
    scale = min(TILE_W / crop.shape[1], TILE_H / crop.shape[0])
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    def at(px: float, py: float) -> tuple[int, int]:
        return int(round((px - x0) * scale)), int(round((py - y0) * scale))

    cv2.rectangle(crop, at(box.x, box.y), at(box.x + box.w, box.y + box.h), BOX_COLOR, 1)
    if seen.head is not None:
        r = seen.head
        cv2.rectangle(crop, at(r.x0, r.y0), at(r.x1, r.y1), HEAD_COLOR, 2)
    if seen.face_rect is not None:
        r = seen.face_rect
        cv2.rectangle(crop, at(r.x0, r.y0), at(r.x1, r.y1), FACE_COLOR, 2)
    tile = np.zeros((TILE_H + 16, TILE_W, 3), np.uint8)
    tile[: crop.shape[0], : crop.shape[1]] = crop
    cv2.putText(tile, label, (2, TILE_H + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return tile


def _sheet(tiles: list[np.ndarray]) -> np.ndarray:
    rows = []
    for start in range(0, len(tiles), COLUMNS):
        row = tiles[start : start + COLUMNS]
        row += [np.zeros_like(tiles[0])] * (COLUMNS - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def _spread(items: list[Seen], count: int) -> list[Seen]:
    """Pick up to ``count`` items spread over the whole clip, not only its first frames."""
    if len(items) <= count:
        return items
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]


def _write_sheet(source: FileSource, picks: list[Seen], path: Path) -> None:
    tiles = []
    for seen in picks:
        image = source.read_frame(seen.frame)
        if image is None:
            continue
        rule = "head" if seen.head is not None else "fallback"
        label = f"f{seen.frame} {seen.box.w}x{seen.box.h} {rule}"
        if seen.face_rect is not None and seen.eligible and not seen.inside:
            label += " OUT"
        tiles.append(_tile(image, seen, label))
    if tiles:
        cv2.imwrite(str(path), _sheet(tiles))
        print(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("clip", type=Path, help="video file to measure")
    parser.add_argument("model_dir", type=Path, help="directory holding the models")
    parser.add_argument("out_dir", type=Path, help="directory for the sheets (created)")
    parser.add_argument("--step", type=int, default=1,
                        help="measure every Nth frame (default 1, every frame)")
    parser.add_argument("--padding", type=float, default=DEFAULT_PADDING,
                        help=f"face padding of the shadow mode (default {DEFAULT_PADDING})")
    parser.add_argument("--tiles", type=int, default=60,
                        help="largest number of tiles per sheet (default 60)")
    args = parser.parse_args(argv)
    if args.step < 1:
        parser.error("--step must be at least 1")

    dirs = [args.model_dir]
    faces_det = FaceYuNetDetector.from_config(FaceYuNetConfig(), dirs)
    persons_det = PersonDnnDetector.from_config(PersonDnnConfig(), dirs)
    segmenter = PersonSegmenter.from_config(SegmentConfig(), dirs)

    cleared: list[Seen] = []
    orphans: list[Seen] = []
    bands: dict[str, list[float]] = {"under 150 px": [], "150 px or more": []}
    frames = 0
    with FileSource(args.clip) as source:
        h, w = source.info.height, source.info.width
        for frame in source:
            if frame.index % args.step:
                continue
            frames += 1
            persons = tuple(
                b for b in persons_det.detect(frame).persons or () if b.w > 0 and b.h > 0
            )
            if not persons:
                continue
            faces = faces_det.detect(frame).faces or ()
            mask = segmenter.detect(frame).mask
            taken: dict[int, Face] = {}
            for face in faces:
                if face.box.w <= 0 or face.box.h <= 0:
                    continue
                i = _clearing(persons, face, h, w)
                # The plugin lets several faces clear the same box; the first one
                # found is measured, the others add no new box.
                if i is not None and i not in taken:
                    taken[i] = face
            orphan_set = set(orphan_indices(persons, faces, None, "face", h, w))
            for i, box in enumerate(persons):
                face = taken.get(i)
                seen = Seen(
                    frame=frame.index,
                    box=box,
                    face_rect=pad_face(face.box, args.padding, h, w) if face else None,
                    face_h=face.box.h if face else 0,
                    top=top_half(box, h, w),
                    eligible=eligible(box),
                    edge=_touches_edge(box, h, w),
                )
                if i in orphan_set:
                    orphans.append(seen)
                elif face is not None and seen.face_rect is not None:
                    cleared.append(seen)
                if mask is not None and box.h * 5 >= box.w * 7:
                    bottom = _band_bottom(mask, box)
                    if bottom is not None:
                        key = "150 px or more" if box.h >= HEAD_MIN_HEIGHT_PX else "under 150 px"
                        bands[key].append(bottom)

        print(f"{args.clip}: {w}x{h}, {frames} frames measured (step {args.step}), "
              f"face padding {args.padding}")
        print()
        print("1. Person boxes cleared by a face: padded face rect as shares of the box")
        for name, get in (
            ("top", lambda s: (s.face_rect.y0 - s.box.y) / s.box.h),
            ("bottom", lambda s: (s.face_rect.y1 - s.box.y) / s.box.h),
            ("left", lambda s: (s.face_rect.x0 - s.box.x) / s.box.w),
            ("right", lambda s: (s.face_rect.x1 - s.box.x) / s.box.w),
            ("face h / box h", lambda s: s.face_h / s.box.h),
            ("box h / box w", lambda s: s.box.h / s.box.w),
        ):
            print(f"   {name:15} {_pct([get(s) for s in cleared])}")
        print(f"   boxes touching a frame edge: {sum(s.edge for s in cleared)} of {len(cleared)}")
        print()
        print(f"2. Segmenter head band ({BAND_SHARE:.0%} of the box height from the mask top; "
              f"mask share >= {MIN_MASK_SHARE:.0%}, h/w >= 1.4)")
        for key, values in bands.items():
            inside = sum(v <= 0.5 for v in values)
            print(f"   {key:15} band bottom {_pct(values)}")
            print(f"   {'':15} inside the top half: {_share(inside, len(values))}")
        print()
        print("3. Proposed rule (padded box width, top edge to y + ceil(h / 2); "
              f"h >= {HEAD_MIN_HEIGHT_PX} px and h/w >= 1.4)")
        accepted = [s for s in cleared if s.eligible]
        refused = [s for s in cleared if not s.eligible]
        held_in = [s for s in accepted if s.inside]
        print(f"   face-cleared, eligible: padded face rect inside the head rect "
              f"{_share(len(held_in), len(accepted))}")
        # A refused box keeps the whole-box fill; the line shows how many faces the
        # top half would have held there, had the rule applied.
        print(f"   face-cleared, not eligible (whole-box fill kept): top half would hold "
              f"{_share(sum(s.inside for s in refused), len(refused))}")
        violations = [s for s in accepted if not s.inside]
        for s in violations:
            r = s.face_rect
            assert r is not None
            print(f"     violation: frame {s.frame} box {s.box.x},{s.box.y} {s.box.w}x{s.box.h} "
                  f"face rect bottom {(r.y1 - s.box.y) / s.box.h:.3f} "
                  f"left {(r.x0 - s.box.x) / s.box.w:.3f} right {(r.x1 - s.box.x) / s.box.w:.3f}")
        heads = sum(s.head is not None for s in orphans)
        print(f"   orphan boxes: head rect {_share(heads, len(orphans))}, "
              f"whole-box fill {_share(len(orphans) - heads, len(orphans))}")
        short = sum(s.box.h < HEAD_MIN_HEIGHT_PX for s in orphans)
        wide = sum(s.box.h >= HEAD_MIN_HEIGHT_PX and not s.eligible for s in orphans)
        print(f"     fallback reasons: under {HEAD_MIN_HEIGHT_PX} px {short}, h/w under 1.4 {wide}")
        print()

        args.out_dir.mkdir(parents=True, exist_ok=True)
        stem = args.clip.stem
        _write_sheet(source, _spread(orphans, args.tiles), args.out_dir / f"{stem}_head_orphans.png")
        rest = [s for s in cleared if s not in violations]
        picks = violations[: args.tiles] + _spread(rest, max(0, args.tiles - len(violations)))
        _write_sheet(source, picks, args.out_dir / f"{stem}_head_cleared.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
