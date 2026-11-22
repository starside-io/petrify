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

"""Write contact sheets of padded face crops, to judge the face padding by eye.

Runs YuNet on sampled frames of a clip and, for each padding value, writes one
sheet ``<clip>_padding_<value>.png`` into the output directory. Each tile shows
the area around one face, scaled to a fixed size, with the face box in green
and the padded cover rect of :func:`petrify_shadow.regions.pad_face` in red.
The crop extends past the largest padded rect, so what the rect leaves out is
visible.

Example::

    PYTHONPATH=. python plugins/petrify-shadow/tools/padding_sheet.py \\
        data/camera.mp4 models /tmp/sheets --paddings 0.2,0.3,0.4

The sheets show frames of the clip: write them outside the repository.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from petrify_shadow.regions import pad_face  # noqa: E402

from petrify.core.config import FaceYuNetConfig  # noqa: E402
from petrify.core.detections import Face  # noqa: E402
from petrify.detectors.face_yunet import FaceYuNetDetector  # noqa: E402
from petrify.sources import FileSource  # noqa: E402

TILE = 192
COLUMNS = 8
BOX_COLOR = (0, 200, 0)
RECT_COLOR = (0, 0, 255)


def _paddings(text: str) -> list[float]:
    values = [float(part) for part in text.split(",") if part.strip()]
    if not values or any(not 0.0 <= value <= 1.0 for value in values):
        raise argparse.ArgumentTypeError("paddings must be numbers from 0 to 1")
    return values


def _view(h: int, w: int, face: Face, reach: float) -> tuple[int, int, int, int]:
    """Return a square crop around ``face`` that holds its rect padded by ``reach``."""
    box = face.box
    side = int(max(box.w, box.h) * (1 + 3 * reach)) + 8
    cx, cy = box.x + box.w // 2, box.y + box.h // 2 - int(0.4 * reach * box.h)
    x0, y0 = max(cx - side // 2, 0), max(cy - side // 2, 0)
    return x0, y0, min(x0 + side, w), min(y0 + side, h)


def _tile(image: np.ndarray, face: Face, padding: float, reach: float, label: str) -> np.ndarray:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = _view(h, w, face, reach)
    crop = image[y0:y1, x0:x1].copy()
    scale = TILE / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    def to_tile(px: float, py: float) -> tuple[int, int]:
        return int(round((px - x0) * scale)), int(round((py - y0) * scale))

    box = face.box
    cv2.rectangle(crop, to_tile(box.x, box.y), to_tile(box.x + box.w, box.y + box.h),
                  BOX_COLOR, 1)
    rect = pad_face(box, padding, h, w)
    if rect is not None:
        cv2.rectangle(crop, to_tile(rect.x0, rect.y0), to_tile(rect.x1, rect.y1), RECT_COLOR, 2)
    tile = np.zeros((TILE + 16, TILE, 3), np.uint8)
    tile[: crop.shape[0], : crop.shape[1]] = crop
    cv2.putText(tile, label, (2, TILE + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return tile


def _sheet(tiles: list[np.ndarray]) -> np.ndarray:
    rows = []
    for start in range(0, len(tiles), COLUMNS):
        row = tiles[start : start + COLUMNS]
        row += [np.zeros_like(tiles[0])] * (COLUMNS - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("clip", type=Path, help="video file to sample")
    parser.add_argument("model_dir", type=Path, help="directory holding the YuNet model")
    parser.add_argument("out_dir", type=Path, help="directory for the sheets (created)")
    parser.add_argument("--paddings", type=_paddings, default=_paddings("0.1,0.2,0.3,0.4,0.5"),
                        help="comma-separated padding values (default 0.1 to 0.5)")
    parser.add_argument("--faces", type=int, default=48,
                        help="largest number of faces per sheet (default 48)")
    args = parser.parse_args(argv)

    detector = FaceYuNetDetector.from_config(FaceYuNetConfig(), [args.model_dir])
    found: list[tuple[int, Face]] = []
    with FileSource(args.clip) as source:
        for frame in source:
            found.extend((frame.index, face) for face in detector.detect(frame).faces or ())
        if not found:
            print(f"no face found in {args.clip}", file=sys.stderr)
            return 1
        # Spread the sample over the whole clip, not only its first faces. Frames
        # are read again by index, so only the sampled ones are kept in memory.
        step = max(1, len(found) // args.faces)
        sample = []
        for index, face in found[::step][: args.faces]:
            image = source.read_frame(index)
            if image is not None:
                sample.append((index, image, face))
    reach = max(args.paddings)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for padding in args.paddings:
        tiles = [
            _tile(image, face, padding, reach, f"f{index} {face.box.w}x{face.box.h}")
            for index, image, face in sample
        ]
        path = args.out_dir / f"{args.clip.stem}_padding_{padding:g}.png"
        cv2.imwrite(str(path), _sheet(tiles))
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
