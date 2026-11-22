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

"""Render time of the shadow mode, informational.

The table is printed (run with ``-s`` to see it). The only assertion is a
generous upper bound, far above the measured times, so a busy machine cannot
make it fail: it catches an accidental per-pixel Python loop or full-frame
work, not a few milliseconds of drift. Detectors are not timed here.
"""

from __future__ import annotations

import statistics
import time

import cv2
import numpy as np
import pytest
from petrify_shadow.mode import ShadowMode
from petrify_shadow.options import STYLE_NAMES, parse_params

from petrify.core.detections import Box, Detections, Face
from petrify.core.frame import Frame, SourceInfo

SIZES = ((480, 640), (1080, 1920))
CASES: tuple[tuple[str, tuple[str, ...], int], ...] = (
    ("facebox, 1 face", ("facebox",), 1),
    ("facebox, 10 faces", ("facebox",), 10),
    ("noise", ("noise",), 1),
    ("gaussian", ("gaussian",), 1),
    ("silhouette", ("silhouette",), 1),
    ("bodybox", ("bodybox",), 1),
    ("silhouette + facebox", ("silhouette", "facebox"), 1),
    ("all nine", STYLE_NAMES, 1),
)
RUNS = 20
BOUND_MS = 50.0
"""Upper bound on the median render time of every case, at every size."""


def _frame_and_detections(h: int, w: int, n_faces: int) -> tuple[Frame, Detections]:
    """Return a random frame with ``n_faces`` faces, one person box and its mask.

    The person stands in the middle, a third of the frame wide and most of its
    height tall, a size a close camera gives. Faces are a sixteenth of the frame
    width, about 120 px at 1080p.
    """
    info = SourceInfo(fps=25.0, width=w, height=h, frame_count=1, seekable=False,
                      live=False)
    image = np.random.default_rng(0).integers(1, 256, (h, w, 3), dtype=np.uint8)
    image.setflags(write=False)
    fw = w // 16
    faces = []
    for k in range(n_faces):
        x = (k % 5) * (w // 5) + w // 20
        y = (k // 5) * (h // 3) + h // 10
        eyes = ((x + fw * 0.3, y + fw * 0.4), (x + fw * 0.7, y + fw * 0.4))
        faces.append(Face(Box(x, y, fw, fw * 13 // 10), landmarks=eyes))
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (w // 8, h * 2 // 5), 0, 0, 360, 1, -1)
    persons = (Box(w // 3, h // 12, w // 3, h * 5 // 6),)
    detections = Detections(faces=tuple(faces), persons=persons, mask=mask.astype(bool))
    return Frame(image=image, index=0, timestamp=0.0, source=info), detections


def _median_ms(mode: ShadowMode, frame: Frame, detections: Detections) -> float:
    mode.render(frame, detections, True)  # Warm up caches.
    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        mode.render(frame, detections, True)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


@pytest.mark.parametrize("size", SIZES, ids=["640x480", "1920x1080"])
def test_render_time_is_printed_and_far_below_the_bound(size: tuple[int, int]) -> None:
    h, w = size
    rows = []
    for name, styles, n_faces in CASES:
        frame, detections = _frame_and_detections(h, w, n_faces)
        mode = ShadowMode(parse_params({"styles": styles}))
        rows.append((name, _median_ms(mode, frame, detections)))
    width = max(len(name) for name, _ in rows)
    print(f"\nshadow render, {w}x{h}, median of {RUNS} runs:")
    for name, ms in rows:
        print(f"  {name:<{width}}  {ms:7.2f} ms")
    slow = [(name, round(ms, 1)) for name, ms in rows if ms >= BOUND_MS]
    assert not slow, f"render slower than {BOUND_MS} ms: {slow}"
