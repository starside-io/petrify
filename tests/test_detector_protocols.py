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

"""Protocol conformance and from-config construction for the detectors."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from petrify.core import (
    BackgroundDiffConfig,
    Box,
    DetectionKind,
    Detections,
    Detector,
    Frame,
    MotionConfig,
    PersonHogConfig,
    SourceInfo,
)
from petrify.detectors import BackgroundDiffer, MotionDetector, PersonDetector

DEMO_CLIP = Path(__file__).resolve().parent.parent / "data" / "camera.mp4"
FIELDS = ("motion", "persons", "background_diff", "faces", "mask")


def _random(w: int, h: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _flat(w: int, h: int, value: int = 100) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _frame(image: np.ndarray, index: int = 0) -> Frame:
    h, w = image.shape[:2]
    info = SourceInfo(fps=25.0, width=w, height=h, frame_count=None, seekable=False, live=False)
    return Frame(image=image, index=index, timestamp=index / 25.0, source=info)


def _set_fields(det: Detections) -> set[str]:
    return {name for name in FIELDS if getattr(det, name) is not None}


def _demo_frames(step: int = 97) -> list[np.ndarray]:
    if not DEMO_CLIP.exists():
        pytest.skip(f"demo clip not available: {DEMO_CLIP}")
    cap = cv2.VideoCapture(str(DEMO_CLIP))
    frames: list[np.ndarray] = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if ok:
            frames.append(img)
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Protocol members
# ---------------------------------------------------------------------------


def test_motion_detector_protocol() -> None:
    det = MotionDetector()
    assert isinstance(det, Detector)
    assert det.provides == frozenset({DetectionKind.MOTION})
    assert det.stateful is True
    out = det.detect(_frame(_random(320, 240)))
    assert _set_fields(out) == {"motion"}
    assert isinstance(out.motion, bool)


def test_motion_detect_matches_is_motion() -> None:
    frames = [_flat(320, 240)] * 3 + [_random(320, 240, seed=s) for s in range(3)]
    a, b = MotionDetector(), MotionDetector()
    for i, img in enumerate(frames):
        assert a.detect(_frame(img, i)).motion == b.is_motion(img)


def test_background_differ_protocol() -> None:
    det = BackgroundDiffer(reference=_flat(320, 240))
    assert isinstance(det, Detector)
    assert det.provides == frozenset({DetectionKind.BACKGROUND_DIFF})
    assert det.stateful is False
    det.reset()
    same = det.detect(_frame(_flat(320, 240)))
    assert _set_fields(same) == {"background_diff"}
    assert same.background_diff is False
    assert det.detect(_frame(_random(320, 240))).background_diff is True


def test_person_detector_protocol() -> None:
    det = PersonDetector()
    assert isinstance(det, Detector)
    assert det.provides == frozenset({DetectionKind.PERSONS})
    assert det.stateful is False
    det.reset()
    out = det.detect(_frame(_flat(200, 300)))
    assert _set_fields(out) == {"persons"}
    assert out.persons == ()


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------


def test_motion_from_config() -> None:
    cfg = MotionConfig(
        history=77, var_threshold=9.5, detect_shadows=True, threshold=321, downsample=0.25
    )
    det = MotionDetector.from_config(cfg)
    assert (det.history, det.var_threshold, det.detect_shadows, det.threshold, det.downsample) == (
        77,
        9.5,
        True,
        321,
        0.25,
    )


def test_person_from_config() -> None:
    cfg = PersonHogConfig(
        downsample=0.75, win_stride=(4, 4), scale=1.1, min_confidence=0.9, min_box_area=42
    )
    det = PersonDetector.from_config(cfg)
    assert det.downsample == 0.75
    assert det.win_stride == (4, 4)
    assert det.scale == 1.1
    assert det.min_confidence == 0.9
    assert det.min_box_area == 42
    assert det.padding == PersonDetector().padding


def test_background_differ_from_config() -> None:
    ref = _flat(320, 240)
    cfg = BackgroundDiffConfig(pixel_threshold=40, area_threshold=123, downsample=0.25)
    det = BackgroundDiffer.from_config(ref, cfg)
    assert (det.pixel_threshold, det.area_threshold, det.downsample) == (40, 123, 0.25)
    assert det._ref_gray.shape == (60, 80)


# ---------------------------------------------------------------------------
# PersonDetector: detect_boxes vs is_person
# ---------------------------------------------------------------------------

# All at or above the 64x128 (w x h) HOG window at full resolution.
SYNTHETIC = [
    _random(64, 128, seed=0),
    _random(100, 200, seed=1),
    _random(200, 300, seed=2),
    _random(320, 240, seed=3),
    _flat(64, 128),
    _flat(320, 240, value=30),
]
DETECTORS = [
    PersonDetector(),
    PersonDetector(downsample=1.0),
    PersonDetector(min_confidence=0.0, min_box_area=0),
    PersonDetector(downsample=1.0, min_confidence=-10.0, min_box_area=0),
]


@pytest.mark.parametrize("det", DETECTORS)
def test_is_person_equals_nonempty_boxes_synthetic(det: PersonDetector) -> None:
    for img in SYNTHETIC:
        assert det.is_person(img) == bool(det.detect_boxes(img))


@pytest.mark.parametrize("det", DETECTORS)
def test_is_person_equals_nonempty_boxes_demo(det: PersonDetector) -> None:
    frames = _demo_frames()
    seen_boxes = False
    for img in frames:
        boxes = det.detect_boxes(img)
        seen_boxes = seen_boxes or bool(boxes)
        assert det.is_person(img) == bool(boxes)
    # The demo clip has people in it, so the non-empty branch is exercised.
    assert seen_boxes


def test_boxes_full_resolution_and_within_bounds() -> None:
    det = PersonDetector(min_confidence=0.0, min_box_area=0)
    frames = _demo_frames()
    count = 0
    for img in frames:
        fh, fw = img.shape[:2]
        for box in det.detect_boxes(img):
            count += 1
            assert isinstance(box, Box)
            assert all(isinstance(v, int) for v in (box.x, box.y, box.w, box.h))
            assert isinstance(box.score, float)
            assert box.score >= det.min_confidence
            # HOG padding could in principle push a box a few pixels past the
            # edge; allow that overhang (scaled to full resolution).
            pad_x = det.padding[0] * 4
            pad_y = det.padding[1] * 4
            assert -pad_x <= box.x and box.x + box.w <= fw + pad_x
            assert -pad_y <= box.y and box.y + box.h <= fh + pad_y
            assert box.w > 0 and box.h > 0
    assert count > 0


def test_boxes_scaled_from_downsampled_space() -> None:
    frames = _demo_frames()
    half = PersonDetector(min_confidence=0.0, min_box_area=0)
    for img in frames:
        small = half._maybe_downsample(img)
        sx = img.shape[1] / small.shape[1]
        sy = img.shape[0] / small.shape[0]
        raw, weights = half._hog.detectMultiScale(
            small, winStride=half.win_stride, padding=half.padding, scale=half.scale
        )
        expected = [
            (round(x * sx), round(y * sy), round(w * sx), round(h * sy))
            for i, (x, y, w, h) in enumerate(raw)
            if float(weights[i]) >= 0.0
        ]
        got = [(b.x, b.y, b.w, b.h) for b in half.detect_boxes(img)]
        # OpenCV may return the same boxes in a different order between calls.
        assert sorted(got) == sorted(expected)


def test_min_box_area_filter_applies() -> None:
    frames = _demo_frames()
    det = PersonDetector(min_confidence=0.0, min_box_area=10**9)
    for img in frames:
        assert det.detect_boxes(img) == ()
        assert det.is_person(img) is False


@pytest.mark.parametrize(("w", "h"), [(63, 128), (64, 127), (50, 400)])
def test_too_small_frame_raises_everywhere(w: int, h: int) -> None:
    det = PersonDetector(downsample=0.5)
    img = _random(w, h)
    with pytest.raises(ValueError, match="minimum is 64x128"):
        det.is_person(img)
    with pytest.raises(ValueError, match="minimum is 64x128"):
        det.detect_boxes(img)
    with pytest.raises(ValueError, match="minimum is 64x128"):
        det.detect(_frame(img))
