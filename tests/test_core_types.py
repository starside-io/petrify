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

"""Tests for the core engine data types, configs, and protocols."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import numpy as np
import pytest

from petrify.cli import _build_parser
from petrify.core import (
    FIELD_OF,
    BackgroundConfig,
    BackgroundDiffConfig,
    Box,
    DetectionKind,
    Detections,
    Detector,
    EngineConfig,
    Face,
    Frame,
    FrameMeta,
    MalformedDetections,
    Mode,
    PersonHogConfig,
    PetrifyConfig,
    RoiConfig,
    Sink,
    Source,
    SourceInfo,
    Stage,
    check_detections,
)
from petrify.detectors.bgdiff import BackgroundDiffer
from petrify.detectors.motion import MotionDetector
from petrify.detectors.person_hog import PersonDetector


def _config(*extra: str) -> PetrifyConfig:
    return PetrifyConfig.from_namespace(_build_parser().parse_args(["in.mp4", "out.mp4", *extra]))


def _info() -> SourceInfo:
    return SourceInfo(fps=25.0, width=4, height=2, frame_count=10, seekable=True, live=False)


def _frame() -> Frame:
    return Frame(image=np.zeros((2, 4, 3), dtype=np.uint8), index=0, timestamp=0.0, source=_info())


# ---------------------------------------------------------------------------
# Frame / SourceInfo
# ---------------------------------------------------------------------------


def test_source_info_constructs_and_is_frozen() -> None:
    info = _info()
    assert info.uri is None
    assert info.frame_count == 10
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.fps = 30.0  # type: ignore[misc]


def test_source_info_unknown_frame_count() -> None:
    info = SourceInfo(fps=30.0, width=640, height=480, frame_count=None, seekable=False,
                      live=True, uri="rtsp://cam")
    assert info.frame_count is None
    assert info.live


def test_frame_constructs_and_is_frozen() -> None:
    frame = _frame()
    assert frame.image.shape == (2, 4, 3)
    assert frame.source.fps == 25.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        frame.index = 1  # type: ignore[misc]


def test_frame_meta_has_no_pixels_and_is_frozen() -> None:
    frame = _frame()
    meta = frame.meta
    assert meta == FrameMeta(index=0, timestamp=0.0, source=_info())
    assert not hasattr(meta, "image")
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.index = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------


def test_detection_kind_values() -> None:
    assert {k.value for k in DetectionKind} == {
        "motion", "persons", "background_diff", "faces", "landmarks", "mask",
    }
    assert DetectionKind.MOTION == "motion"


def test_detections_default_all_none() -> None:
    d = Detections()
    for f in dataclasses.fields(d):
        assert getattr(d, f.name) is None


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, False),
        ({"motion": True}, True),
        ({"motion": False}, False),
        ({"persons": ()}, False),
        ({"persons": (Box(0, 0, 1, 1),)}, True),
        ({"background_diff": True}, True),
        ({"background_diff": False}, False),
        ({"motion": False, "persons": (), "background_diff": False}, False),
        ({"motion": None, "persons": None, "background_diff": True}, True),
        ({"faces": (Face(Box(0, 0, 1, 1)),)}, True),
        ({"mask": np.ones((2, 2), dtype=np.uint8)}, True),
        ({"faces": ()}, False),
        ({"mask": np.zeros((2, 2), dtype=bool)}, False),
        ({"mask": np.array([[False, True], [False, False]])}, True),
    ],
)
def test_subject_present(kwargs: dict, expected: bool) -> None:
    assert Detections(**kwargs).subject_present is expected


def test_merged_keeps_and_overrides() -> None:
    boxes = (Box(1, 2, 3, 4, score=0.9),)
    base = Detections(motion=False, persons=boxes)
    other = Detections(motion=True, background_diff=False)
    out = base.merged(other)
    assert out.motion is True
    assert out.persons == boxes
    assert out.background_diff is False
    assert out.faces is None
    assert out.mask is None
    # Originals are untouched.
    assert base.motion is False
    assert base.background_diff is None


def test_merged_with_empty_other_keeps_everything() -> None:
    mask = np.ones((2, 2), dtype=np.uint8)
    base = Detections(motion=True, mask=mask)
    out = base.merged(Detections())
    assert out.motion is True
    assert out.mask is mask


def test_merged_ors_flags() -> None:
    for name in ("motion", "background_diff"):
        for a, b in ((False, False), (False, True), (True, False), (True, True)):
            out = Detections(**{name: a}).merged(Detections(**{name: b}))
            assert getattr(out, name) is (a or b)


def test_merged_ors_numpy_flags() -> None:
    out = Detections(motion=np.bool_(False)).merged(Detections(motion=np.bool_(True)))
    assert out.motion is True


@pytest.mark.parametrize("name", ["persons", "faces"])
def test_merged_concatenates_tuples_self_first(name: str) -> None:
    a: tuple[object, ...]
    b: tuple[object, ...]
    if name == "persons":
        a = (Box(0, 0, 1, 1), Box(1, 1, 1, 1))
        b = (Box(2, 2, 2, 2),)
    else:
        a = (Face(Box(0, 0, 1, 1)),)
        b = (Face(Box(2, 2, 2, 2), landmarks=((1.0, 2.0),)), Face(Box(3, 3, 1, 1)))
    out = Detections(**{name: a}).merged(Detections(**{name: b}))
    result = getattr(out, name)
    assert isinstance(result, tuple)
    assert result == (*a, *b)
    assert all(x is y for x, y in zip(result, (*a, *b), strict=True))


def test_merged_concatenates_empty_tuples() -> None:
    boxes = (Box(0, 0, 1, 1),)
    assert Detections(persons=()).merged(Detections(persons=boxes)).persons == boxes
    assert Detections(persons=boxes).merged(Detections(persons=())).persons == boxes


def test_merged_ors_masks() -> None:
    a = np.array([[True, False], [False, False]])
    b = np.array([[False, False], [False, True]])
    out = Detections(mask=a).merged(Detections(mask=b))
    assert out.mask is not None
    assert out.mask.dtype == np.bool_
    np.testing.assert_array_equal(out.mask, np.array([[True, False], [False, True]]))


def test_merged_mask_shape_mismatch_raises() -> None:
    a = Detections(mask=np.zeros((2, 2), dtype=bool))
    b = Detections(mask=np.zeros((2, 3), dtype=bool))
    with pytest.raises(MalformedDetections, match="shape"):
        a.merged(b)
    assert issubclass(MalformedDetections, ValueError)


def _full() -> Detections:
    return Detections(
        motion=True,
        persons=(Box(0, 0, 1, 1),),
        background_diff=False,
        faces=(Face(Box(1, 1, 1, 1)),),
        mask=np.zeros((2, 4), dtype=bool),
    )


@pytest.mark.parametrize("empty_first", [True, False])
def test_merged_keeps_identity_when_one_side_is_none(empty_first: bool) -> None:
    full = _full()
    out = Detections().merged(full) if empty_first else full.merged(Detections())
    for f in dataclasses.fields(full):
        assert getattr(out, f.name) is getattr(full, f.name)


def test_merged_does_not_mutate_inputs() -> None:
    a_mask = np.array([[True, False], [False, False]])
    b_mask = np.array([[False, False], [False, True]])
    a_boxes = (Box(0, 0, 1, 1),)
    b_boxes = (Box(2, 2, 1, 1),)
    a_faces = (Face(Box(0, 0, 1, 1)),)
    b_faces = (Face(Box(1, 1, 1, 1)),)
    a = Detections(motion=False, persons=a_boxes, background_diff=True, faces=a_faces,
                   mask=a_mask)
    b = Detections(motion=True, persons=b_boxes, background_diff=False, faces=b_faces,
                   mask=b_mask)
    a_mask_copy, b_mask_copy = a_mask.copy(), b_mask.copy()
    out = a.merged(b)
    assert (a.motion, a.persons, a.background_diff, a.faces) == (False, a_boxes, True, a_faces)
    assert (b.motion, b.persons, b.background_diff, b.faces) == (True, b_boxes, False, b_faces)
    assert a.mask is a_mask
    assert b.mask is b_mask
    np.testing.assert_array_equal(a_mask, a_mask_copy)
    np.testing.assert_array_equal(b_mask, b_mask_copy)
    assert out.mask is not a_mask
    assert out.mask is not b_mask


def test_field_of_covers_every_kind() -> None:
    field_names = {f.name for f in dataclasses.fields(Detections)}
    assert set(FIELD_OF) == set(DetectionKind)
    assert set(FIELD_OF.values()) == field_names
    assert FIELD_OF[DetectionKind.LANDMARKS] == "faces"
    for kind in DetectionKind:
        if kind is not DetectionKind.LANDMARKS:
            assert FIELD_OF[kind] == kind.value


_ALL = frozenset(DetectionKind)
_SHAPE = (2, 4, 3)


def test_check_detections_accepts_well_formed_and_returns_same_object() -> None:
    d = Detections(
        motion=np.bool_(True),
        persons=(Box(np.int32(0), np.int64(1), 2, 0, score=0.5), Box(0, 0, 1, 1)),  # type: ignore[arg-type]
        background_diff=False,
        faces=(Face(Box(0, 0, 1, 1), landmarks=((0.0, 1.0),)),),
        mask=np.zeros((2, 4), dtype=bool),
    )
    assert check_detections(d, _ALL, _SHAPE) is d
    empty = Detections()
    assert check_detections(empty, frozenset(), _SHAPE) is empty


def test_check_detections_faces_without_landmarks_ok_when_landmarks_not_provided() -> None:
    d = Detections(faces=(Face(Box(0, 0, 1, 1)),))
    assert check_detections(d, frozenset({DetectionKind.FACES}), _SHAPE) is d


def test_check_detections_landmarks_kind_permits_faces_field() -> None:
    d = Detections(faces=(Face(Box(0, 0, 1, 1), landmarks=((0.0, 0.0),)),))
    assert check_detections(d, frozenset({DetectionKind.LANDMARKS}), _SHAPE) is d


def test_check_detections_rejects_non_detections() -> None:
    with pytest.raises(MalformedDetections, match="expected Detections"):
        check_detections({"motion": True}, _ALL, _SHAPE)
    with pytest.raises(MalformedDetections, match="expected Detections"):
        check_detections(None, _ALL, _SHAPE)


@pytest.mark.parametrize(
    ("kwargs", "provides"),
    [
        ({"motion": True}, {DetectionKind.PERSONS}),
        ({"persons": ()}, {DetectionKind.MOTION}),
        ({"background_diff": False}, {DetectionKind.MOTION}),
        ({"faces": ()}, {DetectionKind.MASK}),
        ({"mask": np.zeros((2, 4), dtype=bool)}, {DetectionKind.FACES}),
        ({"motion": False}, set()),
    ],
)
def test_check_detections_rejects_undeclared_field(
    kwargs: dict[str, object], provides: set[DetectionKind]
) -> None:
    with pytest.raises(MalformedDetections, match="no provided kind"):
        check_detections(Detections(**kwargs), frozenset(provides), _SHAPE)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["motion", "background_diff"])
@pytest.mark.parametrize("value", [1, 0, 1.0, "yes", np.array([True]), np.int64(1)])
def test_check_detections_rejects_non_bool_flag(name: str, value: object) -> None:
    with pytest.raises(MalformedDetections, match=f"{name} must be a bool"):
        check_detections(Detections(**{name: value}), _ALL, _SHAPE)  # type: ignore[arg-type]


_BAD_BOXES = [
    pytest.param((0, 0, 1, 1), "expected a Box", id="plain-tuple"),
    pytest.param(Box(0.0, 0, 1, 1), "Box.x must be an integer", id="float-x"),  # type: ignore[arg-type]
    pytest.param(Box(0, np.float32(1), 1, 1), "Box.y must be an integer", id="numpy-float-y"),  # type: ignore[arg-type]
    pytest.param(Box(0, 0, True, 1), "Box.w must be an integer", id="bool-w"),
    pytest.param(Box(0, 0, 1, np.bool_(True)), "Box.h must be an integer", id="numpy-bool-h"),  # type: ignore[arg-type]
    pytest.param(Box(0, 0, "1", 1), "Box.w must be an integer", id="str-w"),  # type: ignore[arg-type]
    pytest.param(Box(0, 0, -1, 1), "non-negative", id="negative-w"),
    pytest.param(Box(0, 0, 1, np.int32(-2)), "non-negative", id="negative-h"),  # type: ignore[arg-type]
]


def test_check_detections_rejects_persons_not_tuple() -> None:
    d = Detections(persons=[Box(0, 0, 1, 1)])  # type: ignore[arg-type]
    with pytest.raises(MalformedDetections, match="persons must be a tuple"):
        check_detections(d, _ALL, _SHAPE)


@pytest.mark.parametrize(("box", "match"), _BAD_BOXES)
def test_check_detections_rejects_bad_person_box(box: object, match: str) -> None:
    d = Detections(persons=(Box(0, 0, 1, 1), box))  # type: ignore[arg-type]
    with pytest.raises(MalformedDetections, match=r"persons\[1\]") as info:
        check_detections(d, _ALL, _SHAPE)
    assert match in str(info.value)


def test_check_detections_rejects_faces_not_tuple() -> None:
    d = Detections(faces=[Face(Box(0, 0, 1, 1))])  # type: ignore[arg-type]
    with pytest.raises(MalformedDetections, match="faces must be a tuple"):
        check_detections(d, _ALL, _SHAPE)


def test_check_detections_rejects_non_face() -> None:
    d = Detections(faces=(Box(0, 0, 1, 1),))  # type: ignore[arg-type]
    with pytest.raises(MalformedDetections, match=r"faces\[0\]: expected a Face"):
        check_detections(d, _ALL, _SHAPE)


@pytest.mark.parametrize(("box", "match"), _BAD_BOXES)
def test_check_detections_rejects_bad_face_box(box: object, match: str) -> None:
    d = Detections(faces=(Face(box, landmarks=()),))  # type: ignore[arg-type]
    with pytest.raises(MalformedDetections, match=r"faces\[0\]\.box") as info:
        check_detections(d, _ALL, _SHAPE)
    assert match in str(info.value)


def test_check_detections_rejects_face_without_landmarks_when_provided() -> None:
    d = Detections(faces=(Face(Box(0, 0, 1, 1), landmarks=((0.0, 0.0),)), Face(Box(0, 0, 1, 1))))
    with pytest.raises(MalformedDetections, match=r"faces\[1\] has no landmarks"):
        check_detections(d, frozenset({DetectionKind.FACES, DetectionKind.LANDMARKS}), _SHAPE)


@pytest.mark.parametrize(
    ("mask", "match"),
    [
        pytest.param([[True] * 4] * 2, "numpy array", id="list"),
        pytest.param(np.zeros((2, 4), dtype=np.uint8), "dtype must be bool", id="uint8"),
        pytest.param(np.zeros((2, 4), dtype=np.float32), "dtype must be bool", id="float"),
        pytest.param(np.zeros((1, 2), dtype=bool), "shape must be", id="downscaled"),
        pytest.param(np.zeros((4, 2), dtype=bool), "shape must be", id="transposed"),
        pytest.param(np.zeros((2, 4, 1), dtype=bool), "shape must be", id="extra-channel"),
        pytest.param(np.zeros((2, 4, 3), dtype=bool), "shape must be", id="frame-shape"),
        pytest.param(np.zeros(8, dtype=bool), "shape must be", id="flat"),
    ],
)
def test_check_detections_rejects_bad_mask(mask: object, match: str) -> None:
    with pytest.raises(MalformedDetections, match=match):
        check_detections(Detections(mask=mask), _ALL, _SHAPE)  # type: ignore[arg-type]


def test_check_detections_mask_accepts_grayscale_frame_shape() -> None:
    d = Detections(mask=np.ones((2, 4), dtype=bool))
    assert check_detections(d, frozenset({DetectionKind.MASK}), (2, 4)) is d


# The built-in detectors, run for real, must produce results that pass the check.


def _image_frame(image: np.ndarray, index: int = 0) -> Frame:
    h, w = image.shape[:2]
    info = SourceInfo(fps=25.0, width=w, height=h, frame_count=None, seekable=False, live=False)
    return Frame(image=image, index=index, timestamp=index / 25.0, source=info)


def _scene(lit: bool) -> np.ndarray:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    if lit:
        image[32:96, 32:96] = 255
    return image


def test_motion_detector_output_passes_check() -> None:
    detector = MotionDetector(threshold=10)
    results = [detector.detect(_image_frame(_scene(lit=i == 29), i)) for i in range(30)]
    assert {r.motion for r in results} == {True, False}
    for r in results:
        assert check_detections(r, detector.provides, (128, 128, 3)) is r


def test_bgdiff_output_passes_check() -> None:
    differ = BackgroundDiffer(_scene(lit=False), area_threshold=10)
    for lit, expected in ((False, False), (True, True)):
        r = differ.detect(_image_frame(_scene(lit)))
        assert r.background_diff is expected
        assert check_detections(r, differ.provides, (128, 128, 3)) is r


def test_bgdiff_not_ready_output_passes_check() -> None:
    class _NotReady:
        version = 0

        def current(self) -> None:
            return None

    differ = BackgroundDiffer.from_model(_NotReady(), BackgroundDiffConfig())  # type: ignore[arg-type]
    r = differ.detect(_image_frame(_scene(lit=False)))
    assert r.background_diff is None
    assert check_detections(r, differ.provides, (128, 128, 3)) is r


def test_person_detector_output_passes_check(monkeypatch: pytest.MonkeyPatch) -> None:
    # Feed detect() the array types cv2.HOGDescriptor.detectMultiScale returns
    # (int32 boxes of shape (N, 4), float64 weights of shape (N,)) so the real
    # conversion to Box runs on them.
    detector = PersonDetector(min_box_area=0, min_confidence=0.0)
    small = np.zeros((64, 64, 3), dtype=np.uint8)
    boxes = np.array([[3, 5, 64, 128], [0, 0, 7, 9]], dtype=np.int32)
    weights = np.array([0.8, 1.25], dtype=np.float64)
    monkeypatch.setattr(detector, "_raw_detections", lambda frame: (small, boxes, weights))
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    r = detector.detect(_image_frame(image))
    assert r.persons is not None
    assert len(r.persons) == 2
    assert all(type(b.score) is float for b in r.persons)
    assert check_detections(r, detector.provides, image.shape) is r


def test_person_detector_empty_output_passes_check(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = PersonDetector()
    small = np.zeros((64, 64, 3), dtype=np.uint8)
    monkeypatch.setattr(detector, "_raw_detections", lambda frame: (small, (), ()))
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    r = detector.detect(_image_frame(image))
    assert r.persons == ()
    assert check_detections(r, detector.provides, image.shape) is r


def test_detections_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        Detections().motion = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_cli_defaults_match_config_defaults() -> None:
    from_cli = _config()
    default = PetrifyConfig()
    assert from_cli.detector == default.detector == "motion"
    assert from_cli.motion == default.motion
    assert from_cli.person == default.person
    assert from_cli.background_diff == default.background_diff
    assert from_cli.background_diff is not None
    assert from_cli.roi == default.roi
    assert from_cli.roi.mode == "auto"
    assert from_cli.roi.box is None
    assert from_cli.background == default.background
    assert from_cli.background.frame_index is None
    assert from_cli.background.scan_step == 5
    assert from_cli.encode == default.encode
    assert from_cli.engine == default.engine
    assert from_cli.engine.lookahead == 10
    assert from_cli.engine.cold_start_frames == 50
    assert from_cli == default


def test_detector_flag() -> None:
    assert _config("--detector", "both").detector == "both"
    assert _config("--detector", "person").detector == "person"


def test_no_bg_diff() -> None:
    assert _config("--no-bg-diff").background_diff is None


def test_bg_diff_flags() -> None:
    cfg = _config("--bg-diff-pixel-threshold", "40", "--bg-diff-area-threshold", "300")
    assert cfg.background_diff == BackgroundDiffConfig(pixel_threshold=40, area_threshold=300)


def test_gap_fill_maps_to_lookahead() -> None:
    assert _config("--gap-fill", "3").engine.lookahead == 3


def test_manual_roi() -> None:
    roi = _config("--roi", "1", "2", "3", "4").roi
    assert roi.mode == "manual"
    assert roi.box == (1, 2, 3, 4)


def test_no_roi() -> None:
    roi = _config("--no-roi").roi
    assert roi.mode == "disabled"
    assert roi.box is None


def test_manual_roi_wins_over_no_roi() -> None:
    assert _config("--roi", "1", "2", "3", "4", "--no-roi").roi.mode == "manual"


def test_roi_scan_flags() -> None:
    roi = _config(
        "--roi-scan-samples", "30",
        "--roi-scan-min-freq", "0.8",
        "--roi-scan-margin", "4",
        "--roi-scan-max-area-frac", "0.2",
        "--roi-scan-edge-threshold", "90",
        "--roi-scan-edge-gap-tolerance", "20",
        "--roi-scan-edge-samples", "5",
    ).roi
    assert roi.mode == "auto"
    assert roi.scan_samples == 30
    assert roi.scan_min_freq == 0.8
    assert roi.scan_margin == 4
    assert roi.scan_max_area_frac == 0.2
    assert roi.scan_edge_threshold == 90.0
    assert roi.scan_edge_gap_tolerance == 20
    assert roi.scan_edge_samples == 5


def test_background_frame() -> None:
    assert _config("--background-frame", "7").background.frame_index == 7


def test_downsample_feeds_motion_and_bg_diff_only() -> None:
    cfg = _config("--downsample", "0.25")
    assert cfg.motion.downsample == 0.25
    assert cfg.background_diff is not None
    assert cfg.background_diff.downsample == 0.25
    assert cfg.person.downsample == 0.5


def test_hog_downsample_feeds_person_only() -> None:
    cfg = _config("--hog-downsample", "0.75")
    assert cfg.person.downsample == 0.75
    assert cfg.motion.downsample == 0.5
    assert cfg.background_diff is not None
    assert cfg.background_diff.downsample == 0.5


def test_motion_flags() -> None:
    cfg = _config("--threshold", "200", "--history", "100", "--var-threshold", "20",
                  "--detect-shadows")
    assert cfg.motion.threshold == 200
    assert cfg.motion.history == 100
    assert cfg.motion.var_threshold == 20.0
    assert cfg.motion.detect_shadows is True


def test_hog_flags() -> None:
    cfg = _config("--hog-scale", "1.2", "--hog-min-confidence", "0.5",
                  "--hog-min-box-area", "900")
    assert cfg.person == PersonHogConfig(scale=1.2, min_confidence=0.5, min_box_area=900)


def test_encode_flags() -> None:
    enc = _config("--crf", "30", "--codec", "libx265", "--preset", "fast",
                  "--hwaccel", "cuda").encode
    assert enc.crf == 30
    assert enc.codec == "libx265"
    assert enc.preset == "fast"
    assert enc.hwaccel == "cuda"


def test_background_scan_person_config_motion_uses_class_defaults() -> None:
    cfg = _config("--hog-downsample", "0.75", "--hog-scale", "1.2",
                  "--hog-min-confidence", "0.5", "--hog-min-box-area", "900")
    scan = cfg.background_scan_person_config()
    assert scan.downsample == 0.75
    assert scan.scale == 1.2
    assert scan.min_confidence == 0.5
    assert scan.win_stride == PersonHogConfig().win_stride
    assert scan.min_box_area == PersonHogConfig().min_box_area == 1500


@pytest.mark.parametrize("mode", ["both", "person"])
def test_background_scan_person_config_reuses_person(mode: str) -> None:
    cfg = _config("--detector", mode, "--hog-min-box-area", "900")
    assert cfg.background_scan_person_config() is cfg.person
    assert cfg.background_scan_person_config().min_box_area == 900


def test_roi_manual_without_box_rejected() -> None:
    with pytest.raises(ValueError):
        RoiConfig(mode="manual")


@pytest.mark.parametrize("mode", ["auto", "disabled"])
def test_roi_box_rejected_outside_manual(mode: str) -> None:
    with pytest.raises(ValueError):
        RoiConfig(mode=mode, box=(1, 2, 3, 4))  # type: ignore[arg-type]


def test_roi_warmup_seconds_default_and_validation() -> None:
    assert RoiConfig().warmup_seconds == 10.0
    assert RoiConfig(warmup_seconds=2.5).warmup_seconds == 2.5
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="warmup_seconds"):
            RoiConfig(warmup_seconds=bad)


def test_l2_roi_max_area_cap_default_and_validation() -> None:
    assert RoiConfig().max_area_cap == 0.15
    assert RoiConfig(max_area_cap=1.0).max_area_cap == 1.0
    for bad in (0.0, -0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="max_area_cap"):
            RoiConfig(max_area_cap=bad)


def test_l1_background_sample_check_defaults_on() -> None:
    assert BackgroundConfig().sample_check is True
    assert BackgroundConfig(sample_check=False).sample_check is False
    assert _config().background.sample_check is True


def test_negative_lookahead_rejected() -> None:
    with pytest.raises(ValueError):
        EngineConfig(lookahead=-1)


def test_negative_cold_start_rejected() -> None:
    with pytest.raises(ValueError):
        EngineConfig(cold_start_frames=-1)


def test_frame_budget_defaults_to_disabled() -> None:
    assert EngineConfig().frame_budget_ms is None
    assert _config().engine.frame_budget_ms is None
    assert EngineConfig(frame_budget_ms=33.3).frame_budget_ms == 33.3


@pytest.mark.parametrize("budget", [0.0, -5.0, float("nan")])
def test_non_positive_frame_budget_rejected(budget: float) -> None:
    with pytest.raises(ValueError, match="frame_budget_ms"):
        EngineConfig(frame_budget_ms=budget)


def test_background_config_strategy_defaults() -> None:
    cfg = BackgroundConfig()
    assert cfg.strategy == "locked"
    assert (cfg.running_window, cfg.running_interval, cfg.calibrate_seconds) == (30, 5, 3.0)
    assert cfg.running_min_samples == 5
    small = BackgroundConfig(running_window=1, running_min_samples=1)
    assert (small.running_window, small.running_min_samples) == (1, 1)
    assert _config().background.strategy == "locked"
    for strategy in ("locked", "running", "calibrate"):
        assert BackgroundConfig(strategy=strategy).strategy == strategy  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strategy": "rolling"},
        {"running_window": 0},
        {"running_interval": 0},
        {"running_min_samples": 0},
        {"running_window": 4},
        {"running_window": 10, "running_min_samples": 11},
        {"calibrate_seconds": 0.0},
        {"calibrate_seconds": -1.0},
        {"calibrate_seconds": float("nan")},
    ],
)
def test_background_config_rejects_bad_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="BackgroundConfig"):
        BackgroundConfig(**kwargs)  # type: ignore[arg-type]


def test_config_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        PetrifyConfig().detector = "both"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class _DummyDetector:
    provides = frozenset({DetectionKind.MOTION})
    stateful = True

    def detect(self, frame: Frame) -> Detections:
        return Detections(motion=False)

    def reset(self) -> None:
        pass


class _DummyMode:
    name = "dummy"
    requires = frozenset({DetectionKind.MOTION})

    def render(self, frame: Frame, detections: Detections, subject_present: bool) -> np.ndarray:
        return frame.image


class _DummyStage:
    def apply(self, frame: Frame, image: np.ndarray) -> np.ndarray:
        return image


class _DummySink:
    critical = True

    def open(self, info: SourceInfo) -> None:
        pass

    def write(self, image: np.ndarray, meta: FrameMeta) -> None:
        pass

    def close(self) -> None:
        pass

    def abort(self) -> None:
        pass


class _DummySource:
    def __init__(self) -> None:
        self.info = _info()

    def __iter__(self) -> Iterator[Frame]:
        yield _frame()

    def close(self) -> None:
        pass


@pytest.mark.parametrize(
    ("obj", "proto"),
    [
        (_DummyDetector(), Detector),
        (_DummyMode(), Mode),
        (_DummyStage(), Stage),
        (_DummySink(), Sink),
        (_DummySource(), Source),
    ],
)
def test_dummy_satisfies_protocol(obj: object, proto: type) -> None:
    assert isinstance(obj, proto)


def test_object_missing_members_fails_protocol() -> None:
    assert not isinstance(_DummyStage(), Sink)
    assert not isinstance(_DummySink(), Detector)
