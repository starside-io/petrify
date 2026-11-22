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

"""The YuNet face detector: row conversion, input checks, model loading, detection."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from petrify.core import DetectionKind, Detections, Detector, FaceYuNetConfig, Frame, SourceInfo
from petrify.core.detections import Box, Face, check_detections
from petrify.detectors import face_yunet
from petrify.detectors.face_yunet import (
    MIN_FRAME_SIDE,
    MODEL,
    FaceYuNetDetector,
    face_from_row,
    faces_from_output,
)
from petrify.models import (
    MODEL_DIR_ENV,
    LoadedModel,
    ModelIntegrityError,
    ModelNotFoundError,
    load_model_bytes,
)


def _frame(image: np.ndarray, index: int = 0) -> Frame:
    h, w = image.shape[:2]
    info = SourceInfo(fps=25.0, width=w, height=h, frame_count=None, seekable=False, live=False)
    return Frame(image=image, index=index, timestamp=index / 25.0, source=info)


def _row(x: float, y: float, w: float, h: float, score: float = 0.9) -> np.ndarray:
    landmarks = [x + 0.3 * w, y + 0.4 * h, x + 0.7 * w, y + 0.4 * h, x + 0.5 * w, y + 0.6 * h,
                 x + 0.35 * w, y + 0.8 * h, x + 0.65 * w, y + 0.8 * h]
    return np.array([x, y, w, h, *landmarks, score], dtype=np.float32)


def _cartoon_face(size: int) -> np.ndarray:
    """Draw a simple frontal face; YuNet scores it about 0.84 to 0.89 at 160 to 320 px."""
    img = np.full((size, size, 3), 60, np.uint8)
    c = size // 2
    s = size / 100.0
    skin = (140, 170, 210)
    cv2.ellipse(img, (c, c), (int(28 * s), int(36 * s)), 0, 0, 360, skin, -1)
    for side in (-1, 1):
        ex, ey = c + side * int(11 * s), c - int(8 * s)
        cv2.ellipse(img, (ex, ey), (int(6 * s), int(3 * s)), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(img, (ex, ey), int(2.5 * s), (30, 20, 10), -1)
        cv2.line(img, (ex - int(7 * s), ey - int(7 * s)), (ex + int(7 * s), ey - int(8 * s)),
                 (40, 30, 20), int(2 * s))
    cv2.line(img, (c, c - int(3 * s)), (c - int(3 * s), c + int(8 * s)), (100, 120, 170), 3)
    cv2.ellipse(img, (c, c + int(17 * s)), (int(9 * s), int(3 * s)), 0, 0, 360,
                (80, 80, 180), -1)
    return cv2.GaussianBlur(img, (7, 7), 0)


def _scene_with_face(width: int, height: int, face_size: int, x: int, y: int) -> np.ndarray:
    img = np.full((height, width, 3), 60, np.uint8)
    img[y:y + face_size, x:x + face_size] = _cartoon_face(face_size)
    return img


# ---------------------------------------------------------------------------
# Model spec
# ---------------------------------------------------------------------------


def test_model_spec_values() -> None:
    assert MODEL.filename == "face_detection_yunet_2023mar.onnx"
    assert MODEL.sha256 == "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
    assert MODEL.size == 232589
    assert MODEL.key == "face_yunet_2023mar"
    assert MODEL.license
    assert MODEL.source_url.startswith("https://")


def test_wrong_bytes_of_right_size_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / MODEL.filename
    path.write_bytes(b"\0" * MODEL.size)
    with pytest.raises(ModelIntegrityError, match="SHA-256 mismatch"):
        load_model_bytes(MODEL, path=path)


def test_from_config_rejects_wrong_bytes_before_building_the_net(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / MODEL.filename).write_bytes(b"x" * MODEL.size)

    def _no_create(*args: object, **kwargs: object) -> None:
        raise AssertionError("the net must not be built from unverified bytes")

    monkeypatch.setattr(face_yunet.cv2.FaceDetectorYN, "create", _no_create)
    with pytest.raises(ModelIntegrityError):
        FaceYuNetDetector.from_config(FaceYuNetConfig(), [tmp_path])


def test_from_config_explicit_path_wins_over_dirs(tmp_path: Path) -> None:
    good_dir = tmp_path / "dirs"
    good_dir.mkdir()
    (good_dir / MODEL.filename).write_bytes(b"x" * MODEL.size)
    missing = tmp_path / "nowhere" / MODEL.filename
    with pytest.raises(ModelNotFoundError) as info:
        FaceYuNetDetector.from_config(FaceYuNetConfig(model_path=str(missing)), [good_dir])
    assert str(missing) in str(info.value)


def test_from_config_not_found_without_dirs() -> None:
    with pytest.raises(ModelNotFoundError):
        FaceYuNetDetector.from_config(FaceYuNetConfig(), [])


# ---------------------------------------------------------------------------
# Row conversion
# ---------------------------------------------------------------------------


def test_row_to_face_box_landmarks_and_score() -> None:
    face = face_from_row(_row(10.0, 20.0, 30.0, 40.0, score=0.75))
    assert face.box == Box(x=10, y=20, w=30, h=40, score=pytest.approx(0.75))
    assert face.landmarks is not None
    assert len(face.landmarks) == 5
    assert face.landmarks[0] == pytest.approx((19.0, 36.0))
    assert all(type(c) is float for point in face.landmarks for c in point)


def test_fractional_box_is_widened_outward() -> None:
    face = face_from_row(_row(10.6, 20.2, 30.1, 40.5))
    # floor(10.6) = 10, ceil(40.7) = 41; floor(20.2) = 20, ceil(60.7) = 61.
    assert (face.box.x, face.box.y, face.box.w, face.box.h) == (10, 20, 31, 41)
    assert all(type(v) is int for v in (face.box.x, face.box.y, face.box.w, face.box.h))


def test_box_partly_outside_is_kept_for_check_detections_to_clip() -> None:
    face = face_from_row(_row(-5.5, -3.0, 20.0, 20.0))
    assert face.box.x == -6
    checked = check_detections(
        Detections(faces=(face,)),
        frozenset({DetectionKind.FACES, DetectionKind.LANDMARKS}),
        (100, 100, 3),
    )
    assert checked.faces is not None
    assert checked.faces[0].box.x == 0
    # The box ends at ceil(-5.5 + 20) = 15.
    assert checked.faces[0].box.w == 15


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4, 9, 14])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_value_raises(index: int, bad: float) -> None:
    row = _row(10, 20, 30, 40)
    row[index] = bad
    with pytest.raises(ValueError, match="non-finite"):
        face_from_row(row)


@pytest.mark.parametrize("index", [2, 3])
def test_negative_size_raises(index: int) -> None:
    row = _row(10, 20, 30, 40)
    row[index] = -1.0
    with pytest.raises(ValueError, match="negative box size"):
        face_from_row(row)


def test_wrong_row_length_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        face_from_row(np.zeros(14, dtype=np.float32))


def test_output_none_is_no_faces() -> None:
    assert faces_from_output(None) == ()


def test_output_converts_every_row() -> None:
    out = np.stack([_row(0, 0, 10, 10), _row(50, 50, 20, 20, score=0.6)])
    faces = faces_from_output(out)
    assert len(faces) == 2
    assert faces[1].box.score == pytest.approx(0.6)


@pytest.mark.parametrize("shape", [(15,), (2, 14), (1, 2, 15)])
def test_output_wrong_shape_raises(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="shape"):
        faces_from_output(np.zeros(shape, dtype=np.float32))


def test_converted_output_passes_check_detections() -> None:
    faces = faces_from_output(np.stack([_row(1.5, 2.5, 30.2, 40.8), _row(90, 90, 50, 50)]))
    checked = check_detections(
        Detections(faces=faces), FaceYuNetDetector.provides, (100, 120, 3)
    )
    assert checked.faces is not None
    assert len(checked.faces) == 2
    assert all(f.landmarks is not None and len(f.landmarks) == 5 for f in checked.faces)


# ---------------------------------------------------------------------------
# Detector with a fake OpenCV net
# ---------------------------------------------------------------------------


class _FakeNet:
    def __init__(self, output: np.ndarray | None = None, error: Exception | None = None) -> None:
        self.output = output
        self.error = error
        self.sizes: list[tuple[int, int]] = []
        self.detect_calls = 0

    def setInputSize(self, size: tuple[int, int]) -> None:  # noqa: N802 (OpenCV name)
        self.sizes.append(tuple(size))

    def detect(self, image: np.ndarray) -> tuple[int, np.ndarray | None]:
        self.detect_calls += 1
        if self.error is not None:
            raise self.error
        return 1, self.output


@pytest.fixture
def fake_net(monkeypatch: pytest.MonkeyPatch) -> _FakeNet:
    net = _FakeNet()
    created: list[tuple[object, ...]] = []

    def _create(*args: object) -> _FakeNet:
        created.append(args)
        return net

    monkeypatch.setattr(face_yunet.cv2.FaceDetectorYN, "create", _create)
    net.created = created  # type: ignore[attr-defined]
    return net


def test_protocol_members(fake_net: _FakeNet) -> None:
    det = FaceYuNetDetector(b"model")
    assert isinstance(det, Detector)
    assert det.provides == frozenset({DetectionKind.FACES, DetectionKind.LANDMARKS})
    assert det.stateful is False
    det.reset()


def test_net_built_from_the_given_bytes_and_thresholds(fake_net: _FakeNet) -> None:
    FaceYuNetDetector(b"abc", score_threshold=0.5, nms_threshold=0.4, top_k=10)
    (args,) = fake_net.created  # type: ignore[attr-defined]
    framework, model, config, _size, score, nms, top_k = args
    assert framework == "onnx"
    assert isinstance(model, np.ndarray) and model.tobytes() == b"abc"
    assert isinstance(config, np.ndarray) and config.size == 0
    assert (score, nms, top_k) == (0.5, 0.4, 10)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"score_threshold": 0.0},
        {"score_threshold": 1.0},
        {"nms_threshold": 0.0},
        {"nms_threshold": 1.5},
        {"top_k": 0},
        {"top_k": True},
        {"score_threshold": float("nan")},
    ],
)
def test_bad_settings_raise(fake_net: _FakeNet, kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        FaceYuNetDetector(b"model", **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("data", [b"", "text", None, np.zeros(4, np.uint8)])
def test_bad_model_bytes_raise(fake_net: _FakeNet, data: object) -> None:
    with pytest.raises(ValueError, match="model_bytes"):
        FaceYuNetDetector(data)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((10, 10), np.uint8),
        np.zeros((10, 10, 1), np.uint8),
        np.zeros((10, 10, 4), np.uint8),
        # Above the side limit, so only the dtype is wrong.
        np.zeros((48, 64, 3), np.float32),
        np.zeros((48, 64, 3), np.uint16),
        np.zeros((0, 10, 3), np.uint8),
        np.zeros((10, 0, 3), np.uint8),
        [[[0, 0, 0]]],
    ],
)
def test_bad_frames_raise_before_opencv(fake_net: _FakeNet, image: object) -> None:
    det = FaceYuNetDetector(b"model")
    frame = Frame(
        image=image,  # type: ignore[arg-type]
        index=0,
        timestamp=0.0,
        source=SourceInfo(fps=25.0, width=10, height=10, frame_count=None, seekable=False,
                          live=False),
    )
    with pytest.raises(ValueError, match="frame image"):
        det.detect(frame)
    assert fake_net.detect_calls == 0


def test_opencv_error_propagates(fake_net: _FakeNet) -> None:
    fake_net.error = cv2.error("boom")
    det = FaceYuNetDetector(b"model")
    with pytest.raises(cv2.error):
        det.detect(_frame(np.zeros((48, 64, 3), np.uint8)))


def test_malformed_output_raises_instead_of_no_faces(fake_net: _FakeNet) -> None:
    fake_net.output = np.zeros((1, 7), np.float32)
    det = FaceYuNetDetector(b"model")
    with pytest.raises(ValueError, match="shape"):
        det.detect(_frame(np.zeros((48, 64, 3), np.uint8)))


def test_input_size_follows_the_frame(fake_net: _FakeNet) -> None:
    det = FaceYuNetDetector(b"model")
    det.detect(_frame(np.zeros((48, 64, 3), np.uint8)))
    det.detect(_frame(np.zeros((48, 64, 3), np.uint8)))
    det.detect(_frame(np.zeros((100, 80, 3), np.uint8)))
    # Width first, as OpenCV expects; set again only when the size changes.
    assert fake_net.sizes == [(64, 48), (80, 100)]


@pytest.mark.parametrize(("index", "value"), [(0, float("nan")), (2, -1.0)])
def test_one_bad_row_fails_the_frame(fake_net: _FakeNet, index: int, value: float) -> None:
    bad = _row(30, 10, 20, 20)
    bad[index] = value
    fake_net.output = np.stack([_row(5, 6, 20, 22), bad])
    det = FaceYuNetDetector(b"model")
    with pytest.raises(ValueError):
        det.detect(_frame(np.zeros((48, 64, 3), np.uint8)))


def test_output_keeps_partly_off_frame_rows() -> None:
    out = np.stack([_row(-4.0, 10.0, 20, 20), _row(10.0, -3.5, 20, 20), _row(5, 5, 10, 10)])
    faces = faces_from_output(out)
    assert [(f.box.x, f.box.y) for f in faces] == [(-4, 10), (10, -4), (5, 5)]


def test_detect_keeps_partly_off_frame_faces(fake_net: _FakeNet) -> None:
    fake_net.output = np.stack([_row(-4.0, 10.0, 20, 20), _row(50.0, -3.5, 20, 20)])
    det = FaceYuNetDetector(b"model")
    image = np.zeros((48, 64, 3), np.uint8)
    raw = det.detect(_frame(image))
    assert raw.faces is not None
    assert [(f.box.x, f.box.y) for f in raw.faces] == [(-4, 10), (50, -4)]
    checked = check_detections(raw, det.provides, image.shape)
    assert checked.faces is not None
    assert [(f.box.x, f.box.y, f.box.w, f.box.h) for f in checked.faces] == [
        (0, 10, 16, 20),
        (50, 0, 14, 17),
    ]


def test_from_config_passes_the_config_settings_to_the_net(
    fake_net: _FakeNet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        face_yunet,
        "load_model",
        lambda spec, path, dirs: LoadedModel(b"verified", tmp_path / MODEL.filename),
    )
    config = FaceYuNetConfig(score_threshold=0.45, nms_threshold=0.7, top_k=123)
    det = FaceYuNetDetector.from_config(config, [tmp_path])
    assert det.model_path == tmp_path / MODEL.filename
    (args,) = fake_net.created  # type: ignore[attr-defined]
    assert args[1].tobytes() == b"verified"
    assert args[4:7] == (0.45, 0.7, 123)
    assert (det.score_threshold, det.nms_threshold, det.top_k) == (0.45, 0.7, 123)


def test_model_path_is_none_when_built_from_bytes_and_read_only(fake_net: _FakeNet) -> None:
    det = FaceYuNetDetector(b"verified")
    assert det.model_path is None
    with pytest.raises(AttributeError):
        det.model_path = Path("x")  # type: ignore[misc]


@pytest.mark.parametrize("name", ["score_threshold", "nms_threshold", "top_k"])
def test_settings_are_read_only(fake_net: _FakeNet, name: str) -> None:
    det = FaceYuNetDetector(b"model")
    with pytest.raises(AttributeError):
        setattr(det, name, 0.5)


@pytest.mark.parametrize("shape", [(32, 64, 3), (64, 32, 3), (24, 1280, 3), (1280, 24, 3)])
def test_frames_at_or_under_the_side_limit_raise(fake_net: _FakeNet, shape: tuple) -> None:
    det = FaceYuNetDetector(b"model")
    with pytest.raises(ValueError, match=f"at least {MIN_FRAME_SIDE} pixels"):
        det.detect(_frame(np.zeros(shape, np.uint8)))
    assert fake_net.detect_calls == 0


@pytest.mark.parametrize("shape", [(33, 64, 3), (64, 33, 3), (33, 33, 3), (33, 1280, 3)])
def test_frames_just_over_the_side_limit_are_accepted(
    fake_net: _FakeNet, shape: tuple
) -> None:
    det = FaceYuNetDetector(b"model")
    assert det.detect(_frame(np.zeros(shape, np.uint8))).faces == ()
    assert fake_net.detect_calls == 1


def test_side_limit_value() -> None:
    assert MIN_FRAME_SIDE == 33


def test_fake_detection_sets_only_faces(fake_net: _FakeNet) -> None:
    fake_net.output = np.stack([_row(5, 6, 20, 22)])
    det = FaceYuNetDetector(b"model")
    result = det.detect(_frame(np.zeros((48, 64, 3), np.uint8)))
    assert result.faces is not None and len(result.faces) == 1
    assert result.motion is None and result.persons is None
    assert result.background_diff is None and result.mask is None


# ---------------------------------------------------------------------------
# Real model (skipped unless PETRIFY_MODEL_DIR holds the verified file)
# ---------------------------------------------------------------------------


def _model_dir() -> Path:
    env_dir = os.environ.get(MODEL_DIR_ENV, "")
    if not env_dir:
        pytest.skip(f"{MODEL_DIR_ENV} is not set; the YuNet model file is needed")
    path = Path(env_dir).expanduser()
    if not (path / MODEL.filename).is_file():
        pytest.skip(f"YuNet model file not found in {MODEL_DIR_ENV}: {path / MODEL.filename}")
    return path


@pytest.fixture(scope="module")
def real_detector() -> FaceYuNetDetector:
    return FaceYuNetDetector.from_config(FaceYuNetConfig(), [_model_dir()])


def _checked(det: FaceYuNetDetector, image: np.ndarray) -> Detections:
    return check_detections(det.detect(_frame(image)), det.provides, image.shape)


def test_real_model_loads_through_from_config(real_detector: FaceYuNetDetector) -> None:
    assert isinstance(real_detector, Detector)


def test_real_model_blank_frame_has_no_faces(real_detector: FaceYuNetDetector) -> None:
    result = _checked(real_detector, np.full((480, 640, 3), 100, np.uint8))
    assert result.faces == ()


def test_real_model_finds_synthetic_face(real_detector: FaceYuNetDetector) -> None:
    image = _scene_with_face(640, 480, face_size=240, x=300, y=150)
    result = _checked(real_detector, image)
    assert result.faces is not None and len(result.faces) == 1
    face = result.faces[0]
    # The drawn face ellipse spans about x 373 to 487 and y 184 to 386.
    cx, cy = face.box.x + face.box.w / 2, face.box.y + face.box.h / 2
    assert 400 < cx < 460 and 240 < cy < 330
    assert face.box.score is not None and face.box.score >= 0.6
    assert face.landmarks is not None and len(face.landmarks) == 5
    right_eye, left_eye = face.landmarks[0], face.landmarks[1]
    assert right_eye[0] < left_eye[0]


def test_real_model_handles_frame_size_changes(real_detector: FaceYuNetDetector) -> None:
    # (width, height, face size, patch x, patch y)
    scenes = [(320, 240, 200, 60, 20), (800, 600, 240, 100, 300), (320, 240, 200, 60, 20)]
    for width, height, size, x, y in scenes:
        image = _scene_with_face(width, height, face_size=size, x=x, y=y)
        raw = real_detector.detect(_frame(image))
        assert raw.faces is not None and len(raw.faces) == 1
        box = raw.faces[0].box
        # The face is drawn centred in its patch; a stale input size would misplace it.
        cx, cy = box.x + box.w / 2, box.y + box.h / 2
        assert abs(cx - (x + size / 2)) < size * 0.15
        assert abs(cy - (y + size / 2)) < size * 0.2
        check_detections(raw, real_detector.provides, image.shape)
    assert _checked(real_detector, np.zeros((97, 131, 3), np.uint8)).faces == ()
    assert _checked(real_detector, np.zeros((33, 33, 3), np.uint8)).faces == ()


def test_real_model_random_noise_output_is_well_formed(
    real_detector: FaceYuNetDetector,
) -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    raw = real_detector.detect(_frame(image))
    assert raw.faces is not None
    # Checked on the detector's own output, before check_detections clips anything.
    for face in raw.faces:
        assert all(type(v) is int for v in (face.box.x, face.box.y, face.box.w, face.box.h))
        assert face.box.w >= 0 and face.box.h >= 0
        assert face.landmarks is not None and len(face.landmarks) == 5
    check_detections(raw, real_detector.provides, image.shape)


def test_real_model_rejects_corrupt_bytes() -> None:
    _model_dir()
    with pytest.raises(cv2.error):
        FaceYuNetDetector(b"not an onnx model" * 100)


def test_face_type_is_the_core_face() -> None:
    assert face_from_row(_row(0, 0, 1, 1)).__class__ is Face
