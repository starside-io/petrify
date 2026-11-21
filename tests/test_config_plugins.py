"""Tests for the plugin mode and new detector sections of petrify.core.config."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

from petrify.cli import _build_parser
from petrify.core import (
    BackgroundConfig,
    FaceYuNetConfig,
    ModelConfig,
    PersonDnnConfig,
    PetrifyConfig,
    SegmentConfig,
    TrackerConfig,
    mode_options_from,
)
from petrify.server.control import config_to_dict

OPTIONS = (("labels", ("a", "b")), ("radius", 7), ("ratio", 0.5), ("solid", True), ("tint", "red"))


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_new_petrify_fields_default_to_off() -> None:
    config = PetrifyConfig()
    assert config.mode == "erase"
    assert config.mode_options == ()
    assert config.faces is None
    assert config.person_dnn is None
    assert config.segment is None
    assert config.tracker is None
    assert config.models == ModelConfig()
    assert config.models.dir is None
    assert config.background.sample_checkers == ("hog",)


def test_section_defaults() -> None:
    assert FaceYuNetConfig() == FaceYuNetConfig(
        score_threshold=0.6, nms_threshold=0.3, top_k=5000, model_path=None
    )
    assert PersonDnnConfig() == PersonDnnConfig(
        score_threshold=0.4, nms_threshold=0.5, model_path=None
    )
    assert SegmentConfig() == SegmentConfig(
        threshold=0.5, dilate_px=0, min_area_share=0.002, model_path=None
    )
    assert TrackerConfig() == TrackerConfig(iou_threshold=0.3, hold_frames=15, max_tracks=256)


def test_from_namespace_leaves_new_fields_at_defaults() -> None:
    config = PetrifyConfig.from_namespace(_build_parser().parse_args(["in.mp4", "out.mp4"]))
    default = PetrifyConfig()
    assert config.mode == default.mode
    assert config.mode_options == default.mode_options
    assert config.faces is None
    assert config.person_dnn is None
    assert config.segment is None
    assert config.tracker is None
    assert config.models == default.models
    assert config.background.sample_checkers == ("hog",)
    assert config == default


def test_from_namespace_with_flags_still_leaves_new_fields_at_defaults() -> None:
    args = _build_parser().parse_args(
        ["in.mp4", "out.mp4", "--detector", "both", "--no-bg-diff", "--no-roi"]
    )
    config = PetrifyConfig.from_namespace(args)
    assert config.detector == "both"
    assert config.background_diff is None
    assert dataclasses.replace(
        config,
        mode="erase", mode_options=(), faces=None, person_dnn=None, segment=None,
        tracker=None, models=ModelConfig(),
    ) == config


# ---------------------------------------------------------------------------
# Detector section bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (FaceYuNetConfig, {"score_threshold": 0.0}),
        (FaceYuNetConfig, {"score_threshold": 1.0}),
        (FaceYuNetConfig, {"score_threshold": float("nan")}),
        (FaceYuNetConfig, {"nms_threshold": 0.0}),
        (FaceYuNetConfig, {"nms_threshold": 1.01}),
        (FaceYuNetConfig, {"top_k": 0}),
        (FaceYuNetConfig, {"model_path": ""}),
        (PersonDnnConfig, {"score_threshold": 0.0}),
        (PersonDnnConfig, {"score_threshold": 1.0}),
        (PersonDnnConfig, {"nms_threshold": 0.0}),
        (PersonDnnConfig, {"nms_threshold": 1.01}),
        (PersonDnnConfig, {"model_path": ""}),
        (SegmentConfig, {"threshold": 0.0}),
        (SegmentConfig, {"threshold": 1.0}),
        (SegmentConfig, {"dilate_px": -1}),
        (SegmentConfig, {"min_area_share": 0.0}),
        (SegmentConfig, {"min_area_share": 1.01}),
        (SegmentConfig, {"model_path": ""}),
        (TrackerConfig, {"iou_threshold": 0.0}),
        (TrackerConfig, {"iou_threshold": 1.01}),
        (TrackerConfig, {"hold_frames": -1}),
        (TrackerConfig, {"max_tracks": 0}),
        (ModelConfig, {"dir": ""}),
    ],
)
def test_section_rejects_out_of_bounds(cls: type, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=cls.__name__):
        cls(**kwargs)


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (FaceYuNetConfig, {"score_threshold": 0.01, "nms_threshold": 1.0, "top_k": 1}),
        (FaceYuNetConfig, {"score_threshold": 0.99, "model_path": "face.onnx"}),
        (PersonDnnConfig, {"score_threshold": 0.01, "nms_threshold": 1.0}),
        (PersonDnnConfig, {"score_threshold": 0.99, "model_path": "person.onnx"}),
        (SegmentConfig, {"threshold": 0.01, "dilate_px": 0, "min_area_share": 1.0}),
        (SegmentConfig, {"threshold": 0.99, "dilate_px": 12, "model_path": "seg.onnx"}),
        (TrackerConfig, {"iou_threshold": 1.0, "hold_frames": 0, "max_tracks": 1}),
        (ModelConfig, {"dir": "/models"}),
    ],
)
def test_section_accepts_edge_values(cls: type, kwargs: dict[str, Any]) -> None:
    section = cls(**kwargs)
    for key, value in kwargs.items():
        assert getattr(section, key) == value


def test_new_sections_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        TrackerConfig().hold_frames = 3  # type: ignore[misc]


# ---------------------------------------------------------------------------
# mode and mode_options
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["a", "erase", "blur_2", "m" + "x" * 31])
def test_mode_accepts_valid_names(mode: str) -> None:
    assert PetrifyConfig(mode=mode).mode == mode


@pytest.mark.parametrize("mode", ["", "Erase", "2blur", "_erase", "blur-x", "m" + "x" * 32])
def test_mode_rejects_invalid_names(mode: str) -> None:
    with pytest.raises(ValueError, match="PetrifyConfig.mode"):
        PetrifyConfig(mode=mode)


def test_mode_options_accepts_every_allowed_type() -> None:
    config = PetrifyConfig(mode="blur", mode_options=OPTIONS)
    assert config.mode_options == OPTIONS
    assert config.mode_options_dict() == dict(OPTIONS)


def test_mode_options_accepts_empty_string_tuple() -> None:
    assert PetrifyConfig(mode_options=(("labels", ()),)).mode_options_dict() == {"labels": ()}


@pytest.mark.parametrize(
    "options",
    [
        (("Radius", 1),),
        (("1radius", 1),),
        (("radius-x", 1),),
        (("r" + "x" * 32, 1),),
        ((1, 1),),
    ],
)
def test_mode_options_rejects_invalid_names(options: Any) -> None:
    with pytest.raises(ValueError, match="name"):
        PetrifyConfig(mode_options=options)


def test_mode_options_rejects_unsorted_names() -> None:
    with pytest.raises(ValueError, match="sorted"):
        PetrifyConfig(mode_options=(("radius", 1), ("labels", "a")))


def test_mode_options_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        PetrifyConfig(mode_options=(("radius", 1), ("radius", 2)))


@pytest.mark.parametrize(
    "value", [["a", "b"], ("a", 1), None, {"a": 1}, float("nan"), float("inf"), b"raw"]
)
def test_mode_options_rejects_invalid_values(value: Any) -> None:
    with pytest.raises(ValueError, match="radius"):
        PetrifyConfig(mode_options=(("radius", value),))


@pytest.mark.parametrize(
    "options", [[("radius", 1)], {"radius": 1}, (("radius",),), (("radius", 1, 2),), ("radius",)]
)
def test_mode_options_rejects_non_pair_shapes(options: Any) -> None:
    with pytest.raises(ValueError, match="mode_options"):
        PetrifyConfig(mode_options=options)


def test_mode_options_from_sorts_and_validates() -> None:
    pairs = mode_options_from({"tint": "red", "radius": 7, "labels": ("a", "b")})
    assert pairs == (("labels", ("a", "b")), ("radius", 7), ("tint", "red"))
    assert PetrifyConfig(mode_options=pairs).mode_options == pairs
    assert mode_options_from({}) == ()


@pytest.mark.parametrize(
    "mapping", [{"Bad": 1}, {"labels": ["a"]}, {1: 1}, {"ratio": float("nan")}]
)
def test_mode_options_from_rejects_invalid(mapping: Any) -> None:
    with pytest.raises(ValueError):
        mode_options_from(mapping)


def test_mode_options_dict_round_trips() -> None:
    mapping = {"tint": "red", "radius": 7, "labels": ("a", "b")}
    config = PetrifyConfig(mode_options=mode_options_from(mapping))
    assert config.mode_options_dict() == mapping
    config.mode_options_dict()["radius"] = 1  # a new dict each call
    assert config.mode_options_dict()["radius"] == 7


# ---------------------------------------------------------------------------
# sample_checkers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "checkers", [("hog",), ("yunet",), ("hog", "yunet", "dnn", "segment"), ("segment", "hog")]
)
def test_sample_checkers_accepts_known_unique_names(checkers: tuple[str, ...]) -> None:
    assert BackgroundConfig(sample_checkers=checkers).sample_checkers == checkers


def test_sample_checkers_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        BackgroundConfig(sample_checkers=())


def test_sample_checkers_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="one of"):
        BackgroundConfig(sample_checkers=("hog", "mog2"))


def test_sample_checkers_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="repeat"):
        BackgroundConfig(sample_checkers=("hog", "hog"))


def test_sample_checkers_rejects_list() -> None:
    with pytest.raises(ValueError, match="tuple"):
        BackgroundConfig(sample_checkers=["hog"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Hashing, equality, asdict and config_to_dict
# ---------------------------------------------------------------------------


def _full_config() -> PetrifyConfig:
    return PetrifyConfig(
        mode="blur",
        mode_options=OPTIONS,
        faces=FaceYuNetConfig(model_path="face.onnx"),
        person_dnn=PersonDnnConfig(),
        segment=SegmentConfig(dilate_px=4),
        tracker=TrackerConfig(),
        models=ModelConfig(dir="/models"),
        background=BackgroundConfig(sample_checkers=("hog", "segment")),
    )


def test_config_with_options_is_hashable_and_equal() -> None:
    a, b = _full_config(), _full_config()
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b, PetrifyConfig()}) == 2
    assert a != dataclasses.replace(a, mode_options=OPTIONS[:-1])
    assert a != dataclasses.replace(a, mode="pixelate")


def test_asdict_keeps_mode_options_as_pairs() -> None:
    data = dataclasses.asdict(_full_config())
    assert data["mode_options"] == OPTIONS
    assert data["faces"]["model_path"] == "face.onnx"
    assert data["tracker"] == {"iou_threshold": 0.3, "hold_frames": 15, "max_tracks": 256}
    assert data["models"] == {"dir": "/models"}
    assert data["background"]["sample_checkers"] == ("hog", "segment")


def test_config_to_dict_renders_mode_options_as_object() -> None:
    data = config_to_dict(_full_config())
    assert json.loads(json.dumps(data, allow_nan=False)) == data
    assert data["mode"] == "blur"
    assert data["mode_options"] == {
        "labels": ["a", "b"], "radius": 7, "ratio": 0.5, "solid": True, "tint": "red",
    }
    assert data["faces"] == {
        "score_threshold": 0.6, "nms_threshold": 0.3, "top_k": 5000, "model_path": "face.onnx",
    }
    assert data["person_dnn"] == {"score_threshold": 0.4, "nms_threshold": 0.5, "model_path": None}
    assert data["segment"] == {
        "threshold": 0.5, "dilate_px": 4, "min_area_share": 0.002, "model_path": None,
    }
    assert data["tracker"] == {"iou_threshold": 0.3, "hold_frames": 15, "max_tracks": 256}
    assert data["models"] == {"dir": "/models"}
    assert data["background"]["sample_checkers"] == ["hog", "segment"]


def test_config_to_dict_defaults() -> None:
    data = config_to_dict(PetrifyConfig())
    assert json.loads(json.dumps(data, allow_nan=False)) == data
    assert data["mode"] == "erase"
    assert data["mode_options"] == {}
    assert data["faces"] is None
    assert data["person_dnn"] is None
    assert data["segment"] is None
    assert data["tracker"] is None
    assert data["models"] == {"dir": None}
    assert data["background"]["sample_checkers"] == ["hog"]


def test_config_to_dict_existing_fields_unchanged() -> None:
    new_keys = {"mode", "mode_options", "faces", "person_dnn", "segment", "tracker", "models"}
    data = config_to_dict(PetrifyConfig())
    assert set(data) == {
        "detector", "motion", "person", "background_diff", "roi", "background", "encode",
        "engine",
    } | new_keys
    assert data["person"]["win_stride"] == [8, 8]
    assert data["roi"]["box"] is None
    assert data["background_diff"] == {
        "pixel_threshold": 25, "area_threshold": 200, "downsample": 0.5,
    }
