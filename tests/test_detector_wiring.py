"""Wiring of the YuNet face, DNN person and segmentation detectors into the pipeline.

The detectors themselves are tested in their own modules; here they are replaced
by fakes (their ``from_config`` is patched), so these tests need no model file.
The end-to-end tests with the real models are in ``tests/test_detector_wiring_e2e.py``.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import logging
import sys
import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import mock

import numpy as np
import pytest

import petrify.pipeline as pipeline
from petrify.background.vetting import detector_check
from petrify.core.config import (
    BackgroundConfig,
    FaceYuNetConfig,
    ModelConfig,
    PersonDnnConfig,
    PersonHogConfig,
    PetrifyConfig,
    SegmentConfig,
    mode_options_from,
)
from petrify.core.detections import Box, DetectionKind, Detections, Face, MalformedDetections
from petrify.core.frame import Frame
from petrify.core.registry import ENTRY_POINT_GROUP, ModeLoadError
from petrify.detectors import FaceYuNetDetector, PersonDnnDetector, PersonSegmenter
from petrify.detectors.person_hog import PersonDetector
from petrify.models import ModelIntegrityError, ModelNotFoundError
from petrify.pipeline import (
    BACKGROUND_CHECK_NOT_USED_WARNING,
    DNN_NOT_USED_WARNING,
    DNN_WITHOUT_DIFFER_WARNING,
    FACES_NOT_USED_WARNING,
    MISSING_PROVIDER_ERROR,
    SAMPLE_CHECK_STUCK_WARNING,
    SAMPLE_REJECTION_STREAK,
    SEGMENT_NOT_USED_WARNING,
    OutputConfig,
    run_stream,
)
from petrify.server.control import RunControl, config_to_dict
from tests.test_run_stream import (
    FakeSource,
    H,
    HookSource,
    ListSink,
    Scene,
    W,
    _fake_person_free,
    _live_config,
    _shows_subject,
    _submit_at,
)

PLUGIN_MODULE = "t8plugin_kinds"

_PLUGIN_SOURCE = '''
import numpy as np

from petrify.core.detections import DetectionKind as K
from petrify.core.plugin import OptionSpec

RENDERS = []  # (mode name, frame index, detections) for every render call.


class KindsMode:
    detect_policy = "all"
    mandatory = frozenset()

    def __init__(self, name, requires):
        self.name = name
        self.requires = requires

    def render(self, frame, detections, subject_present):
        RENDERS.append((self.name, frame.index, detections))
        return np.zeros_like(frame.image)


class KindsFactory:
    summary = "Record the detections, for tests."
    options = ()
    needs_background = False
    roi_overlay = False

    def __init__(self, name, requires, policy="all"):
        self.name = name
        self._requires = frozenset(requires)
        self.policy = policy

    def requires(self, options):
        return self._requires

    def mandatory(self, options):
        return frozenset()

    def weakened(self, options):
        return ()

    def create(self, ctx, previous=None):
        requires = self.requires(ctx.options)
        if isinstance(previous, KindsMode) and previous.requires == requires:
            return previous
        mode = KindsMode(self.name, requires)
        mode.detect_policy = self.policy
        return mode


USES = {
    "persons": {K.PERSONS},
    "faces": {K.PERSONS, K.FACES},
    "mask": {K.PERSONS, K.MASK},
}


class OptionFactory(KindsFactory):
    """What the mode requires follows its runtime option "use"."""

    options = (
        OptionSpec("use", "choice", "persons", "what the mode uses",
                   choices=tuple(USES), runtime=True),
    )

    def requires(self, options):
        return frozenset(USES[options["use"]])


FACES = KindsFactory("faces", {K.FACES})
LANDMARKS = KindsFactory("landmarks", {K.FACES, K.LANDMARKS})
MASK = KindsFactory("mask", {K.MASK})
PERSONS = KindsFactory("persons", {K.PERSONS})
MOTION_FACES = KindsFactory("motionfaces", {K.MOTION, K.FACES})
EVERYTHING = KindsFactory("everything", {K.MOTION, K.PERSONS, K.FACES, K.LANDMARKS, K.MASK})
FIRST_HIT_FACES = KindsFactory("fhfaces", {K.FACES, K.LANDMARKS}, "first_hit")
FIRST_HIT_MASK = KindsFactory("fhmask", {K.MASK}, "first_hit")
FIRST_HIT_MOTION_FACES = KindsFactory("fhmotionfaces", {K.MOTION, K.FACES}, "first_hit")
FIRST_HIT_FACES_MASK = KindsFactory("fhfacesmask", {K.FACES, K.MASK}, "first_hit")
FIRST_HIT_PERSONS_MASK = KindsFactory("fhpersonsmask", {K.PERSONS, K.MASK}, "first_hit")
OPTIONAL = OptionFactory("optional", set())
FIRST_HIT_OPTIONAL = OptionFactory("fhoptional", set(), "first_hit")
'''

_ENTRY_POINTS = {
    "faces": "FACES",
    "landmarks": "LANDMARKS",
    "mask": "MASK",
    "persons": "PERSONS",
    "motionfaces": "MOTION_FACES",
    "everything": "EVERYTHING",
    "fhfaces": "FIRST_HIT_FACES",
    "fhmask": "FIRST_HIT_MASK",
    "fhmotionfaces": "FIRST_HIT_MOTION_FACES",
    "fhfacesmask": "FIRST_HIT_FACES_MASK",
    "fhpersonsmask": "FIRST_HIT_PERSONS_MASK",
    "optional": "OPTIONAL",
    "fhoptional": "FIRST_HIT_OPTIONAL",
}


@pytest.fixture
def plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """Install a fake distribution with test-only modes; yield the imported plugin module."""
    info = tmp_path / "petrify_t8_kinds-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: petrify-t8-kinds\nVersion: 0.1\n")
    lines = "".join(f"{name} = {PLUGIN_MODULE}:{attr}\n" for name, attr in _ENTRY_POINTS.items())
    (info / "entry_points.txt").write_text(f"[{ENTRY_POINT_GROUP}]\n{lines}")
    (tmp_path / f"{PLUGIN_MODULE}.py").write_text(textwrap.dedent(_PLUGIN_SOURCE))
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        import t8plugin_kinds  # type: ignore[import-not-found]

        yield t8plugin_kinds
    finally:
        sys.modules.pop(PLUGIN_MODULE, None)


# ---------------------------------------------------------------------------
# Fake model detectors
# ---------------------------------------------------------------------------

_FACE = Face(box=Box(x=10, y=10, w=20, h=20, score=0.9),
             landmarks=((12.0, 15.0), (25.0, 15.0), (18.0, 20.0), (13.0, 25.0), (24.0, 25.0)))


class FakeModelDetector:
    """Stands in for one of the three model detectors; reports on frames showing the subject."""

    stateful = False

    def __init__(self, cls: type, settings: object, dirs: Any) -> None:
        self.provides = cls.provides  # type: ignore[attr-defined]
        self.settings = settings
        self.dirs = tuple(dirs)
        self.kind = cls

    def detect(self, frame: Frame) -> Detections:
        seen = _shows_subject(frame.image)
        if self.kind is FaceYuNetDetector:
            return Detections(faces=(_FACE,) if seen else ())
        if self.kind is PersonDnnDetector:
            return Detections(persons=(Box(x=0, y=0, w=40, h=80, score=0.8),) if seen else ())
        mask = np.zeros(frame.image.shape[:2], dtype=bool)
        if seen:
            mask[:20, :20] = True
        return Detections(mask=mask)

    def reset(self) -> None:
        pass


class ModelBuilds:
    """Patches the three ``from_config`` methods; records every build, can fail one class."""

    def __init__(self) -> None:
        self.calls: list[tuple[type, object]] = []
        self.fail: dict[type, Exception] = {}

    def build(self, cls: type) -> Callable[..., FakeModelDetector]:
        def from_config(settings: object, dirs: Any = ()) -> FakeModelDetector:
            self.calls.append((cls, settings))
            if cls in self.fail:
                raise self.fail[cls]
            return FakeModelDetector(cls, settings, dirs)

        return from_config

    def classes(self) -> list[type]:
        return [cls for cls, _ in self.calls]


@pytest.fixture
def builds() -> Iterator[ModelBuilds]:
    recorder = ModelBuilds()
    with (
        mock.patch.object(FaceYuNetDetector, "from_config", side_effect=recorder.build(
            FaceYuNetDetector)),
        mock.patch.object(PersonDnnDetector, "from_config", side_effect=recorder.build(
            PersonDnnDetector)),
        mock.patch.object(PersonSegmenter, "from_config", side_effect=recorder.build(
            PersonSegmenter)),
    ):
        yield recorder


@pytest.fixture
def fake_person_check() -> Iterator[None]:
    with mock.patch.object(pipeline, "person_free", side_effect=_fake_person_free):
        yield


def _plugin_config(mode: str, **overrides: Any) -> PetrifyConfig:
    overrides.setdefault("detector", "both")
    return _live_config("running", mode=mode, **overrides)


ON = {
    "faces": FaceYuNetConfig(),
    "person_dnn": PersonDnnConfig(),
    "segment": SegmentConfig(),
}


def _sections(names: tuple[str, ...]) -> dict[str, object]:
    return {name: ON[name] for name in names}


def _missing(mode: str, kind: str, provider: str) -> str:
    return MISSING_PROVIDER_ERROR.format(mode=mode, kind=kind, provider=provider)


FACES_FLAG = "the YuNet face detector with --faces"
MASK_FLAG = "the person segmenter with --segment"


# ---------------------------------------------------------------------------
# Selection matrix: what the mode requires x what the config enables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("enabled", [(), ("faces",), ("segment",), ("person_dnn",),
                                     ("faces", "segment", "person_dnn")])
@pytest.mark.parametrize(
    ("mode", "built"),
    [
        ("faces", {"faces"}),
        ("landmarks", {"faces"}),
        ("mask", {"segment"}),
        ("persons", {"person_dnn"}),
        ("motionfaces", {"faces"}),
        ("everything", {"faces", "segment", "person_dnn"}),
    ],
)
@pytest.mark.usefixtures("plugin")
def test_selection_matrix(
    builds: ModelBuilds, mode: str, built: set[str], enabled: tuple[str, ...]
) -> None:
    config = _plugin_config(mode, **_sections(enabled))
    needed = {"faces", "segment"} & built
    missing = needed - set(enabled)
    if missing:
        with pytest.raises(ValueError, match="no enabled detector provides it"):
            pipeline._build_components(FakeSource(), config)
        assert builds.calls == []
        return
    components, _ = pipeline._build_components(FakeSource(), config)
    expected = built & set(enabled)
    assert (components.faces is not None) == ("faces" in expected)
    assert (components.segment is not None) == ("segment" in expected)
    if DetectionKind.PERSONS in components.requires:
        dnn = "person_dnn" in enabled
        assert components.person is not None
        assert isinstance(components.person, FakeModelDetector) is dnn
        assert isinstance(components.person, PersonDetector) is not dnn
    else:
        assert components.person is None
    classes = {FaceYuNetDetector: "faces", PersonDnnDetector: "person_dnn",
               PersonSegmenter: "segment"}
    assert {classes[cls] for cls in builds.classes()} == expected
    assert len(builds.calls) == len(expected)  # Each model is loaded once.
    # The engine order: motion, person, faces, mask, then the differ.
    assert components.detectors == [
        d for d in (components.motion, components.person, components.faces,
                    components.segment, components.differ) if d is not None
    ]


@pytest.mark.parametrize("detector", ["motion", "person", "both"])
@pytest.mark.usefixtures("fake_person_check")
def test_erase_builds_no_model_detector_by_default_even_with_faces_and_segment_on(
    builds: ModelBuilds, detector: str
) -> None:
    components, _ = pipeline._build_components(FakeSource(), _live_config("running"))
    assert builds.calls == []
    config = _live_config("running", detector=detector, faces=FaceYuNetConfig(),
                          segment=SegmentConfig())
    components, _ = pipeline._build_components(FakeSource(), config)
    assert builds.calls == []  # Erase requires neither faces nor a mask.
    assert components.faces is None and components.segment is None
    assert (components.person is not None) == (detector != "motion")
    assert components.person is None or isinstance(components.person, PersonDetector)


@pytest.mark.parametrize("detector", ["motion", "person", "both"])
@pytest.mark.usefixtures("fake_person_check")
def test_erase_with_the_dnn_selected_uses_it_in_place_of_hog(
    builds: ModelBuilds, detector: str
) -> None:
    config = _live_config("running", detector=detector, person_dnn=PersonDnnConfig())
    with mock.patch.object(pipeline.PersonDetector, "from_config",
                           wraps=pipeline.PersonDetector.from_config) as hog:
        components, _ = pipeline._build_components(FakeSource(), config)
    if detector == "motion":
        assert components.person is None
        assert builds.calls == []
    else:
        assert isinstance(components.person, FakeModelDetector)
        assert components.person.kind is PersonDnnDetector
        assert builds.calls == [(PersonDnnDetector, PersonDnnConfig())]
        assert components.person in components.detectors
    # HOG is built only for the background sample check, with the pre-scan settings.
    hog.assert_called_once_with(config.background_scan_person_config())


@pytest.mark.usefixtures("fake_person_check")
def test_erase_with_the_dnn_freezes_the_frames_the_dnn_sees(builds: ModelBuilds) -> None:
    config = _live_config("running", detector="person", person_dnn=PersonDnnConfig(),
                          background_diff=None)
    sink = ListSink()
    report = run_stream(FakeSource(Scene(total=120)), config, OutputConfig(), sinks=[sink])
    assert report.stats.failed_closed < report.stats.frames
    assert not any(_shows_subject(image) for image in sink.images)


# ---------------------------------------------------------------------------
# Missing provider
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "kind", "provider"),
    [
        ("faces", "faces", FACES_FLAG),
        ("motionfaces", "faces", FACES_FLAG),
        ("mask", "mask", MASK_FLAG),
    ],
)
@pytest.mark.usefixtures("plugin")
def test_a_required_kind_without_an_enabled_provider_fails_before_any_frame(
    builds: ModelBuilds, mode: str, kind: str, provider: str
) -> None:
    src = FakeSource()
    sink = ListSink()
    with (
        mock.patch.object(pipeline, "_roi_stage", side_effect=AssertionError("ROI built")),
        pytest.raises(ValueError) as excinfo,
    ):
        run_stream(src, _plugin_config(mode), OutputConfig(), sinks=[sink])
    message = str(excinfo.value)
    assert message == _missing(mode, kind, provider)
    assert "\n" not in message
    assert sink.opened == 0
    assert src.yielded == []
    assert builds.calls == []


@pytest.mark.usefixtures("plugin")
def test_landmarks_without_faces_names_the_first_missing_kind() -> None:
    with pytest.raises(ValueError) as excinfo:
        pipeline._build_components(FakeSource(), _plugin_config("landmarks"))
    assert str(excinfo.value) == _missing("landmarks", "faces", FACES_FLAG)


def test_the_missing_provider_error_names_the_kind_and_the_flag() -> None:
    message = _missing("blur", "mask", MASK_FLAG)
    assert message == (
        "The mode 'blur' requires mask but no enabled detector provides it; enable the "
        "person segmenter with --segment"
    )


@pytest.mark.usefixtures("plugin")
def test_a_reconfiguration_keeps_the_provider_check(builds: ModelBuilds) -> None:
    config = _plugin_config("faces", faces=FaceYuNetConfig())
    components, _ = pipeline._build_components(FakeSource(), config)
    with pytest.raises(ValueError, match="requires faces but no enabled detector"):
        pipeline._rebuild(components, dataclasses.replace(config, faces=None))


# ---------------------------------------------------------------------------
# Preflight: every model loaded before anything reads frames
# ---------------------------------------------------------------------------

_NOT_FOUND = ModelNotFoundError("Model 'x' was not found.\nLocations searched, in order:")


@pytest.mark.parametrize(
    ("overrides", "failing"),
    [
        ({"detector": "both", "person_dnn": PersonDnnConfig()}, PersonDnnDetector),
        ({"background": BackgroundConfig(strategy="running", sample_checkers=("hog", "yunet"))},
         FaceYuNetDetector),
        ({"background": BackgroundConfig(strategy="calibrate", sample_checkers=("dnn",))},
         PersonDnnDetector),
    ],
    ids=["frame-detector", "running-sample-check", "calibrate-sample-check"],
)
def test_a_model_error_stops_the_run_before_the_roi_scan_and_the_background(
    builds: ModelBuilds, overrides: dict[str, Any], failing: type
) -> None:
    builds.fail[failing] = _NOT_FOUND
    config = _live_config("running", **overrides)
    src = FakeSource()
    sink = ListSink()
    with (
        mock.patch.object(pipeline, "_roi_stage", side_effect=AssertionError("ROI scan")) as roi,
        mock.patch.object(pipeline, "_background_for",
                          side_effect=AssertionError("background")) as background,
        mock.patch.object(pipeline, "locked_background",
                          side_effect=AssertionError("pre-scan")) as prescan,
        pytest.raises(ModelNotFoundError),
    ):
        run_stream(src, config, OutputConfig(), sinks=[sink])
    roi.assert_not_called()
    background.assert_not_called()
    prescan.assert_not_called()
    assert src.yielded == []
    assert sink.opened == 0


def test_a_model_error_comes_before_the_locked_pre_scan(builds: ModelBuilds) -> None:
    builds.fail[PersonDnnDetector] = ModelIntegrityError("SHA-256 mismatch")
    config = _live_config("locked", detector="person", person_dnn=PersonDnnConfig())
    with (
        mock.patch.object(pipeline, "_roi_stage", side_effect=AssertionError("ROI scan")),
        mock.patch.object(pipeline, "locked_background",
                          side_effect=AssertionError("pre-scan")) as prescan,
        pytest.raises(ModelIntegrityError),
    ):
        pipeline._build_components(FakeSource(live=False), config)
    prescan.assert_not_called()


def test_model_dir_feeds_the_model_search(
    builds: ModelBuilds, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PETRIFY_MODEL_DIR", str(tmp_path / "env"))
    config = _live_config("running", detector="both", person_dnn=PersonDnnConfig(),
                          models=ModelConfig(dir=str(tmp_path / "explicit")),
                          background=BackgroundConfig(strategy="running",
                                                      sample_checkers=("dnn",)))
    with mock.patch.object(pipeline, "person_free", side_effect=_fake_person_free):
        components, _ = pipeline._build_components(FakeSource(), config)
    assert isinstance(components.person, FakeModelDetector)
    assert components.person.dirs[:2] == (tmp_path / "explicit", tmp_path / "env")


# ---------------------------------------------------------------------------
# Each model is loaded once per run
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("plugin")
def test_a_reconfiguration_that_keeps_the_detectors_reads_no_model(builds: ModelBuilds) -> None:
    config = _plugin_config("everything", **ON)
    components, _ = pipeline._build_components(FakeSource(), config)
    assert len(builds.calls) == 3
    changed = dataclasses.replace(config, person=PersonHogConfig(min_confidence=0.9))
    rebuilt, kwargs = pipeline._rebuild(components, changed)
    assert len(builds.calls) == 3
    assert "detectors" not in kwargs  # HOG settings do not matter with the DNN selected.
    assert rebuilt.person is components.person
    assert rebuilt.faces is components.faces
    assert rebuilt.segment is components.segment


@pytest.mark.usefixtures("fake_person_check")
def test_the_dnn_shared_by_the_frame_detector_and_the_sample_check_is_loaded_once(
    builds: ModelBuilds,
) -> None:
    config = _live_config("running", detector="both", person_dnn=PersonDnnConfig(),
                          background=BackgroundConfig(strategy="running",
                                                      sample_checkers=("hog", "dnn")))
    pipeline._build_components(FakeSource(), config)
    assert builds.calls == [(PersonDnnDetector, PersonDnnConfig())]


@pytest.mark.usefixtures("fake_person_check")
def test_a_detector_switched_on_later_is_loaded_then_and_kept(builds: ModelBuilds) -> None:
    config = _live_config("running", detector="motion", person_dnn=PersonDnnConfig())
    components, _ = pipeline._build_components(FakeSource(), config)
    assert builds.calls == []
    both = dataclasses.replace(config, detector="both")
    on, kwargs = pipeline._rebuild(components, both)
    assert builds.classes() == [PersonDnnDetector]
    assert "detectors" in kwargs
    off, _ = pipeline._rebuild(on, config)
    again, _ = pipeline._rebuild(off, both)
    assert again.person is on.person
    assert builds.classes() == [PersonDnnDetector]


@pytest.mark.usefixtures("fake_person_check")
def test_a_model_failure_on_reconfiguration_is_a_rejected_setting(builds: ModelBuilds) -> None:
    config = _live_config("running", detector="motion", person_dnn=PersonDnnConfig())
    components, _ = pipeline._build_components(FakeSource(), config)
    builds.fail[PersonDnnDetector] = _NOT_FOUND
    with pytest.raises(ValueError, match="Model 'x' was not found"):
        pipeline._rebuild(components, dataclasses.replace(config, detector="both"))


@pytest.mark.usefixtures("fake_person_check")
def test_a_model_failure_on_a_web_change_keeps_the_run_going(builds: ModelBuilds) -> None:
    builds.fail[PersonDnnDetector] = _NOT_FOUND
    config = _live_config("running", detector="motion", person_dnn=PersonDnnConfig())
    control = RunControl(config, (W, H))
    hooks = {40: _submit_at(control, {"detector": "both"})}
    sink = ListSink()
    report = run_stream(HookSource(Scene(total=100), hooks=hooks), config, OutputConfig(),
                        sinks=[sink], control=control)
    assert report.stats.frames == 100
    assert control.applied.detector == "motion"
    error = control.snapshot().last_error
    assert error is not None and "Model 'x' was not found" in error and "\n" not in error


# ---------------------------------------------------------------------------
# Background sample checkers
# ---------------------------------------------------------------------------


def _check_for(config: PetrifyConfig) -> Any:
    with mock.patch.object(pipeline, "person_free", return_value=lambda image: True) as check:
        pipeline._background_for(FakeSource(), config)
    check.assert_called_once()
    return check.call_args.args[0]


def test_the_default_sample_check_is_the_hog_detector_alone(builds: ModelBuilds) -> None:
    check = _check_for(_live_config("running"))
    assert isinstance(check, PersonDetector)
    assert builds.calls == []


@pytest.mark.parametrize(
    ("name", "cls"),
    [("yunet", FaceYuNetDetector), ("dnn", PersonDnnDetector)],
)
def test_each_model_checker_name_works(builds: ModelBuilds, name: str, cls: type) -> None:
    config = _live_config("running", background=BackgroundConfig(
        strategy="running", sample_checkers=(name,)))
    check = _check_for(config)
    assert builds.classes() == [cls]
    empty = Scene(total=1, enter=10).image(0)
    assert not check.is_person(empty)
    with_subject = Scene().image(150)
    assert _shows_subject(with_subject)
    assert check.is_person(with_subject)


def test_a_model_checker_uses_its_section_when_enabled_else_the_defaults(
    builds: ModelBuilds,
) -> None:
    background = BackgroundConfig(strategy="running", sample_checkers=("yunet", "dnn"))
    _check_for(_live_config("running", background=background))
    assert builds.calls == [
        (FaceYuNetDetector, FaceYuNetConfig()),
        (PersonDnnDetector, PersonDnnConfig()),
    ]
    builds.calls.clear()
    tuned = {"faces": FaceYuNetConfig(score_threshold=0.8),
             "person_dnn": PersonDnnConfig(score_threshold=0.7)}
    _check_for(_live_config("running", background=background, **tuned))
    assert builds.calls == [
        (FaceYuNetDetector, tuned["faces"]),
        (PersonDnnDetector, tuned["person_dnn"]),
    ]


class _Check:
    def __init__(self, answer: bool | Exception, log: list[str], name: str) -> None:
        self.answer = answer
        self.log = log
        self.name = name

    def is_person(self, frame: np.ndarray) -> bool:
        self.log.append(self.name)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


@pytest.mark.parametrize(
    ("answers", "expected", "ran"),
    [
        ((False, False, False), False, ["hog", "dnn", "yunet"]),
        ((False, True, False), True, ["hog", "dnn"]),
        ((True, False, False), True, ["hog"]),
    ],
)
def test_several_checkers_reject_a_sample_when_any_sees_a_person(
    answers: tuple[bool, ...], expected: bool, ran: list[str]
) -> None:
    log: list[str] = []
    names = ("hog", "dnn", "yunet")
    fakes = {name: _Check(answer, log, name) for name, answer in zip(names, answers, strict=True)}
    config = _live_config("running", background=BackgroundConfig(
        strategy="running", sample_checkers=names))
    with mock.patch.object(pipeline, "_sample_checker",
                           side_effect=lambda name, cfg, models: fakes[name]):
        check = pipeline._sample_check(config, pipeline._ModelDetectors(config))
    assert check.is_person(np.zeros((8, 8, 3), np.uint8)) is expected
    assert log == ran


@pytest.mark.parametrize("name", ["mog2", "segment"])
def test_an_unknown_checker_name_is_refused(builds: ModelBuilds, name: str) -> None:
    config = _live_config("running")
    with pytest.raises(ValueError, match=f"sample checker '{name}' is not available"):
        pipeline._sample_checker(name, config, pipeline._ModelDetectors(config))


def test_every_name_the_config_accepts_has_a_checker(builds: ModelBuilds) -> None:
    from petrify.core.config import SAMPLE_CHECKERS

    config = _live_config("running")
    models = pipeline._ModelDetectors(config)
    for name in SAMPLE_CHECKERS:
        assert hasattr(pipeline._sample_checker(name, config, models), "is_person")


def test_the_sample_check_runs_on_the_samples_and_keeps_the_subject_out(
    builds: ModelBuilds,
) -> None:
    # A person present from the first frame: only the check keeps them out of the
    # running background, as in test_run_stream's L1 tests, here with the DNN checker.
    config = _live_config("running", background=BackgroundConfig(
        strategy="running", sample_checkers=("dnn",)))
    scene = Scene(total=160, enter=0, stand=5, leave=70, gone=90)
    sink = ListSink()
    run_stream(FakeSource(scene), config, OutputConfig(), sinks=[sink])
    assert not any(_shows_subject(image) for image in sink.images)


def test_detector_check_validates_the_result() -> None:
    class Broken:
        provides = frozenset({DetectionKind.FACES})
        stateful = False

        def detect(self, frame: Frame) -> Detections:
            return Detections(persons=())  # A field the detector does not provide.

        def reset(self) -> None:
            pass

    with pytest.raises(MalformedDetections):
        detector_check(Broken()).is_person(np.zeros((40, 40, 3), np.uint8))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Config and config_to_dict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, 1, 0.0, 0.25, 1.0])
@pytest.mark.parametrize("name", ["pad_x", "pad_y"])
def test_person_dnn_pads_accept_numbers_from_0_to_1(name: str, value: float) -> None:
    assert getattr(PersonDnnConfig(**{name: value}), name) == value


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf"), True, "0.1", None])
@pytest.mark.parametrize("name", ["pad_x", "pad_y"])
def test_person_dnn_pads_reject_other_values(name: str, value: object) -> None:
    with pytest.raises(ValueError, match=rf"PersonDnnConfig.{name} must be a number in \[0, 1\]"):
        PersonDnnConfig(**{name: value})  # type: ignore[arg-type]


def test_person_dnn_pad_defaults() -> None:
    assert (PersonDnnConfig().pad_x, PersonDnnConfig().pad_y) == (0.10, 0.07)


def test_from_config_forwards_the_pads(monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2

    from petrify.detectors import person_dnn

    monkeypatch.setattr(person_dnn, "load_model_bytes", lambda spec, path=None, dirs=(): b"x")

    class Net:
        def getUnconnectedOutLayersNames(self) -> tuple[str, ...]:  # noqa: N802 (OpenCV API)
            return ()

    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda buffer: Net())
    det = PersonDnnDetector.from_config(PersonDnnConfig(pad_x=0.3, pad_y=0.0), [])
    assert (det.pad_x, det.pad_y) == (0.3, 0.0)


def test_config_to_dict_with_the_new_detectors_is_plain_json() -> None:
    config = PetrifyConfig(
        detector="both",
        faces=FaceYuNetConfig(),
        person_dnn=PersonDnnConfig(pad_x=0.2),
        segment=SegmentConfig(),
        background=BackgroundConfig(strategy="running", sample_checkers=("hog", "dnn")),
    )
    data = config_to_dict(config)
    assert json.loads(json.dumps(data, allow_nan=False)) == data
    assert data["person_dnn"]["pad_x"] == 0.2
    assert data["person_dnn"]["pad_y"] == 0.07
    assert data["background"]["sample_checkers"] == ["hog", "dnn"]


@pytest.mark.usefixtures("fake_person_check")
def test_the_settings_page_snapshot_works_with_the_new_detectors(builds: ModelBuilds) -> None:
    config = _live_config("running", detector="both", person_dnn=PersonDnnConfig(),
                          faces=FaceYuNetConfig(), segment=SegmentConfig())
    control = RunControl(config, (W, H))
    hooks = {30: _submit_at(control, {"motion": {"threshold": 200}})}
    run_stream(HookSource(Scene(total=60), hooks=hooks), config, OutputConfig(),
               sinks=[ListSink()], control=control)
    snapshot = control.snapshot()
    assert snapshot.last_error is None
    assert snapshot.config["person_dnn"]["pad_x"] == 0.10
    assert control.applied.motion.threshold == 200
    assert control.applied.person_dnn == PersonDnnConfig()


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


def test_the_models_module_does_not_import_opencv() -> None:
    import petrify.models

    tree = ast.parse(Path(petrify.models.__file__).read_text())
    imported = {alias.name.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.Import) for alias in node.names}
    imported |= {(node.module or "").split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom)}
    assert "cv2" not in imported
    assert "numpy" not in imported


# ---------------------------------------------------------------------------
# T8b: startup warnings for detectors that do not cover what the user asked
# ---------------------------------------------------------------------------


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and r.name == "petrify.pipeline"]


@pytest.mark.usefixtures("fake_person_check")
@pytest.mark.parametrize(
    ("overrides", "warned"),
    [
        ({"detector": "both", "person_dnn": PersonDnnConfig(), "background_diff": None}, True),
        ({"detector": "person", "person_dnn": PersonDnnConfig(), "background_diff": None},
         True),
        ({"detector": "both", "person_dnn": PersonDnnConfig()}, False),
        ({"detector": "both", "background_diff": None}, False),
    ],
    ids=["dnn-no-differ", "dnn-person-no-differ", "dnn-with-differ", "hog-no-differ"],
)
def test_dnn_without_the_differ_warns_once(
    builds: ModelBuilds, caplog: pytest.LogCaptureFixture, overrides: dict[str, Any],
    warned: bool,
) -> None:
    config = _live_config("running", **overrides)
    with caplog.at_level(logging.WARNING, logger="petrify.pipeline"):
        run_stream(FakeSource(Scene(total=20)), config, OutputConfig(), sinks=[ListSink()])
    assert _warnings(caplog).count(DNN_WITHOUT_DIFFER_WARNING) == (1 if warned else 0)


@pytest.mark.usefixtures("plugin")
def test_dnn_in_a_mode_without_the_differ_warns(
    builds: ModelBuilds, caplog: pytest.LogCaptureFixture
) -> None:
    config = _plugin_config("persons", person_dnn=PersonDnnConfig())
    with caplog.at_level(logging.WARNING, logger="petrify.pipeline"):
        pipeline._build_components(FakeSource(), config)
    assert _warnings(caplog).count(DNN_WITHOUT_DIFFER_WARNING) == 1


def test_the_dnn_warning_states_the_size_limit() -> None:
    assert "110 px tall on a 1080p frame" in DNN_WITHOUT_DIFFER_WARNING
    assert "--no-bg-diff" in DNN_WITHOUT_DIFFER_WARNING


@pytest.mark.usefixtures("fake_person_check")
def test_dnn_with_detector_motion_warns_that_it_has_no_effect(
    builds: ModelBuilds, caplog: pytest.LogCaptureFixture
) -> None:
    config = _live_config("running", detector="motion", person_dnn=PersonDnnConfig())
    with caplog.at_level(logging.WARNING, logger="petrify.pipeline"):
        pipeline._build_components(FakeSource(), config)
    assert _warnings(caplog).count(DNN_NOT_USED_WARNING) == 1
    assert "switched to person or both" in DNN_NOT_USED_WARNING
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="petrify.pipeline"):
        pipeline._build_components(FakeSource(), dataclasses.replace(config, detector="both"))
    assert DNN_NOT_USED_WARNING not in _warnings(caplog)


@pytest.mark.usefixtures("fake_person_check")
def test_faces_and_segment_with_erase_warn_that_they_have_no_effect(
    builds: ModelBuilds, caplog: pytest.LogCaptureFixture
) -> None:
    config = _live_config("running", faces=FaceYuNetConfig(), segment=SegmentConfig())
    with caplog.at_level(logging.WARNING, logger="petrify.pipeline"):
        pipeline._build_components(FakeSource(), config)
    warnings = _warnings(caplog)
    assert warnings.count(FACES_NOT_USED_WARNING.format(mode="erase")) == 1
    assert warnings.count(SEGMENT_NOT_USED_WARNING.format(mode="erase")) == 1


@pytest.mark.usefixtures("plugin")
def test_faces_and_segment_used_by_the_mode_do_not_warn(
    builds: ModelBuilds, caplog: pytest.LogCaptureFixture
) -> None:
    config = _plugin_config("everything", **ON)
    with caplog.at_level(logging.WARNING, logger="petrify.pipeline"):
        pipeline._build_components(FakeSource(), config)
    warnings = _warnings(caplog)
    assert FACES_NOT_USED_WARNING.format(mode="everything") not in warnings
    assert SEGMENT_NOT_USED_WARNING.format(mode="everything") not in warnings
    assert DNN_NOT_USED_WARNING not in warnings


@pytest.mark.usefixtures("plugin")
def test_background_check_in_a_mode_without_a_background_warns(
    builds: ModelBuilds, caplog: pytest.LogCaptureFixture
) -> None:
    background = BackgroundConfig(strategy="running", sample_checkers=("hog", "dnn"))
    config = _plugin_config("persons", background=background)
    with caplog.at_level(logging.WARNING, logger="petrify.pipeline"):
        pipeline._build_components(FakeSource(), config)
    assert _warnings(caplog).count(BACKGROUND_CHECK_NOT_USED_WARNING.format(mode="persons")) == 1
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="petrify.pipeline"):
        pipeline._build_components(FakeSource(), _plugin_config("persons"))
    assert _warnings(caplog) == []


@pytest.mark.usefixtures("fake_person_check")
def test_background_check_with_erase_does_not_warn(
    builds: ModelBuilds, caplog: pytest.LogCaptureFixture
) -> None:
    background = BackgroundConfig(strategy="running", sample_checkers=("hog", "dnn"))
    with caplog.at_level(logging.WARNING, logger="petrify.pipeline"):
        pipeline._build_components(FakeSource(), _live_config("running", background=background))
    assert not any("--background-check has no effect" in w for w in _warnings(caplog))


def test_the_ready_line_names_the_model_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from petrify.models import load_model_bytes
    from tests.test_models import _place, _spec

    spec = _spec()
    model_file = _place(tmp_path / "models")
    monkeypatch.setitem(pipeline._MODEL_OF, PersonDnnDetector, spec)

    def from_config(settings: object, dirs: Any = ()) -> FakeModelDetector:
        load_model_bytes(spec, dirs=dirs)
        return FakeModelDetector(PersonDnnDetector, settings, dirs)

    config = _live_config("running", models=ModelConfig(dir=str(tmp_path / "models")))
    models = pipeline._ModelDetectors(config)
    with (
        mock.patch.object(PersonDnnDetector, "from_config", side_effect=from_config),
        caplog.at_level(logging.INFO, logger="petrify.pipeline"),
    ):
        models.get(PersonDnnDetector, PersonDnnConfig())
    assert f"Detector PersonDnnDetector ready: model {model_file} loaded and verified" in [
        r.getMessage() for r in caplog.records
    ]


def test_pop_loaded_path_reports_the_file_once(tmp_path: Path) -> None:
    from petrify.models import load_model_bytes, pop_loaded_path
    from tests.test_models import _place, _spec

    spec = _spec()
    target = _place(tmp_path / "b")
    load_model_bytes(spec, dirs=[tmp_path / "a", tmp_path / "b"])
    assert pop_loaded_path(spec) == target
    assert pop_loaded_path(spec) is None
    with pytest.raises(ModelNotFoundError):
        load_model_bytes(spec, dirs=[tmp_path / "a"])
    assert pop_loaded_path(spec) is None


# ---------------------------------------------------------------------------
# T8b: the stuck sample check
# ---------------------------------------------------------------------------


class _Model:
    def __init__(self, ready: bool = False) -> None:
        self.ready = ready

    def current(self) -> np.ndarray | None:
        return np.zeros((2, 2, 3), np.uint8) if self.ready else None


def _watch(answers: list[bool | Exception], ready: bool = False) -> Any:
    queue = list(answers)

    def sample_filter(image: np.ndarray) -> bool:
        answer = queue.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    watch = pipeline._RejectionWatch(sample_filter, ("hog", "dnn"))
    watch.model = _Model(ready)  # type: ignore[assignment]
    return watch


def _feed(watch: Any, count: int) -> list[bool | str]:
    image = np.zeros((2, 2, 3), np.uint8)
    out: list[bool | str] = []
    for _ in range(count):
        try:
            out.append(watch(image))
        except RuntimeError:
            out.append("error")
    return out


STUCK = SAMPLE_CHECK_STUCK_WARNING.format(checkers="hog, dnn", count=SAMPLE_REJECTION_STREAK)


def test_a_stuck_sample_check_warns_once_after_the_streak(
    caplog: pytest.LogCaptureFixture,
) -> None:
    n = SAMPLE_REJECTION_STREAK
    watch = _watch([False] * (3 * n))
    with caplog.at_level(logging.WARNING, logger="petrify.pipeline"):
        assert _feed(watch, n - 1) == [False] * (n - 1)
        assert _warnings(caplog) == []
        assert _feed(watch, 2 * n + 1) == [False] * (2 * n + 1)
    assert _warnings(caplog) == [STUCK]
    assert "hog, dnn" in STUCK and "every frame is written black" in STUCK


def test_an_accepted_sample_restarts_the_streak(caplog: pytest.LogCaptureFixture) -> None:
    n = SAMPLE_REJECTION_STREAK
    watch = _watch([False] * (n - 1) + [True] + [False] * (n - 1))
    with caplog.at_level(logging.WARNING, logger="petrify.pipeline"):
        assert _feed(watch, 2 * n - 1).count(True) == 1
    assert _warnings(caplog) == []


def test_filter_errors_count_as_rejections_and_still_propagate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    n = SAMPLE_REJECTION_STREAK
    watch = _watch([RuntimeError("detector failed")] * n)
    with caplog.at_level(logging.WARNING, logger="petrify.pipeline"):
        assert _feed(watch, n) == ["error"] * n
    assert _warnings(caplog) == [STUCK]


def test_no_warning_once_the_background_is_ready(caplog: pytest.LogCaptureFixture) -> None:
    n = SAMPLE_REJECTION_STREAK
    watch = _watch([False] * (2 * n), ready=True)
    with caplog.at_level(logging.WARNING, logger="petrify.pipeline"):
        _feed(watch, 2 * n)
    assert _warnings(caplog) == []


class _AlwaysPerson:
    """A sample check that sees a person in every sample."""

    def is_person(self, frame: np.ndarray) -> bool:
        return True


@pytest.mark.parametrize("strategy", ["running", "calibrate"])
def test_a_run_whose_check_rejects_every_sample_warns_and_stays_black(
    caplog: pytest.LogCaptureFixture, strategy: str
) -> None:
    config = _live_config(strategy, background=BackgroundConfig(
        strategy=strategy, calibrate_seconds=2.0, running_interval=1,  # type: ignore[arg-type]
        sample_checkers=("hog", "yunet")))
    sink = ListSink()
    with (
        mock.patch.object(pipeline, "_sample_check", return_value=_AlwaysPerson()),
        caplog.at_level(logging.WARNING, logger="petrify.pipeline"),
    ):
        report = run_stream(FakeSource(Scene(total=200, enter=500)), config, OutputConfig(),
                            sinks=[sink])
    stuck = [w for w in _warnings(caplog) if w.startswith("The background sample check")]
    assert stuck == [SAMPLE_CHECK_STUCK_WARNING.format(
        checkers="hog, yunet", count=SAMPLE_REJECTION_STREAK)]
    assert report.stats.failed_closed == report.stats.frames  # Fail closed, unchanged.
    assert all(not image.any() for image in sink.images)


# ---------------------------------------------------------------------------
# T8b: a model checker that raises rejects the sample
# ---------------------------------------------------------------------------


class _RaisingDetector:
    provides = frozenset({DetectionKind.PERSONS})
    stateful = False

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: Frame) -> Detections:
        self.calls += 1
        raise RuntimeError("inference failed")

    def reset(self) -> None:
        pass


@pytest.mark.parametrize("strategy", ["running", "calibrate"])
def test_a_raising_model_checker_rejects_the_sample_and_stores_nothing(strategy: str) -> None:
    detector = _RaisingDetector()
    config = _live_config(strategy, background=BackgroundConfig(
        strategy=strategy, calibrate_seconds=1.0, running_interval=1,  # type: ignore[arg-type]
        running_min_samples=1, sample_checkers=("dnn",)))
    with mock.patch.object(PersonDnnDetector, "from_config", return_value=detector):
        model, _ = pipeline._background_for(FakeSource(live=False), config)
    assert not isinstance(model, np.ndarray)
    empty = Scene(total=1, enter=10).image(0)
    for i in range(30):
        model.update(Frame(image=empty, index=i, timestamp=i / 10, source=FakeSource().info),
                     False)
    assert detector.calls == 30
    assert model.current() is None
    assert model.rejected == 30  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# T8b: a face model error at startup fails the run
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("plugin")
@pytest.mark.parametrize(("mode", "failing"), [("faces", FaceYuNetDetector),
                                               ("mask", PersonSegmenter)])
def test_a_face_or_mask_model_error_at_startup_fails_the_run(
    builds: ModelBuilds, mode: str, failing: type
) -> None:
    builds.fail[failing] = _NOT_FOUND
    src = FakeSource()
    sink = ListSink()
    with (
        mock.patch.object(pipeline, "_roi_stage", side_effect=AssertionError("ROI scan")),
        pytest.raises(ModelNotFoundError),
    ):
        run_stream(src, _plugin_config(mode, **ON), OutputConfig(), sinks=[sink])
    assert src.yielded == []
    assert sink.opened == 0


# ---------------------------------------------------------------------------
# T8b: a runtime option that adds faces or a mask builds the detector
# ---------------------------------------------------------------------------


def _use(config: PetrifyConfig, value: str) -> PetrifyConfig:
    return dataclasses.replace(config, mode_options=mode_options_from({"use": value}))


@pytest.mark.usefixtures("plugin")
@pytest.mark.parametrize(("use", "field", "cls"), [("faces", "faces", FaceYuNetDetector),
                                                   ("mask", "segment", PersonSegmenter)])
def test_an_option_that_adds_a_kind_builds_its_detector_on_rebuild(
    builds: ModelBuilds, use: str, field: str, cls: type
) -> None:
    config = _use(_plugin_config("optional", **ON), "persons")
    components, _ = pipeline._build_components(FakeSource(), config)
    assert components.faces is None and components.segment is None
    rebuilt, kwargs = pipeline._rebuild(components, _use(config, use))
    built = getattr(rebuilt, field)
    assert isinstance(built, FakeModelDetector) and built.kind is cls
    assert built in kwargs["detectors"]
    back, kwargs = pipeline._rebuild(rebuilt, config)
    assert getattr(back, field) is None
    assert built not in kwargs["detectors"]


@pytest.mark.usefixtures("plugin")
def test_an_option_that_adds_faces_without_the_detector_is_rejected(builds: ModelBuilds) -> None:
    config = _use(_plugin_config("optional"), "persons")
    components, _ = pipeline._build_components(FakeSource(), config)
    with pytest.raises(ValueError, match="requires faces but no enabled detector"):
        pipeline._rebuild(components, _use(config, "faces"))


# ---------------------------------------------------------------------------
# T8b: first_hit modes that need faces, landmarks or a mask with another kind
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("plugin")
@pytest.mark.parametrize(
    ("mode", "kinds", "own"),
    [
        ("fhmotionfaces", "{faces, motion}", "{faces}"),
        ("fhpersonsmask", "{mask, persons}", "{mask}"),
        ("fhfacesmask", "{faces, mask}", "{faces}"),
    ],
)
def test_first_hit_mode_mixing_kinds_is_refused_at_startup(
    builds: ModelBuilds, mode: str, kinds: str, own: str
) -> None:
    src = FakeSource()
    sink = ListSink()
    with pytest.raises(ModeLoadError) as excinfo:
        run_stream(src, _plugin_config(mode, **ON), OutputConfig(), sinks=[sink])
    message = str(excinfo.value)
    assert message == (
        f"mode {mode!r}: requires {kinds} with the 'first_hit' detection policy, which "
        f"skips the detector of {own} on frames where another detector fired; the mode "
        'must declare detect_policy = "all"'
    )
    assert "\n" not in message
    assert sink.opened == 0 and src.yielded == []


@pytest.mark.usefixtures("plugin")
@pytest.mark.parametrize("mode", ["fhfaces", "fhmask"])
def test_first_hit_mode_with_one_detector_kinds_runs(builds: ModelBuilds, mode: str) -> None:
    report = run_stream(FakeSource(Scene(total=20)), _plugin_config(mode, **ON), OutputConfig(),
                        sinks=[ListSink()])
    assert report.stats.frames == 20


@pytest.mark.usefixtures("plugin")
def test_first_hit_option_that_mixes_kinds_is_rejected_on_rebuild(builds: ModelBuilds) -> None:
    config = _use(_plugin_config("fhoptional", **ON), "persons")
    components, _ = pipeline._build_components(FakeSource(), config)
    with pytest.raises(ValueError, match='must declare detect_policy = "all"'):
        pipeline._rebuild(components, _use(config, "faces"))
