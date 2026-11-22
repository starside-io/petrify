"""Tests for run_stream with modes loaded through the mode registry (plugin modes)."""

from __future__ import annotations

import dataclasses
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import mock

import cv2
import numpy as np
import pytest

import petrify.pipeline as pipeline
from petrify.core import (
    BackgroundConfig,
    Frame,
    PetrifyConfig,
    RoiConfig,
)
from petrify.core.config import TrackerConfig
from petrify.core.detections import Box, DetectionKind, Detections
from petrify.core.registry import ENTRY_POINT_GROUP, ModeLoadError
from petrify.detectors.bgdiff import BackgroundDiffer
from petrify.detectors.motion import MotionDetector
from petrify.detectors.person_hog import PersonDetector
from petrify.modes.erase import EraseMode
from petrify.pipeline import OutputConfig, run_stream
from petrify.server.control import RunControl
from petrify.sources import FileSource
from petrify.stages.roi_overlay import RoiOverlay, WarmupRoiOverlay
from tests.test_run_stream import (
    BACKGROUND,
    ENCODE,
    FPS,
    QUIET,
    SUBJECT,
    FakeSource,
    H,
    HookSource,
    ListSink,
    Scene,
    W,
    _fake_person_free,
    _live_config,
)

PAINT = 77  # Every pixel of a plugin mode's output.
PLUGIN_MODULE = "t11plugin_paint"
BROKEN_MODULE = "t11plugin_broken"

_PLUGIN_SOURCE = f'''
import numpy as np

from petrify.core.detections import DetectionKind
from petrify.core.plugin import OptionSpec

CONTEXTS = []  # Every ModeContext a factory's create received.
RENDERS = []  # (mode name, frame index, detections) for every render call.


class PaintMode:
    detect_policy = "all"
    mandatory = frozenset()

    def __init__(self, name, requires, options):
        self.name = name
        self.requires = requires
        self.options = options

    def render(self, frame, detections, subject_present):
        RENDERS.append((self.name, frame.index, detections))
        return np.full_like(frame.image, {PAINT})


class PaintFactory:
    summary = "Paint every frame flat, for tests."
    options = (
        OptionSpec("strength", "int", 3, "strength", minimum=1, maximum=10, runtime=True),
        OptionSpec("fail", "choice", "none", "make create fail",
                   choices=("none", "value", "runtime")),
    )

    def __init__(self, name, requires, needs_background=False, roi_overlay=False):
        self.name = name
        self._requires = frozenset(requires)
        self.needs_background = needs_background
        self.roi_overlay = roi_overlay

    def requires(self, options):
        return self._requires

    def mandatory(self, options):
        return frozenset()

    def weakened(self, options):
        return ()

    def create(self, ctx, previous=None):
        CONTEXTS.append(ctx)
        if ctx.options["fail"] == "value":
            raise ValueError("this strength cannot be used")
        if ctx.options["fail"] == "runtime":
            raise RuntimeError("plugin bug")
        options = dict(ctx.options)
        if isinstance(previous, PaintMode) and previous.options == options:
            return previous
        return PaintMode(self.name, self._requires, options)


PAINT = PaintFactory("paint", {{DetectionKind.PERSONS}})
MOTION_PAINT = PaintFactory("motionpaint", {{DetectionKind.MOTION}})
DIFF_PAINT = PaintFactory("diffpaint", {{DetectionKind.BACKGROUND_DIFF}})
'''

_ENTRY_POINTS = {
    "paint": f"{PLUGIN_MODULE}:PAINT",
    "motionpaint": f"{PLUGIN_MODULE}:MOTION_PAINT",
    "diffpaint": f"{PLUGIN_MODULE}:DIFF_PAINT",
    "broken": f"{BROKEN_MODULE}:FACTORY",
}


@pytest.fixture
def plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """Install a fake distribution with the plugin modes; yield the imported plugin module."""
    info = tmp_path / "petrify_t11_plugins-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: petrify-t11-plugins\nVersion: 0.1\n")
    lines = "".join(f"{name} = {target}\n" for name, target in _ENTRY_POINTS.items())
    (info / "entry_points.txt").write_text(f"[{ENTRY_POINT_GROUP}]\n{lines}")
    (tmp_path / f"{PLUGIN_MODULE}.py").write_text(textwrap.dedent(_PLUGIN_SOURCE))
    (tmp_path / f"{BROKEN_MODULE}.py").write_text("raise KeyError('broken at import')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        import t11plugin_paint  # type: ignore[import-not-found]

        yield t11plugin_paint
    finally:
        for name in (PLUGIN_MODULE, BROKEN_MODULE):
            sys.modules.pop(name, None)


class FakePersons:
    """A stand-in for the HOG person detector: one box on the frames in ``hits``."""

    provides = frozenset({DetectionKind.PERSONS})
    stateful = False

    def __init__(self, hits: dict[int, Box] | None = None) -> None:
        self.hits = hits or {}

    def detect(self, frame: Frame) -> Detections:
        box = self.hits.get(frame.index)
        return Detections(persons=(box,) if box is not None else ())

    def reset(self) -> None:
        pass


@pytest.fixture
def fake_persons() -> Iterator[FakePersons]:
    detector = FakePersons()
    with mock.patch.object(pipeline.PersonDetector, "from_config", return_value=detector):
        yield detector


@pytest.fixture
def fake_person_check() -> Iterator[None]:
    with mock.patch.object(pipeline, "person_free", side_effect=_fake_person_free):
        yield


def _plugin_config(mode: str = "paint", **overrides: Any) -> PetrifyConfig:
    overrides.setdefault("detector", "motion" if mode == "motionpaint" else "person")
    return _live_config("running", mode=mode, **overrides)


def _painted(image: np.ndarray) -> bool:
    return bool((image == PAINT).all())


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("fake_persons")
def test_plugin_mode_runs_end_to_end_and_its_output_reaches_the_sink(
    plugin: ModuleType,
) -> None:
    sink = ListSink()
    report = run_stream(FakeSource(Scene(total=40)), _plugin_config(), OutputConfig(),
                        sinks=[sink])
    assert report.stats.frames == 40
    assert report.stats.failed_closed == 0
    assert len(sink.images) == 40
    assert all(_painted(image) for image in sink.images)
    assert [index for name, index, _ in plugin.RENDERS] == list(range(40))


@pytest.mark.usefixtures("fake_persons")
def test_mode_options_reach_the_context_with_defaults_filled(plugin: ModuleType) -> None:
    config = _plugin_config(mode_options=(("strength", 7),))
    components, _ = pipeline._build_components(FakeSource(), config)
    (ctx,) = plugin.CONTEXTS
    assert dict(ctx.options) == {"strength": 7, "fail": "none"}
    assert ctx.config.mode_options == (("fail", "none"), ("strength", 7))
    assert ctx.source is components.source
    assert components.options == {"strength": 7, "fail": "none"}
    assert components.config is config


@pytest.mark.usefixtures("fake_persons", "plugin")
def test_a_mode_option_not_in_its_declared_type_is_rejected() -> None:
    config = _plugin_config(mode_options=(("strength", 7.0),))
    with pytest.raises(ValueError, match="'strength' is 7.0 in the config"):
        pipeline._build_components(FakeSource(), config)


@pytest.mark.usefixtures("plugin")
@pytest.mark.parametrize(
    ("mode", "message"),
    [("nosuchmode", "unknown mode 'nosuchmode'"), ("broken", "KeyError.*broken at import")],
)
def test_unknown_or_broken_mode_fails_at_startup_before_any_sink_opens(
    mode: str, message: str
) -> None:
    src = FakeSource()
    sink = ListSink()
    with pytest.raises(ModeLoadError, match=message):
        run_stream(src, _plugin_config(mode=mode), OutputConfig(), sinks=[sink])
    assert sink.opened == 0
    assert src.closed == 1


# ---------------------------------------------------------------------------
# ROI overlay
# ---------------------------------------------------------------------------

ROI_BOX = (296, 0, 24, 16)  # Covers the ticking patch Scene(tick_roi=True) draws.


@pytest.mark.usefixtures("fake_persons", "plugin")
def test_mode_without_roi_overlay_gets_no_roi_stage_and_no_live_roi() -> None:
    config = _plugin_config(roi=RoiConfig(mode="manual", box=ROI_BOX))
    components, _ = pipeline._build_components(FakeSource(), config)
    assert components.roi_stage is None
    assert components.stages == []
    src = FakeSource(Scene(total=30, tick_roi=True))
    sink = ListSink()
    report = run_stream(src, config, OutputConfig(), sinks=[sink])
    assert report.roi is None
    x, y, w, h = ROI_BOX
    # The live ROI ticks, so a pasted ROI would differ from the painted output.
    assert len({img[y : y + h, x : x + w].tobytes() for img in src.yielded}) > 1
    assert all(_painted(image) for image in sink.images)


@pytest.mark.usefixtures("fake_persons", "plugin")
def test_mode_without_roi_overlay_skips_roi_auto_detection() -> None:
    config = _plugin_config(roi=RoiConfig(mode="auto"))
    with mock.patch.object(pipeline, "_roi_stage", side_effect=AssertionError("ROI built")):
        components, _ = pipeline._build_components(FakeSource(), config)
    assert components.roi_stage is None


@pytest.mark.usefixtures("fake_person_check")
def test_erase_still_pastes_the_live_roi() -> None:
    config = _live_config("calibrate", roi=RoiConfig(mode="manual", box=ROI_BOX))
    components, _ = pipeline._build_components(FakeSource(), config)
    assert isinstance(components.roi_stage, RoiOverlay)
    assert components.stages == [components.roi_stage]
    src = FakeSource(Scene(total=80, enter=40, stand=45, leave=60, gone=65, tick_roi=True))
    sink = ListSink()
    report = run_stream(src, config, OutputConfig(), sinks=[sink])
    assert report.roi == ROI_BOX
    x, y, w, h = ROI_BOX
    rendered = [k for k, img in enumerate(sink.images) if img.any()]
    assert rendered
    for k in rendered:
        assert np.array_equal(sink.images[k][y : y + h, x : x + w],
                              src.yielded[k][y : y + h, x : x + w]), k


# ---------------------------------------------------------------------------
# Background and detector selection
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("fake_persons", "plugin")
def test_mode_without_background_runs_without_one_being_built() -> None:
    # A locked background cannot be built on a live source: building it would raise.
    config = _plugin_config(background=BackgroundConfig(strategy="locked"))
    sink = ListSink()
    with mock.patch.object(pipeline, "_background_for",
                           side_effect=AssertionError("background built")):
        report = run_stream(FakeSource(Scene(total=30)), config, OutputConfig(), sinks=[sink])
        components, bg_idx = pipeline._build_components(FakeSource(), config)
    assert report.background_index is None
    assert report.stats.failed_closed == 0
    assert all(_painted(image) for image in sink.images)
    assert components.background is None and bg_idx is None
    assert components.differ is None


@pytest.mark.usefixtures("plugin")
def test_mode_requiring_the_differ_without_a_background_is_a_startup_error() -> None:
    sink = ListSink()
    with pytest.raises(ValueError, match="requires the background differ but does not use"):
        run_stream(FakeSource(), _plugin_config(mode="diffpaint"), OutputConfig(),
                   sinks=[sink])
    assert sink.opened == 0


@pytest.mark.usefixtures("plugin")
def test_detectors_follow_what_the_mode_requires() -> None:
    persons, _ = pipeline._build_components(FakeSource(), _plugin_config(detector="both"))
    assert persons.motion is None
    assert isinstance(persons.person, PersonDetector)
    assert persons.detectors == [persons.person]
    motion, _ = pipeline._build_components(
        FakeSource(), _plugin_config(mode="motionpaint", detector="both")
    )
    assert isinstance(motion.motion, MotionDetector)
    assert motion.person is None
    assert motion.detectors == [motion.motion]


# ---------------------------------------------------------------------------
# Tracker refiners
# ---------------------------------------------------------------------------


def test_tracker_holds_a_person_box_for_the_hold_period(
    plugin: ModuleType, fake_persons: FakePersons
) -> None:
    box = Box(100, 80, 40, 90)
    fake_persons.hits = {20: box}
    config = _plugin_config(tracker=TrackerConfig(hold_frames=5))
    run_stream(FakeSource(Scene(total=40)), config, OutputConfig(), sinks=[ListSink()])
    with_box = [index for _, index, det in plugin.RENDERS if box in (det.persons or ())]
    assert with_box == [20, 21, 22, 23, 24, 25]


@pytest.mark.usefixtures("fake_person_check")
def test_tracker_with_erase_is_a_clear_startup_error() -> None:
    sink = ListSink()
    config = _live_config("calibrate", tracker=TrackerConfig())
    with pytest.raises(ValueError, match="mode 'erase'.*'first_hit'"):
        run_stream(FakeSource(), config, OutputConfig(), sinks=[sink])
    assert sink.opened == 0


@pytest.mark.usefixtures("fake_persons", "plugin")
def test_rebuild_hands_the_refiners_over_with_new_detectors() -> None:
    config = _plugin_config(detector="both", tracker=TrackerConfig())
    components, _ = pipeline._build_components(FakeSource(), config)
    (refiner,) = components.refiners
    changed = dataclasses.replace(config, detector="person", person=dataclasses.replace(
        config.person, min_confidence=0.9))
    with mock.patch.object(pipeline.PersonDetector, "from_config",
                           return_value=FakePersons()):
        built, kwargs = pipeline._rebuild(components, changed)
    assert "detectors" in kwargs
    assert kwargs["refiners"] == [refiner]


# ---------------------------------------------------------------------------
# Run-time changes
# ---------------------------------------------------------------------------


class InjectingControl(RunControl):
    """A RunControl whose next pending config can be set directly."""

    def __init__(self, config: PetrifyConfig) -> None:
        super().__init__(config, (W, H))
        self.queued: PetrifyConfig | None = None

    def take_pending(self) -> PetrifyConfig | None:
        queued, self.queued = self.queued, None
        return queued if queued is not None else super().take_pending()


@pytest.mark.usefixtures("fake_persons")
def test_run_time_create_errors_are_rejected_and_the_run_continues(
    plugin: ModuleType,
) -> None:
    config = _plugin_config()
    control = InjectingControl(config)
    errors: list[str | None] = []

    def queue(fail: str) -> Any:
        def hook() -> None:
            control.queued = dataclasses.replace(
                config, mode_options=(("fail", fail), ("strength", 3))
            )
        return hook

    def record() -> None:
        errors.append(control.snapshot().last_error)

    hooks = {30: queue("value"), 40: record, 60: queue("runtime"), 70: record}
    sink = ListSink()
    report = run_stream(HookSource(QUIET, hooks=hooks), config, OutputConfig(), sinks=[sink],
                        control=control)
    assert len(sink.images) == QUIET.total  # The run went on to the end.
    assert report.stats.reconfigurations == 0
    assert all(_painted(image) for image in sink.images)
    assert errors[0] == "this strength cannot be used"
    assert errors[1] is not None and "RuntimeError: plugin bug" in errors[1]
    assert control.applied is config
    assert control.snapshot().ended is True


@pytest.mark.usefixtures("fake_persons")
def test_run_time_option_change_keeps_or_replaces_the_mode(plugin: ModuleType) -> None:
    config = _plugin_config()
    components, _ = pipeline._build_components(FakeSource(), config)
    same, kwargs = pipeline._rebuild(
        components, dataclasses.replace(config, engine=dataclasses.replace(config.engine,
                                                                            lookahead=4))
    )
    assert same.mode is components.mode and "mode" not in kwargs
    stronger, kwargs = pipeline._rebuild(
        components, dataclasses.replace(config, mode_options=(("strength", 9),))
    )
    assert kwargs["mode"] is stronger.mode is not components.mode
    assert stronger.mode.options == {"strength": 9, "fail": "none"}


@pytest.mark.usefixtures("fake_persons")
def test_mode_without_a_model_does_not_break_the_web_publisher(plugin: ModuleType) -> None:
    config = _plugin_config()
    control = RunControl(config, (W, H))
    sink = ListSink()
    report = run_stream(FakeSource(Scene(total=30)), config, OutputConfig(), sinks=[sink],
                        control=control)
    assert report.stats.frames == 30
    final = control.snapshot()
    assert final.ended is True
    assert final.background.ready is False
    assert len(sink.images) == 30
    assert not hasattr(plugin.PaintMode, "model")


# ---------------------------------------------------------------------------
# Erase parity
# ---------------------------------------------------------------------------

_DETECTOR_TYPES = {
    "motion": [MotionDetector],
    "person": [PersonDetector],
    "both": [MotionDetector, PersonDetector],
}


@pytest.mark.usefixtures("fake_person_check")
@pytest.mark.parametrize("detector", ["motion", "person", "both"])
@pytest.mark.parametrize("strategy", ["running", "calibrate"])
@pytest.mark.parametrize("bg_diff", [True, False])
def test_erase_components_match_the_pre_change_build_for_online_backgrounds(
    detector: str, strategy: str, bg_diff: bool
) -> None:
    overrides: dict[str, Any] = {"detector": detector}
    if not bg_diff:
        overrides["background_diff"] = None
    config = _live_config(strategy, **overrides)
    components, bg_idx = pipeline._build_components(FakeSource(), config)
    model_type = (
        pipeline.RunningBackground if strategy == "running" else pipeline.CalibratedBackground
    )
    assert isinstance(components.background, model_type)
    assert bg_idx is None
    expected = _DETECTOR_TYPES[detector] + ([BackgroundDiffer] if bg_diff else [])
    assert [type(d) for d in components.detectors] == expected
    if bg_diff:
        assert components.differ is not None
        assert components.differ._model is components.background
    assert [type(s) for s in components.stages] == [RoiOverlay]
    assert components.roi_stage.roi is None  # type: ignore[union-attr]
    assert type(components.mode) is EraseMode
    assert components.mode.model is components.background  # type: ignore[attr-defined]
    assert components.mode.require_background_diff is bg_diff  # type: ignore[attr-defined]
    assert components.refiners == ()


@pytest.mark.usefixtures("fake_person_check")
def test_erase_gets_the_warmup_roi_stage_on_a_live_source() -> None:
    config = _live_config("running", roi=RoiConfig(mode="auto"))
    components, _ = pipeline._build_components(FakeSource(), config)
    assert [type(s) for s in components.stages] == [WarmupRoiOverlay]


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Static textured background; a green block crosses frames 30 to 39 of 50."""
    path = tmp_path_factory.mktemp("clip") / "synthetic.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert writer.isOpened()
    try:
        for i in range(50):
            frame = BACKGROUND.copy()
            if 30 <= i < 40:
                frame[100:180, 20 + (i - 30) * 12 : 70 + (i - 30) * 12] = SUBJECT
            writer.write(frame)
    finally:
        writer.release()
    return path


@pytest.mark.parametrize("bg_diff", [True, False])
def test_erase_components_match_the_pre_change_build_for_the_locked_image(
    clip: Path, bg_diff: bool
) -> None:
    overrides: dict[str, Any] = {} if bg_diff else {"background_diff": None}
    config = PetrifyConfig(encode=ENCODE, detector="both",
                           roi=RoiConfig(mode="manual", box=ROI_BOX), **overrides)
    source = FileSource(clip)
    try:
        components, bg_idx = pipeline._build_components(source, config)
    finally:
        source.close()
    assert isinstance(components.background, np.ndarray)
    assert bg_idx is not None
    expected = [MotionDetector, PersonDetector] + ([BackgroundDiffer] if bg_diff else [])
    assert [type(d) for d in components.detectors] == expected
    if bg_diff:
        assert components.differ is not None and components.differ._model is None
    assert [type(s) for s in components.stages] == [RoiOverlay]
    assert components.roi_stage.roi == ROI_BOX  # type: ignore[union-attr]
    assert type(components.mode) is EraseMode
    assert components.mode.background is components.background  # type: ignore[attr-defined]
    assert components.mode.require_background_diff is False  # type: ignore[attr-defined]
    assert components.refiners == ()
