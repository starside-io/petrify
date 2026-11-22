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

"""Contract tests of the shadow mode through the real registry, pipeline, CLI and control.

Every test registers the plugin with the ``shadow_entry_point`` fixture, so the
mode is found and loaded the way an installed plugin is. The detectors are
replaced by scripted fakes (their ``from_config`` is patched), so no model file
is read and every detection is known in advance.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any
from unittest import mock

import numpy as np
import pytest
from conftest import ShadowEntryPoint
from petrify_shadow import SHADOW
from petrify_shadow.factory import (
    BODYBOX_NEEDS_PERSONS,
    WEAK_DILATE,
    WEAK_NO_BODY,
    WEAK_PADDING,
    WEAK_REVERSIBLE,
    WEAK_SEGMENT_MISS,
    WEAK_STRENGTH,
)
from petrify_shadow.mode import ShadowMode
from petrify_shadow.options import OPTIONS, STYLE_NAMES, parse_params
from petrify_shadow.regions import Rect, held_growth, pad_face

import petrify.pipeline as pipeline
import petrify.sources
from petrify.cli import main
from petrify.core import Frame, PetrifyConfig
from petrify.core.config import (
    EngineConfig,
    FaceYuNetConfig,
    OptionValue,
    SegmentConfig,
    TrackerConfig,
    mode_options_from,
)
from petrify.core.detections import Box, DetectionKind, Detections, Face
from petrify.core.plugin import ModeContext, resolve_option_text, resolve_options
from petrify.core.registry import build_mode, list_modes, load_mode
from petrify.detectors import FaceYuNetDetector, PersonDnnDetector, PersonSegmenter
from petrify.detectors.motion import MotionDetector
from petrify.detectors.person_hog import PersonDetector
from petrify.pipeline import (
    MISSING_PROVIDER_ERROR,
    TRACKER_UNTRACKED_WARNING,
    OutputConfig,
    run_stream,
)
from petrify.server.control import RunControl
from tests.test_cli import _report, _StubSource
from tests.test_run_stream import FakeSource, H, HookSource, ListSink, Scene, W, _live_config

K = DetectionKind
FACES_FLAG = "the YuNet face detector with --faces"
MASK_FLAG = "the person segmenter with --segment"
EMPTY_STYLES_ERROR = "the shadow mode needs at least one style in 'styles'"
OUTLINE_ALONE_ERROR = (
    "the outline style does not hide anyone on its own: add another style to 'styles'"
)
ORIGIN = "petrify-shadow 0.1.0"

FACE_BOX = Box(x=120, y=60, w=40, h=48)
# One pixel to the right: a box that alternates between the two is never held.
FACE_BOX_SHIFTED = Box(x=121, y=60, w=40, h=48)
PERSON_BOX = Box(x=100, y=40, w=80, h=200)


def _missing(kind: str, provider: str) -> str:
    return MISSING_PROVIDER_ERROR.format(mode="shadow", kind=kind, provider=provider)


def _quiet(total: int) -> Scene:
    """A scene without a subject: every frame is the same background texture."""
    return Scene(total=total, enter=10_000, stand=10_000, leave=10_000, gone=10_000)


def _face(box: Box) -> Face:
    """A face with five landmarks inside its box, as YuNet reports them."""
    x, y, w, h = box.x, box.y, box.w, box.h
    points = (
        (x + 0.3 * w, y + 0.4 * h),
        (x + 0.7 * w, y + 0.4 * h),
        (x + 0.5 * w, y + 0.6 * h),
        (x + 0.35 * w, y + 0.8 * h),
        (x + 0.65 * w, y + 0.8 * h),
    )
    return Face(box=box, landmarks=points)


def _mask() -> np.ndarray:
    """A person mask away from the face boxes."""
    mask = np.zeros((H, W), dtype=bool)
    mask[150:230, 20:100] = True
    return mask


# ---------------------------------------------------------------------------
# Scripted fake detectors
# ---------------------------------------------------------------------------


class Scripted:
    """A fake detector: the scripted result for a frame index, else its empty result.

    A scripted exception is raised instead of returned.
    """

    stateful = False
    model_path = None  # Read by the pipeline's model detector cache when it logs.

    def __init__(
        self,
        provides: frozenset[DetectionKind],
        empty: Callable[[Frame], Detections],
    ) -> None:
        self.provides = provides
        self.empty = empty
        self.script: dict[int, Detections | Exception] = {}
        self.every: Callable[[int], Detections] | None = None
        self.calls: list[int] = []

    def detect(self, frame: Frame) -> Detections:
        self.calls.append(frame.index)
        item = self.script.get(frame.index)
        if isinstance(item, Exception):
            raise item
        if item is not None:
            return item
        if self.every is not None:
            return self.every(frame.index)
        return self.empty(frame)

    def reset(self) -> None:
        pass


@dataclass
class Fakes:
    motion: Scripted = field(default_factory=lambda: Scripted(
        frozenset({K.MOTION}), lambda frame: Detections(motion=False)))
    persons: Scripted = field(default_factory=lambda: Scripted(
        frozenset({K.PERSONS}), lambda frame: Detections(persons=())))
    faces: Scripted = field(default_factory=lambda: Scripted(
        frozenset({K.FACES, K.LANDMARKS}), lambda frame: Detections(faces=())))
    segment: Scripted = field(default_factory=lambda: Scripted(
        frozenset({K.MASK}),
        lambda frame: Detections(mask=np.zeros(frame.image.shape[:2], dtype=bool))))


@pytest.fixture
def fakes(shadow_entry_point: ShadowEntryPoint) -> Iterator[Fakes]:
    """Register the plugin and replace every detector the pipeline can build by a fake.

    HOG and the DNN both hand out the same person fake.
    """
    found = Fakes()
    with (
        mock.patch.object(MotionDetector, "from_config", return_value=found.motion),
        mock.patch.object(PersonDetector, "from_config", return_value=found.persons),
        mock.patch.object(PersonDnnDetector, "from_config", return_value=found.persons),
        mock.patch.object(FaceYuNetDetector, "from_config", return_value=found.faces),
        mock.patch.object(PersonSegmenter, "from_config", return_value=found.segment),
    ):
        yield found


def _config(
    styles: tuple[str, ...] = ("facebox",),
    *,
    detector: str = "both",
    faces: bool = True,
    segment: bool = False,
    lookahead: int = 0,
    tracker: TrackerConfig | None = None,
    **options: OptionValue,
) -> PetrifyConfig:
    """A live shadow config; no cold start and no lookahead unless asked for."""
    return _live_config(
        "running",
        mode="shadow",
        detector=detector,
        faces=FaceYuNetConfig() if faces else None,
        segment=SegmentConfig() if segment else None,
        tracker=tracker,
        engine=EngineConfig(lookahead=lookahead, cold_start_frames=0),
        mode_options=mode_options_from({"styles": styles, **options}),
    )


class _BuiltComponents:
    """Wraps ``pipeline._build_components`` to keep what it built."""

    def __init__(self) -> None:
        self.real = pipeline._build_components
        self.built: list[Any] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = self.real(*args, **kwargs)
        self.built.append(result[0])
        return result


@pytest.fixture
def components() -> Iterator[_BuiltComponents]:
    spy = _BuiltComponents()
    with mock.patch.object(pipeline, "_build_components", side_effect=spy):
        yield spy


class _RenderSpy:
    """Records the ShadowMode instance that rendered each frame."""

    def __init__(self) -> None:
        self.real = ShadowMode.render
        self.modes: list[ShadowMode] = []

    def __call__(self, mode: ShadowMode, frame: Frame, *args: Any) -> np.ndarray:
        self.modes.append(mode)
        return self.real(mode, frame, *args)


@pytest.fixture
def renders() -> Iterator[_RenderSpy]:
    spy = _RenderSpy()
    with mock.patch.object(ShadowMode, "render", autospec=True, side_effect=spy):
        yield spy


def _region(rect: Rect) -> np.ndarray:
    region = np.zeros((H, W), dtype=bool)
    region[rect.y0 : rect.y1, rect.x0 : rect.x1] = True
    return region


def _assert_only_covered(out: np.ndarray, given: np.ndarray, rect: Rect, label: str) -> None:
    """``rect`` is black in ``out`` and every other pixel is the input's."""
    region = _region(rect)
    assert not out[region].any(), label
    assert np.array_equal(out[~region], given[~region]), label


def _face_rect(box: Box, held: int = 0, padding: float = 0.3) -> Rect:
    rect = pad_face(box, padding, H, W, grow=held_growth(held))
    assert rect is not None
    return rect


def _control(config: PetrifyConfig) -> RunControl:
    return RunControl(config, (W, H), modes=list_modes(), factory=load_mode("shadow"))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_list_modes_shows_shadow_as_a_selectable_plugin_with_its_distribution(
    shadow_entry_point: ShadowEntryPoint,
) -> None:
    (entry,) = [e for e in list_modes() if e.name == "shadow"]
    assert entry.problem is None
    assert entry.builtin is False
    assert (entry.distribution, entry.version) == ("petrify-shadow", "0.1.0")
    assert entry.origin == ORIGIN


def test_load_mode_returns_the_shadow_factory_object(
    shadow_entry_point: ShadowEntryPoint,
) -> None:
    assert load_mode("shadow") is SHADOW


def _ctx(options: Mapping[str, OptionValue], detector: str = "both") -> ModeContext:
    resolved = resolve_options(OPTIONS, options)
    config = PetrifyConfig(mode="shadow", detector=detector,
                           mode_options=mode_options_from(resolved))
    info = FakeSource().info
    return ModeContext(info, config, resolved)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"styles": ("bodybox",)},
        {"styles": ("silhouette", "facebox")},
        {"styles": ("outline", "silhouette_color"), "color": "red", "dilate": 0.1},
        {"styles": ("gaussian",), "strength": "low", "padding": 0.0},
        {"styles": ("pixelate", "eyebar"), "strength": "medium", "padding": 1.0},
        {"styles": STYLE_NAMES, "color": "white"},
    ],
    ids=["defaults", "bodybox", "silhouette-facebox", "outline-with-fill", "gaussian-low",
         "pixelate-eyebar", "all-nine"],
)
def test_build_mode_through_the_registry_accepts_valid_option_sets(
    shadow_entry_point: ShadowEntryPoint, options: dict[str, OptionValue]
) -> None:
    mode = build_mode(load_mode("shadow"), _ctx(options))
    assert isinstance(mode, ShadowMode)
    assert mode.params == parse_params(options)


@pytest.mark.parametrize(
    ("styles", "message"),
    [((), EMPTY_STYLES_ERROR), (("outline",), OUTLINE_ALONE_ERROR)],
    ids=["empty", "outline-alone"],
)
def test_build_mode_through_the_registry_rejects_a_style_list_that_hides_no_one(
    shadow_entry_point: ShadowEntryPoint, styles: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError) as excinfo:
        build_mode(load_mode("shadow"), _ctx({"styles": styles}))
    assert str(excinfo.value) == message


# ---------------------------------------------------------------------------
# run_stream startup matrix
# ---------------------------------------------------------------------------

# Each style set with the detectors it needs: faces, segment, and the detector setting.
STARTUP = {
    "facebox": (("facebox",), True, False),
    "silhouette": (("silhouette",), False, True),
    "bodybox": (("bodybox",), False, False),
    "silhouette-facebox": (("silhouette", "facebox"), True, True),
    "bodybox-facebox": (("bodybox", "facebox"), True, False),
    "all-nine": (STYLE_NAMES, True, True),
}


@pytest.mark.parametrize("case", list(STARTUP))
def test_run_starts_without_roi_and_fails_no_frame_closed_when_the_providers_are_enabled(
    fakes: Fakes,
    components: _BuiltComponents,
    caplog: pytest.LogCaptureFixture,
    case: str,
) -> None:
    styles, faces, segment = STARTUP[case]
    config = _config(styles, faces=faces, segment=segment)
    src = FakeSource(_quiet(12))
    sink = ListSink()
    with caplog.at_level(logging.INFO, logger="petrify.pipeline"):
        report = run_stream(src, config, OutputConfig(), sinks=[sink])
    (built,) = components.built
    assert built.roi_stage is None
    assert built.stages == []
    assert "ROI: not used, the mode 'shadow' does not allow the overlay" in [
        r.getMessage() for r in caplog.records
    ]
    assert (built.faces is fakes.faces) is faces
    assert (built.segment is fakes.segment) is segment
    assert report.stats.frames == 12
    assert report.stats.failed_closed == 0
    # Nothing was detected, so every frame is the input unchanged.
    assert all(np.array_equal(o, i) for o, i in zip(sink.images, src.yielded, strict=True))


MISSING = [
    ("facebox", {"faces": False}, _missing("faces", FACES_FLAG)),
    ("silhouette", {"segment": False}, _missing("mask", MASK_FLAG)),
    ("bodybox", {"detector": "motion"}, BODYBOX_NEEDS_PERSONS),
    ("silhouette-facebox", {"faces": False}, _missing("faces", FACES_FLAG)),
    ("silhouette-facebox", {"segment": False}, _missing("mask", MASK_FLAG)),
    ("bodybox-facebox", {"faces": False}, _missing("faces", FACES_FLAG)),
    ("bodybox-facebox", {"detector": "motion"}, BODYBOX_NEEDS_PERSONS),
    ("all-nine", {"faces": False}, _missing("faces", FACES_FLAG)),
    ("all-nine", {"segment": False}, _missing("mask", MASK_FLAG)),
    ("all-nine", {"detector": "motion"}, BODYBOX_NEEDS_PERSONS),
]


@pytest.mark.parametrize(
    ("case", "change", "message"),
    MISSING,
    ids=[f"{case}-without-{next(iter(change))}" for case, change, _ in MISSING],
)
def test_run_with_a_provider_missing_fails_before_any_frame_with_the_one_line_error(
    fakes: Fakes, case: str, change: dict[str, Any], message: str
) -> None:
    styles, faces, segment = STARTUP[case]
    settings: dict[str, Any] = {"faces": faces, "segment": segment, "detector": "both"}
    settings.update(change)
    config = _config(styles, **settings)
    src = FakeSource(_quiet(12))
    sink = ListSink()
    with pytest.raises(ValueError) as excinfo:
        run_stream(src, config, OutputConfig(), sinks=[sink])
    assert str(excinfo.value) == message
    assert "\n" not in message
    assert src.yielded == []
    assert sink.opened == 0


# ---------------------------------------------------------------------------
# Output through run_stream
# ---------------------------------------------------------------------------


def test_frames_without_a_detection_are_unchanged_and_a_face_has_only_its_padded_box_covered(
    fakes: Fakes,
) -> None:
    for i in (5, 6, 7):
        fakes.faces.script[i] = Detections(faces=(_face(FACE_BOX),))
    src = FakeSource(_quiet(12))
    sink = ListSink()
    report = run_stream(src, _config(), OutputConfig(), sinks=[sink])
    assert report.stats.failed_closed == 0
    for i, (out, given) in enumerate(zip(sink.images, src.yielded, strict=True)):
        if i in (5, 6, 7):
            # The same box on consecutive frames counts as held, so its cover grows.
            _assert_only_covered(out, given, _face_rect(FACE_BOX, held=i - 5), f"frame {i}")
        else:
            assert np.array_equal(out, given), f"frame {i}"


def test_with_lookahead_gap_filled_frames_without_a_detection_of_their_own_are_black(
    fakes: Fakes,
) -> None:
    fakes.faces.script[10] = Detections(faces=(_face(FACE_BOX),))
    fakes.faces.script[13] = Detections(faces=(_face(FACE_BOX),))
    src = FakeSource(_quiet(20))
    sink = ListSink()
    report = run_stream(src, _config(lookahead=2), OutputConfig(), sinks=[sink])
    assert report.stats.failed_closed == 0
    assert len(sink.images) == 20
    gap_filled = {8, 9, 11, 12, 14, 15}  # Within 2 frames of a face, without one.
    for i, (out, given) in enumerate(zip(sink.images, src.yielded, strict=True)):
        if i in (10, 13):
            _assert_only_covered(out, given, _face_rect(FACE_BOX), f"frame {i}")
        elif i in gap_filled:
            assert not out.any(), f"frame {i}"
        else:
            assert np.array_equal(out, given), f"frame {i}"


def test_a_frame_whose_detector_raises_is_black_and_failed_closed_and_the_next_recovers(
    fakes: Fakes,
) -> None:
    fakes.faces.script[4] = RuntimeError("fake face detector failure")
    fakes.faces.script[5] = Detections(faces=(_face(FACE_BOX),))
    src = FakeSource(_quiet(8))
    sink = ListSink()
    report = run_stream(src, _config(), OutputConfig(), sinks=[sink])
    assert report.stats.failed_closed == 1
    assert len(sink.images) == 8
    assert not sink.images[4].any()
    _assert_only_covered(sink.images[5], src.yielded[5], _face_rect(FACE_BOX), "frame 5")
    for i in (0, 1, 2, 3, 6, 7):
        assert np.array_equal(sink.images[i], src.yielded[i]), f"frame {i}"


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


def _untracked(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "does not hold" in r.getMessage()]


def test_track_with_detector_motion_starts_with_the_faces_refiner_only_and_warns_for_persons(
    fakes: Fakes, components: _BuiltComponents, caplog: pytest.LogCaptureFixture
) -> None:
    config = _config(detector="motion", tracker=TrackerConfig(hold_frames=5))
    with caplog.at_level(logging.INFO, logger="petrify.pipeline"):
        report = run_stream(FakeSource(_quiet(6)), config, OutputConfig(), sinks=[ListSink()])
    assert report.stats.failed_closed == 0
    (built,) = components.built
    (refiner,) = built.refiners
    assert refiner.kind is K.FACES
    assert refiner.provides == frozenset({K.FACES, K.LANDMARKS})
    assert _untracked(caplog) == [
        TRACKER_UNTRACKED_WARNING.format(kind="persons", mode="shadow", detector="motion")
    ]


def test_track_with_detector_both_has_both_refiners_and_no_untracked_warning(
    fakes: Fakes, components: _BuiltComponents, caplog: pytest.LogCaptureFixture
) -> None:
    config = _config(detector="both", tracker=TrackerConfig(hold_frames=5))
    with caplog.at_level(logging.INFO, logger="petrify.pipeline"):
        report = run_stream(FakeSource(_quiet(6)), config, OutputConfig(), sinks=[ListSink()])
    assert report.stats.failed_closed == 0
    (built,) = components.built
    assert [r.kind for r in built.refiners] == [K.PERSONS, K.FACES]
    assert _untracked(caplog) == []


def test_a_face_missing_for_fewer_frames_than_the_hold_stays_covered_at_its_grown_last_box(
    fakes: Fakes,
) -> None:
    for i in (5, 6, 7, 11):
        fakes.faces.script[i] = Detections(faces=(_face(FACE_BOX),))
    config = _config(detector="motion", tracker=TrackerConfig(hold_frames=5))
    src = FakeSource(_quiet(12))
    sink = ListSink()
    report = run_stream(src, config, OutputConfig(), sinks=[sink])
    assert report.stats.failed_closed == 0
    for i, (out, given) in enumerate(zip(sink.images, src.yielded, strict=True)):
        if i >= 5:
            # Missed on 8 to 10 and held by the tracker at the same box, so the
            # mode keeps counting it as held and its cover keeps growing.
            _assert_only_covered(out, given, _face_rect(FACE_BOX, held=i - 5), f"frame {i}")
        else:
            assert np.array_equal(out, given), f"frame {i}"


# ---------------------------------------------------------------------------
# Live changes through RunControl
# ---------------------------------------------------------------------------


def _alternating_faces(index: int) -> Detections:
    box = FACE_BOX if index % 2 == 0 else FACE_BOX_SHIFTED
    return Detections(faces=(_face(box),))


def _expected(options: Mapping[str, OptionValue], frame: np.ndarray, index: int) -> np.ndarray:
    """What a fresh shadow mode renders for the scripted detections of frame ``index``."""
    detections = Detections(persons=(), faces=_alternating_faces(index).faces, mask=_mask())
    image = Frame(image=frame, index=index, timestamp=index / 10.0, source=FakeSource().info)
    return ShadowMode(parse_params(options)).render(image, detections, True)


def test_live_option_changes_take_effect_on_the_next_frame_without_failed_closed_frames(
    fakes: Fakes,
) -> None:
    fakes.faces.every = _alternating_faces
    fakes.segment.every = lambda index: Detections(mask=_mask())
    config = _config(("silhouette", "facebox"), detector="person", segment=True)
    control = _control(config)
    changes: dict[int, dict[str, OptionValue]] = {
        10: {"color": "white"},
        20: {"padding": 0.0},
        30: {"dilate": 0.1},
        40: {"styles": ["silhouette", "gaussian"]},
        50: {"strength": "low"},
    }
    hooks = {
        index: (lambda change=change: control.submit({"mode_options": change}))
        for index, change in changes.items()
    }
    src = HookSource(_quiet(60), hooks=hooks)
    sink = ListSink()
    report = run_stream(src, config, OutputConfig(), sinks=[sink], control=control)
    assert report.stats.failed_closed == 0
    assert report.stats.reconfigurations == len(changes)
    assert control.snapshot().last_error is None
    options: dict[str, OptionValue] = {"styles": ("silhouette", "facebox")}
    for i, (out, given) in enumerate(zip(sink.images, src.yielded, strict=True)):
        change = changes.get(i - 1)
        if change is not None:
            # A change submitted before frame k is applied after it: frame k + 1
            # is the first one drawn with it, and it does change that frame.
            before = _expected(options, given, i)
            options.update({k: tuple(v) if isinstance(v, list) else v
                            for k, v in change.items()})
            assert not np.array_equal(before, _expected(options, given, i)), f"frame {i}"
        assert np.array_equal(out, _expected(options, given, i)), f"frame {i}"


REJECTED = [
    ("face-style-without-faces", ("silhouette",), {"faces": False, "segment": True},
     {"mode_options": {"styles": ["facebox", "silhouette"]}}, _missing("faces", FACES_FLAG)),
    ("mask-style-without-segment", ("facebox",), {},
     {"mode_options": {"styles": ["facebox", "silhouette"]}}, _missing("mask", MASK_FLAG)),
    ("empty-styles", ("facebox",), {},
     {"mode_options": {"styles": []}}, EMPTY_STYLES_ERROR),
    ("outline-alone", ("facebox",), {"segment": True},
     {"mode_options": {"styles": ["outline"]}}, OUTLINE_ALONE_ERROR),
    ("motion-with-bodybox", ("bodybox",), {"detector": "person", "faces": False},
     {"detector": "motion"}, BODYBOX_NEEDS_PERSONS),
]


@pytest.mark.parametrize(
    ("styles", "settings", "patch", "message"),
    [case[1:] for case in REJECTED],
    ids=[case[0] for case in REJECTED],
)
def test_a_live_change_the_mode_cannot_run_is_rejected_and_the_previous_styles_keep_running(
    fakes: Fakes,
    renders: _RenderSpy,
    styles: tuple[str, ...],
    settings: dict[str, Any],
    patch: dict[str, Any],
    message: str,
) -> None:
    fakes.faces.every = _alternating_faces
    fakes.segment.every = lambda index: Detections(mask=_mask())
    fakes.persons.every = lambda index: Detections(persons=(PERSON_BOX,))
    config = _config(styles, **settings)
    control = _control(config)
    seen: list[Any] = []
    hooks = {5: lambda: control.submit(patch), 8: lambda: seen.append(control.snapshot())}
    sink = ListSink()
    report = run_stream(HookSource(_quiet(12), hooks=hooks), config, OutputConfig(),
                        sinks=[sink], control=control)
    (snapshot,) = seen
    assert snapshot.last_error == message
    assert control.applied is config
    assert report.stats.reconfigurations == 0
    assert report.stats.failed_closed == 0
    assert len(sink.images) == 12
    # One mode instance rendered every frame, with the styles the run started with.
    assert len({id(mode) for mode in renders.modes}) == 1
    assert renders.modes[-1].params.styles == frozenset(styles)
    (shown,) = [o for o in snapshot.to_dict()["mode"]["options"] if o["name"] == "styles"]
    assert shown["value"] == list(styles)


def _mode_options(snapshot: Any) -> dict[str, dict[str, Any]]:
    return {o["name"]: o for o in snapshot.to_dict()["mode"]["options"]}


def test_the_snapshot_lists_the_five_options_and_the_mode_weakened_sentences_and_follows_changes(
    fakes: Fakes,
) -> None:
    config = _config(("facebox",), segment=True)
    control = _control(config)
    # Without a factory the control lists only petrify's own weakened sentences.
    builtin = RunControl(config, (W, H)).snapshot().weakened
    seen: list[Any] = []
    change = {"styles": ["silhouette", "gaussian"], "padding": 0.1, "strength": "medium",
              "dilate": 0.0}
    hooks = {
        1: lambda: seen.append(control.snapshot()),
        3: lambda: control.submit({"mode_options": change}),
        6: lambda: seen.append(control.snapshot()),
    }
    report = run_stream(HookSource(_quiet(10), hooks=hooks), config, OutputConfig(),
                        sinks=[ListSink()], control=control)
    assert report.stats.failed_closed == 0
    before, after = seen
    assert _mode_options(before) == {
        "styles": {**_mode_options(before)["styles"], "kind": "choices", "value": ["facebox"],
                   "default": ["facebox"], "choices": list(STYLE_NAMES), "minimum": None,
                   "maximum": None, "runtime": True},
        "padding": {**_mode_options(before)["padding"], "kind": "float", "value": 0.3,
                    "default": 0.3, "choices": [], "minimum": 0.0, "maximum": 1.0,
                    "runtime": True},
        "strength": {**_mode_options(before)["strength"], "kind": "choice", "value": "high",
                     "default": "high", "choices": ["high", "medium", "low"], "minimum": None,
                     "maximum": None, "runtime": True},
        "color": {**_mode_options(before)["color"], "kind": "choice", "value": "black",
                  "default": "black",
                  "choices": ["black", "white", "gray", "red", "green", "blue"],
                  "minimum": None, "maximum": None, "runtime": True},
        "dilate": {**_mode_options(before)["dilate"], "kind": "float", "value": 0.01,
                   "default": 0.01, "choices": [], "minimum": 0.0, "maximum": 0.1,
                   "runtime": True},
    }
    assert [o["name"] for o in before.to_dict()["mode"]["options"]] == [
        "styles", "padding", "strength", "color", "dilate"]
    assert before.weakened == (*builtin, WEAK_NO_BODY)

    values = {name: o["value"] for name, o in _mode_options(after).items()}
    assert values == {"styles": ["silhouette", "gaussian"], "padding": 0.1,
                      "strength": "medium", "color": "black", "dilate": 0.0}
    assert after.weakened == (
        *builtin,
        WEAK_REVERSIBLE,
        WEAK_STRENGTH.format(v="medium"),
        WEAK_PADDING.format(v="0.1"),
        WEAK_DILATE.format(v="0"),
        WEAK_SEGMENT_MISS,
    )
    assert after.last_error is None


def test_equal_style_sets_in_another_order_keep_the_running_mode_and_its_held_faces(
    fakes: Fakes, renders: _RenderSpy
) -> None:
    fakes.faces.every = lambda index: Detections(faces=(_face(FACE_BOX),))
    fakes.segment.every = lambda index: Detections(mask=_mask())
    # The command line keeps the typed order, the page sends the option's order.
    typed = resolve_option_text(OPTIONS, ["styles=silhouette,facebox"])["styles"]
    assert typed == ("silhouette", "facebox")
    config = _config(typed, detector="person", segment=True)  # type: ignore[arg-type]
    control = _control(config)
    hooks = {5: lambda: control.submit({"mode_options": {"styles": ["facebox", "silhouette"]}})}
    src = HookSource(_quiet(12), hooks=hooks)
    sink = ListSink()
    report = run_stream(src, config, OutputConfig(), sinks=[sink], control=control)
    # The change was applied, not rejected, and replaced nothing in the engine.
    assert control.applied.mode_options_dict()["styles"] == ("facebox", "silhouette")
    assert control.snapshot().last_error is None
    assert report.stats.reconfigurations == 0
    assert report.stats.failed_closed == 0
    assert len({id(mode) for mode in renders.modes}) == 1
    # The stationary face keeps growing across the change: the held memory survived.
    for i, (out, given) in enumerate(zip(sink.images, src.yielded, strict=True)):
        region = _region(_face_rect(FACE_BOX, held=i))
        assert not out[region].any(), f"frame {i}"
    smaller = _region(_face_rect(FACE_BOX, held=0))
    grown = _region(_face_rect(FACE_BOX, held=6))
    ring = grown & ~smaller & ~_mask()
    assert ring.any()
    # A mode rebuilt at the change would start again from the ungrown cover on frame 6.
    assert np.array_equal(sink.images[6][ring], np.zeros_like(src.yielded[6][ring]))
    assert src.yielded[6][ring].any()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

IN = "in.mp4"
OUT = "out.mp4"


@pytest.fixture
def cli_run_stream(shadow_entry_point: ShadowEntryPoint) -> Iterator[mock.MagicMock]:
    with (
        mock.patch.object(pipeline, "run_stream", return_value=_report()) as patched,
        mock.patch.object(petrify.sources, "open_source", return_value=_StubSource()),
    ):
        yield patched


def _usage_error(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2
    return capsys.readouterr().err


def test_cli_list_modes_prints_shadow_with_its_plugin_origin(
    shadow_entry_point: ShadowEntryPoint, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "--list-modes"]) == 0
    assert f"  shadow (plugin {ORIGIN})" in capsys.readouterr().out.splitlines()


def test_cli_list_modes_with_mode_shadow_prints_the_five_options(
    shadow_entry_point: ShadowEntryPoint, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "--list-modes", "--mode", "shadow"]) == 0
    detail = capsys.readouterr().out.split("\n\n", 1)[1]
    helps = {spec.name: " ".join(spec.help.split()) for spec in OPTIONS}
    assert detail == (
        f"Mode shadow (plugin {ORIGIN}): Obscure faces and bodies while a subject is present.\n"
        "Options (set with --mode-option NAME=VALUE):\n"
        "  styles\n    kind: choices\n    default: facebox\n"
        f"    choices: {', '.join(STYLE_NAMES)}\n"
        f"    changeable while running: yes\n    help: {helps['styles']}\n"
        "  padding\n    kind: float\n    default: 0.3\n    range: 0.0 to 1.0\n"
        f"    changeable while running: yes\n    help: {helps['padding']}\n"
        "  strength\n    kind: choice\n    default: high\n    choices: high, medium, low\n"
        f"    changeable while running: yes\n    help: {helps['strength']}\n"
        "  color\n    kind: choice\n    default: black\n"
        "    choices: black, white, gray, red, green, blue\n"
        f"    changeable while running: yes\n    help: {helps['color']}\n"
        "  dilate\n    kind: float\n    default: 0.01\n    range: 0.0 to 0.1\n"
        f"    changeable while running: yes\n    help: {helps['dilate']}\n"
    )


def test_cli_mode_option_styles_reaches_the_config_as_a_tuple_in_typed_order(
    cli_run_stream: mock.MagicMock,
) -> None:
    argv = ["run", "--source", IN, "--output", OUT, "--mode", "shadow",
            "--mode-option", "styles=silhouette,facebox"]
    assert main(argv) == 0
    _, config, _ = cli_run_stream.call_args.args
    assert config.mode == "shadow"
    assert config.mode_options_dict() == {
        "styles": ("silhouette", "facebox"), "padding": 0.3, "strength": "high",
        "color": "black", "dilate": 0.01,
    }


@pytest.mark.parametrize(
    ("assignment", "message"),
    [
        ("padding=2", "option 'padding' must be at most 1.0"),
        ("styles=nope", "option 'styles' must be one of gaussian, pixelate"),
        ("color=pink", "option 'color' must be one of black, white, gray, red, green, blue"),
    ],
)
def test_cli_a_bad_mode_option_value_is_a_usage_error_naming_the_option(
    cli_run_stream: mock.MagicMock,
    capsys: pytest.CaptureFixture[str],
    assignment: str,
    message: str,
) -> None:
    argv = ["run", "--source", IN, "--output", OUT, "--mode", "shadow",
            "--mode-option", assignment]
    err = _usage_error(argv, capsys)
    assert "petrify run: error: --mode-option: " in err
    assert message in err
    cli_run_stream.assert_not_called()


def test_cli_logs_the_plugin_trust_warning_when_shadow_is_selected(
    cli_run_stream: mock.MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    assert main(["run", "--source", IN, "--output", OUT, "--mode", "shadow"]) == 0
    warnings = [r.getMessage() for r in caplog.records
                if r.levelno == logging.WARNING and "installed plugin" in r.getMessage()]
    assert warnings == [
        f"The mode 'shadow' comes from the installed plugin {ORIGIN}. It runs in the "
        "petrify process with petrify's permissions and receives the unprocessed frames, "
        "people included."
    ]
