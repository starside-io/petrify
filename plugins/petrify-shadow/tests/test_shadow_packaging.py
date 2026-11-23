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

"""Packaging tests: the plugin is found, loaded and run through its real entry point."""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from unittest import mock

import numpy as np
import pytest
from conftest import ShadowEntryPoint, read_pyproject
from petrify_shadow import SHADOW, __version__
from petrify_shadow.mode import ShadowMode
from petrify_shadow.options import OPTIONS, parse_params
from petrify_shadow.regions import pad_person

import petrify.pipeline as pipeline
from petrify.core import Frame, PetrifyConfig
from petrify.core.config import mode_options_from
from petrify.core.detections import Box, DetectionKind, Detections
from petrify.core.frame import SourceInfo
from petrify.core.plugin import ModeContext, resolve_options
from petrify.core.registry import ENTRY_POINT_GROUP, build_mode, list_modes, load_mode
from petrify.pipeline import OutputConfig, run_stream
from tests.test_run_stream import FakeSource, ListSink, Scene, _live_config
from tests.test_run_stream import H as SCENE_H
from tests.test_run_stream import W as SCENE_W
from tests.test_run_stream_plugins import FakePersons

EXPECTED_TARGET = "petrify_shadow:SHADOW"
INFO = SourceInfo(fps=10.0, width=64, height=48, frame_count=None, seekable=False,
                  live=True, uri="test")


def _frame() -> Frame:
    rng = np.random.default_rng(0)
    image = rng.integers(1, 256, (INFO.height, INFO.width, 3), dtype=np.uint8)
    return Frame(image=image, index=0, timestamp=0.0, source=INFO)


# ---------------------------------------------------------------------------
# Entry point and registry
# ---------------------------------------------------------------------------


def test_fixture_entry_point_is_the_pyproject_line(shadow_entry_point: ShadowEntryPoint) -> None:
    project = read_pyproject()["project"]
    assert project["entry-points"][ENTRY_POINT_GROUP] == {"shadow": EXPECTED_TARGET}
    (found,) = [ep for ep in entry_points(group=ENTRY_POINT_GROUP) if ep.name == "shadow"]
    assert (found.name, found.value) == ("shadow", EXPECTED_TARGET)
    assert found.dist is not None
    assert (found.dist.metadata["Name"], found.dist.version) == (project["name"],
                                                                  project["version"])
    assert project["version"] == __version__


def test_list_modes_lists_shadow_as_a_selectable_plugin(
    shadow_entry_point: ShadowEntryPoint,
) -> None:
    (entry,) = [e for e in list_modes() if e.name == "shadow"]
    assert entry.problem is None
    assert entry.builtin is False
    assert entry.target == EXPECTED_TARGET
    assert (entry.distribution, entry.version) == ("petrify-shadow", "0.1.0")


def test_load_mode_returns_the_shadow_factory(shadow_entry_point: ShadowEntryPoint) -> None:
    assert load_mode("shadow") is SHADOW


def _default_ctx() -> ModeContext:
    options = resolve_options(OPTIONS, {})
    config = PetrifyConfig(mode="shadow", mode_options=mode_options_from(options))
    return ModeContext(INFO, config, options)


def test_build_mode_accepts_the_shadow_mode(shadow_entry_point: ShadowEntryPoint) -> None:
    factory = load_mode("shadow")
    ctx = _default_ctx()
    mode = build_mode(factory, ctx)
    assert isinstance(mode, ShadowMode)
    # The default style is facebox, a face style.
    assert mode.requires == frozenset({
        DetectionKind.MOTION, DetectionKind.PERSONS, DetectionKind.FACES, DetectionKind.LANDMARKS,
    })
    assert mode.detect_policy == "all"
    # The same options give equal params, so the previous mode is kept.
    assert build_mode(factory, ctx, mode) is mode


def test_factory_mode_and_entry_point_share_the_name_shadow(
    shadow_entry_point: ShadowEntryPoint,
) -> None:
    mode = SHADOW.create(_default_ctx())
    assert shadow_entry_point.name == SHADOW.name == mode.name == "shadow"


def test_factory_declares_a_region_mode_without_background_or_roi() -> None:
    assert SHADOW.options is OPTIONS
    assert SHADOW.needs_background is False
    assert SHADOW.roi_overlay is False


# ---------------------------------------------------------------------------
# Render rule
# ---------------------------------------------------------------------------


def test_render_returns_the_input_itself_without_a_subject() -> None:
    frame = _frame()
    assert ShadowMode(parse_params({})).render(frame, Detections(), False) is frame.image


def test_render_returns_a_new_black_frame_when_no_face_is_found() -> None:
    frame = _frame()
    frame.image.flags.writeable = False  # Any write to the input raises.
    before = frame.image.copy()
    out = ShadowMode(parse_params({})).render(frame, Detections(faces=()), True)
    assert type(out) is np.ndarray
    assert out.shape == frame.image.shape
    assert out.dtype == frame.image.dtype
    assert not out.any()
    assert not np.may_share_memory(out, frame.image)
    assert np.array_equal(frame.image, before)


# ---------------------------------------------------------------------------
# Through run_stream
# ---------------------------------------------------------------------------

SCENE = Scene(total=60, enter=20, stand=25, leave=35, gone=40)
PERSON = Box(x=100, y=90, w=60, h=80)


def test_run_stream_covers_the_person_box(
    shadow_entry_point: ShadowEntryPoint, caplog: pytest.LogCaptureFixture
) -> None:
    persons = FakePersons({i: PERSON for i in range(SCENE.enter, SCENE.gone)})
    # bodybox needs only person boxes, which the fake detector provides; the
    # default facebox would need the YuNet face detector.
    config = _live_config(
        "running", mode="shadow", detector="person", mode_options=(("styles", ("bodybox",)),)
    )
    src = FakeSource(SCENE)
    sink = ListSink()
    with (
        mock.patch.object(pipeline.PersonDetector, "from_config", return_value=persons),
        caplog.at_level(logging.INFO, logger="petrify.pipeline"),
    ):
        components, _ = pipeline._build_components(FakeSource(), config)
        report = run_stream(src, config, OutputConfig(), sinks=[sink])
    assert components.roi_stage is None
    assert components.stages == []
    assert any(r.getMessage().startswith("ROI: not used") for r in caplog.records)
    assert report.stats.frames == SCENE.total
    assert report.stats.failed_closed == 0
    assert len(sink.images) == SCENE.total
    for i, (out, given) in enumerate(zip(sink.images, src.yielded, strict=True)):
        # The fake box is the same on every frame but fresh (held 0) on each, so
        # it is never grown and stays a region however long it lasts.
        if SCENE.enter <= i < SCENE.gone:
            # The padded person box is filled black and the rest is shown. The
            # fake box does not follow the scene's subject: this checks the
            # mechanics end to end, not the detector.
            rect = pad_person(PERSON, SCENE_H, SCENE_W)
            assert rect is not None
            box = np.zeros((SCENE_H, SCENE_W), bool)
            box[rect.y0 : rect.y1, rect.x0 : rect.x1] = True
            assert not out[box].any(), f"frame {i}"
            assert np.array_equal(out[~box], given[~box]), f"frame {i}"
        else:
            # No person box: black while the subject is held, else unchanged.
            assert not out.any() or np.array_equal(out, given), f"frame {i}"
    # Frames well after the subject left (past the lookahead) pass through unchanged.
    assert np.array_equal(sink.images[-1], src.yielded[-1])
