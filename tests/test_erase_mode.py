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

"""Tests for EraseMode."""

from __future__ import annotations

import numpy as np
import pytest

from petrify.background import LockedBackground, RunningBackground
from petrify.core import (
    BackgroundModel,
    BackgroundNotReady,
    DetectionKind,
    Detections,
    Frame,
    Mode,
    ModeContext,
    ModeFactory,
    PetrifyConfig,
    SourceInfo,
    build_mode,
    load_mode,
)
from petrify.modes import ERASE, EraseMode
from petrify.pipeline import _background_differ, _requires_differ

W, H = 160, 120


def _frame(image: np.ndarray) -> Frame:
    h, w = image.shape[:2]
    info = SourceInfo(fps=25.0, width=w, height=h, frame_count=None, seekable=False, live=False)
    return Frame(image=image, index=0, timestamp=0.0, source=info)


def _bg() -> np.ndarray:
    return np.full((H, W, 3), 50, dtype=np.uint8)


def _live() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (H, W, 3), dtype=np.uint8)


def test_satisfies_mode() -> None:
    mode = EraseMode(_bg())
    assert isinstance(mode, Mode)
    assert mode.name == "erase"
    assert mode.requires == frozenset(
        {DetectionKind.MOTION, DetectionKind.PERSONS, DetectionKind.BACKGROUND_DIFF}
    )


def test_present_returns_background_copy() -> None:
    bg = _bg()
    mode = EraseMode(bg)
    out = mode.render(_frame(_live()), Detections(), subject_present=True)
    assert np.array_equal(out, bg)
    assert out is not bg
    out[:] = 0
    assert np.array_equal(mode.background, _bg())


def test_absent_returns_frame_image_itself() -> None:
    live = _live()
    out = EraseMode(_bg()).render(_frame(live), Detections(motion=True), subject_present=False)
    assert out is live


def test_shape_mismatch_raises() -> None:
    mode = EraseMode(_bg())
    wrong = np.zeros((H, W + 1, 3), dtype=np.uint8)
    for present in (True, False):
        with pytest.raises(ValueError, match="does not match background shape"):
            mode.render(_frame(wrong), Detections(), subject_present=present)


@pytest.mark.parametrize("shape", [(H, W), (H, W, 1), (H, W, 4)])
def test_background_must_be_three_channel(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="3-channel"):
        EraseMode(np.zeros(shape, dtype=np.uint8))


# ----------------------------------------------------------------------
# With a BackgroundModel
# ----------------------------------------------------------------------


class _SpyModel:
    """Model that becomes ready after ``ready_after`` updates and records every update."""

    def __init__(self, ready_after: int, image: np.ndarray) -> None:
        self.version = 0
        self.ready_after = ready_after
        self.image = image
        self.updates: list[tuple[Frame, bool]] = []

    def current(self) -> np.ndarray | None:
        return self.image if len(self.updates) >= self.ready_after else None

    def update(self, frame: Frame, subject_present: bool) -> None:
        self.updates.append((frame, subject_present))


def test_ndarray_is_wrapped_in_locked_background() -> None:
    bg = _bg()
    mode = EraseMode(bg)
    assert isinstance(mode.model, LockedBackground)
    assert isinstance(mode.model, BackgroundModel)
    assert mode.background is bg


def test_not_ready_raises_for_every_frame_and_still_updates() -> None:
    model = _SpyModel(ready_after=4, image=_bg())
    mode = EraseMode(model)
    frames = [_frame(_live()) for _ in range(4)]
    for i, (frame, present) in enumerate(zip(frames, (False, True, False, True))):
        with pytest.raises(BackgroundNotReady):
            mode.render(frame, Detections(), subject_present=present)
        assert len(model.updates) == i + 1
        assert model.updates[-1][0] is frame
        assert model.updates[-1][1] is present
    assert issubclass(BackgroundNotReady, RuntimeError)


def test_ready_model_renders_and_updates_once_per_frame() -> None:
    bg = _bg()
    model = _SpyModel(ready_after=0, image=bg)
    mode = EraseMode(model)
    live = _live()
    frozen = mode.render(_frame(live), Detections(), subject_present=True)
    assert np.array_equal(frozen, bg)
    assert frozen is not bg
    frozen[:] = 0
    assert np.array_equal(bg, _bg())
    assert len(model.updates) == 1
    passed = mode.render(_frame(live), Detections(), subject_present=False)
    assert passed is live
    assert len(model.updates) == 2
    assert [present for _, present in model.updates] == [True, False]


def test_model_render_uses_latest_background() -> None:
    model = _SpyModel(ready_after=0, image=_bg())
    mode = EraseMode(model)
    model.image = np.full((H, W, 3), 99, dtype=np.uint8)
    out = mode.render(_frame(_live()), Detections(), subject_present=True)
    assert np.all(out == 99)


def test_model_shape_mismatch_raises() -> None:
    mode = EraseMode(_SpyModel(ready_after=0, image=_bg()))
    wrong = np.zeros((H, W + 1, 3), dtype=np.uint8)
    for present in (True, False):
        with pytest.raises(ValueError, match="does not match background shape"):
            mode.render(_frame(wrong), Detections(), subject_present=present)


def test_running_background_model_in_erase_mode() -> None:
    model = RunningBackground(window=3, interval=1, min_samples=2)
    mode = EraseMode(model)
    bg = _bg()
    for _ in range(2):  # The second update makes the model ready.
        with pytest.raises(BackgroundNotReady):
            mode.render(_frame(bg.copy()), Detections(), subject_present=False)
    assert model.version == 1
    out = mode.render(_frame(bg.copy()), Detections(), subject_present=False)
    assert np.array_equal(out, bg)
    assert model.version == 2
    frozen = mode.render(_frame(_live()), Detections(), subject_present=True)
    assert np.array_equal(frozen, bg)
    assert model.version == 2  # Subject frames are not sampled.


@pytest.mark.parametrize("require", [False, True])
@pytest.mark.parametrize("other_fired", [False, True])
@pytest.mark.parametrize("bg_diff", [None, False, True])
@pytest.mark.parametrize("present", [False, True])
def test_require_background_diff(
    require: bool, other_fired: bool, bg_diff: bool | None, present: bool
) -> None:
    bg = _bg()
    model = _SpyModel(ready_after=0, image=bg)
    mode = EraseMode(model, require_background_diff=require)
    live = _live()
    detections = Detections(motion=other_fired, background_diff=bg_diff)
    frame = _frame(live)
    unvouched = bg_diff is None and not other_fired
    if require and unvouched:
        with pytest.raises(BackgroundNotReady):
            mode.render(frame, detections, subject_present=present)
    else:
        out = mode.render(frame, detections, subject_present=present)
        if present:
            assert np.array_equal(out, bg)
        else:
            assert out is live
    assert len(model.updates) == 1
    assert model.updates[0][0] is frame
    assert model.updates[0][1] is present


def test_require_background_diff_defaults_off() -> None:
    assert EraseMode(_bg()).require_background_diff is False


# ----------------------------------------------------------------------
# The ERASE factory
# ----------------------------------------------------------------------

_INFO = SourceInfo(fps=25.0, width=W, height=H, frame_count=None, seekable=False, live=True)


def _ctx(
    background: np.ndarray | BackgroundModel, differ: bool = True
) -> ModeContext:
    config = PetrifyConfig() if differ else PetrifyConfig(background_diff=None)
    return ModeContext(_INFO, config, {}, background)


def test_erase_factory_attributes() -> None:
    assert isinstance(ERASE, ModeFactory)
    assert ERASE.name == "erase"
    assert ERASE.summary and "\n" not in ERASE.summary
    assert ERASE.options == ()
    assert ERASE.needs_background is True
    assert ERASE.roi_overlay is True
    assert ERASE.requires({}) == EraseMode.requires
    assert ERASE.mandatory({}) == frozenset()
    assert ERASE.weakened({}) == ()


def test_load_mode_erase_returns_factory() -> None:
    assert load_mode("erase") is ERASE


@pytest.mark.parametrize("model", [False, True])
@pytest.mark.parametrize("differ", [False, True])
def test_erase_factory_matches_pipeline_rule(model: bool, differ: bool) -> None:
    background: np.ndarray | BackgroundModel = (
        RunningBackground(window=3, interval=1, min_samples=1) if model else _bg()
    )
    ctx = _ctx(background, differ)
    mode = build_mode(ERASE, ctx)
    assert isinstance(mode, EraseMode)
    assert mode.require_background_diff is (model and differ)
    # The same decision as the pipeline, from the same background and config.
    pipeline_differ = _background_differ(background, ctx.config, live=False)
    assert mode.require_background_diff is _requires_differ(background, pipeline_differ)
    if model:
        assert mode.model is background
    else:
        assert mode.background is background


@pytest.mark.parametrize("model", [False, True])
def test_erase_factory_keeps_previous_when_nothing_changed(model: bool) -> None:
    background: np.ndarray | BackgroundModel = (
        RunningBackground(window=3, interval=1, min_samples=1) if model else _bg()
    )
    first = build_mode(ERASE, _ctx(background))
    assert build_mode(ERASE, _ctx(background), first) is first


def test_erase_factory_replaces_previous_when_background_changes() -> None:
    first = build_mode(ERASE, _ctx(_bg()))
    other = _bg()
    second = build_mode(ERASE, _ctx(other), first)
    assert second is not first
    assert isinstance(second, EraseMode) and second.background is other
    model = RunningBackground(window=3, interval=1, min_samples=1)
    third = build_mode(ERASE, _ctx(model), second)
    assert third is not second
    assert isinstance(third, EraseMode) and third.model is model


def test_erase_factory_replaces_previous_when_model_changes() -> None:
    # Same require_background_diff (True) on both sides: only the model differs.
    old = RunningBackground(window=3, interval=1, min_samples=1)
    new = RunningBackground(window=3, interval=1, min_samples=1)
    first = build_mode(ERASE, _ctx(old))
    second = build_mode(ERASE, _ctx(new), first)
    assert second is not first
    assert isinstance(second, EraseMode) and second.model is new
    assert second.require_background_diff is True


def test_erase_factory_replaces_previous_when_differ_requirement_changes() -> None:
    model = RunningBackground(window=3, interval=1, min_samples=1)
    with_differ = build_mode(ERASE, _ctx(model, differ=True))
    without = build_mode(ERASE, _ctx(model, differ=False), with_differ)
    assert without is not with_differ
    assert isinstance(without, EraseMode)
    assert without.model is model and without.require_background_diff is False
    again = build_mode(ERASE, _ctx(model, differ=True), without)
    assert again is not without
    assert isinstance(again, EraseMode) and again.require_background_diff is True


def test_erase_factory_keeps_previous_for_array_when_differ_toggles() -> None:
    # With a locked image the differ never makes the flag True, so nothing changes.
    bg = _bg()
    first = build_mode(ERASE, _ctx(bg, differ=True))
    assert build_mode(ERASE, _ctx(bg, differ=False), first) is first


def test_erase_factory_replaces_previous_of_another_type() -> None:
    class Other:
        name = "erase"
        requires = EraseMode.requires

        def render(self, frame: Frame, detections: Detections, subject_present: bool) -> np.ndarray:
            return frame.image.copy()

    bg = _bg()
    mode = build_mode(ERASE, _ctx(bg), Other())  # type: ignore[arg-type]
    assert isinstance(mode, EraseMode)


def test_erase_factory_needs_background() -> None:
    with pytest.raises(ValueError, match="needs a background"):
        ERASE.create(ModeContext(_INFO, PetrifyConfig(), {}))
