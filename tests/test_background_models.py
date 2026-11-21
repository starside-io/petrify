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

"""Tests for the online background models and their end-to-end use in the engine."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator, Sequence

import numpy as np
import pytest

import petrify.background.running as running_mod
from petrify.background import (
    CalibratedBackground,
    LockedBackground,
    RunningBackground,
    frames_for,
)
from petrify.background.running import median_image
from petrify.core import (
    BackgroundDiffConfig,
    BackgroundModel,
    BackgroundNotReady,
    Detections,
    Engine,
    EngineConfig,
    Frame,
    FrameMeta,
    SourceInfo,
)
from petrify.detectors import BackgroundDiffer, MotionDetector
from petrify.modes import EraseMode

W, H = 160, 120
BLOCK = 60  # Side of the synthetic subject, large enough for the motion threshold.
INFO = SourceInfo(fps=25.0, width=W, height=H, frame_count=None, seekable=False, live=True)


def _texture(seed: int = 0) -> np.ndarray:
    """Static textured background that never reaches 255."""
    return np.random.default_rng(seed).integers(0, 200, (H, W, 3), dtype=np.uint8)


def _with_block(bg: np.ndarray, x: int, y: int = 30) -> np.ndarray:
    img = bg.copy()
    img[y : y + BLOCK, x : x + BLOCK] = 255
    return img


def _frame(image: np.ndarray, index: int = 0) -> Frame:
    return Frame(image=image, index=index, timestamp=index / INFO.fps, source=INFO)


def _has_block(image: np.ndarray) -> bool:
    return bool((image == 255).all(axis=2).any())


# ----------------------------------------------------------------------
# Protocol and LockedBackground
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [LockedBackground(_texture()), RunningBackground(), CalibratedBackground(frames=10)],
    ids=["locked", "running", "calibrate"],
)
def test_models_satisfy_protocol(model: BackgroundModel) -> None:
    assert isinstance(model, BackgroundModel)
    assert model.version == 0


def test_locked_background_is_fixed() -> None:
    bg = _texture()
    model = LockedBackground(bg)
    assert model.current() is bg
    for i in range(5):
        model.update(_frame(_texture(i + 1), i), subject_present=i % 2 == 0)
    assert model.current() is bg
    assert model.version == 0


# ----------------------------------------------------------------------
# median_image
# ----------------------------------------------------------------------


def test_median_image_odd_and_even_and_input_untouched() -> None:
    samples = np.array([[[[1, 2, 3]]], [[[9, 9, 9]]], [[[5, 0, 7]]]], dtype=np.uint8)
    before = samples.copy()
    assert median_image(samples).tolist() == [[[5, 2, 7]]]
    assert np.array_equal(samples, before)
    even = np.array([[[[1]]], [[[4]]], [[[2]]], [[[8]]]], dtype=np.uint8)
    assert median_image(even).tolist() == [[[2]]]  # Lower median.
    assert median_image(even).dtype == np.uint8


# ----------------------------------------------------------------------
# RunningBackground
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window": 0},
        {"interval": 0},
        {"min_samples": 0},
        {"window": 3, "min_samples": 4},
        {"window": 2.5},
        {"interval": True},
    ],
)
def test_running_rejects_bad_arguments(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RunningBackground(**kwargs)  # type: ignore[arg-type]


def test_running_ready_at_min_samples_and_respects_interval() -> None:
    model = RunningBackground(window=10, interval=3, min_samples=3)
    bg = _texture()
    # Candidates 0, 3, 6 are stored: ready after the 7th clean frame.
    for i in range(6):
        model.update(_frame(bg.copy(), i), subject_present=False)
        assert model.current() is None
        assert model.version == 0
    assert model.samples == 2
    model.update(_frame(bg.copy(), 6), subject_present=False)
    assert model.samples == 3
    assert model.version == 1
    current = model.current()
    assert current is not None
    assert np.array_equal(current, bg)


def test_running_ignores_subject_frames() -> None:
    model = RunningBackground(window=5, interval=1, min_samples=1)
    for i in range(10):
        model.update(_frame(_with_block(_texture(), 10), i), subject_present=True)
    assert model.samples == 0
    assert model.current() is None
    assert model.version == 0


def test_running_subject_frames_do_not_advance_interval() -> None:
    model = RunningBackground(window=5, interval=2, min_samples=1)
    model.update(_frame(_texture()), subject_present=False)  # Candidate 0: stored.
    model.update(_frame(_texture()), subject_present=True)
    model.update(_frame(_texture()), subject_present=False)  # Candidate 1: skipped.
    assert model.samples == 1
    model.update(_frame(_texture()), subject_present=False)  # Candidate 2: stored.
    assert model.samples == 2
    assert model.version == 2


def test_running_window_is_a_ring_buffer() -> None:
    model = RunningBackground(window=3, interval=1, min_samples=1)
    for level in (10, 20, 30, 40, 50):
        model.update(_frame(np.full((H, W, 3), level, dtype=np.uint8)), subject_present=False)
    assert model.samples == 3
    current = model.current()
    assert current is not None
    assert np.all(current == 40)  # Median of the last three: 30, 40, 50.
    assert model.version == 5


def test_running_does_not_mutate_returned_arrays() -> None:
    model = RunningBackground(window=3, interval=1, min_samples=1)
    model.update(_frame(np.full((H, W, 3), 10, dtype=np.uint8)), subject_present=False)
    first = model.current()
    assert first is not None
    kept = first.copy()
    for level in (100, 150, 200, 250):
        model.update(_frame(np.full((H, W, 3), level, dtype=np.uint8)), subject_present=False)
    assert model.current() is not first
    assert np.array_equal(first, kept)


def test_running_stores_copies() -> None:
    model = RunningBackground(window=3, interval=1, min_samples=1)
    image = np.full((H, W, 3), 10, dtype=np.uint8)
    model.update(_frame(image), subject_present=False)
    image[:] = 99
    model.update(_frame(np.full((H, W, 3), 10, dtype=np.uint8)), subject_present=False)
    current = model.current()
    assert current is not None
    assert np.all(current == 10)


def test_running_median_removes_transient_object() -> None:
    bg = _texture()
    model = RunningBackground(window=9, interval=1, min_samples=9)
    for i in range(9):
        image = _with_block(bg, 20 + 10 * i) if i in (1, 4, 7) else bg.copy()
        model.update(_frame(image, i), subject_present=False)
    current = model.current()
    assert current is not None
    assert not _has_block(current)
    assert np.array_equal(current, bg)


def test_running_shape_mismatch_raises() -> None:
    model = RunningBackground(window=3, interval=1, min_samples=1)
    model.update(_frame(_texture()), subject_present=False)
    wrong = np.zeros((H, W + 1, 3), dtype=np.uint8)
    for present in (False, True):
        with pytest.raises(ValueError, match="does not match"):
            model.update(_frame(wrong), subject_present=present)


# ----------------------------------------------------------------------
# RunningBackground with recompute_in_background
# ----------------------------------------------------------------------

THREAD_NAME = "petrify-running-background"
TIMEOUT = 5.0


def _recompute_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == THREAD_NAME and t.is_alive()]


class _GatedMedian:
    """Stands in for ``median_image``: each call waits for one ``release``."""

    def __init__(self) -> None:
        self.calls = 0
        self.max_alive = 0
        self.fail: Exception | None = None
        self.gated = True
        self._started = threading.Semaphore(0)
        self._release = threading.Semaphore(0)

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.max_alive = max(self.max_alive, len(_recompute_threads()))
        self._started.release()
        if self.gated:
            assert self._release.acquire(timeout=TIMEOUT), "recompute was never released"
        if self.fail is not None:
            raise self.fail
        return _REAL_MEDIAN(samples)

    def wait_started(self) -> None:
        assert self._started.acquire(timeout=TIMEOUT), "recompute did not start"

    def release(self) -> None:
        self._release.release()


_REAL_MEDIAN = median_image


@pytest.fixture
def gate(monkeypatch: pytest.MonkeyPatch) -> Iterator[_GatedMedian]:
    gated = _GatedMedian()
    monkeypatch.setattr(running_mod, "median_image", gated)
    yield gated
    # Never leave a recompute blocked or a thread running past the test.
    gated.gated = False
    for _ in range(10):
        gated.release()
    for thread in _recompute_threads():
        thread.join(TIMEOUT)
    assert _recompute_threads() == []


def _level(level: int) -> np.ndarray:
    return np.full((H, W, 3), level, dtype=np.uint8)


def _bg_model(window: int = 1) -> RunningBackground:
    return RunningBackground(
        window=window, interval=1, min_samples=1, recompute_in_background=True
    )


def test_background_update_does_not_wait_for_recompute(gate: _GatedMedian) -> None:
    model = _bg_model()
    model.update(_frame(_level(10)), subject_present=False)  # Returns while gated.
    gate.wait_started()
    assert model.current() is None
    assert model.version == 0
    assert not model.wait_idle(timeout=0.01)
    gate.release()
    assert model.wait_idle(timeout=TIMEOUT)
    current = model.current()
    assert current is not None
    assert np.all(current == 10)
    assert model.version == 1
    assert _recompute_threads() == []


def test_background_current_switches_only_when_recompute_finishes(gate: _GatedMedian) -> None:
    model = _bg_model()
    model.update(_frame(_level(10)), subject_present=False)
    gate.wait_started()
    gate.release()
    assert model.wait_idle(timeout=TIMEOUT)
    first = model.current()
    assert first is not None
    model.update(_frame(_level(20)), subject_present=False)
    gate.wait_started()
    assert model.current() is first
    assert model.version == 1
    gate.release()
    assert model.wait_idle(timeout=TIMEOUT)
    second = model.current()
    assert second is not None and second is not first
    assert np.all(second == 20)
    assert np.all(first == 10)  # The published array was never mutated.
    assert model.version == 2


def test_background_samples_during_recompute_trigger_one_follow_up(gate: _GatedMedian) -> None:
    model = _bg_model(window=3)
    model.update(_frame(_level(10)), subject_present=False)
    gate.wait_started()
    for level in (20, 30, 40):
        model.update(_frame(_level(level)), subject_present=False)
    assert len(_recompute_threads()) == 1
    gate.release()
    gate.wait_started()  # The single follow-up.
    assert gate.calls == 2
    assert model.version == 1
    gate.release()
    assert model.wait_idle(timeout=TIMEOUT)
    assert gate.calls == 2
    assert model.version == 2
    current = model.current()
    assert current is not None
    assert np.all(current == 30)  # Median of the newest three: 20, 30, 40.
    assert gate.max_alive == 1


def test_background_never_more_than_one_recompute_thread(gate: _GatedMedian) -> None:
    gate.gated = False
    model = _bg_model(window=5)
    for i in range(200):
        model.update(_frame(_level(i % 250)), subject_present=False)
        assert len(_recompute_threads()) <= 1
    assert model.wait_idle(timeout=TIMEOUT)
    assert gate.max_alive == 1
    assert 1 <= gate.calls <= 200
    assert model.version == gate.calls
    assert _recompute_threads() == []


def test_background_recompute_error_surfaces_on_next_update(gate: _GatedMedian) -> None:
    gate.fail = OSError("boom")
    model = _bg_model()
    model.update(_frame(_level(10)), subject_present=False)
    gate.wait_started()
    gate.release()
    assert model.wait_idle(timeout=TIMEOUT)
    assert model.current() is None
    for present in (False, True, False):  # Sticky: every later update raises.
        with pytest.raises(OSError, match="boom"):
            model.update(_frame(_level(20)), subject_present=present)
    assert gate.calls == 1


def test_background_recompute_error_fails_erase_mode_closed(gate: _GatedMedian) -> None:
    gate.gated = False
    model = _bg_model()
    mode = EraseMode(model)
    with pytest.raises(BackgroundNotReady):
        mode.render(_frame(_level(10)), Detections(), subject_present=False)
    assert model.wait_idle(timeout=TIMEOUT)
    gate.fail = OSError("boom")
    mode.render(_frame(_level(20)), Detections(), subject_present=False)
    assert model.wait_idle(timeout=TIMEOUT)
    for _ in range(3):
        with pytest.raises(OSError, match="boom"):
            mode.render(_frame(_level(30)), Detections(), subject_present=False)


def test_background_wait_idle_without_any_recompute() -> None:
    model = _bg_model()
    assert model.wait_idle(timeout=0.0)
    model.update(_frame(_level(10)), subject_present=True)  # Not a candidate.
    assert model.wait_idle()
    assert _recompute_threads() == []


def test_background_and_synchronous_modes_agree() -> None:
    rng = np.random.default_rng(3)
    images = [rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(40)]
    flags = [bool(x) for x in rng.integers(0, 2, 40)]
    sync = RunningBackground(window=7, interval=2, min_samples=3)
    threaded = RunningBackground(
        window=7, interval=2, min_samples=3, recompute_in_background=True
    )
    for i, (image, present) in enumerate(zip(images, flags)):
        sync.update(_frame(image, i), present)
        threaded.update(_frame(image, i), present)
    assert threaded.wait_idle(timeout=TIMEOUT)
    assert sync.samples == threaded.samples
    a, b = sync.current(), threaded.current()
    assert a is not None and b is not None
    assert np.array_equal(a, b)
    assert 1 <= threaded.version <= sync.version
    assert _recompute_threads() == []


# ----------------------------------------------------------------------
# Sample filter (RunningBackground)
# ----------------------------------------------------------------------

REJECT = 99  # Image level the recording filter rejects.


class _RecordingFilter:
    """Sample filter that records the level of every image it sees.

    Rejects images of level REJECT and raises ``error`` for images of level
    ``raise_level``.
    """

    def __init__(self, error: Exception | None = None, raise_level: int | None = None) -> None:
        self.seen: list[int] = []
        self.error = error
        self.raise_level = raise_level

    def __call__(self, image: np.ndarray) -> bool:
        level = int(image[0, 0, 0])
        self.seen.append(level)
        if self.error is not None and level == self.raise_level:
            raise self.error
        return level != REJECT


def _filter_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "sample filter raised" in r.getMessage()
    ]


@pytest.mark.parametrize("threaded", [False, True], ids=["sync", "threaded"])
def test_running_filter_called_once_per_would_be_stored_sample(threaded: bool) -> None:
    check = _RecordingFilter()
    model = RunningBackground(
        window=10,
        interval=3,
        min_samples=1,
        recompute_in_background=threaded,
        sample_filter=check,
    )
    for i in range(12):
        # Subject frames are not candidates and never reach the filter.
        model.update(_frame(_level(200), 2 * i), subject_present=True)
        model.update(_frame(_level(i), 2 * i + 1), subject_present=False)
    assert model.wait_idle(timeout=TIMEOUT)
    assert check.seen == [0, 3, 6, 9]  # Candidates 0, 3, 6 and 9 only.
    assert model.samples == 4
    assert model.rejected == 0
    assert _recompute_threads() == []


@pytest.mark.parametrize("threaded", [False, True], ids=["sync", "threaded"])
def test_running_rejected_samples_are_not_stored_and_trigger_no_recompute(threaded: bool) -> None:
    check = _RecordingFilter()
    model = RunningBackground(
        window=10,
        interval=1,
        min_samples=1,
        recompute_in_background=threaded,
        sample_filter=check,
    )
    for i in range(5):
        model.update(_frame(_level(REJECT), i), subject_present=False)
        assert _recompute_threads() == []
    assert model.wait_idle(timeout=0.0)
    assert model.samples == 0
    assert model.version == 0
    assert model.current() is None
    assert model.rejected == 5
    # Accepted samples work as usual; the rejected ones never enter the median.
    model.update(_frame(_level(10), 5), subject_present=False)
    assert model.wait_idle(timeout=TIMEOUT)
    model.update(_frame(_level(REJECT), 6), subject_present=False)
    assert model.wait_idle(timeout=TIMEOUT)
    assert model.samples == 1
    assert model.version == 1
    current = model.current()
    assert current is not None and np.all(current == 10)
    assert model.rejected == 6
    assert len(check.seen) == 7
    assert _recompute_threads() == []


def test_running_rejection_does_not_shift_the_interval() -> None:
    check = _RecordingFilter()
    model = RunningBackground(window=10, interval=2, min_samples=1, sample_filter=check)
    levels = [REJECT, 1, 2, 3, REJECT, 5, 6]
    for i, level in enumerate(levels):
        model.update(_frame(_level(level), i), subject_present=False)
    # Candidates 0, 2, 4 and 6 are offered; 0 and 4 are rejected.
    assert check.seen == [REJECT, 2, REJECT, 6]
    assert model.samples == 2
    assert model.rejected == 2


@pytest.mark.parametrize("threaded", [False, True], ids=["sync", "threaded"])
def test_running_filter_exception_rejects_and_warns_once(
    threaded: bool, caplog: pytest.LogCaptureFixture
) -> None:
    check = _RecordingFilter(error=RuntimeError("detector exploded"), raise_level=50)
    model = RunningBackground(
        window=10,
        interval=1,
        min_samples=2,
        recompute_in_background=threaded,
        sample_filter=check,
    )
    with caplog.at_level(logging.WARNING, logger="petrify.background.running"):
        for i in range(4):
            model.update(_frame(_level(50), i), subject_present=False)  # Does not raise.
        assert model.current() is None
        assert model.rejected == 4
        assert model.samples == 0
        for i in range(4, 6):
            model.update(_frame(_level(20), i), subject_present=False)
        assert model.wait_idle(timeout=TIMEOUT)
    warnings = _filter_warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "RuntimeError" in message and "detector exploded" in message
    assert model.rejected == 4
    assert model.samples == 2
    current = model.current()
    assert current is not None and np.all(current == 20)
    assert _recompute_threads() == []


def test_running_without_filter_has_no_rejections() -> None:
    model = RunningBackground(window=5, interval=1, min_samples=1)
    for i in range(3):
        model.update(_frame(_level(REJECT), i), subject_present=False)
    assert model.rejected == 0
    assert model.samples == 3


# ----------------------------------------------------------------------
# CalibratedBackground
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [{"frames": 0}, {"frames": 4}, {"frames": 10, "min_samples": 0}, {"frames": 40, "min_samples": 31}],
)
def test_calibrate_rejects_bad_arguments(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        CalibratedBackground(**kwargs)


def test_calibrate_readiness_and_remaining() -> None:
    bg = _texture()
    model = CalibratedBackground(frames=6)
    assert not model.ready
    assert model.remaining == 6
    for i in range(5):
        # The subject flag is ignored during calibration.
        model.update(_frame(bg.copy(), i), subject_present=i % 2 == 0)
        assert not model.ready
        assert model.current() is None
        assert model.remaining == 5 - i
        assert model.version == 0
    model.update(_frame(bg.copy(), 5), subject_present=True)
    assert model.ready
    assert model.remaining == 0
    assert model.version == 1
    current = model.current()
    assert current is not None
    assert np.array_equal(current, bg)


def test_calibrate_never_changes_after_ready() -> None:
    model = CalibratedBackground(frames=5)
    for i in range(5):
        model.update(_frame(np.full((H, W, 3), 10, dtype=np.uint8), i), subject_present=False)
    current = model.current()
    assert current is not None
    kept = current.copy()
    for i in range(20):
        model.update(_frame(np.full((H, W, 3), 200, dtype=np.uint8), i), subject_present=False)
    assert model.current() is current
    assert np.array_equal(current, kept)
    assert model.version == 1
    assert model.remaining == 0


def test_calibrate_keeps_at_most_30_evenly_spread_samples() -> None:
    frames = 100
    model = CalibratedBackground(frames=frames)
    assert len(model._picks) == 30
    picks = sorted(model._picks)
    assert picks[0] == 0
    assert picks[-1] == frames - 1
    gaps = np.diff(picks)
    assert gaps.min() >= 3 and gaps.max() <= 4
    # Pixel value = frame index: the median of the kept samples is a middle index.
    for i in range(frames):
        model.update(_frame(np.full((H, W, 3), i, dtype=np.uint8), i), subject_present=False)
    current = model.current()
    assert current is not None
    assert int(current[0, 0, 0]) == picks[14]


def test_calibrate_median_removes_minority_object() -> None:
    bg = _texture()
    model = CalibratedBackground(frames=7)
    for i in range(7):
        image = _with_block(bg, 10 + 15 * i) if i in (2, 5) else bg.copy()
        model.update(_frame(image, i), subject_present=False)
    current = model.current()
    assert current is not None
    assert np.array_equal(current, bg)


def test_calibrate_shape_mismatch_raises() -> None:
    model = CalibratedBackground(frames=5)
    model.update(_frame(_texture()), subject_present=False)
    with pytest.raises(ValueError, match="does not match"):
        model.update(_frame(np.zeros((H, W + 1, 3), dtype=np.uint8)), subject_present=False)



def _calibrate_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and "waiting for a scene without a person" in r.getMessage()
    ]


def test_calibrate_rejected_frames_do_not_advance() -> None:
    check = _RecordingFilter()
    model = CalibratedBackground(frames=5, sample_filter=check)
    fed = 0
    for n, level in enumerate([10, 20, 30, 40, 50]):
        for _ in range(3):
            model.update(_frame(_level(REJECT), fed), subject_present=False)
            fed += 1
            assert not model.ready
            assert model.remaining == 5 - n
        model.update(_frame(_level(level), fed), subject_present=False)
        fed += 1
        assert model.remaining == 4 - n
    assert model.ready
    assert model.version == 1
    assert model.rejected == 15
    current = model.current()
    assert current is not None and np.all(current == 30)  # Median of the accepted only.
    # Once ready the filter is no longer called.
    calls = len(check.seen)
    model.update(_frame(_level(REJECT), fed), subject_present=False)
    assert len(check.seen) == calls
    assert model.rejected == 15


def test_calibrate_subsamples_over_accepted_frames() -> None:
    frames = 100
    model = CalibratedBackground(frames=frames, sample_filter=lambda image: image[0, 0, 0] != 255)
    picks = sorted(model._picks)
    accepted = 0
    for i in range(3 * frames):
        if i % 3 == 0:
            model.update(_frame(np.full((H, W, 3), accepted, dtype=np.uint8), i), False)
            accepted += 1
        else:
            model.update(_frame(np.full((H, W, 3), 255, dtype=np.uint8), i), False)
        if model.ready:
            break
    assert accepted == frames
    assert model.rejected == 2 * (frames - 1)
    current = model.current()
    assert current is not None
    # Pixel value = accepted-frame index: the same picks as without a filter.
    assert int(current[0, 0, 0]) == picks[14]


def test_calibrate_warns_every_100_consecutive_rejections(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = CalibratedBackground(frames=5, sample_filter=_RecordingFilter())
    with caplog.at_level(logging.WARNING, logger="petrify.background.calibrate"):
        for i in range(99):
            model.update(_frame(_level(REJECT), i), subject_present=False)
        assert _calibrate_warnings(caplog) == []
        model.update(_frame(_level(REJECT), 99), subject_present=False)
        assert len(_calibrate_warnings(caplog)) == 1
        for i in range(100, 250):
            model.update(_frame(_level(REJECT), i), subject_present=False)
        assert len(_calibrate_warnings(caplog)) == 2
        # An accepted frame resets the consecutive count.
        model.update(_frame(_level(10), 250), subject_present=False)
        for i in range(251, 350):
            model.update(_frame(_level(REJECT), i), subject_present=False)
        assert len(_calibrate_warnings(caplog)) == 2
        model.update(_frame(_level(REJECT), 350), subject_present=False)
    warnings = _calibrate_warnings(caplog)
    assert len(warnings) == 3
    assert "100 consecutive" in warnings[0].getMessage()
    assert "200 consecutive" in warnings[1].getMessage()
    assert "100 consecutive" in warnings[2].getMessage()
    assert "4 accepted frames still needed" in warnings[2].getMessage()
    assert model.rejected == 350
    assert model.remaining == 4
    assert not model.ready


def test_calibrate_never_ready_when_everything_is_rejected() -> None:
    model = CalibratedBackground(frames=5, sample_filter=lambda image: False)
    for i in range(300):
        model.update(_frame(_level(10), i), subject_present=False)
    assert not model.ready
    assert model.current() is None
    assert model.version == 0
    assert model.remaining == 5
    assert model.rejected == 300


def test_calibrate_filter_exception_rejects_and_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    check = _RecordingFilter(error=ValueError("frame too small"), raise_level=50)
    model = CalibratedBackground(frames=5, sample_filter=check)
    with caplog.at_level(logging.WARNING, logger="petrify.background.running"):
        for i in range(10):
            model.update(_frame(_level(50), i), subject_present=False)  # Does not raise.
        assert not model.ready
        for i in range(10, 15):
            model.update(_frame(_level(20), i), subject_present=False)
    warnings = _filter_warnings(caplog)
    assert len(warnings) == 1
    assert "ValueError" in warnings[0].getMessage()
    assert "frame too small" in warnings[0].getMessage()
    assert model.rejected == 10
    assert model.ready
    current = model.current()
    assert current is not None and np.all(current == 20)


def test_calibrate_without_filter_has_no_rejections() -> None:
    model = CalibratedBackground(frames=5)
    for i in range(5):
        model.update(_frame(_level(REJECT), i), subject_present=False)
    assert model.ready
    assert model.rejected == 0


def test_frames_for() -> None:
    assert frames_for(3.0, 25.0) == 75
    assert frames_for(0.1, 10.0) == 5  # At least min_samples.
    assert frames_for(0.1, 10.0, min_samples=2) == 2
    assert frames_for(1.0, 29.97) == 30
    for seconds, fps in ((0.0, 25.0), (1.0, 0.0), (-1.0, 25.0)):
        with pytest.raises(ValueError):
            frames_for(seconds, fps)


# ----------------------------------------------------------------------
# End to end through the engine
# ----------------------------------------------------------------------


class _Recording:
    """Delegating model that records, per update, whether it was ready before it."""

    def __init__(self, inner: BackgroundModel) -> None:
        self.inner = inner
        self.not_ready: list[int] = []

    @property
    def version(self) -> int:
        return self.inner.version

    def current(self) -> np.ndarray | None:
        return self.inner.current()

    def update(self, frame: Frame, subject_present: bool) -> None:
        if self.inner.current() is None:
            self.not_ready.append(frame.index)
        self.inner.update(frame, subject_present)


class _ListSink:
    critical = True

    def __init__(self) -> None:
        self.images: list[np.ndarray] = []
        self.metas: list[FrameMeta] = []

    def open(self, info: SourceInfo) -> None:
        pass

    def write(self, image: np.ndarray, meta: FrameMeta) -> None:
        self.images.append(image.copy())
        self.metas.append(meta)

    def close(self) -> None:
        pass

    def abort(self) -> None:
        pass


def _stream(bg: np.ndarray, n: int, subject: Sequence[range]) -> list[np.ndarray]:
    """``n`` frames of ``bg`` with a block moving across it during ``subject`` ranges."""
    images = []
    for i in range(n):
        image = bg.copy()
        for r in subject:
            if i in r:
                image = _with_block(bg, 5 + 6 * (i - r.start))
        images.append(image)
    return images


LOOKAHEAD = 2


def _run(
    model: BackgroundModel, images: list[np.ndarray], *, require: bool = True
) -> tuple[_Recording, _ListSink, int]:
    recording = _Recording(model)
    sink = _ListSink()
    engine = Engine(
        detectors=[
            MotionDetector(),
            BackgroundDiffer.from_model(recording, BackgroundDiffConfig()),
        ],
        mode=EraseMode(recording, require_background_diff=require),
        stages=[],
        sinks=[sink],
        config=EngineConfig(lookahead=LOOKAHEAD, cold_start_frames=0),
    )
    engine.open(INFO)
    for i, image in enumerate(images):
        engine.step(_frame(image, i))
    engine.flush()
    stats = engine.close()
    assert stats.frames == len(images)
    return recording, sink, stats.failed_closed


def _black(sink: _ListSink) -> set[int]:
    return {i for i, out in enumerate(sink.images) if not out.any()}


def _assert_erased(
    bg: np.ndarray,
    images: list[np.ndarray],
    recording: _Recording,
    sink: _ListSink,
    failed: int,
    require: bool,
) -> None:
    not_ready = set(recording.not_ready)
    # With require_background_diff, the frames detected before the model was ready
    # but rendered after it (at most LOOKAHEAD of them) are black too.
    allowed_black = set(not_ready)
    if require and not_ready:
        last = max(not_ready)
        allowed_black |= set(range(last + 1, last + 1 + LOOKAHEAD))
    black = _black(sink)
    assert not_ready <= black
    assert black <= allowed_black
    assert failed == len(black)
    subject = {i for i, image in enumerate(images) if _has_block(image)}
    assert [m.index for m in sink.metas] == list(range(len(images)))
    for i, out in enumerate(sink.images):
        assert not _has_block(out), f"frame {i} leaked the subject"
        if i in black:
            continue
        near = any(abs(i - s) <= LOOKAHEAD for s in subject)
        if i in subject or near:
            assert np.array_equal(out, bg), f"frame {i} near the subject must show the background"
        else:
            assert np.array_equal(out, images[i])


@pytest.mark.parametrize("require", [False, True])
def test_engine_running_background_end_to_end(require: bool) -> None:
    bg = _texture()
    # A subject during warmup delays readiness; later ones must be erased.
    images = _stream(bg, 70, [range(3, 6), range(30, 42), range(55, 60)])
    model = RunningBackground(window=10, interval=2, min_samples=3)
    recording, sink, failed = _run(model, images, require=require)
    not_ready = recording.not_ready
    assert not_ready == list(range(len(not_ready)))  # A prefix of the stream.
    assert 5 <= len(not_ready) < 30
    if not require:
        assert failed == len(not_ready)
    _assert_erased(bg, images, recording, sink, failed, require)


@pytest.mark.parametrize("require", [False, True])
def test_engine_calibrated_background_end_to_end(require: bool) -> None:
    bg = _texture()
    calibration = 12
    images = _stream(bg, 50, [range(25, 35)])
    model = CalibratedBackground(frames=calibration)
    recording, sink, failed = _run(model, images, require=require)
    # Black during the whole calibration even though no subject is in view.
    assert recording.not_ready == list(range(calibration))
    if not require:
        assert failed == calibration
    assert model.ready
    _assert_erased(bg, images, recording, sink, failed, require)


# A still subject: the block appears and never moves again. MOG2 fires only on its
# first frame or two and does not fire when it leaves (the empty background is still
# in its mixture). The frames detected before the model is ready but rendered after
# it (the last LOOKAHEAD not-ready detections) are vouched for by no detector.


def _still(bg: np.ndarray, n: int, start: int, stop: int | None = None) -> list[np.ndarray]:
    end = n if stop is None else stop
    return [_with_block(bg, 50) if start <= i < end else bg.copy() for i in range(n)]


def _still_running(stop: int | None = None) -> tuple[np.ndarray, list[np.ndarray], BackgroundModel, int]:
    """Block from frame 12; the model becomes ready after rendering frame 17.

    Seven clean samples (frames 3 to 9) come before the block and nine are needed:
    the model becomes ready on two samples that already contain the block, and the
    clean samples keep the majority of the window afterwards.
    """
    bg = _texture()
    model = RunningBackground(window=30, interval=1, min_samples=9)
    return bg, _still(bg, 60, 12, stop), model, 18


def _still_calibrate(stop: int | None = None) -> tuple[np.ndarray, list[np.ndarray], BackgroundModel, int]:
    """Block from frame 8; four of the twelve calibration frames contain it."""
    bg = _texture()
    return bg, _still(bg, 50, 8, stop), CalibratedBackground(frames=12), 12


SCENARIOS = pytest.mark.parametrize(
    "scenario", [_still_running, _still_calibrate], ids=["running", "calibrate"]
)


@SCENARIOS
@pytest.mark.parametrize("require", [False, True])
def test_engine_still_subject_present_until_well_after_ready(scenario, require: bool) -> None:  # type: ignore[no-untyped-def]
    """A subject still from before readiness until the end never leaks.

    This holds with the flag off too: the frames in the gap are within LOOKAHEAD of
    the first frames the ready differ flags, so the gap fill freezes them.
    """
    bg, images, model, ready_at = scenario()
    recording, sink, failed = _run(model, images, require=require)
    assert max(recording.not_ready) + 1 == ready_at
    assert any(_has_block(images[i]) for i in recording.not_ready)  # Present before ready.
    assert np.array_equal(model.current(), bg)
    for i, out in enumerate(sink.images):
        assert not _has_block(out), f"frame {i} leaked the still subject"
    black = _black(sink)
    assert failed == len(black)
    gap = set(range(ready_at, ready_at + LOOKAHEAD))
    assert (gap <= black) is require
    for i in range(ready_at, len(images)):
        if i not in black:
            assert np.array_equal(sink.images[i], bg), f"frame {i} must show the background"


@SCENARIOS
@pytest.mark.parametrize("require", [False, True])
def test_engine_still_subject_leaving_at_ready(scenario, require: bool) -> None:  # type: ignore[no-untyped-def]
    """A still subject that steps out right when the model becomes ready.

    The gap frames still show it, and nothing after them fires (it left), so with
    ``require_background_diff=False`` they pass through. With True they are black.
    """
    _, _, _, ready_at = scenario()
    stop = ready_at + LOOKAHEAD  # Present in exactly the gap frames, gone after.
    bg, images, model, _ = scenario(stop)
    assert all(_has_block(images[i]) for i in range(ready_at, stop))
    recording, sink, failed = _run(model, images, require=require)
    assert max(recording.not_ready) + 1 == ready_at
    leaked = [i for i, out in enumerate(sink.images) if _has_block(out)]
    if not require:
        assert leaked == list(range(ready_at, stop))
        return
    assert leaked == []
    black = _black(sink)
    assert failed == len(black)
    assert set(range(ready_at, stop)) <= black
    for i in range(stop, len(images)):
        assert np.array_equal(sink.images[i], images[i])


# A subject in view from the very first frame, still, that leaves at frame
# PRESENT_UNTIL. MOG2 learns it as background, so the erase mode feeds those frames
# to the model as "no subject". Without a filter the model absorbs it and shows it
# on every frozen frame after it left (the known limitation); a filter that
# recognizes it keeps it out.

PRESENT_UNTIL = 20
STARTUP_FRAMES = 80


def _startup_running(sample_filter) -> BackgroundModel:  # type: ignore[no-untyped-def]
    return RunningBackground(window=10, interval=2, min_samples=3, sample_filter=sample_filter)


def _startup_calibrate(sample_filter) -> BackgroundModel:  # type: ignore[no-untyped-def]
    return CalibratedBackground(frames=12, sample_filter=sample_filter)


STARTUP_MODELS = pytest.mark.parametrize(
    "make", [_startup_running, _startup_calibrate], ids=["running", "calibrate"]
)


def _no_block(image: np.ndarray) -> bool:
    return not _has_block(image)


@STARTUP_MODELS
@pytest.mark.parametrize("require", [False, True])
def test_engine_subject_at_startup_without_filter_is_absorbed(make, require: bool) -> None:  # type: ignore[no-untyped-def]
    """Documents the known limitation: the subject is shown after it left."""
    bg = _texture()
    images = _still(bg, STARTUP_FRAMES, 0, PRESENT_UNTIL)
    model = make(None)
    _, sink, _ = _run(model, images, require=require)
    current = model.current()
    assert current is not None and _has_block(current)  # Baked into the background.
    leaked_after = [
        i for i in range(PRESENT_UNTIL, STARTUP_FRAMES) if _has_block(sink.images[i])
    ]
    assert leaked_after == list(range(PRESENT_UNTIL, STARTUP_FRAMES))


@STARTUP_MODELS
@pytest.mark.parametrize("require", [False, True])
def test_engine_subject_at_startup_with_filter_never_shown(make, require: bool) -> None:  # type: ignore[no-untyped-def]
    """Black until the empty scene was seen long enough, the live frame afterwards."""
    bg = _texture()
    images = _still(bg, STARTUP_FRAMES, 0, PRESENT_UNTIL)
    model = make(_no_block)
    recording, sink, failed = _run(model, images, require=require)
    for i, out in enumerate(sink.images):
        assert not _has_block(out), f"frame {i} leaked the subject"
    not_ready = recording.not_ready
    assert not_ready == list(range(len(not_ready)))  # A prefix of the stream.
    assert len(not_ready) > PRESENT_UNTIL  # Ready only on samples taken after it left.
    assert model.rejected > 0
    current = model.current()
    assert current is not None and np.array_equal(current, bg)
    black = _black(sink)
    first_live = len(not_ready) + (LOOKAHEAD if require else 0)
    assert black == set(range(first_live))
    assert failed == len(black)
    for i in range(first_live, STARTUP_FRAMES):
        assert np.array_equal(sink.images[i], images[i]), f"frame {i} must be the live frame"
