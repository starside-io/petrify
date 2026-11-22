"""End-to-end runs with the real YuNet face, DNN person and segmentation models.

Skipped unless ``PETRIFY_MODEL_DIR`` names a directory holding the three model
files, and, like the golden harness, unless the demo clips are in ``data/``.
"""

from __future__ import annotations

import dataclasses
import itertools
import os
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from petrify.cli import _build_run_parser, _resolve_run
from petrify.core.config import PetrifyConfig
from petrify.core.engine import FrameTrace
from petrify.core.frame import Frame
from petrify.models import KNOWN_MODELS, MODEL_DIR_ENV
from petrify.pipeline import OutputConfig, run_stream
from petrify.sources import FileSource
from tests.golden.cases import clip_path
from tests.test_detector_wiring import plugin  # noqa: F401 (pytest fixture)
from tests.test_run_stream import ListSink


def _model_dir() -> Path | None:
    value = os.environ.get(MODEL_DIR_ENV, "")
    if not value:
        return None
    directory = Path(value).expanduser()
    if all((directory / spec.filename).is_file() for spec in KNOWN_MODELS.values()):
        return directory
    return None


needs_models = pytest.mark.skipif(
    _model_dir() is None,
    reason=f"model files not found: set {MODEL_DIR_ENV} to the directory that holds "
    + ", ".join(spec.filename for spec in KNOWN_MODELS.values()),
)

CAMERA2 = clip_path("camera2.mp4")
CAMERA = clip_path("camera.mp4")


def needs_clip(path: Path) -> pytest.MarkDecorator:
    return pytest.mark.skipif(not path.exists(), reason=f"demo clip not available: {path}")


class FirstFrames(FileSource):
    """A video file source that yields only its first ``limit`` frames.

    The ROI scan and the locked pre-scan still see the whole clip, as in a full run.
    """

    def __init__(self, path: Path, limit: int) -> None:
        super().__init__(path)
        self.limit = limit

    def __iter__(self) -> Iterator[Frame]:
        return itertools.islice(super().__iter__(), self.limit)


def _config(source: Path, *flags: str) -> PetrifyConfig:
    parser = _build_run_parser()
    args = parser.parse_args(["--source", str(source), "--output", "unused.mp4", *flags])
    return _resolve_run(parser, args).config


def _frozen(config: PetrifyConfig, limit: int) -> tuple[set[int], Any]:
    traces: list[FrameTrace] = []
    report = run_stream(FirstFrames(CAMERA2, limit), config, OutputConfig(),
                        sinks=[ListSink()], observer=traces.append)
    assert report.stats.frames == limit
    assert report.stats.failed_closed == 0
    return {t.index for t in traces if t.final}, report


@needs_models
@needs_clip(CAMERA2)
def test_erase_with_the_dnn_freezes_at_least_the_frames_hog_freezes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MODEL_DIR_ENV, str(_model_dir()))
    hog, _ = _frozen(_config(CAMERA2, "--detector", "person"), 60)
    dnn, _ = _frozen(_config(CAMERA2, "--detector", "person", "--person-detector", "dnn"), 60)
    assert hog <= dnn, sorted(hog - dnn)
    # With the CLI defaults the 50-frame cold start and the background differ freeze
    # most of these frames whatever the person detector does. Without them the person
    # detector alone decides: HOG finds nobody here, the DNN finds the people.
    hog, _ = _frozen(_detector_only(_config(CAMERA2, "--detector", "person", "--no-bg-diff")), 60)
    dnn, _ = _frozen(_detector_only(_config(
        CAMERA2, "--detector", "person", "--no-bg-diff", "--person-detector", "dnn")), 60)
    assert hog <= dnn, sorted(hog - dnn)
    assert len(dnn) > len(hog)


def _detector_only(config: PetrifyConfig) -> PetrifyConfig:
    return dataclasses.replace(
        config, engine=dataclasses.replace(config.engine, cold_start_frames=0)
    )


def _plugin_run(mode: str, source: Path, limit: int, *flags: str) -> Any:
    config = _config(source, "--mode", mode, "--no-roi", *flags)
    return run_stream(FirstFrames(source, limit), config, OutputConfig(), sinks=[ListSink()])


@needs_models
@needs_clip(CAMERA)
def test_a_faces_mode_receives_faces_with_landmarks(
    plugin: ModuleType,  # noqa: F811 (pytest fixture)
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MODEL_DIR_ENV, str(_model_dir()))
    report = _plugin_run("landmarks", CAMERA, 150, "--faces")
    assert report.stats.failed_closed == 0
    renders = [d for name, _, d in plugin.RENDERS if name == "landmarks"]
    assert len(renders) == 150
    assert all(d.faces is not None for d in renders)
    faces = [face for d in renders for face in d.faces]
    assert faces, "YuNet found no face in the first 150 frames of camera.mp4"
    assert all(len(face.landmarks) == 5 for face in faces)


@needs_models
@needs_clip(CAMERA2)
def test_a_mask_mode_receives_a_mask(
    plugin: ModuleType,  # noqa: F811 (pytest fixture)
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MODEL_DIR_ENV, str(_model_dir()))
    report = _plugin_run("mask", CAMERA2, 60, "--segment")
    assert report.stats.failed_closed == 0
    renders = [d for name, _, d in plugin.RENDERS if name == "mask"]
    assert len(renders) == 60
    for d in renders:
        assert isinstance(d.mask, np.ndarray)
        assert d.mask.dtype == np.bool_ and d.mask.shape == (480, 640)
    assert any(d.mask.any() for d in renders)


@needs_models
@needs_clip(CAMERA2)
def test_the_dnn_sample_check_runs_on_a_real_running_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MODEL_DIR_ENV, str(_model_dir()))
    config = _config(CAMERA2, "--background", "running", "--background-check", "hog,dnn")
    assert config.background.sample_checkers == ("hog", "dnn")
    report = run_stream(FirstFrames(CAMERA2, 60), config, OutputConfig(), sinks=[ListSink()])
    assert report.stats.frames == 60


@needs_models
@needs_clip(CAMERA)
def test_the_dnn_on_1080p_finds_a_near_person_and_misses_distant_ones() -> None:
    """Pin the size limit stated in the --person-detector help and the startup warning.

    The DNN sees the frame scaled down to 416 px, so a person far down the 1080p
    corridor is about 13 px wide there and is missed, while HOG still fires. The
    background differ covers that gap only while it is enabled. If the misses
    below stop happening, the stated limit (about 110 px tall on 1080p) is stale.
    """
    from petrify.core.config import PersonDnnConfig, PersonHogConfig
    from petrify.detectors import PersonDetector, PersonDnnDetector

    directory = _model_dir()
    assert directory is not None
    dnn = PersonDnnDetector.from_config(PersonDnnConfig(), [directory])
    hog = PersonDetector.from_config(PersonHogConfig())
    source = FileSource(CAMERA)
    try:
        near = source.read_frame(350)
        distant = [source.read_frame(index) for index in (40, 1100)]
    finally:
        source.close()
    assert near is not None and all(image is not None for image in distant)
    boxes = dnn.detect_boxes(near)
    assert boxes, "the DNN found no person on frame 350, where the person is large"
    assert max(box.h for box in boxes) > 110
    for image in distant:
        assert hog.is_person(image)  # A person is there.
        assert dnn.detect_boxes(image) == ()  # The known gap: too small for the DNN.
