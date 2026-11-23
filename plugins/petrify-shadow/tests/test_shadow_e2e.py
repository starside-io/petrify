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

"""End-to-end runs of the shadow mode with the real detector models on the demo clips.

Skipped unless ``PETRIFY_MODEL_DIR`` names a directory holding the three model
files and the demo clips are in ``data/``, as in
``tests/test_detector_wiring_e2e.py``.

Each run goes through the real command line parser and :func:`run_stream` on a
frame range of a clip. The sink checks every output frame against its input as
it arrives, so a 1080p run never holds more than the lookahead in memory:

- an independent detector (a second instance of the same model, run on the
  input frame) says where the face, mask or person is;
- the frame is classified as untouched (equal to the input), black (all zero)
  or drawn (anything else).

The per-run counts are printed, not asserted. What is asserted: no frame failed
closed; every frame the engine passed through (the mode received no subject) is
byte-identical to its input; and on every frame that is not black, whatever the
independent detector found is covered.

Both clips show a subject (motion, a person box or mask pixels) on nearly every
frame, and each run starts with the 50-frame cold start, so on the ranges below
no frame passes through and the pass-through check has nothing to compare.
"""

from __future__ import annotations

import dataclasses
import itertools
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import petrify_shadow.mode as shadow_mode
import pytest
from conftest import ShadowEntryPoint
from petrify_shadow.regions import pad_face

from petrify.cli import _build_run_parser, _mode_options_config, _resolve_run
from petrify.core.config import FaceYuNetConfig, PersonDnnConfig, PetrifyConfig, SegmentConfig
from petrify.core.detections import Box
from petrify.core.engine import FrameTrace
from petrify.core.frame import Frame, FrameMeta, SourceInfo
from petrify.core.registry import load_mode
from petrify.detectors import PersonDnnDetector
from petrify.detectors.face_yunet import FaceYuNetDetector
from petrify.detectors.segment import PersonSegmenter
from petrify.models import MODEL_DIR_ENV
from petrify.pipeline import OutputConfig, run_stream
from petrify.sources import FileSource
from tests.test_detector_wiring_e2e import CAMERA, CAMERA2, _model_dir, needs_clip, needs_models

pytestmark = pytest.mark.models

# Frame ranges, from an independent pass of the three models over every frame.
# Their lengths keep the file near 90 seconds: the DNN person detector alone
# takes 50 to 100 ms a frame, and every check runs a second model instance.
# camera2.mp4 (640x480): the person is distant (under about 125 px tall) up to
# frame 250 and close (250 to 300 px tall, faces 20 to 30 px wide) from 440 on.
CAMERA2_DISTANT = range(120, 160)
CAMERA2_CLOSE = range(460, 500)
CAMERA2_BODYBOX = range(130, 180)
# camera.mp4 (1920x1080): YuNet finds a face on every frame from 101 to 764.
CAMERA_NOISE = range(300, 600)
CAMERA_FACEBOX = range(300, 360)

# Share of a face box's pixels that must differ from the input under noise. A
# noise pixel equals its input on all three channels with probability 1/256^3;
# the margin is for pixels a black orphan fill covers where the input is black.
NOISE_CHANGED_SHARE = 0.98


class ClipRange(FileSource):
    """Frames ``frames.start`` to ``frames.stop - 1`` of a clip, numbered from 0.

    Every yielded input image is kept until the sink has checked its output.
    Frames are decoded from the start of the clip, not seeked to, so each one is
    the frame a whole-clip run would see.
    """

    def __init__(self, path: Path, frames: range) -> None:
        super().__init__(path)
        self.frames = frames
        self.inputs: dict[int, np.ndarray] = {}

    def __iter__(self) -> Iterator[Frame]:
        clip = itertools.islice(super().__iter__(), self.frames.start, self.frames.stop)
        for index, frame in enumerate(clip):
            image = frame.image.copy()
            self.inputs[index] = image
            yield Frame(image=image, index=index, timestamp=frame.timestamp, source=frame.source)


@dataclasses.dataclass
class Verdict:
    """What the checking sink found on one output frame."""

    untouched: bool
    black: bool
    problems: list[str]


Check = Callable[[int, np.ndarray, np.ndarray], list[str]]
"""Return the coverage problems of an output frame, given (clip index, input, output)."""


class CheckingSink:
    """A list sink that keeps a verdict per frame instead of the frame itself."""

    critical = True

    def __init__(self, source: ClipRange, check: Check) -> None:
        self.source = source
        self.check = check
        self.verdicts: list[Verdict] = []

    def open(self, info: SourceInfo) -> None:
        pass

    def write(self, image: np.ndarray, meta: FrameMeta) -> None:
        assert meta.index == len(self.verdicts)
        given = self.source.inputs.pop(meta.index)
        black = not image.any()
        index = self.source.frames.start + meta.index
        problems = [] if black else self.check(index, given, image)
        self.verdicts.append(Verdict(np.array_equal(image, given), black, problems))

    def close(self) -> None:
        pass

    def abort(self) -> None:
        pass


def _config(source: Path, styles: str, *flags: str) -> PetrifyConfig:
    """Return the config ``petrify run`` builds for these flags and shadow styles."""
    parser = _build_run_parser()
    argv = ["--source", str(source), "--output", "unused.mp4", "--mode", "shadow",
            "--mode-option", f"styles={styles}", *flags]
    args = parser.parse_args(argv)
    config = _resolve_run(parser, args).config
    return _mode_options_config(parser, args, load_mode("shadow"), config)


def _run(name: str, source: Path, frames: range, config: PetrifyConfig, check: Check) -> None:
    """Run ``config`` on ``frames`` of ``source``, print the counts, assert the invariants."""
    clip = ClipRange(source, frames)
    sink = CheckingSink(clip, check)
    traces: list[FrameTrace] = []
    start = time.perf_counter()
    report = run_stream(clip, config, OutputConfig(), sinks=[sink], observer=traces.append)
    seconds = time.perf_counter() - start
    verdicts = sink.verdicts
    assert report.stats.frames == len(frames) == len(verdicts) == len(traces)
    untouched = sum(v.untouched for v in verdicts)
    black = sum(v.black for v in verdicts)
    print(
        f"\n{name}: frames {frames.start} to {frames.stop - 1}: untouched {untouched}, "
        f"black {black}, drawn {len(verdicts) - untouched - black} ({seconds:.1f} s)"
    )
    assert report.stats.failed_closed == 0
    for trace, verdict in zip(traces, verdicts, strict=True):
        assert not trace.failed_closed, f"frame {frames.start + trace.index} failed closed"
        if not trace.final:
            assert verdict.untouched, f"frame {frames.start + trace.index} passed but changed"
        assert not verdict.problems, (frames.start + trace.index, verdict.problems)


@pytest.fixture
def models(shadow_entry_point: ShadowEntryPoint, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Register the shadow mode and point the pipeline at the model directory."""
    directory = _model_dir()
    assert directory is not None
    monkeypatch.setenv(MODEL_DIR_ENV, str(directory))
    return directory


def _segmenter_check(directory: Path) -> Check:
    segmenter = PersonSegmenter.from_config(SegmentConfig(), [directory])

    def check(index: int, given: np.ndarray, out: np.ndarray) -> list[str]:
        frame = Frame(image=given, index=0, timestamp=0.0, source=None)  # type: ignore[arg-type]
        mask = segmenter.detect(frame).mask
        assert mask is not None
        # The silhouette, the facebox and the orphan fills are all black here.
        shown = int(np.count_nonzero(out[mask].any(axis=1)))
        return [f"{shown} of {int(mask.sum())} mask pixels not black"] if shown else []

    return check


_FACES: dict[tuple[str, int], tuple[Box, ...]] = {}
"""Independent YuNet face boxes per (clip name, frame index), shared by the runs."""


def _face_check(
    directory: Path, clip: Path, covered: Callable[[np.ndarray, np.ndarray], str | None]
) -> Check:
    detector = FaceYuNetDetector.from_config(FaceYuNetConfig(), [directory])

    def faces(index: int, given: np.ndarray) -> tuple[Box, ...]:
        key = (clip.name, index)
        if key not in _FACES:
            frame = Frame(image=given, index=0, timestamp=0.0, source=None)  # type: ignore[arg-type]
            _FACES[key] = tuple(face.box for face in detector.detect(frame).faces or ())
        return _FACES[key]

    def check(index: int, given: np.ndarray, out: np.ndarray) -> list[str]:
        problems = []
        for b in faces(index, given):
            region = np.s_[max(b.y, 0) : b.y + b.h, max(b.x, 0) : b.x + b.w]
            problem = covered(given[region], out[region])
            if problem is not None:
                problems.append(f"face {b}: {problem}")
        return problems

    return check


def _noise_covered(given: np.ndarray, out: np.ndarray) -> str | None:
    changed = float(np.mean((given != out).any(axis=2)))
    return None if changed >= NOISE_CHANGED_SHARE else f"only {changed:.3f} of pixels changed"


def _fill_covered(given: np.ndarray, out: np.ndarray) -> str | None:
    shown = int(np.count_nonzero(out.any(axis=2)))
    return None if shown == 0 else f"{shown} of {out.shape[0] * out.shape[1]} pixels not black"


@needs_models
@needs_clip(CAMERA2)
@pytest.mark.parametrize("frames", [CAMERA2_DISTANT, CAMERA2_CLOSE], ids=["distant", "close"])
def test_silhouette_and_facebox_cover_every_mask_pixel_on_camera2(
    models: Path, frames: range
) -> None:
    config = _config(CAMERA2, "silhouette,facebox", "--detector", "both",
                     "--person-detector", "dnn", "--faces", "--segment")
    _run(f"camera2 silhouette,facebox {frames.start}", CAMERA2, frames, config,
         _segmenter_check(models))


@needs_models
@needs_clip(CAMERA)
def test_noise_changes_every_found_face_on_camera(models: Path) -> None:
    config = _config(CAMERA, "noise", "--detector", "both", "--person-detector", "dnn",
                     "--faces", "--track")
    _run("camera noise", CAMERA, CAMERA_NOISE, config, _face_check(models, CAMERA, _noise_covered))


@needs_models
@needs_clip(CAMERA)
def test_facebox_fills_every_found_face_on_camera(models: Path) -> None:
    config = _config(CAMERA, "facebox", "--detector", "both", "--person-detector", "dnn",
                     "--faces", "--track")
    _run("camera facebox", CAMERA, CAMERA_FACEBOX, config,
         _face_check(models, CAMERA, _fill_covered))


@needs_models
@pytest.mark.parametrize(
    ("clip", "frames", "styles", "flags"),
    [
        (CAMERA2, CAMERA2_CLOSE, "facebox", ("--faces", "--track")),
        (CAMERA, CAMERA_FACEBOX, "silhouette,facebox", ("--faces", "--segment", "--track")),
    ],
    ids=["camera2-facebox", "camera-silhouette-facebox"],
)
def test_orphan_head_covers_every_found_face(
    models: Path, monkeypatch: pytest.MonkeyPatch, clip: Path, frames: range, styles: str,
    flags: tuple[str, ...],
) -> None:
    if not clip.exists():
        pytest.skip(f"demo clip not available: {clip}")
    # Count, per orphan box, whether it got a head rect or kept the whole-box fill.
    counts = {"orphans": 0, "head": 0}
    real_reasons, real_head = shadow_mode.orphan_reasons, shadow_mode.head_rect

    def reasons(*args: object) -> object:
        out = real_reasons(*args)  # type: ignore[arg-type]
        counts["orphans"] += len(out)
        return out

    def head(*args: object) -> object:
        out = real_head(*args)  # type: ignore[arg-type]
        counts["head"] += out is not None
        return out

    monkeypatch.setattr(shadow_mode, "orphan_reasons", reasons)
    monkeypatch.setattr(shadow_mode, "head_rect", head)
    detector = FaceYuNetDetector.from_config(FaceYuNetConfig(), [models])

    def check(index: int, given: np.ndarray, out: np.ndarray) -> list[str]:
        frame = Frame(image=given, index=0, timestamp=0.0, source=None)  # type: ignore[arg-type]
        h, w = given.shape[:2]
        problems = []
        for face in detector.detect(frame).faces or ():
            # Padded as the mode pads a fresh face; every cover here is black.
            rect = pad_face(face.box, 0.3, h, w)
            if rect is None:
                continue
            shown = int(np.count_nonzero(out[rect.y0 : rect.y1, rect.x0 : rect.x1].any(axis=2)))
            if shown:
                problems.append(f"face {face.box}: {shown} pixels of its padded rect shown")
        return problems

    config = _config(clip, styles, "--detector", "both", "--person-detector", "dnn",
                     "--mode-option", "orphan=head", *flags)
    assert config.mode_options_dict()["orphan"] == "head"
    _run(f"{clip.name} {styles} orphan=head", clip, frames, config, check)
    print(f"orphan boxes {counts['orphans']}: head rect {counts['head']}, "
          f"whole-box fill {counts['orphans'] - counts['head']}")


@needs_models
@needs_clip(CAMERA2)
def test_bodybox_fills_every_found_person_box_on_camera2(models: Path) -> None:
    detector = PersonDnnDetector.from_config(PersonDnnConfig(), [models])

    def check(index: int, given: np.ndarray, out: np.ndarray) -> list[str]:
        problems = []
        for b in detector.detect_boxes(given):
            region = out[max(b.y, 0) : b.y + b.h, max(b.x, 0) : b.x + b.w]
            shown = int(np.count_nonzero(region.any(axis=2)))
            if shown:
                problems.append(f"person {b}: {shown} pixels not black")
        return problems

    config = _config(CAMERA2, "bodybox", "--detector", "both", "--person-detector", "dnn")
    _run("camera2 bodybox", CAMERA2, CAMERA2_BODYBOX, config, check)

