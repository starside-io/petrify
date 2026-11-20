"""Tests for petrify.pipeline.run_file, the streaming-engine file runner."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

import cv2
import numpy as np
import pytest

import petrify.sinks.file as sink_mod
from petrify.core import (
    BackgroundConfig,
    EncodeConfig,
    FrameTrace,
    PersonHogConfig,
    PetrifyConfig,
    RoiConfig,
)
from petrify.pipeline import RunReport, run_file

# Height must stay >= 256 so the default HOG downsample (0.5) keeps frames at least as
# large as the 64x128 HOG window.
_W, _H, _N = 320, 256, 40

ENCODE = EncodeConfig(codec="libx265", crf=19, preset="fast", hwaccel=None)


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Static textured background with a bright square crossing frames 20 to 29."""
    path = tmp_path_factory.mktemp("clip") / "synthetic.mp4"
    rng = np.random.default_rng(99)
    background = rng.integers(40, 120, size=(_H, _W, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (_W, _H))
    assert writer.isOpened()
    try:
        for i in range(_N):
            frame = background.copy()
            if 20 <= i < 30:
                x = 20 + (i - 20) * 20
                frame[100:160, x : x + 50] = 255
            writer.write(frame)
    finally:
        writer.release()
    return path


class _EncoderSpy:
    """Stand-in for ``ffmpeg_encode`` that counts frames in the intermediate file."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.frame_counts: list[int] = []

    def __call__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        cap = cv2.VideoCapture(str(kwargs["video_path"]))
        try:
            self.frame_counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        finally:
            cap.release()


def _config(**overrides: Any) -> PetrifyConfig:
    base: dict[str, Any] = {
        "roi": RoiConfig(mode="disabled"),
        "background": BackgroundConfig(frame_index=3),
        "encode": ENCODE,
    }
    base.update(overrides)
    return PetrifyConfig(**base)


def test_run_file_report_and_encode(clip: Path, tmp_path: Path) -> None:
    spy = _EncoderSpy()
    traces: list[FrameTrace] = []
    out = tmp_path / "out.mp4"
    with mock.patch.object(sink_mod, "ffmpeg_encode", new=spy):
        report = run_file(clip, out, _config(), traces.append)

    assert isinstance(report, RunReport)
    assert report.background_index == 3
    assert report.roi is None
    assert report.stats.frames == _N
    assert report.stats.frozen + report.stats.passed_through == _N
    assert report.stats.failed_closed == 0
    assert [t.index for t in traces] == list(range(_N))
    assert all(t.final for t in traces[:50])

    assert len(spy.calls) == 1
    call = spy.calls[0]
    assert Path(call["output_path"]) == out
    assert Path(call["original_audio_source"]) == clip
    assert (call["codec"], call["crf"], call["preset"], call["hwaccel"]) == (
        "libx265",
        19,
        "fast",
        None,
    )
    assert spy.frame_counts == [_N]
    assert not os.path.exists(call["video_path"])


def test_run_file_manual_roi_reported(clip: Path, tmp_path: Path) -> None:
    config = _config(roi=RoiConfig(mode="manual", box=(0, 0, 40, 20)))
    with mock.patch.object(sink_mod, "ffmpeg_encode", new=_EncoderSpy()):
        report = run_file(clip, tmp_path / "out.mp4", config)
    assert report.roi == (0, 0, 40, 20)


def test_run_file_background_scan_uses_scan_person_config(clip: Path, tmp_path: Path) -> None:
    config = _config(detector="motion", person=PersonHogConfig(min_box_area=4000))
    expected = config.background_scan_person_config()
    assert expected != config.person
    background = np.zeros((_H, _W, 3), dtype=np.uint8)
    selector = mock.Mock(return_value=(background, 3))
    with (
        mock.patch.object(sink_mod, "ffmpeg_encode", new=_EncoderSpy()),
        mock.patch("petrify.pipeline.locked_background", new=selector),
    ):
        report = run_file(clip, tmp_path / "out.mp4", config)
    selector.assert_called_once()
    args = selector.call_args.args
    assert args[1] == config.background
    assert args[2] == expected
    assert report.background_index == 3


def test_run_file_failure_skips_encode_and_removes_temp(clip: Path, tmp_path: Path) -> None:
    spy = _EncoderSpy()
    temp_paths: list[str] = []
    real_mkstemp = tempfile.mkstemp

    def recording_mkstemp(*args: Any, **kwargs: Any) -> tuple[int, str]:
        fd, path = real_mkstemp(*args, **kwargs)
        temp_paths.append(path)
        return fd, path

    def observer(trace: FrameTrace) -> None:
        if trace.index == 15:
            raise RuntimeError("stop mid-run")

    with (
        mock.patch.object(sink_mod, "ffmpeg_encode", new=spy),
        mock.patch.object(sink_mod.tempfile, "mkstemp", new=recording_mkstemp),
        pytest.raises(RuntimeError, match="stop mid-run"),
    ):
        run_file(clip, tmp_path / "out.mp4", _config(), observer)

    assert spy.calls == []
    assert len(temp_paths) == 1
    assert not os.path.exists(temp_paths[0])
    assert not (tmp_path / "out.mp4").exists()


def test_run_file_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_file(tmp_path / "missing.mp4", tmp_path / "out.mp4", _config())
