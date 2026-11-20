"""Tests for the golden capture harness, using a synthetic clip (no demo clips needed)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from unittest import mock

import cv2
import numpy as np
import pytest

import petrify.cli
import petrify.sinks.file
from petrify.stages.roi_overlay import Compositor
from tests.golden.cases import compare_records
from tests.golden.harness import (
    SCHEMA_VERSION,
    GoldenRecord,
    capture_engine,
    capture_legacy,
    from_json,
    to_json,
)

# Height must stay >= 256 so the default HOG downsample (0.5) keeps frames at least as
# large as the 64x128 HOG window; smaller frames crash the native detector.
_W, _H, _N = 320, 256, 80


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a textured static background with a bright square crossing the middle span."""
    path = tmp_path_factory.mktemp("clip") / "synthetic.mp4"
    rng = np.random.default_rng(1234)
    background = rng.integers(40, 120, size=(_H, _W, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (_W, _H))
    assert writer.isOpened()
    try:
        for i in range(_N):
            frame = background.copy()
            if 30 <= i < 55:
                x = 20 + (i - 30) * 10
                frame[100:160, x : x + 50] = 255
            writer.write(frame)
    finally:
        writer.release()
    return path


@pytest.fixture(scope="module")
def default_record(clip: Path) -> GoldenRecord:
    return capture_legacy(clip)


def test_record_lengths(default_record: GoldenRecord) -> None:
    n = default_record.total_frames
    assert n > 0
    assert len(default_record.raw) == n
    assert len(default_record.filled) == n
    assert len(default_record.final) == n


def test_stages_are_monotonic(default_record: GoldenRecord) -> None:
    r = default_record
    assert all(f for raw, f in zip(r.raw, r.filled) if raw)
    assert all(fin for f, fin in zip(r.filled, r.final) if f)


def test_cold_start_frozen(default_record: GoldenRecord) -> None:
    n = min(50, default_record.total_frames)
    assert all(default_record.final[:n])


def test_metadata(default_record: GoldenRecord, clip: Path) -> None:
    assert default_record.schema_version == 3
    assert default_record.clip == clip.name
    assert default_record.argv == []
    assert (default_record.width, default_record.height) == (_W, _H)


def test_gap_fill_zero_keeps_raw(clip: Path) -> None:
    record = capture_legacy(clip, ["--gap-fill", "0"])
    assert record.filled == record.raw


def test_background_frame_override(clip: Path) -> None:
    assert capture_legacy(clip, ["--background-frame", "0"]).background_index == 0


def test_no_roi(clip: Path) -> None:
    assert capture_legacy(clip, ["--no-roi"]).roi is None


def test_manual_roi(clip: Path) -> None:
    assert capture_legacy(clip, ["--roi", "0", "0", "10", "10"]).roi == [0, 0, 10, 10]


def test_capture_is_deterministic(clip: Path) -> None:
    argv = ["--background-frame", "0", "--no-roi"]
    assert to_json(capture_legacy(clip, argv)) == to_json(capture_legacy(clip, argv))


def test_json_round_trip(default_record: GoldenRecord) -> None:
    text = to_json(default_record)
    assert text.endswith("\n")
    assert from_json(text) == default_record


def test_json_frames_are_bit_strings(default_record: GoldenRecord) -> None:
    data = json.loads(to_json(default_record))
    for field in ("raw", "filled", "final"):
        assert isinstance(data[field], str)
        assert len(data[field]) == default_record.total_frames
        assert set(data[field]) <= {"0", "1"}


def test_real_ffmpeg_not_invoked(clip: Path) -> None:
    with (
        mock.patch("petrify.encoder.subprocess.run") as run_mock,
        mock.patch("petrify.encoder.shutil.which") as which_mock,
    ):
        capture_legacy(clip, ["--no-roi", "--background-frame", "0"])
    run_mock.assert_not_called()
    which_mock.assert_not_called()


def test_digests_cover_every_written_frame(clip: Path) -> None:
    real_compose = Compositor.compose
    composed: list[int] = []

    def counting_compose(self: Compositor, *args: Any, **kwargs: Any) -> Any:
        composed.append(1)
        return real_compose(self, *args, **kwargs)

    with mock.patch.object(Compositor, "compose", new=counting_compose):
        record = capture_legacy(clip, ["--no-roi", "--background-frame", "0"])
    assert len(record.frame_digests) == len(composed)
    assert len(record.frame_digests) == record.total_frames == _N


def test_digests_are_hex(default_record: GoldenRecord) -> None:
    assert default_record.frame_digests
    assert all(re.fullmatch(r"[0-9a-f]{16}", d) for d in default_record.frame_digests)


def test_digests_are_deterministic(clip: Path) -> None:
    assert capture_legacy(clip).frame_digests == capture_legacy(clip).frame_digests


def test_roi_changes_digests(clip: Path) -> None:
    # The ROI covers the band the bright square crosses, so pasting it live over the
    # frozen background must change those frames.
    argv = ["--background-frame", "0"]
    no_roi = capture_legacy(clip, [*argv, "--no-roi"]).frame_digests
    manual = capture_legacy(clip, [*argv, "--roi", "0", "100", "320", "60"]).frame_digests
    assert len(no_roi) == len(manual)
    assert no_roi != manual


def test_real_video_writer_restored(clip: Path) -> None:
    real = cv2.VideoWriter
    capture_legacy(clip, ["--no-roi", "--background-frame", "0"])
    assert cv2.VideoWriter is real


def test_from_json_rejects_schema_version_1(default_record: GoldenRecord) -> None:
    data = json.loads(to_json(default_record))
    data["schema_version"] = 1
    with pytest.raises(ValueError, match="schema_version"):
        from_json(json.dumps(data))
    assert SCHEMA_VERSION == 3


def test_output_matches_clip(default_record: GoldenRecord) -> None:
    out = default_record.output
    assert (out["width"], out["height"]) == (_W, _H)
    assert out["fps"] == default_record.fps == 25.0


def test_encode_defaults_match_cli(default_record: GoldenRecord) -> None:
    defaults = petrify.cli._build_parser().parse_args(["in.mp4", "out.mp4"])
    assert default_record.output["encode"] == {
        "codec": defaults.codec,
        "crf": defaults.crf,
        "preset": defaults.preset,
        "hwaccel": defaults.hwaccel,
        "audio_from_input": True,
        "output_is_requested_path": True,
    }


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--detector", "both"],
        ["--no-roi", "--background-frame", "0"],
        ["--gap-fill", "0"],
        ["--no-bg-diff"],
    ],
    ids=["defaults", "detector-both", "no-roi-bg-0", "gap-fill-0", "no-bg-diff"],
)
def test_engine_matches_legacy(clip: Path, argv: list[str]) -> None:
    engine = capture_engine(clip, argv)
    assert compare_records(engine, capture_legacy(clip, argv)) == []
    assert engine.total_frames == len(engine.raw) == len(engine.frame_digests) == _N


def test_engine_capture_restores_patches(clip: Path) -> None:
    real_writer = cv2.VideoWriter
    real_encode = petrify.sinks.file.ffmpeg_encode
    capture_engine(clip, ["--no-roi", "--background-frame", "0"])
    assert cv2.VideoWriter is real_writer
    assert petrify.sinks.file.ffmpeg_encode is real_encode


def test_encode_options_recorded(clip: Path) -> None:
    argv = ["--no-roi", "--background-frame", "0"]
    argv += ["--crf", "31", "--codec", "libx265", "--preset", "veryfast"]
    encode = capture_legacy(clip, argv).output["encode"]
    assert (encode["crf"], encode["codec"], encode["preset"]) == (31, "libx265", "veryfast")
