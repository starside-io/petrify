"""CLI routing tests: ``petrify.cli.main`` drives the streaming engine via ``run_file``."""

from __future__ import annotations

import dataclasses
import logging
import re
import socket
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

import petrify
import petrify.pipeline
import petrify.sources
from petrify.cli import _build_parser, _build_run_parser, _resolve_run, main
from petrify.core.config import (
    BackgroundConfig,
    BackgroundDiffConfig,
    EncodeConfig,
    EngineConfig,
    MotionConfig,
    PersonHogConfig,
    PetrifyConfig,
    RoiConfig,
)
from petrify.core.engine import EngineStats
from petrify.core.frame import SourceInfo
from petrify.pipeline import OutputConfig, RunReport

IN = "in.mp4"
OUT = "out.mp4"

NON_DEFAULT_ARGV = (
    # ROI / Compositor
    "--roi", "10", "20", "320", "80",
    "--roi-scan-samples", "30",
    "--roi-scan-min-freq", "0.7",
    "--roi-scan-margin", "4",
    "--roi-scan-max-area-frac", "0.2",
    "--roi-scan-edge-threshold", "70",
    "--roi-scan-edge-gap-tolerance", "20",
    "--roi-scan-edge-samples", "5",
    # Detector mode
    "--detector", "both",
    "--hog-downsample", "0.75",
    "--hog-scale", "1.1",
    "--hog-min-confidence", "0.5",
    "--hog-min-box-area", "2000",
    # Temporal gap fill
    "--gap-fill", "3",
    # Background reference
    "--background-frame", "7",
    "--bg-diff-pixel-threshold", "30",
    "--bg-diff-area-threshold", "300",
    # Motion detection
    "--threshold", "250",
    "--history", "100",
    "--var-threshold", "20",
    "--detect-shadows",
    "--downsample", "0.25",
    # Encoding
    "--codec", "libx265",
    "--crf", "18",
    "--preset", "slow",
    "--hwaccel", "videotoolbox",
    # Logging
    "--verbose",
)


def _report(failed_closed: int = 0) -> RunReport:
    frames = 10
    return RunReport(
        stats=EngineStats(
            frames=frames,
            frozen=0,
            passed_through=frames - failed_closed,
            failed_closed=failed_closed,
        ),
        background_index=0,
        roi=None,
    )


@pytest.fixture
def run_file() -> Iterator[mock.MagicMock]:
    with mock.patch.object(petrify.pipeline, "run_file", return_value=_report()) as patched:
        yield patched


@pytest.fixture
def old_run() -> Iterator[mock.MagicMock]:
    with mock.patch.object(petrify.pipeline, "run") as patched:
        yield patched


@pytest.mark.parametrize("extra", [(), NON_DEFAULT_ARGV], ids=["default", "non-default"])
def test_main_calls_run_file_with_parsed_config(
    run_file: mock.MagicMock, old_run: mock.MagicMock, extra: tuple[str, ...]
) -> None:
    argv = [IN, OUT, *extra]

    assert main(argv) == 0

    expected_config = PetrifyConfig.from_namespace(_build_parser().parse_args(argv))
    run_file.assert_called_once_with(IN, OUT, expected_config)
    old_run.assert_not_called()


def test_non_default_argv_changes_config() -> None:
    default = PetrifyConfig.from_namespace(_build_parser().parse_args([IN, OUT]))
    custom = PetrifyConfig.from_namespace(
        _build_parser().parse_args([IN, OUT, *NON_DEFAULT_ARGV])
    )
    assert custom != default


def test_failed_closed_frames_still_exit_zero(run_file: mock.MagicMock) -> None:
    run_file.return_value = _report(failed_closed=3)
    assert main([IN, OUT]) == 0


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (FileNotFoundError("missing.mp4"), 2),
        (KeyboardInterrupt(), 130),
        (RuntimeError("boom"), 1),
    ],
    ids=["file-not-found", "keyboard-interrupt", "other-exception"],
)
def test_exit_codes_on_errors(
    run_file: mock.MagicMock, error: BaseException, code: int
) -> None:
    run_file.side_effect = error
    assert main([IN, OUT]) == code


@pytest.mark.parametrize("value", ["0", "-0.5", "1.5"])
def test_invalid_downsample_exits_through_parser_error(
    run_file: mock.MagicMock, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([IN, OUT, "--downsample", value])
    assert excinfo.value.code == 2
    assert "--downsample must be in (0, 1]" in capsys.readouterr().err
    run_file.assert_not_called()


def test_negative_gap_fill_exits_through_parser_error(
    run_file: mock.MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([IN, OUT, "--gap-fill", "-3"])
    assert excinfo.value.code == 2
    assert "--gap-fill must be >= 0, got -3" in capsys.readouterr().err
    run_file.assert_not_called()


def test_version(run_file: mock.MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert f"petrify {petrify.__version__}" in capsys.readouterr().out
    run_file.assert_not_called()


def test_help(run_file: mock.MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "usage: petrify" in out
    assert "petrify run --help" in out
    run_file.assert_not_called()


def test_file_form_namespace_and_config_unchanged() -> None:
    """The file form parses exactly as before the run form existed."""
    args = _build_parser().parse_args([IN, OUT, *NON_DEFAULT_ARGV])
    assert vars(args) == {
        "input": IN, "output": OUT,
        "roi": [10, 20, 320, 80], "no_roi": False,
        "roi_scan_samples": 30, "roi_scan_min_freq": 0.7, "roi_scan_margin": 4,
        "roi_scan_max_area_frac": 0.2, "roi_scan_edge_threshold": 70.0,
        "roi_scan_edge_gap_tolerance": 20, "roi_scan_edge_samples": 5,
        "detector": "both", "hog_downsample": 0.75, "hog_scale": 1.1,
        "hog_min_confidence": 0.5, "hog_min_box_area": 2000,
        "gap_fill": 3, "background_frame": 7, "no_bg_diff": False,
        "bg_diff_pixel_threshold": 30, "bg_diff_area_threshold": 300,
        "threshold": 250, "history": 100, "var_threshold": 20.0, "detect_shadows": True,
        "downsample": 0.25,
        "codec": "libx265", "crf": 18, "preset": "slow", "hwaccel": "videotoolbox",
        "verbose": True,
    }
    assert PetrifyConfig.from_namespace(args) == PetrifyConfig(
        detector="both",
        motion=MotionConfig(history=100, var_threshold=20.0, detect_shadows=True,
                            threshold=250, downsample=0.25),
        person=PersonHogConfig(downsample=0.75, scale=1.1, min_confidence=0.5,
                               min_box_area=2000),
        background_diff=BackgroundDiffConfig(pixel_threshold=30, area_threshold=300,
                                             downsample=0.25),
        roi=RoiConfig(mode="manual", box=(10, 20, 320, 80), scan_samples=30,
                      scan_min_freq=0.7, scan_margin=4, scan_max_area_frac=0.2,
                      scan_edge_threshold=70.0, scan_edge_gap_tolerance=20,
                      scan_edge_samples=5),
        background=BackgroundConfig(strategy="locked", frame_index=7),
        encode=EncodeConfig(codec="libx265", crf=18, preset="slow", hwaccel="videotoolbox"),
        engine=EngineConfig(lookahead=3),
    )
    assert PetrifyConfig.from_namespace(_build_parser().parse_args([IN, OUT])) == PetrifyConfig()


def test_k1_run_without_source_is_the_file_form(
    run_file: mock.MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    # "run" with no --source is an INPUT named run: the file form's usage applies.
    with pytest.raises(SystemExit) as excinfo:
        main(["run"])
    assert excinfo.value.code == 2
    assert "INPUT OUTPUT" in capsys.readouterr().err
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "--output", OUT])
    assert excinfo.value.code == 2
    assert "unrecognized arguments: --output" in capsys.readouterr().err
    run_file.assert_not_called()


# ---------------------------------------------------------------------------
# petrify run
# ---------------------------------------------------------------------------

SECRET = "Zq9CliSecretPassw0rd"
SECRET_URL = f"rtsp://admin:{SECRET}@cam.example:554/live?token=Tk7CliToken"


class _StubSource:
    def __init__(self, live: bool = False) -> None:
        self.info = SourceInfo(fps=25.0, width=320, height=256, frame_count=None,
                               seekable=not live, live=live, uri="stub")

    def close(self) -> None:
        pass


@pytest.fixture
def run_stream() -> Iterator[mock.MagicMock]:
    with (
        mock.patch.object(petrify.pipeline, "run_stream", return_value=_report()) as patched,
        mock.patch.object(petrify.sources, "open_source", return_value=_StubSource()),
    ):
        yield patched


def _resolved(argv: list[str]) -> tuple[PetrifyConfig, OutputConfig]:
    parser = _build_run_parser()
    settings = _resolve_run(parser, parser.parse_args(argv))
    return settings.config, settings.outputs


@pytest.fixture
def image_dir(tmp_path: Path) -> str:
    return str(tmp_path)


@pytest.mark.parametrize(
    ("spec", "background", "lookahead", "roi_mode"),
    [
        ("clip.mp4", "locked", 10, "auto"),
        ("frames/*.png", "locked", 10, "auto"),
        ("camera:0", "running", 3, "disabled"),
        ("rtsp://cam.example/live", "running", 10, "auto"),
        ("https://cam.example/mjpeg", "running", 10, "auto"),
    ],
)
def test_run_defaults_per_source_kind(
    spec: str, background: str, lookahead: int, roi_mode: str
) -> None:
    config, outputs = _resolved(["--source", spec, "--output", OUT])
    assert config.background.strategy == background
    assert config.background.sample_check is True
    assert config.engine.lookahead == lookahead
    assert config.engine.frame_budget_ms is None
    assert config.roi == RoiConfig(mode=roi_mode)  # type: ignore[arg-type]
    assert outputs == OutputConfig(file=Path(OUT))


def test_run_defaults_for_image_directory(image_dir: str) -> None:
    config, _ = _resolved(["--source", image_dir, "--preview"])
    assert (config.background.strategy, config.engine.lookahead) == ("locked", 10)


def test_run_file_source_defaults_match_file_form() -> None:
    config, _ = _resolved(["--source", IN, "--output", OUT])
    assert config == PetrifyConfig.from_namespace(_build_parser().parse_args([IN, OUT]))


def test_run_shared_options_match_file_form() -> None:
    shared = ["--lookahead" if a == "--gap-fill" else a for a in NON_DEFAULT_ARGV]
    config, _ = _resolved(["--source", IN, "--output", OUT, *shared])
    assert config == PetrifyConfig.from_namespace(
        _build_parser().parse_args([IN, OUT, *NON_DEFAULT_ARGV])
    )


def test_run_options_reach_the_config(image_dir: str) -> None:
    config, outputs = _resolved([
        "--source", "camera:1", "--virtualcam", "--virtualcam-device", "OBS", "--preview",
        "--output", OUT, "--background", "calibrate", "--calibrate-seconds", "4.5",
        "--lookahead", "0", "--frame-budget-ms", "40", "--roi-auto", "--roi-warmup", "6",
        "--roi-max-area", "0.05", "--no-background-check",
    ])
    assert config.background.strategy == "calibrate"
    assert config.background.calibrate_seconds == 4.5
    assert config.engine.lookahead == 0
    assert config.engine.frame_budget_ms == 40.0
    assert config.roi.mode == "auto"
    assert config.roi.warmup_seconds == 6.0
    assert config.roi.max_area_cap == 0.05
    assert config.background.sample_check is False
    assert outputs == OutputConfig(file=Path(OUT), virtualcam=True, virtualcam_device="OBS",
                                   preview=True)
    config, _ = _resolved(["--source", "clip.mp4", "--output", OUT, "--background", "running"])
    assert config.background.strategy == "running"
    config, _ = _resolved(["--source", "a.mp4", "--output", OUT, "--background-frame", "4"])
    assert config.background.frame_index == 4


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--source", IN], "at least one output is required"),
        (["--source", "camera:0", "--output", OUT, "--background", "locked"],
         "--background locked needs a video file"),
        (["--source", SECRET_URL, "--output", OUT, "--background", "locked"],
         "--background locked needs a video file"),
        (["--source", IN, "--output", OUT, "--virtualcam-device", "x"],
         "--virtualcam-device requires --virtualcam"),
        (["--source", IN, "--output", OUT, "--gap-fill", "3"], "use --lookahead"),
        (["--source", IN, "--output", OUT, "--lookahead", "-1"], "--lookahead must be >= 0"),
        (["--source", IN, "--output", OUT, "--frame-budget-ms", "0"],
         "--frame-budget-ms must be > 0"),
        (["--source", IN, "--output", OUT, "--background", "calibrate",
          "--calibrate-seconds", "-2"], "--calibrate-seconds must be > 0"),
        (["--source", IN, "--output", OUT, "--roi-warmup", "0"], "--roi-warmup must be > 0"),
        (["--source", "f/*.png", "--output", OUT, "--fps", "0"], "--fps must be > 0"),
        (["--source", IN, "--output", OUT, "--fps", "10"], "--fps applies only"),
        (["--source", IN, "--output", OUT, "--downsample", "1.5"], "--downsample must be in"),
        (["--source", IN, "--output", OUT, "--background-frame", "-1"],
         "--background-frame must be >= 0"),
        (["--source", "camera:0", "--output", OUT, "--background-frame", "3"],
         "--background-frame requires --background locked"),
        (["--source", IN, "--output", OUT, "--calibrate-seconds", "3"],
         "--calibrate-seconds requires --background calibrate"),
        (["--source", "camera:x", "--output", OUT], "camera:N"),
        (["--source", f"ftp://u:{SECRET}@h/x", "--output", OUT], "supported schemes"),
        (["--source", IN, "--output", OUT, "--mode", "shadow"], "invalid choice"),
    ],
)
def test_run_usage_errors(
    run_stream: mock.MagicMock,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    message: str,
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["run", *argv])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert message in err
    assert SECRET not in err
    run_stream.assert_not_called()


def test_unknown_mode_lists_available_modes(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["run", "--source", IN, "--output", OUT, "--mode", "shadow"])
    err = capsys.readouterr().err
    assert "erase" in err.split("invalid choice", 1)[1]


def test_run_calls_run_stream(run_stream: mock.MagicMock) -> None:
    assert main(["run", "--source", IN, "--output", OUT, "--lookahead", "5"]) == 0
    run_stream.assert_called_once()
    source, config, outputs = run_stream.call_args.args
    assert isinstance(source, _StubSource)
    assert config.engine.lookahead == 5
    assert outputs == OutputConfig(file=Path(OUT))
    petrify.sources.open_source.assert_called_once_with(IN, fps=None)  # type: ignore[attr-defined]


def test_run_prints_banner_but_not_for_help(
    run_stream: mock.MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "--source", IN, "--output", OUT]) == 0
    assert "Freeze the world" in capsys.readouterr().err
    run_stream.reset_mock()
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "Freeze the world" not in captured.err
    assert "usage: petrify run" in captured.out
    assert "--lookahead" in captured.out and "--gap-fill" in captured.out  # Named in help text.
    run_stream.assert_not_called()


@pytest.mark.parametrize(
    ("result", "code"),
    [
        (_report(), 0),
        (dataclasses.replace(_report(failed_closed=2), interrupted=True), 0),
        (FileNotFoundError("missing.mp4"), 2),
        (KeyboardInterrupt(), 130),
        (RuntimeError("boom"), 1),
    ],
    ids=["ok", "live-interrupted", "file-not-found", "keyboard-interrupt", "other-exception"],
)
def test_run_exit_codes(run_stream: mock.MagicMock, result: object, code: int) -> None:
    if isinstance(result, BaseException):
        run_stream.side_effect = result
    else:
        run_stream.return_value = result
    assert main(["run", "--source", IN, "--output", OUT]) == code


def test_run_missing_input_file_exits_2(tmp_path: Path) -> None:
    assert main(["run", "--source", str(tmp_path / "missing.mp4"), "--output", OUT]) == 2


def test_run_never_logs_the_stream_url(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    real_open = petrify.sources.open_source

    class _Closed:
        def __init__(self, url: str) -> None:
            pass

        def isOpened(self) -> bool:  # noqa: N802 (mirrors the OpenCV API)
            return False

        def release(self) -> None:
            pass

    def fake_open(spec: str, **kwargs: object) -> object:
        return real_open(spec, capture_factory=_Closed, **kwargs)  # type: ignore[arg-type]

    with mock.patch.object(petrify.sources, "open_source", side_effect=fake_open):
        code = main(["run", "--source", SECRET_URL, "--preview", "-v"])
    assert code == 1
    text = capsys.readouterr().err + caplog.text
    assert "rtsp://***@cam.example:554" in text
    assert SECRET not in text and "Tk7CliToken" not in text


def test_run_logs_the_redacted_source_uri(caplog: pytest.LogCaptureFixture) -> None:
    stub = _StubSource(live=True)
    stub.info = dataclasses.replace(stub.info, uri="rtsp://***@cam.example:554")
    with (
        mock.patch.object(petrify.pipeline, "run_stream", return_value=_report()),
        mock.patch.object(petrify.sources, "open_source", return_value=stub),
        caplog.at_level(logging.INFO, logger="petrify.cli"),
    ):
        assert main(["run", "--source", SECRET_URL, "--preview"]) == 0
    assert "rtsp://***@cam.example:554" in caplog.text
    assert SECRET not in caplog.text


# ---------------------------------------------------------------------------
# P2-9c
# ---------------------------------------------------------------------------

C1_SECRET = "Zq9MistypedSecret"
C1_SPECS = [
    f"camera:rtsp://u:{C1_SECRET}@h/x",
    f"rtsp:/u:{C1_SECRET}@h/x",
    f"u:{C1_SECRET}@h:554/x",
    f"rtsp://u:{C1_SECRET}@h/x",
]


def _main_text(
    argv: list[str], capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> tuple[int | str | None, str]:
    """Run the real CLI; return the exit code and everything it printed or logged."""
    caplog.clear()
    try:
        code: int | str | None = main(argv)
        exc_text = ""
    except SystemExit as exc:
        code, exc_text = exc.code, str(exc)
    captured = capsys.readouterr()
    return code, captured.err + captured.out + caplog.text + exc_text


@pytest.mark.parametrize("verbose", [[], ["--verbose"]], ids=["quiet", "verbose"])
@pytest.mark.parametrize("spec", C1_SPECS)
def test_c1_mistyped_specs_never_show_the_password_in_the_run_form(
    spec: str,
    verbose: list[str],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG):
        code, text = _main_text(["run", "--source", spec, "--output", OUT, *verbose],
                                capsys, caplog)
    assert code in (1, 2)
    assert C1_SECRET not in text
    assert "<redacted>" in text or "://" in text


@pytest.mark.parametrize("verbose", [[], ["--verbose"]], ids=["quiet", "verbose"])
@pytest.mark.parametrize("spec", C1_SPECS)
def test_c1_mistyped_specs_never_show_the_password_in_the_file_form(
    spec: str,
    verbose: list[str],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG):
        code, text = _main_text([spec, OUT, *verbose], capsys, caplog)
    assert code == 2
    assert "Input file not found" in text
    assert C1_SECRET not in text
    # The exception the file form handles holds no secret either.
    with pytest.raises(FileNotFoundError) as excinfo:
        petrify.pipeline.run_file(spec, OUT, PetrifyConfig())
    assert C1_SECRET not in str(excinfo.value)


def test_c1_plain_missing_input_is_shown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    missing = str(tmp_path / "plain.mp4")
    code, text = _main_text([missing, OUT], capsys, caplog)
    assert code == 2
    assert f"Input file not found: {missing}" in text


def test_l2_camera_roi_is_disabled_unless_asked() -> None:
    assert _resolved(["--source", "camera:0", "--preview"])[0].roi.mode == "disabled"
    assert _resolved(["--source", "camera:0", "--preview", "--roi-auto"])[0].roi.mode == "auto"
    config, _ = _resolved(["--source", "camera:0", "--preview", "--roi", "1", "2", "3", "4"])
    assert (config.roi.mode, config.roi.box) == ("manual", (1, 2, 3, 4))
    for spec in ("rtsp://cam.example/live", "frames/*.png", IN):
        assert _resolved(["--source", spec, "--preview"])[0].roi.mode == "auto"
    config, _ = _resolved(["--source", "rtsp://cam.example/live", "--preview",
                           "--roi-max-area", "0.3", "--roi-warmup", "4"])
    assert (config.roi.max_area_cap, config.roi.warmup_seconds) == (0.3, 4.0)
    assert _resolved(["--source", IN, "--preview"])[0].roi.max_area_cap == 0.15


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--source", IN, "--roi-auto", "--no-roi"], "cannot be combined"),
        (["--source", IN, "--roi-auto", "--roi", "1", "2", "3", "4"], "cannot be combined"),
        (["--source", IN, "--no-roi", "--roi", "1", "2", "3", "4"], "cannot be combined"),
        (["--source", IN, "--no-roi", "--roi-warmup", "3"], "--roi-warmup needs ROI auto"),
        (["--source", IN, "--roi", "1", "2", "3", "4", "--roi-max-area", "0.2"],
         "--roi-max-area needs ROI auto"),
        (["--source", "camera:0", "--roi-warmup", "3"], "add --roi-auto"),
        (["--source", "camera:0", "--roi-max-area", "0.2"], "add --roi-auto"),
        (["--source", "camera:0", "--roi-auto", "--roi-max-area", "0"],
         "--roi-max-area must be in (0, 1]"),
        (["--source", IN, "--roi-max-area", "1.5"], "--roi-max-area must be in (0, 1]"),
    ],
)
def test_l2_roi_usage_errors(
    run_stream: mock.MagicMock, capsys: pytest.CaptureFixture[str], argv: list[str], message: str
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["run", *argv, "--preview"])
    assert excinfo.value.code == 2
    assert message in capsys.readouterr().err
    run_stream.assert_not_called()


def test_l3_lookahead_defaults_and_cap(
    run_stream: mock.MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _resolved(["--source", "rtsp://cam.example/live", "--preview"])[0].engine.lookahead == 10
    assert _resolved(["--source", "camera:0", "--preview"])[0].engine.lookahead == 3
    assert _resolved(["--source", IN, "--preview", "--lookahead", "600"])[0].engine.lookahead == 600
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "--source", IN, "--preview", "--lookahead", "601"])
    assert excinfo.value.code == 2
    assert "--lookahead must be <= 600" in capsys.readouterr().err
    run_stream.assert_not_called()
    help_text = " ".join(_build_run_parser().format_help().split())
    assert "A smaller lookahead lowers the latency but lets more of a slowly entering" in help_text


def test_l1_no_background_check_flag_reaches_the_config() -> None:
    assert _resolved(["--source", "camera:0", "--preview"])[0].background.sample_check is True
    config, _ = _resolved(["--source", "camera:0", "--preview", "--no-background-check"])
    assert config.background.sample_check is False
    help_text = " ".join(_build_run_parser().format_help().split())
    assert "--no-background-check UNSAFE" in help_text


@pytest.mark.parametrize(
    "argv",
    [
        ["--verbose", "run", "--source", IN, "--output", OUT],
        ["-v", "--crf", "18", "run", "--source", IN, "--output", OUT],
        ["run", "--output", OUT, "--source", IN],
        ["--roi", "1", "2", "3", "4", "run", f"--source={IN}", "--output", OUT],
    ],
)
def test_k1_options_before_run_select_the_run_form(
    run_stream: mock.MagicMock, run_file: mock.MagicMock, argv: list[str]
) -> None:
    assert main(argv) == 0
    run_stream.assert_called_once()
    run_file.assert_not_called()
    _, config, outputs = run_stream.call_args.args
    assert outputs == OutputConfig(file=Path(OUT))
    if "--crf" in argv:
        assert config.encode.crf == 18
    if "--roi" in argv:
        assert config.roi.box == (1, 2, 3, 4)


def test_k1_a_file_named_run_is_the_file_form(
    run_stream: mock.MagicMock, run_file: mock.MagicMock
) -> None:
    assert main(["run", OUT]) == 0
    assert main(["--verbose", "run", OUT]) == 0
    assert [c.args[:2] for c in run_file.call_args_list] == [("run", OUT), ("run", OUT)]
    run_stream.assert_not_called()


def test_k1_dispatch_rule_is_in_the_top_level_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "The run form is used when the first argument that is not an option is 'run'" in out
    assert "petrify run out.mp4" in out


def _hwaccel_help(parser: object) -> str:
    (action,) = [a for a in parser._actions if "--hwaccel" in a.option_strings]  # type: ignore[attr-defined]
    return str(action.help)


def test_k2_run_hwaccel_help_is_accurate() -> None:
    file_help = _hwaccel_help(_build_parser())
    run_help = _hwaccel_help(_build_run_parser())
    assert file_help == (
        "FFmpeg -hwaccel value for hardware-accelerated *decoding* of the "
        "intermediate file.  Examples: cuda, videotoolbox, dxva2.  "
        "Pair with a matching hardware encoder via --codec.  Default: None."
    )
    assert "intermediate file" not in run_help
    assert "raw frames" in run_help and "--codec" in run_help


# ---------------------------------------------------------------------------
# petrify run --web (P3-3)
# ---------------------------------------------------------------------------

_WEB_LINE = re.compile(r"http://127\.0\.0\.1:(\d+)/#token=([A-Za-z0-9_-]+)")


def test_web_alone_is_a_valid_output() -> None:
    config, outputs = _resolved(["--source", IN, "--web"])
    assert not outputs.selected  # Not a file, virtual camera or preview ...
    assert config.engine.lookahead == 10  # ... but the run resolves.


def test_web_options_default() -> None:
    args = _build_run_parser().parse_args(["--source", IN, "--output", OUT])
    assert (args.web, args.web_port, args.web_raw_preview, args.realtime) == (
        False, 8765, False, False)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--source", IN, "--output", OUT, "--web-raw-preview"],
         "--web-raw-preview requires --web"),
        (["--source", IN, "--web", "--web-port", "70000"], "--web-port must be in 0..65535"),
        (["--source", IN, "--web", "--web-port", "-1"], "--web-port must be in 0..65535"),
        (["--source", IN], "--preview or --web"),
    ],
)
def test_web_usage_errors(
    run_stream: mock.MagicMock, capsys: pytest.CaptureFixture[str], argv: list[str],
    message: str,
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["run", *argv])
    assert excinfo.value.code == 2
    assert message in capsys.readouterr().err
    run_stream.assert_not_called()


def test_web_help_mentions_the_raw_feed_and_realtime(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["run", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "UNPROCESSED input feed" in out
    assert "Ignored for live sources" in out
    assert "0 picks a free port" in out


def test_run_without_web_calls_run_stream_exactly_as_before(run_stream: mock.MagicMock) -> None:
    assert main(["run", "--source", IN, "--output", OUT]) == 0
    assert len(run_stream.call_args.args) == 3
    assert run_stream.call_args.kwargs == {}


def test_realtime_without_web_reaches_run_stream(run_stream: mock.MagicMock) -> None:
    assert main(["run", "--source", IN, "--output", OUT, "--realtime"]) == 0
    assert run_stream.call_args.kwargs == {"realtime": True}


def _web_run(argv: list[str], capsys: pytest.CaptureFixture[str],
             caplog: pytest.LogCaptureFixture) -> tuple[mock.MagicMock, str, str]:
    seen: dict[str, Any] = {}

    def fake_run_stream(source: Any, config: Any, outputs: Any, **kwargs: Any) -> RunReport:
        # The server is up while the run is in progress.
        seen["kwargs"] = kwargs
        seen["threads"] = [t.name for t in threading.enumerate()]
        return _report()

    with (
        caplog.at_level(logging.DEBUG),
        mock.patch.object(petrify.pipeline, "run_stream", side_effect=fake_run_stream) as rs,
        mock.patch.object(petrify.sources, "open_source", return_value=_StubSource()),
    ):
        assert main(["run", *argv]) == 0
    err = capsys.readouterr().err
    rs.seen = seen  # type: ignore[attr-defined]
    return rs, err, caplog.text


def test_web_startup_line_is_printed_once_and_is_the_only_place_of_the_token(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    from petrify.server.control import RunControl
    from petrify.sinks.mjpeg import MjpegSink

    rs, err, logs = _web_run(["--source", IN, "--web", "--web-port", "0"], capsys, caplog)
    lines = [line for line in err.splitlines() if "#token=" in line]
    assert len(lines) == 1
    assert "gives control of this run to whoever has it" in lines[0]
    match = _WEB_LINE.search(lines[0])
    assert match is not None
    port, token = int(match.group(1)), match.group(2)
    assert port > 0 and len(token) >= 40
    assert err.count(token) == 1
    assert token not in logs
    kwargs = rs.seen["kwargs"]  # type: ignore[attr-defined]
    assert isinstance(kwargs["control"], RunControl)
    assert isinstance(kwargs["web_sink"], MjpegSink)
    assert kwargs["raw_preview"] is None
    assert kwargs["realtime"] is False
    assert "petrify-web" in rs.seen["threads"]  # type: ignore[attr-defined]
    # The server was stopped when the run ended.
    assert "petrify-web" not in [t.name for t in threading.enumerate()]
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        assert s.connect_ex(("127.0.0.1", port)) != 0


def test_web_raw_preview_and_realtime_reach_run_stream(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    from petrify.sinks.mjpeg import MjpegSink

    rs, _, _ = _web_run(["--source", IN, "--web", "--web-port", "0", "--web-raw-preview",
                         "--realtime"], capsys, caplog)
    kwargs = rs.seen["kwargs"]  # type: ignore[attr-defined]
    assert isinstance(kwargs["raw_preview"], MjpegSink)
    assert kwargs["raw_preview"] is not kwargs["web_sink"]
    assert kwargs["realtime"] is True


def test_web_tokens_differ_per_run(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    _, first, _ = _web_run(["--source", IN, "--web", "--web-port", "0"], capsys, caplog)
    _, second, _ = _web_run(["--source", IN, "--web", "--web-port", "0"], capsys, caplog)
    a, b = _WEB_LINE.search(first), _WEB_LINE.search(second)
    assert a is not None and b is not None and a.group(2) != b.group(2)


def test_web_server_is_stopped_when_the_run_fails(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        mock.patch.object(petrify.pipeline, "run_stream", side_effect=OSError("boom")),
        mock.patch.object(petrify.sources, "open_source", return_value=_StubSource()),
    ):
        assert main(["run", "--source", IN, "--web", "--web-port", "0"]) == 1
    assert "petrify-web" not in [t.name for t in threading.enumerate()]


def test_missing_fastapi_is_a_clear_error(
    run_stream: mock.MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    with mock.patch.dict(sys.modules, {"fastapi": None}):
        assert main(["run", "--source", IN, "--web"]) == 1
    assert "petrify[web]" in caplog.text
    run_stream.assert_not_called()
    petrify.sources.open_source.assert_not_called()  # type: ignore[attr-defined]


def test_importing_the_cli_does_not_import_the_web_stack() -> None:
    code = (
        "import sys\n"
        "import petrify, petrify.cli, petrify.pipeline, petrify.server, petrify.server.app\n"
        "print(sorted(m for m in ('fastapi', 'uvicorn', 'starlette') if m in sys.modules))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == "[]"


# ---------------------------------------------------------------------------
# petrify run: mode selection from the registry (T12)
# ---------------------------------------------------------------------------

T12_MODULE = "t12cli_kinds"
T12_OTHER = "t12cli_other"
T12_BROKEN = "t12cli_broken"
T12_SLOW = "t12cli_slow"
T12_DIST = "petrify-t12-cli"
T12_ORIGIN = f"{T12_DIST} 0.1"

_T12_SOURCE = '''
import numpy as np

from petrify.core.detections import DetectionKind
from petrify.core.plugin import OptionSpec


class KindsMode:
    detect_policy = "all"
    mandatory = frozenset()

    def __init__(self, name):
        self.name = name
        self.requires = frozenset({DetectionKind.PERSONS})

    def render(self, frame, detections, subject_present):
        return np.zeros_like(frame.image)


class KindsFactory:
    needs_background = False
    roi_overlay = False

    def __init__(self, name, summary, options):
        self.name = name
        self.summary = summary
        self.options = options

    def requires(self, options):
        return frozenset({DetectionKind.PERSONS})

    def mandatory(self, options):
        return frozenset()

    def weakened(self, options):
        return ()

    def create(self, ctx, previous=None):
        return KindsMode(self.name)


KINDS = KindsFactory("kinds", "One option of each kind, for tests.", (
    OptionSpec("flag", "bool", False, "a flag", runtime=True),
    OptionSpec("count", "int", 3, "a count", minimum=1, maximum=10),
    OptionSpec("ratio", "float", 0.5, "a ratio", minimum=0.0),
    OptionSpec("style", "choice", "flat", "a style", choices=("flat", "round")),
    OptionSpec("parts", "choices", ("head",), "some parts", choices=("head", "hands", "feet")),
))
DUP = KindsFactory("dup", "Registered twice.", ())
'''

_T12_ENTRY_POINTS = {
    "kinds": f"{T12_MODULE}:KINDS",
    "other": f"{T12_OTHER}:FACTORY",
    "broken": f"{T12_BROKEN}:FACTORY",
    "slow": f"{T12_SLOW}:FACTORY",
    "dup": f"{T12_MODULE}:DUP",
    "erase": f"{T12_OTHER}:FACTORY",
    "Bad-Name": f"{T12_OTHER}:FACTORY",
    "badvalue": "not a target",
}
_T12_MODULES = (T12_MODULE, T12_OTHER, T12_BROKEN, T12_SLOW)


def _t12_dist(root: Path, name: str, version: str, entries: dict[str, str]) -> None:
    info = root / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    lines = "".join(f"{key} = {target}\n" for key, target in entries.items())
    (info / "entry_points.txt").write_text(f"[petrify.modes]\n{lines}")


@pytest.fixture
def t12_plugins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Install fake distributions with plugin modes on a temporary ``sys.path`` entry.

    Each plugin module leaves a marker file when imported, so a test can tell
    whether it was imported even after ``sys.modules`` is cleaned.
    """
    root = tmp_path / "site"
    root.mkdir()
    _t12_dist(root, T12_DIST, "0.1", _T12_ENTRY_POINTS)
    _t12_dist(root, "petrify-t12-rival", "2.0", {"dup": f"{T12_OTHER}:FACTORY"})

    def marker(name: str) -> str:
        return f"open({str(root / ('imported_' + name))!r}, 'w').close()\n"

    (root / f"{T12_MODULE}.py").write_text(marker(T12_MODULE) + _T12_SOURCE)
    (root / f"{T12_OTHER}.py").write_text(marker(T12_OTHER) + "FACTORY = None\n")
    (root / f"{T12_BROKEN}.py").write_text(
        marker(T12_BROKEN) + "raise KeyError('broken at import\\nsecond line')\n"
    )
    (root / f"{T12_SLOW}.py").write_text(
        marker(T12_SLOW) + "import time\ntime.sleep(5)\nFACTORY = None\n"
    )
    monkeypatch.syspath_prepend(str(root))
    try:
        yield root
    finally:
        for name in _T12_MODULES:
            sys.modules.pop(name, None)


def _imported(root: Path, name: str) -> bool:
    return name in sys.modules or (root / f"imported_{name}").exists()


def _run_args(*extra: str) -> list[str]:
    return ["run", "--source", IN, "--output", OUT, *extra]


def _usage_error(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "usage: petrify run" in err
    return err


def test_t12_plugin_mode_reaches_run_stream(
    t12_plugins: Path, run_stream: mock.MagicMock
) -> None:
    assert main(_run_args("--mode", "kinds")) == 0
    _, config, _ = run_stream.call_args.args
    assert config.mode == "kinds"
    # Every declared option is stored, defaults filled in, in its declared type.
    assert config.mode_options == (
        ("count", 3), ("flag", False), ("parts", ("head",)), ("ratio", 0.5), ("style", "flat"),
    )
    assert not _imported(t12_plugins, T12_OTHER)


def test_t12_default_mode_is_erase(run_stream: mock.MagicMock) -> None:
    assert main(_run_args()) == 0
    _, config, _ = run_stream.call_args.args
    assert (config.mode, config.mode_options) == ("erase", ())
    assert main(_run_args("--mode", "erase")) == 0
    assert run_stream.call_args.args[1].mode == "erase"


def test_t12_unknown_mode_lists_the_selectable_modes(
    t12_plugins: Path, run_stream: mock.MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    err = _usage_error(_run_args("--mode", "shadow"), capsys)
    listed = err.split("invalid choice", 1)[1]
    for name in ("erase", "kinds", "broken", "slow", "other"):
        assert name in listed
    # Unselectable entries are not offered.
    assert "dup" not in listed and "Bad-Name" not in listed and "badvalue" not in listed
    run_stream.assert_not_called()


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("dup", "ambiguous: the mode name 'dup' is registered more than once"),
        ("Bad-Name", "invalid mode name"),
        ("badvalue", "invalid entry point value"),
    ],
)
def test_t12_unselectable_mode_gives_its_reason(
    t12_plugins: Path,
    run_stream: mock.MagicMock,
    capsys: pytest.CaptureFixture[str],
    name: str,
    reason: str,
) -> None:
    err = _usage_error(_run_args("--mode", name), capsys)
    assert "invalid choice" not in err
    assert f"argument --mode: the mode {name!r} cannot be selected" in err
    assert reason in err
    assert T12_ORIGIN in err
    run_stream.assert_not_called()


@pytest.mark.parametrize(
    ("assignment", "key", "value"),
    [
        ("flag=yes", "flag", True),
        ("flag=OFF", "flag", False),
        ("count=7", "count", 7),
        ("ratio=2", "ratio", 2.0),
        ("ratio=0.25", "ratio", 0.25),
        ("style=round", "style", "round"),
        ("parts=feet,head", "parts", ("feet", "head")),
        ("parts=", "parts", ()),
    ],
)
def test_t12_mode_option_lands_in_the_config_in_its_declared_type(
    t12_plugins: Path, run_stream: mock.MagicMock, assignment: str, key: str, value: object
) -> None:
    assert main(_run_args("--mode", "kinds", "--mode-option", assignment)) == 0
    options = run_stream.call_args.args[1].mode_options_dict()
    assert options[key] == value
    assert type(options[key]) is type(value)
    assert set(options) == {"flag", "count", "ratio", "style", "parts"}


def test_t12_several_mode_options(t12_plugins: Path, run_stream: mock.MagicMock) -> None:
    argv = _run_args("--mode", "kinds", "--mode-option", "count=2", "--mode-option", "flag=1")
    assert main(argv) == 0
    options = run_stream.call_args.args[1].mode_options_dict()
    assert (options["count"], options["flag"]) == (2, True)


@pytest.mark.parametrize(
    ("assignments", "message"),
    [
        (["size=3"], "unknown mode option 'size'"),
        (["count=abc"], "option 'count' must be a whole number"),
        (["count=11"], "option 'count' must be at most 10"),
        (["ratio=nan"], "option 'ratio' must be a number"),
        (["flag=maybe"], "option 'flag' must be true or false"),
        (["style=square"], "option 'style' must be one of flat, round"),
        (["parts=head,head"], "option 'parts' must not repeat a choice"),
        (["count=2", "count=3"], "mode option 'count' is given more than once"),
        (["count"], "mode options must be written KEY=VALUE, got 'count'"),
    ],
)
def test_t12_mode_option_usage_errors(
    t12_plugins: Path,
    run_stream: mock.MagicMock,
    capsys: pytest.CaptureFixture[str],
    assignments: list[str],
    message: str,
) -> None:
    argv = _run_args("--mode", "kinds")
    for assignment in assignments:
        argv += ["--mode-option", assignment]
    err = _usage_error(argv, capsys)
    assert "petrify run: error: --mode-option: " in err
    assert message in err
    assert (
        "Options of the mode 'kinds': flag (bool), count (int), ratio (float), "
        "style (choice), parts (choices)"
    ) in err
    run_stream.assert_not_called()
    petrify.sources.open_source.assert_not_called()  # type: ignore[attr-defined]


def test_t12_mode_option_with_erase_is_a_usage_error(
    run_stream: mock.MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    err = _usage_error(_run_args("--mode-option", "x=1"), capsys)
    assert "--mode-option: the mode 'erase' has no options" in err
    run_stream.assert_not_called()


def test_t12_list_modes_alone(t12_plugins: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with mock.patch.object(petrify.sources, "open_source") as open_source:
        assert main(["run", "--list-modes"]) == 0
    open_source.assert_not_called()
    captured = capsys.readouterr()
    assert "Freeze the world" not in captured.err
    out = captured.out
    lines = out.splitlines()
    assert lines[0] == "Modes (select one with --mode NAME):"
    assert lines[1] == (
        "  erase (built in): Show the clean background while a subject is present, "
        "the live frame otherwise."
    )
    assert f"  kinds (plugin {T12_ORIGIN})" in lines
    assert f"  broken (plugin {T12_ORIGIN})" in lines
    assert f"  slow (plugin {T12_ORIGIN})" in lines
    assert (
        f"  erase (plugin {T12_ORIGIN}): not selectable: shadowed by the built-in mode "
        "'erase'; it can never be selected"
    ) in lines
    assert any(
        line.startswith("  dup (plugin petrify-t12-rival 2.0): not selectable: ambiguous")
        for line in lines
    )
    assert any(
        line.startswith(f"  Bad-Name (plugin {T12_ORIGIN}): not selectable: invalid mode name")
        for line in lines
    )
    assert any(
        line.startswith(f"  badvalue (plugin {T12_ORIGIN}): not selectable: invalid entry point")
        for line in lines
    )
    # A plugin's summary needs its code, so it is not shown, and nothing was imported.
    assert "One option of each kind" not in out
    assert "Options" not in out
    for name in _T12_MODULES:
        assert not _imported(t12_plugins, name)


def test_t12_list_modes_with_a_plugin_mode_prints_its_options(
    t12_plugins: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "--list-modes", "--mode", "kinds"]) == 0
    out = capsys.readouterr().out
    detail = out.split("\n\n", 1)[1]
    assert detail == (
        f"Mode kinds (plugin {T12_ORIGIN}): One option of each kind, for tests.\n"
        "Options (set with --mode-option NAME=VALUE):\n"
        "  flag\n    kind: bool\n    default: false\n"
        "    changeable while running: yes\n    help: a flag\n"
        "  count\n    kind: int\n    default: 3\n    range: 1 to 10\n"
        "    changeable while running: no\n    help: a count\n"
        "  ratio\n    kind: float\n    default: 0.5\n    range: at least 0.0\n"
        "    changeable while running: no\n    help: a ratio\n"
        "  style\n    kind: choice\n    default: flat\n    choices: flat, round\n"
        "    changeable while running: no\n    help: a style\n"
        "  parts\n    kind: choices\n    default: head\n    choices: head, hands, feet\n"
        "    changeable while running: no\n    help: some parts\n"
    )
    assert _imported(t12_plugins, T12_MODULE)
    for name in (T12_OTHER, T12_BROKEN, T12_SLOW):
        assert not _imported(t12_plugins, name)


def test_t12_list_modes_with_erase(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--list-modes", "--mode", "erase"]) == 0
    assert capsys.readouterr().out.endswith(
        "\n\nMode erase (built in): Show the clean background while a subject is present, "
        "the live frame otherwise.\nOptions: none\n"
    )


def test_t12_list_modes_with_a_broken_plugin(
    t12_plugins: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    assert main(["run", "--list-modes", "--mode", "broken"]) == 1
    assert f"  broken (plugin {T12_ORIGIN})" in capsys.readouterr().out
    (record,) = [r for r in caplog.records if r.levelno >= logging.ERROR]
    message = record.getMessage()
    assert message.startswith(f"Fatal error: mode 'broken' ({T12_ORIGIN}) could not be imported")
    assert "\n" not in message


def test_t12_list_modes_selects_the_run_form() -> None:
    from petrify.cli import _run_form_argv

    assert _run_form_argv(["run", "--list-modes"]) == ["--list-modes"]
    assert _run_form_argv(["--verbose", "run", "--list-modes", "--mode", "x"]) == [
        "--verbose", "--list-modes", "--mode", "x"]
    assert _run_form_argv(["--mode-option", "run", "x.mp4", "--list-modes"]) is None


def test_t12_source_is_required_without_list_modes(
    run_stream: mock.MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = _build_run_parser()
    with pytest.raises(SystemExit) as excinfo:
        _resolve_run(parser, parser.parse_args(["--output", OUT]))
    assert excinfo.value.code == 2
    assert "the following arguments are required: --source" in capsys.readouterr().err
    run_stream.assert_not_called()


@pytest.mark.parametrize("verbose", [False, True], ids=["quiet", "verbose"])
def test_t12_broken_plugin_is_a_one_line_error(
    t12_plugins: Path,
    run_stream: mock.MagicMock,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    verbose: bool,
) -> None:
    argv = _run_args("--mode", "broken", *(["--verbose"] if verbose else []))
    assert main(argv) == 1
    (record,) = [r for r in caplog.records if r.levelno >= logging.ERROR]
    message = record.getMessage()
    assert message.startswith(f"Fatal error: mode 'broken' ({T12_ORIGIN}) could not be imported")
    assert "KeyError" in message and "\n" not in message
    # A traceback only on the CLI's existing verbose path.
    assert bool(record.exc_info) is verbose
    if not verbose:
        assert "Traceback" not in capsys.readouterr().err + caplog.text
    run_stream.assert_not_called()
    petrify.sources.open_source.assert_not_called()  # type: ignore[attr-defined]


def test_t12_mode_error_from_run_stream_is_reported_the_same_way(
    t12_plugins: Path, run_stream: mock.MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    from petrify.core.registry import ModeLoadError

    run_stream.side_effect = ModeLoadError("mode 'kinds': create failed: RuntimeError: bug\nx")
    assert main(_run_args("--mode", "kinds")) == 1
    (record,) = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert record.getMessage() == (
        "Fatal error: mode 'kinds': create failed: RuntimeError: bug x "
        f"(mode 'kinds', {T12_ORIGIN})"
    )
    assert not record.exc_info


def _trust_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and "installed plugin" in r.getMessage()]


def test_t12_trust_warning_for_a_plugin_mode(
    t12_plugins: Path, run_stream: mock.MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    assert main(_run_args("--mode", "kinds")) == 0
    assert _trust_warnings(caplog) == [
        f"The mode 'kinds' comes from the installed plugin {T12_ORIGIN}. It runs in the "
        "petrify process with petrify's permissions and receives the unprocessed frames, "
        "people included."
    ]


def test_t12_no_trust_warning_for_erase(
    t12_plugins: Path, run_stream: mock.MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    assert main(_run_args()) == 0
    assert main(_run_args("--mode", "erase")) == 0
    assert _trust_warnings(caplog) == []
    for name in _T12_MODULES:
        assert not _imported(t12_plugins, name)


def test_t12_tracker_flags_reach_the_config() -> None:
    from petrify.core.config import TrackerConfig

    assert _resolved(["--source", IN, "--output", OUT])[0].tracker is None
    assert _resolved(["--source", IN, "--output", OUT, "--track"])[0].tracker == TrackerConfig()
    config, _ = _resolved(["--source", IN, "--output", OUT, "--track-hold", "40"])
    assert config.tracker == TrackerConfig(hold_frames=40)
    config, _ = _resolved(["--source", IN, "--output", OUT, "--track", "--track-hold", "0"])
    assert config.tracker == TrackerConfig(hold_frames=0)
    assert TrackerConfig().hold_frames == 15  # The default the --track-hold help names.
    assert "(implies --track). Default: 15." in " ".join(
        _build_run_parser().format_help().split())


@pytest.mark.parametrize("value", ["-1", "10001"])
def test_t12_track_hold_out_of_range_is_a_usage_error(
    run_stream: mock.MagicMock, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    err = _usage_error(_run_args("--track-hold", value), capsys)
    assert "--track-hold: TrackerConfig.hold_frames must be a whole number from 0 to 10000" in err
    run_stream.assert_not_called()


def test_t12_tracker_with_erase_is_a_clear_startup_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    import cv2
    import numpy as np

    from tests.test_run_stream import ListSink

    for i in range(3):
        cv2.imwrite(str(tmp_path / f"f{i:03d}.png"), np.full((256, 320, 3), 90, np.uint8))
    with mock.patch.object(petrify.pipeline, "_build_sinks", return_value=[ListSink()]):
        code = main(["run", "--source", str(tmp_path), "--preview", "--track",
                     "--background", "running"])
    assert code == 1
    (record,) = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert record.getMessage().startswith(
        "Fatal error: The settings cannot run with the mode 'erase': "
    )
    assert "\n" not in record.getMessage()
    assert not record.exc_info
    assert "Traceback" not in capsys.readouterr().err


def test_t12_model_dir_reaches_the_config(
    run_stream: mock.MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    from petrify.core.config import ModelConfig

    assert _resolved(["--source", IN, "--output", OUT])[0].models == ModelConfig()
    config, _ = _resolved(["--source", IN, "--output", OUT, "--model-dir", "/opt/models"])
    assert config.models == ModelConfig(dir="/opt/models")
    assert main(_run_args("--model-dir", "/opt/models")) == 0
    assert run_stream.call_args.args[1].models == ModelConfig(dir="/opt/models")
    err = _usage_error(_run_args("--model-dir", " "), capsys)
    assert "--model-dir: ModelConfig.dir must be None or a path" in err


def test_t12_file_form_ignores_broken_and_slow_plugins(
    t12_plugins: Path, run_file: mock.MagicMock
) -> None:
    import petrify.core.registry

    with mock.patch.object(
        petrify.core.registry, "list_modes", side_effect=AssertionError("discovery ran")
    ):
        assert main([IN, OUT]) == 0
        assert main(["run", OUT]) == 0
        with pytest.raises(SystemExit) as excinfo:
            main(["--help"])
        assert excinfo.value.code == 0
    assert run_file.call_count == 2
    assert run_file.call_args_list[0].args[2] == PetrifyConfig()
    for name in _T12_MODULES:
        assert not _imported(t12_plugins, name)


def test_t12_building_the_run_parser_does_not_discover_modes() -> None:
    import petrify.core.registry

    # Only the run form's own help lists the choices and so discovers the modes.
    with mock.patch.object(
        petrify.core.registry, "list_modes", side_effect=AssertionError("discovery ran")
    ):
        _build_run_parser().format_usage()
        args = _build_run_parser().parse_args(["--source", IN, "--mode", "erase"])
    assert args.mode == "erase"


# ---------------------------------------------------------------------------
# T13: the web run gives the control the loaded mode and the listed modes
# ---------------------------------------------------------------------------


def test_t13_web_run_control_knows_the_erase_mode(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    from petrify.modes.erase import ERASE

    rs, _, _ = _web_run(["--source", IN, "--web", "--web-port", "0"], capsys, caplog)
    data = rs.seen["kwargs"]["control"].snapshot().to_dict()  # type: ignore[attr-defined]
    assert data["mode"] == {"name": "erase", "origin": "built in", "builtin": True,
                            "summary": ERASE.summary, "options": []}
    erase = [m for m in data["modes"]["listed"] if m["name"] == "erase"]
    assert erase == [{"name": "erase", "origin": "built in", "builtin": True,
                      "selectable": True, "problem": None, "active": True}]


def test_t13_web_run_control_knows_a_plugin_mode_and_imports_no_other(
    t12_plugins: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    rs, _, _ = _web_run(["--source", IN, "--web", "--web-port", "0", "--mode", "kinds",
                         "--mode-option", "count=4"], capsys, caplog)
    control = rs.seen["kwargs"]["control"]  # type: ignore[attr-defined]
    data = control.snapshot().to_dict()
    assert (data["mode"]["name"], data["mode"]["origin"], data["mode"]["builtin"]) == (
        "kinds", T12_ORIGIN, False)
    options = {opt["name"]: opt for opt in data["mode"]["options"]}
    assert options["count"]["value"] == 4 and options["flag"]["runtime"] is True
    listed = {(m["name"], m["origin"]): m for m in data["modes"]["listed"]}
    assert listed[("kinds", T12_ORIGIN)]["active"] is True
    assert listed[("broken", T12_ORIGIN)]["selectable"] is True
    assert listed[("erase", T12_ORIGIN)]["selectable"] is False
    # The runtime option is accepted through the control the page uses.
    assert control.submit({"mode_options": {"flag": True}}).mode_options_dict()["flag"] is True
    for name in (T12_OTHER, T12_BROKEN, T12_SLOW):
        assert not _imported(t12_plugins, name)


# ---------------------------------------------------------------------------
# T8: detectors with model files
# ---------------------------------------------------------------------------


def test_t8_no_new_flag_leaves_the_new_sections_off() -> None:
    for spec in (IN, "camera:0"):
        config, _ = _resolved(["--source", spec, "--output", OUT])
        assert (config.person_dnn, config.faces, config.segment) == (None, None, None)
        assert config.background.sample_checkers == ("hog",)


def test_t8_detector_flags_reach_the_config() -> None:
    from petrify.core.config import FaceYuNetConfig, PersonDnnConfig, SegmentConfig

    config, _ = _resolved(["--source", IN, "--output", OUT, "--person-detector", "dnn",
                           "--faces", "--segment"])
    assert config.person_dnn == PersonDnnConfig()
    assert config.faces == FaceYuNetConfig()
    assert config.segment == SegmentConfig()
    assert _resolved(["--source", IN, "--output", OUT, "--person-detector", "hog"])[
        0].person_dnn is None
    config, _ = _resolved(["--source", IN, "--output", OUT, "--person-detector", "dnn",
                           "--dnn-score-threshold", "0.55", "--faces",
                           "--face-score-threshold", "0.7", "--segment",
                           "--segment-threshold", "0.4"])
    assert config.person_dnn == PersonDnnConfig(score_threshold=0.55)
    assert config.faces == FaceYuNetConfig(score_threshold=0.7)
    assert config.segment == SegmentConfig(threshold=0.4)


def test_t8_help_defaults_match_the_config() -> None:
    from petrify.core.config import FaceYuNetConfig, PersonDnnConfig, SegmentConfig

    text = " ".join(_build_run_parser().format_help().split())
    assert ("dnn). It also applies when dnn checks the background samples "
            f"(--background-check). Default: {PersonDnnConfig().score_threshold}.") in text
    assert ("(with --faces). It also applies when yunet checks the background samples "
            f"(--background-check). Default: {FaceYuNetConfig().score_threshold}.") in text
    assert "The list replaces the default, so include hog to keep it" in text
    assert "misses people smaller than roughly 110 px tall on a 1080p frame" in text
    assert f"--segment). Default: {SegmentConfig().threshold}." in text


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        ("hog", ("hog",)),
        ("dnn", ("dnn",)),
        ("hog,yunet,dnn", ("hog", "yunet", "dnn")),
        (" dnn , hog ", ("dnn", "hog")),
    ],
)
def test_t8_background_check_names_reach_the_config(names: str, expected: tuple[str, ...]) -> None:
    config, _ = _resolved(["--source", "camera:0", "--preview", "--background-check", names])
    assert config.background.sample_checkers == expected
    config, _ = _resolved(["--source", IN, "--output", OUT, "--background", "calibrate",
                           "--background-check", names])
    assert config.background.sample_checkers == expected


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--dnn-score-threshold", "0.5"], "--dnn-score-threshold requires --person-detector dnn"),
        (["--face-score-threshold", "0.5"], "--face-score-threshold requires --faces"),
        (["--segment-threshold", "0.5"], "--segment-threshold requires --segment"),
        (["--person-detector", "dnn", "--dnn-score-threshold", "1"],
         "PersonDnnConfig.score_threshold must be a number in (0, 1)"),
        (["--faces", "--face-score-threshold", "0"],
         "FaceYuNetConfig.score_threshold must be a number in (0, 1)"),
        (["--segment", "--segment-threshold", "nan"],
         "SegmentConfig.threshold must be a number in (0, 1)"),
        (["--person-detector", "cnn"], "invalid choice: 'cnn'"),
        (["--background", "running", "--background-check", "hog,mog2"],
         "--background-check: BackgroundConfig.sample_checkers entries must be one of"),
        (["--background", "running", "--background-check", "hog,segment"],
         "--background-check: BackgroundConfig.sample_checkers entries must be one of hog, "
         "yunet, dnn, got 'segment'"),
        (["--background", "running", "--background-check", "hog,hog"],
         "--background-check: BackgroundConfig.sample_checkers must not repeat"),
        (["--background", "running", "--background-check", ""],
         "--background-check: BackgroundConfig.sample_checkers entries must be one of"),
        (["--background", "running", "--background-check", "dnn", "--no-background-check"],
         "--background-check cannot be combined with --no-background-check"),
        (["--background-check", "dnn"], "--background-check applies to --background running"),
    ],
)
def test_t8_detector_usage_errors(
    run_stream: mock.MagicMock,
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
    message: str,
) -> None:
    err = _usage_error(_run_args(*extra), capsys)
    assert message in " ".join(err.split())
    run_stream.assert_not_called()


@pytest.mark.parametrize("verbose", [False, True])
def test_t8_model_error_is_one_line_and_exit_1(
    run_stream: mock.MagicMock,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    verbose: bool,
) -> None:
    from petrify.models import ModelNotFoundError

    # A ModelNotFoundError is also a FileNotFoundError; it must not become exit code 2.
    run_stream.side_effect = ModelNotFoundError("Model 'm' was not found.\nLocations:\n  1. /x")
    argv = _run_args("--person-detector", "dnn", *(["--verbose"] if verbose else []))
    assert main(argv) == 1
    (record,) = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert record.getMessage() == "Fatal error: Model 'm' was not found. Locations: 1. /x"
    assert bool(record.exc_info) is verbose
    if not verbose:
        assert "Traceback" not in capsys.readouterr().err + caplog.text


def test_t8_model_dir_and_flags_reach_run_stream(run_stream: mock.MagicMock) -> None:
    from petrify.core.config import ModelConfig, PersonDnnConfig

    assert main(_run_args("--person-detector", "dnn", "--model-dir", "/opt/models")) == 0
    config = run_stream.call_args.args[1]
    assert config.person_dnn == PersonDnnConfig()
    assert config.models == ModelConfig(dir="/opt/models")


def test_t8_missing_model_fails_fast_with_one_line_and_exit_1(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cv2
    import numpy as np

    from tests.test_run_stream import ListSink

    monkeypatch.delenv("PETRIFY_MODEL_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # No per-user model directory.
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"f{i:03d}.png"), np.full((256, 320, 3), 90, np.uint8))
    with (
        mock.patch.object(petrify.pipeline, "_build_sinks", return_value=[ListSink()]),
        mock.patch.object(petrify.pipeline, "locked_background",
                          side_effect=AssertionError("pre-scan ran")) as prescan,
    ):
        code = main(["run", "--source", str(tmp_path), "--preview", "--detector", "both",
                     "--person-detector", "dnn", "--model-dir", str(tmp_path / "none")])
    assert code == 1
    prescan.assert_not_called()
    (record,) = [r for r in caplog.records if r.levelno >= logging.ERROR]
    message = record.getMessage()
    assert message.startswith(
        "Fatal error: Model 'object_detection_nanodet_2022nov' "
        "(file 'object_detection_nanodet_2022nov.onnx') was not found."
    )
    assert str(tmp_path / "none") in message
    assert "\n" not in message
    assert "Traceback" not in capsys.readouterr().err
