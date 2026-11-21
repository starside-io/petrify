"""CLI routing tests: ``petrify.cli.main`` drives the streaming engine via ``run_file``."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterator
from pathlib import Path
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
