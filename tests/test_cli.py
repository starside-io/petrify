"""CLI routing tests: ``petrify.cli.main`` drives the streaming engine via ``run_file``."""

from __future__ import annotations

from collections.abc import Iterator
from unittest import mock

import pytest

import petrify
import petrify.pipeline
from petrify.cli import _build_parser, main
from petrify.core.config import PetrifyConfig
from petrify.core.engine import EngineStats
from petrify.pipeline import RunReport

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
    assert "usage: petrify" in capsys.readouterr().out
    run_file.assert_not_called()
