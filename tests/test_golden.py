"""Parity tests: re-run the legacy pipeline and the streaming engine against committed golden fixtures."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from tests.golden.cases import (
    CASES,
    DIGEST_FIELD,
    GoldenCase,
    _compare_frames,
    compare_records,
    describe_failure,
    file_sha256,
    load_fixture,
)
from tests.golden.harness import (
    GoldenRecord,
    capture_cli_output,
    capture_engine,
    capture_legacy,
    capture_run_cli_output,
)

Producer = Callable[[Path, Sequence[str]], GoldenRecord]


CLI_CASES = tuple(c for c in CASES if c.clip == "camera2.mp4")


def _load_checked_fixture(case: GoldenCase) -> GoldenRecord:
    """Fail on a missing or mismatched fixture, skip without the clip, return the fixture."""
    if not case.fixture_path.exists():
        pytest.fail(
            f"Golden fixture missing for case {case.case_id!r}: {case.fixture_path}. "
            "Generate it with: python -m tests.golden.regenerate --case " + case.case_id
        )
    expected = load_fixture(case.case_id)
    if expected.argv != list(case.argv) or expected.clip != case.clip:
        pytest.fail(
            f"Golden fixture for case {case.case_id!r} was captured with clip "
            f"{expected.clip!r} and argv {expected.argv!r}, but the case declares clip "
            f"{case.clip!r} and argv {list(case.argv)!r}.",
            pytrace=False,
        )

    if not case.clip_path.exists():
        pytest.skip(f"demo clip not available: {case.clip_path}")
    actual_sha = file_sha256(case.clip_path)
    if actual_sha != expected.clip_sha256:
        pytest.skip(
            f"clip {case.clip_path} has sha256 {actual_sha}, fixture was captured from "
            f"{expected.clip_sha256}"
        )
    return expected


def _check_case(case: GoldenCase, producer: Producer) -> None:
    """Check the shared preconditions, then compare the full record with the fixture."""
    expected = _load_checked_fixture(case)
    actual = producer(case.clip_path, case.argv)
    diffs = compare_records(actual, expected)
    if diffs:
        pytest.fail(describe_failure(case.case_id, diffs, actual, expected), pytrace=False)


@pytest.mark.golden
@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_legacy_matches_golden(case: GoldenCase) -> None:
    _check_case(case, capture_legacy)


@pytest.mark.golden
@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_engine_matches_golden(case: GoldenCase) -> None:
    _check_case(case, capture_engine)


@pytest.mark.golden
@pytest.mark.parametrize("case", CLI_CASES, ids=[c.case_id for c in CLI_CASES])
def test_cli_matches_golden(case: GoldenCase) -> None:
    expected = _load_checked_fixture(case)
    frame_digests, output = capture_cli_output(case.clip_path, case.argv)
    diffs: list[str] = []
    if output != expected.output:
        diffs.append(f"output: actual {output!r}, expected {expected.output!r}")
    diffs.extend(_compare_frames(DIGEST_FIELD, frame_digests, expected.frame_digests))
    if diffs:
        lines = [f"CLI golden mismatch for case {case.case_id!r}:"]
        lines.extend(f"  - {d}" for d in diffs)
        pytest.fail("\n".join(lines), pytrace=False)


# ``petrify run`` takes --lookahead where the file form takes --gap-fill (same
# meaning); every other flag of these cases is accepted unchanged.
_RUN_FLAG_RENAMES = {"--gap-fill": "--lookahead"}


def _run_argv(argv: Sequence[str]) -> tuple[str, ...]:
    return tuple(_RUN_FLAG_RENAMES.get(a, a) for a in argv)


@pytest.mark.golden
@pytest.mark.parametrize("case", CLI_CASES, ids=[c.case_id for c in CLI_CASES])
def test_run_cli_matches_golden(case: GoldenCase) -> None:
    expected = _load_checked_fixture(case)
    frame_digests = capture_run_cli_output(case.clip_path, _run_argv(case.argv))
    diffs = _compare_frames(DIGEST_FIELD, frame_digests, expected.frame_digests)
    if diffs:
        lines = [f"petrify run golden mismatch for case {case.case_id!r}:"]
        lines.extend(f"  - {d}" for d in diffs)
        pytest.fail("\n".join(lines), pytrace=False)
