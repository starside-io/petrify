"""Parity tests: re-run the legacy pipeline and the streaming engine against committed golden fixtures."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from tests.golden.cases import (
    CASES,
    GoldenCase,
    compare_records,
    describe_failure,
    file_sha256,
    load_fixture,
)
from tests.golden.harness import GoldenRecord, capture_engine, capture_legacy

Producer = Callable[[Path, Sequence[str]], GoldenRecord]


def _check_case(case: GoldenCase, producer: Producer) -> None:
    """Fail on a missing or mismatched fixture, skip without the clip, then compare."""
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
