"""Parity test: re-run the legacy pipeline and compare it against committed golden fixtures."""

from __future__ import annotations

import pytest

from tests.golden.cases import (
    CASES,
    GoldenCase,
    compare_records,
    describe_failure,
    file_sha256,
    load_fixture,
)
from tests.golden.harness import capture_legacy


@pytest.mark.golden
@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_legacy_matches_golden(case: GoldenCase) -> None:
    if not case.fixture_path.exists():
        pytest.fail(
            f"Golden fixture missing for case {case.case_id!r}: {case.fixture_path}. "
            "Generate it with: python -m tests.golden.regenerate --case " + case.case_id
        )
    expected = load_fixture(case.case_id)

    if not case.clip_path.exists():
        pytest.skip(f"demo clip not available: {case.clip_path}")
    actual_sha = file_sha256(case.clip_path)
    if actual_sha != expected.clip_sha256:
        pytest.skip(
            f"clip {case.clip_path} has sha256 {actual_sha}, fixture was captured from "
            f"{expected.clip_sha256}"
        )

    actual = capture_legacy(case.clip_path, case.argv)
    diffs = compare_records(actual, expected)
    if diffs:
        pytest.fail(describe_failure(case.case_id, diffs, actual, expected), pytrace=False)
