"""Regenerate golden fixtures from the legacy pipeline.

Usage: python -m tests.golden.regenerate [--case ID ...] [--overwrite]

Each case is captured twice and the two ``to_json`` outputs must be
byte-identical before the fixture is written. Existing fixtures are never
overwritten unless ``--overwrite`` is passed.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence

from tests.golden.cases import CASES, CASES_BY_ID, GoldenCase, compare_records
from tests.golden.harness import GoldenRecord, capture_legacy, to_json


class CaseError(Exception):
    pass


def _timed_capture(case: GoldenCase, label: str) -> tuple[GoldenRecord, str]:
    start = time.perf_counter()
    record = capture_legacy(case.clip_path, case.argv)
    elapsed = time.perf_counter() - start
    print(f"[{case.case_id}] capture {label}: {elapsed:.1f} s", flush=True)
    return record, to_json(record)


def _regenerate(case: GoldenCase, overwrite: bool) -> None:
    target = case.fixture_path
    if target.exists() and not overwrite:
        raise CaseError(f"fixture already exists at {target} (pass --overwrite to replace it)")
    if not case.clip_path.exists():
        raise CaseError(f"clip not found: {case.clip_path}")

    first, first_json = _timed_capture(case, "1/2")
    second, second_json = _timed_capture(case, "2/2")
    if first_json != second_json:
        diffs = compare_records(second, first)
        other = [
            name
            for name in ("clip", "clip_sha256", "argv", "opencv_version", "schema_version")
            if getattr(first, name) != getattr(second, name)
        ]
        detail = "; ".join(diffs + [f"{name} differs" for name in other])
        raise CaseError(
            "determinism gate failed: two captures produced different JSON"
            + (f" ({detail})" if detail else "")
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(first_json)
    print(f"[{case.case_id}] wrote {target}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.golden.regenerate")
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        metavar="ID",
        choices=sorted(CASES_BY_ID),
        help="Only regenerate this case (repeatable). Default: all cases.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace fixtures that already exist.",
    )
    args = parser.parse_args(argv)

    selected = [CASES_BY_ID[c] for c in args.cases] if args.cases else list(CASES)
    failures: list[str] = []
    for case in selected:
        try:
            _regenerate(case, args.overwrite)
        except CaseError as exc:
            print(f"[{case.case_id}] ERROR: {exc}", file=sys.stderr, flush=True)
            failures.append(case.case_id)

    if failures:
        print(f"Failed cases: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"OK: {len(selected)} case(s) regenerated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
