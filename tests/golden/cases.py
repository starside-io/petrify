"""Golden case matrix, fixture paths, and a producer-neutral record comparison.

Nothing in this module runs a pipeline. ``compare_records`` only looks at two
``GoldenRecord`` values, so any producer (legacy or streaming) can be checked
against the committed fixtures with it.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Sequence
from pathlib import Path

from tests.golden.harness import GoldenRecord, from_json

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

FRAME_FIELDS = ("raw", "filled", "final")
DIGEST_FIELD = "frame_digests"
SCALAR_FIELDS = ("background_index", "roi", "total_frames", "width", "height", "fps")

_MAX_LISTED_INDICES = 10


@dataclasses.dataclass(frozen=True)
class GoldenCase:
    case_id: str
    clip: str
    argv: tuple[str, ...] = ()

    @property
    def clip_path(self) -> Path:
        return clip_path(self.clip)

    @property
    def fixture_path(self) -> Path:
        return fixture_path(self.case_id)


CASES: tuple[GoldenCase, ...] = (
    GoldenCase("camera-default", "camera.mp4"),
    GoldenCase("camera-detector-both", "camera.mp4", ("--detector", "both")),
    GoldenCase("camera2-default", "camera2.mp4"),
    GoldenCase("camera2-detector-both", "camera2.mp4", ("--detector", "both")),
    GoldenCase("camera2-detector-person", "camera2.mp4", ("--detector", "person")),
    GoldenCase("camera2-no-bg-diff", "camera2.mp4", ("--no-bg-diff",)),
    GoldenCase("camera2-gap-fill-0", "camera2.mp4", ("--gap-fill", "0")),
    GoldenCase("camera2-background-frame-0", "camera2.mp4", ("--background-frame", "0")),
)

CASES_BY_ID: dict[str, GoldenCase] = {case.case_id: case for case in CASES}


def clip_path(clip: str) -> Path:
    """Return the path of a demo clip by file name."""
    return DATA_DIR / clip


def fixture_path(case_id: str) -> Path:
    """Return the path of the committed fixture for *case_id*."""
    return FIXTURES_DIR / f"{case_id}.json"


def load_fixture(case_id: str) -> GoldenRecord:
    """Read and parse the fixture for *case_id* (raises FileNotFoundError if absent)."""
    return from_json(fixture_path(case_id).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compare_frames(field: str, actual: Sequence[object], expected: Sequence[object]) -> list[str]:
    diffs: list[str] = []
    if len(actual) != len(expected):
        diffs.append(f"{field}: length differs (actual {len(actual)}, expected {len(expected)})")
    differing = [i for i, (a, e) in enumerate(zip(actual, expected)) if a != e]
    if differing:
        shown = ", ".join(str(i) for i in differing[:_MAX_LISTED_INDICES])
        more = "" if len(differing) <= _MAX_LISTED_INDICES else ", ..."
        diffs.append(
            f"{field}: {len(differing)} differing frame(s) "
            f"(of {min(len(actual), len(expected))} compared), first indices: {shown}{more}"
        )
    return diffs


def compare_records(actual: GoldenRecord, expected: GoldenRecord) -> list[str]:
    """Return human-readable differences between two records, empty when they match.

    Compared: the per-frame fields (raw, filled, final), the per-frame output digests
    (frame_digests), background_index, roi, total_frames, width, height, fps. Exact
    equality, no tolerance.
    ``opencv_version`` is deliberately not compared (see ``describe_failure``).
    """
    diffs: list[str] = []
    for field in SCALAR_FIELDS:
        a = getattr(actual, field)
        e = getattr(expected, field)
        if a != e:
            diffs.append(f"{field}: actual {a!r}, expected {e!r}")
    for field in FRAME_FIELDS:
        diffs.extend(_compare_frames(field, getattr(actual, field), getattr(expected, field)))
    diffs.extend(
        _compare_frames(DIGEST_FIELD, getattr(actual, DIGEST_FIELD), getattr(expected, DIGEST_FIELD))
    )
    return diffs


def describe_failure(
    case_id: str, diffs: Sequence[str], actual: GoldenRecord, expected: GoldenRecord
) -> str:
    """Format *diffs* into one failure message, noting an OpenCV version mismatch if any."""
    lines = [f"Golden mismatch for case {case_id!r}:"]
    lines.extend(f"  - {d}" for d in diffs)
    if actual.opencv_version != expected.opencv_version:
        lines.append(
            f"  Note: OpenCV version differs (actual {actual.opencv_version}, "
            f"fixture {expected.opencv_version}), which may explain the mismatch."
        )
    return "\n".join(lines)
