"""Golden capture harness for the legacy two-pass pipeline.

Runs the real, unmodified ``petrify.pipeline.run()`` and records its per-frame
classification by spying on the helpers it calls. The JSON format produced by
``to_json`` is producer-neutral so another engine can be compared against it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import tempfile
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any
from unittest import mock

import cv2

import petrify.cli
import petrify.pipeline

SCHEMA_VERSION = 1

_FRAME_FIELDS = ("raw", "filled", "final")


@dataclasses.dataclass
class GoldenRecord:
    schema_version: int
    clip: str
    clip_sha256: str
    argv: list[str]
    opencv_version: str
    fps: float
    width: int
    height: int
    total_frames: int
    background_index: int
    roi: list[int] | None
    raw: list[bool]
    filled: list[bool]
    final: list[bool]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe(path: Path) -> tuple[float, int, int, int]:
    """Read fps, width, height, total_frames with the same expressions as ``run()``."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise OSError(f"Cannot open video file: {path}")
        fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return fps, width, height, total_frames


@contextlib.contextmanager
def _spy(name: str, results: dict[str, list[Any]]) -> Iterator[None]:
    """Replace ``petrify.pipeline.<name>`` with a pass-through that records return values."""
    real: Callable[..., Any] = getattr(petrify.pipeline, name)
    calls: list[Any] = []
    results[name] = calls

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        value = real(*args, **kwargs)
        calls.append(value)
        return value

    with mock.patch.object(petrify.pipeline, name, new=wrapper):
        yield


def _single(results: dict[str, list[Any]], name: str) -> Any:
    calls = results[name]
    if len(calls) != 1:
        raise RuntimeError(
            f"Expected petrify.pipeline.{name} to be called exactly once, got {len(calls)} calls."
        )
    return calls[0]


def capture_legacy(clip_path: str | Path, extra_argv: Sequence[str] = ()) -> GoldenRecord:
    """Run the legacy pipeline on *clip_path* and return its per-frame classification."""
    clip = Path(clip_path)
    argv = [str(a) for a in extra_argv]
    spied = (
        "_classify_frames",
        "_fill_gaps",
        "_freeze_cold_start",
        "_find_clean_background",
        "auto_detect_roi",
    )
    results: dict[str, list[Any]] = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_out = Path(tmp_dir) / "out.mp4"
        args = petrify.cli._build_parser().parse_args([str(clip), str(tmp_out), *argv])
        with contextlib.ExitStack() as stack:
            for name in spied:
                stack.enter_context(_spy(name, results))
            stack.enter_context(
                mock.patch.object(petrify.pipeline, "ffmpeg_encode", new=lambda **_: None)
            )
            petrify.pipeline.run(args)

    raw = _single(results, "_classify_frames")
    filled = _single(results, "_fill_gaps")
    final = _single(results, "_freeze_cold_start")

    if args.background_frame is not None:
        background_index = int(args.background_frame)
    else:
        background_index = int(_single(results, "_find_clean_background")[1])

    roi: list[int] | None
    if args.roi:
        roi = [int(v) for v in args.roi]
    elif args.no_roi:
        roi = None
    else:
        detected = _single(results, "auto_detect_roi")
        roi = None if detected is None else [int(v) for v in detected]

    fps, width, height, total_frames = _probe(clip)

    return GoldenRecord(
        schema_version=SCHEMA_VERSION,
        clip=clip.name,
        clip_sha256=_sha256(clip),
        argv=argv,
        opencv_version=cv2.__version__,
        fps=fps,
        width=width,
        height=height,
        total_frames=total_frames,
        background_index=background_index,
        roi=roi,
        raw=[bool(v) for v in raw],
        filled=[bool(v) for v in filled],
        final=[bool(v) for v in final],
    )


def _encode_bits(values: Sequence[bool]) -> str:
    return "".join("1" if v else "0" for v in values)


def _decode_bits(text: str) -> list[bool]:
    if set(text) - {"0", "1"}:
        raise ValueError("Per-frame field must contain only '0' and '1' characters.")
    return [c == "1" for c in text]


def to_json(record: GoldenRecord) -> str:
    """Serialize *record* deterministically (sorted keys, trailing newline)."""
    data = dataclasses.asdict(record)
    for field in _FRAME_FIELDS:
        data[field] = _encode_bits(data[field])
    return json.dumps(data, sort_keys=True, indent=2) + "\n"


def from_json(text: str) -> GoldenRecord:
    """Parse JSON produced by ``to_json`` back into a ``GoldenRecord``."""
    data = json.loads(text)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version: {data.get('schema_version')!r}")
    for field in _FRAME_FIELDS:
        data[field] = _decode_bits(data[field])
    return GoldenRecord(**data)
