"""Golden capture harness for the legacy two-pass pipeline.

Runs the real, unmodified ``petrify.pipeline.run()`` and records its per-frame
classification by spying on the helpers it calls. The JSON format produced by
``to_json`` is producer-neutral so another engine can be compared against it.
``capture_engine`` is that second producer: it runs the streaming engine
(``petrify.pipeline.run_file``) and records the same fields from its observer,
report, and output stand-ins.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import inspect
import json
import tempfile
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import mock

import cv2

import petrify.cli
import petrify.encoder
import petrify.pipeline
import petrify.sinks.file
from petrify.core.config import PetrifyConfig
from petrify.core.engine import FrameTrace

SCHEMA_VERSION = 3

_FRAME_FIELDS = ("raw", "filled", "final")

_WRITER_PARAMS = ("filename", "fourcc", "fps", "frameSize")


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
    frame_digests: list[str]
    # Output stage request: {"fps", "width", "height", "encode": {"codec", "crf",
    # "preset", "hwaccel", "audio_from_input", "output_is_requested_path"}}.
    output: dict[str, Any]


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


def _frame_digest(frame: Any) -> str:
    """Hash a frame's shape, dtype, and raw C-order bytes into 16 hex characters."""
    digest = hashlib.blake2b(digest_size=8)
    digest.update(repr(tuple(int(d) for d in frame.shape)).encode("ascii"))
    digest.update(frame.dtype.str.encode("ascii"))
    digest.update(frame.tobytes(order="C"))
    return digest.hexdigest()


@contextlib.contextmanager
def _record_written_frames(
    digests: list[str], writers: list[dict[str, Any]]
) -> Iterator[None]:
    """Replace ``cv2.VideoWriter`` with a stand-in that records a digest of each written frame.

    Each construction appends the requested ``fps``, ``width`` and ``height`` to *writers*.
    """

    class _DigestWriter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if len(args) > len(_WRITER_PARAMS) + 1:
                raise RuntimeError(f"Unsupported video writer signature: {len(args)} args.")
            params = dict(zip(_WRITER_PARAMS, args))
            params.update(kwargs)
            missing = [p for p in ("fps", "frameSize") if p not in params]
            if missing:
                raise RuntimeError(f"Video writer constructed without {', '.join(missing)}.")
            frame_w, frame_h = params["frameSize"]
            writers.append(
                {"fps": float(params["fps"]), "width": int(frame_w), "height": int(frame_h)}
            )

        def isOpened(self) -> bool:
            return True

        def write(self, frame: Any) -> None:
            digests.append(_frame_digest(frame))

        def release(self) -> None:
            pass

    with mock.patch.object(cv2, "VideoWriter", new=_DigestWriter):
        yield


@contextlib.contextmanager
def _record_encode_requests(
    requests: list[inspect.BoundArguments], module: ModuleType = petrify.pipeline
) -> Iterator[None]:
    """Replace *module*'s encode step with a no-op that records its bound arguments."""
    signature = inspect.signature(petrify.encoder.ffmpeg_encode)

    def recorder(*args: Any, **kwargs: Any) -> None:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        requests.append(bound)

    with mock.patch.object(module, "ffmpeg_encode", new=recorder):
        yield


def _output_request(
    writers: list[dict[str, Any]],
    encodes: list[inspect.BoundArguments],
    clip: Path,
    requested_output: Path,
) -> dict[str, Any]:
    """Build the record's ``output`` field, failing loudly unless one writer and one encode."""
    if len(writers) != 1:
        raise RuntimeError(
            f"Expected the output video writer to be opened exactly once, got {len(writers)}."
        )
    if len(encodes) != 1:
        raise RuntimeError(
            f"Expected the encode step to be requested exactly once, got {len(encodes)}."
        )
    encode = encodes[0].arguments
    return {
        **writers[0],
        "encode": {
            "codec": encode["codec"],
            "crf": int(encode["crf"]),
            "preset": encode["preset"],
            "hwaccel": encode["hwaccel"],
            "audio_from_input": _same_path(encode["original_audio_source"], clip),
            "output_is_requested_path": _same_path(encode["output_path"], requested_output),
        },
    }


def _build_record(
    clip: Path,
    argv: list[str],
    *,
    background_index: int,
    roi: list[int] | None,
    raw: Sequence[bool],
    filled: Sequence[bool],
    final: Sequence[bool],
    frame_digests: list[str],
    output: dict[str, Any],
) -> GoldenRecord:
    """Assemble a record, adding the clip probe and metadata shared by every producer."""
    if not frame_digests:
        raise RuntimeError("Expected at least one composited frame to be written, got none.")
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
        frame_digests=frame_digests,
        output=output,
    )


def _same_path(a: Any, b: Path) -> bool:
    return Path(a).resolve() == b.resolve()


def _single(results: dict[str, list[Any]], name: str) -> Any:
    calls = results[name]
    if len(calls) != 1:
        raise RuntimeError(
            f"Expected petrify.pipeline.{name} to be called exactly once, got {len(calls)} calls."
        )
    return calls[0]


def capture_legacy(clip_path: str | Path, extra_argv: Sequence[str] = ()) -> GoldenRecord:
    """Run the legacy pipeline on *clip_path* and return its per-frame record."""
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
    frame_digests: list[str] = []
    writers: list[dict[str, Any]] = []
    encodes: list[inspect.BoundArguments] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_out = Path(tmp_dir) / "out.mp4"
        args = petrify.cli._build_parser().parse_args([str(clip), str(tmp_out), *argv])
        with contextlib.ExitStack() as stack:
            for name in spied:
                stack.enter_context(_spy(name, results))
            stack.enter_context(_record_encode_requests(encodes))
            stack.enter_context(_record_written_frames(frame_digests, writers))
            petrify.pipeline.run(args)

        output = _output_request(writers, encodes, clip, tmp_out)

    if not frame_digests:
        raise RuntimeError("Expected at least one composited frame to be written, got none.")

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

    return _build_record(
        clip,
        argv,
        background_index=background_index,
        roi=roi,
        raw=raw,
        filled=filled,
        final=final,
        frame_digests=frame_digests,
        output=output,
    )


def capture_engine(clip_path: str | Path, extra_argv: Sequence[str] = ()) -> GoldenRecord:
    """Run the streaming engine (``run_file``) on *clip_path* and return its per-frame record.

    The record is built only from what the run exposes: the observer traces, the
    returned report, and the video writer and encoder stand-ins, plus the clip probe.
    """
    clip = Path(clip_path)
    argv = [str(a) for a in extra_argv]
    traces: list[FrameTrace] = []
    frame_digests: list[str] = []
    writers: list[dict[str, Any]] = []
    encodes: list[inspect.BoundArguments] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_out = Path(tmp_dir) / "out.mp4"
        args = petrify.cli._build_parser().parse_args([str(clip), str(tmp_out), *argv])
        config = PetrifyConfig.from_namespace(args)
        with contextlib.ExitStack() as stack:
            stack.enter_context(_record_encode_requests(encodes, petrify.sinks.file))
            stack.enter_context(_record_written_frames(frame_digests, writers))
            report = petrify.pipeline.run_file(args.input, args.output, config, traces.append)

        output = _output_request(writers, encodes, clip, tmp_out)

    if [t.index for t in traces] != list(range(len(traces))):
        raise RuntimeError("Expected the engine to report frames in stream order, exactly once.")

    return _build_record(
        clip,
        argv,
        background_index=int(report.background_index),
        roi=None if report.roi is None else [int(v) for v in report.roi],
        raw=[t.raw for t in traces],
        filled=[t.filled for t in traces],
        final=[t.final for t in traces],
        frame_digests=frame_digests,
        output=output,
    )


def capture_cli_output(
    clip_path: str | Path, extra_argv: Sequence[str] = ()
) -> tuple[list[str], dict[str, Any]]:
    """Run ``petrify.cli.main`` on *clip_path* and return its frame digests and output request.

    Only the written frames and the output stage request are recorded: the CLI exposes
    no per-frame classification.
    """
    clip = Path(clip_path)
    argv = [str(a) for a in extra_argv]
    frame_digests: list[str] = []
    writers: list[dict[str, Any]] = []
    encodes: list[inspect.BoundArguments] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_out = Path(tmp_dir) / "out.mp4"
        with contextlib.ExitStack() as stack:
            stack.enter_context(_record_encode_requests(encodes, petrify.sinks.file))
            stack.enter_context(_record_written_frames(frame_digests, writers))
            code = petrify.cli.main([str(clip), str(tmp_out), *argv])
        if code != 0:
            raise RuntimeError(f"Expected petrify.cli.main to exit 0, got {code}.")

        output = _output_request(writers, encodes, clip, tmp_out)

    if not frame_digests:
        raise RuntimeError("Expected at least one composited frame to be written, got none.")
    return frame_digests, output


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
