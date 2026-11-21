"""Subprocess helper for the FfmpegPipeSink Ctrl-C test (not collected by pytest).

Usage: python _helper_pipe_sink_sigint.py OUTPUT FRAMES_BEFORE FRAMES_AFTER

Opens the sink, writes FRAMES_BEFORE frames, prints "ready <ffmpeg pid>", waits
for SIGINT (KeyboardInterrupt), writes FRAMES_AFTER more frames and closes the
sink, then prints "closed". Exit code 0 on success, 1 when the sink failed, 3
when no SIGINT arrived in time.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from petrify.core import EncodeConfig, FrameMeta, SourceInfo
from petrify.sinks import FfmpegPipeSink

W, H = 64, 48
INFO = SourceInfo(fps=25.0, width=W, height=H, frame_count=None, seekable=False, live=True)
WAIT_FOR_SIGINT_S = 60.0


def _write(sink: FfmpegPipeSink, i: int) -> None:
    image = np.full((H, W, 3), (i * 9) % 256, dtype=np.uint8)
    sink.write(image, FrameMeta(index=i, timestamp=i / INFO.fps, source=INFO))


def main() -> int:
    output, before, after = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    sink = FfmpegPipeSink(output, EncodeConfig(preset="ultrafast"))
    sink.open(INFO)
    try:
        for i in range(before):
            _write(sink, i)
        try:
            print(f"ready {sink._proc.pid}", flush=True)
            deadline = time.monotonic() + WAIT_FOR_SIGINT_S
            while time.monotonic() < deadline:
                time.sleep(0.01)
            print("no SIGINT", flush=True)
            return 3
        except KeyboardInterrupt:
            pass
        try:
            for i in range(before, before + after):
                _write(sink, i)
            sink.close()
        except Exception as exc:
            print(f"failed: {exc!r}", flush=True)
            return 1
        print("closed", flush=True)
        return 0
    finally:
        sink.abort()


if __name__ == "__main__":
    sys.exit(main())
