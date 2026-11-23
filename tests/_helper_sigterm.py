"""Subprocess helper for the SIGTERM and SIGINT run tests (not collected by pytest).

Usage: python _helper_sigterm.py OUTPUT READY_AFTER

Runs ``petrify run --source camera:0 --output OUTPUT`` through ``petrify.cli.main``
with the camera replaced by a synthetic live source. After READY_AFTER frames
it prints "ready" and keeps delivering frames until a signal stops the run or
the wait runs out. The exit code is the CLI's own, so the test compares the
codes of a SIGTERM and a SIGINT run. Exit code 3 means no signal arrived.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from unittest import mock

import numpy as np

import petrify.sources
from petrify.cli import main
from petrify.core import Frame, SourceInfo

# At least 320x256, so a 0.5 HOG downsample never goes below the 64x128 window.
W, H = 320, 256
FPS = 25.0
WAIT_FOR_SIGNAL_S = 60.0


class _SyntheticCamera:
    """A live source with a fixed textured scene; ``dropped`` as a live source has."""

    def __init__(self, ready_after: int) -> None:
        self.info = SourceInfo(fps=FPS, width=W, height=H, frame_count=None,
                               seekable=False, live=True, uri="camera:0")
        self.ready_after = ready_after
        self.dropped = 0
        rng = np.random.default_rng(7)
        self._scene = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)

    def __iter__(self) -> Iterator[Frame]:
        deadline = time.monotonic() + WAIT_FOR_SIGNAL_S
        i = 0
        while time.monotonic() < deadline:
            if i == self.ready_after:
                print("ready", flush=True)
            yield Frame(image=self._scene.copy(), index=i, timestamp=i / FPS, source=self.info)
            i += 1
            time.sleep(1.0 / FPS)
        print("no signal", flush=True)
        sys.exit(3)

    def close(self) -> None:
        pass


def run(output: str, ready_after: int) -> int:
    camera = _SyntheticCamera(ready_after)
    with mock.patch.object(petrify.sources, "open_source", return_value=camera):
        return main(["run", "--source", "camera:0", "--output", output,
                     "--preset", "ultrafast"])


if __name__ == "__main__":
    sys.exit(run(sys.argv[1], int(sys.argv[2])))
