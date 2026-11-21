# Copyright 2026 petrify contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-License-Identifier: Apache-2.0

"""Tests for the live camera and network stream sources, using a fake capture only."""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Iterator, Sequence
from typing import Any

import cv2
import numpy as np
import pytest

from petrify.core.plugin import Source
from petrify.sources.camera import CameraSource
from petrify.sources.stream import StreamSource, redact_url

WIDTH = 4
HEIGHT = 3
THREAD_PREFIX = "petrify-capture"
SECRET = "Zq9SuperSecretPassw0rd"
TOKEN = "Tk7UniqueTokenValue"
SECRET_URL = f"rtsp://admin:{SECRET}@cam.example:554/live/ch1?token={TOKEN}"
WAIT = 5.0  # Upper bound for event waits; never reached when the code is correct.


def make_image(value: int, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


class FakeCapture:
    """Stand-in for ``cv2.VideoCapture`` driven by a script of read results.

    Script items: an image (successful read), ``None`` (failed read), or an
    exception instance (raised by ``read``). After the script runs out every
    read fails. When gated, each read waits for a permit from ``allow``.
    """

    def __init__(
        self,
        script: Sequence[np.ndarray | BaseException | None] = (),
        *,
        opened: bool = True,
        width: float = WIDTH,
        height: float = HEIGHT,
        fps: float = 30.0,
        gated: bool = False,
        honor_set: bool = True,
    ) -> None:
        self.opened = opened
        self.props: dict[int, float] = {
            cv2.CAP_PROP_FRAME_WIDTH: width,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
            cv2.CAP_PROP_FPS: fps,
        }
        self.honor_set = honor_set
        self.set_calls: list[tuple[int, float]] = []
        self.release_count = 0
        self._script = list(script)
        self._cond = threading.Condition()
        self._permits = 0 if gated else math.inf
        self.reads_entered = 0

    def isOpened(self) -> bool:  # noqa: N802 (mirrors the OpenCV API)
        return self.opened

    def get(self, prop: int) -> float:
        return self.props.get(prop, 0.0)

    def set(self, prop: int, value: float) -> bool:
        self.set_calls.append((prop, value))
        if self.honor_set:
            self.props[prop] = value
        return self.honor_set

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._cond:
            self.reads_entered += 1
            self._cond.notify_all()
            while self._permits <= 0:
                self._cond.wait()
            self._permits -= 1
            item = self._script.pop(0) if self._script else None
        if item is None:
            return False, None
        if isinstance(item, BaseException):
            raise item
        return True, item

    def release(self) -> None:
        self.release_count += 1

    def allow(self, n: int) -> None:
        """Let ``n`` more reads proceed."""
        with self._cond:
            self._permits += n
            self._cond.notify_all()

    def open_gate(self) -> None:
        with self._cond:
            self._permits = math.inf
            self._cond.notify_all()

    def wait_entered(self, n: int) -> None:
        """Wait until ``read`` has been entered ``n`` times (so read ``n - 1`` was published)."""
        with self._cond:
            assert self._cond.wait_for(lambda: self.reads_entered >= n, WAIT)


class FakeClock:
    """Deterministic clock: returns 0.0, 1.0, 2.0, ... on successive calls."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._now = -1.0

    def __call__(self) -> float:
        with self._lock:
            self._now += 1.0
            return self._now


def capture_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith(THREAD_PREFIX) and t.is_alive()]


@pytest.fixture(autouse=True)
def no_leaked_threads() -> Iterator[None]:
    yield
    assert capture_threads() == []


def stream_source(fake: FakeCapture, **kwargs: Any) -> StreamSource:
    return StreamSource("rtsp://cam.example/live", capture_factory=lambda _url: fake, **kwargs)


def camera_source(fake: FakeCapture, **kwargs: Any) -> CameraSource:
    return CameraSource(capture_factory=lambda _index: fake, **kwargs)


def shutdown(src: StreamSource | CameraSource, fake: FakeCapture) -> None:
    fake.open_gate()
    src.close()


# --- frame delivery -----------------------------------------------------------------------


@pytest.mark.parametrize("factory", [stream_source, camera_source])
def test_frames_in_order_with_contiguous_indices(factory: Any) -> None:
    fake = FakeCapture([make_image(v) for v in (10, 20, 30)], gated=True)
    src = factory(fake, max_read_failures=3, clock=FakeClock())
    it = iter(src)
    frames = []
    for _ in range(3):
        fake.allow(1)
        frames.append(next(it))
    fake.open_gate()
    with pytest.raises(StopIteration):
        next(it)

    assert [f.index for f in frames] == [0, 1, 2]
    assert [int(f.image[0, 0, 0]) for f in frames] == [10, 20, 30]
    # The clock returns 0.0 at start, then 1.0, 2.0, 3.0 at each capture.
    assert [f.timestamp for f in frames] == [1.0, 2.0, 3.0]
    assert all(f.source is src.info for f in frames)
    assert src.dropped == 0
    src.close()


def test_latest_frame_semantics_drop_older_frames() -> None:
    fake = FakeCapture([make_image(v) for v in range(1, 9)], gated=True)
    src = stream_source(fake, max_read_failures=3)
    it = iter(src)

    fake.allow(1)
    first = next(it)
    # Consumer is slow: frames 2..6 are captured before it asks again.
    fake.allow(5)
    fake.wait_entered(7)
    assert src.dropped == 4
    second = next(it)

    fake.allow(2)
    fake.wait_entered(9)
    third = next(it)
    assert src.dropped == 5

    fake.open_gate()
    rest = list(it)
    values = [int(f.image[0, 0, 0]) for f in (first, second, third, *rest)]

    assert values[:3] == [1, 6, 8]
    assert rest == []
    assert values == sorted(set(values))  # Strictly newer each time, no repeats.
    assert [f.index for f in (first, second, third)] == [0, 1, 2]
    src.close()


def test_yielded_image_is_not_the_capture_buffer() -> None:
    buffer = make_image(7)
    fake = FakeCapture([buffer], gated=True)
    src = stream_source(fake, max_read_failures=1)
    it = iter(src)
    fake.allow(1)
    frame = next(it)
    buffer[:] = 99  # A backend reusing its buffer must not affect the yielded frame.
    assert int(frame.image[0, 0, 0]) == 7
    shutdown(src, fake)


def test_blocked_consumer_ends_when_closed_from_another_thread() -> None:
    fake = FakeCapture()  # Every read fails, so no frame ever arrives.
    src = stream_source(fake, max_read_failures=10**9)
    result: list[Any] = []
    consumer = threading.Thread(target=lambda: result.append(list(src)))
    consumer.start()
    fake.wait_entered(3)

    src.close()
    consumer.join(WAIT)

    assert not consumer.is_alive()
    assert result == [[]]
    assert fake.release_count == 1


# --- end of stream and errors -----------------------------------------------------------


def test_stream_ends_after_consecutive_failures_and_success_resets_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = FakeCapture([None, None, make_image(1), None, None, make_image(2), None, None, None])
    src = stream_source(fake, max_read_failures=3)
    with caplog.at_level(logging.WARNING, logger="petrify.sources.stream"):
        frames = list(src)

    # Both images were captured (failures were reset), the last one is always delivered.
    assert int(frames[-1].image[0, 0, 0]) == 2
    assert len(frames) + src.dropped == 2
    assert fake.reads_entered == 9
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "3 consecutive failed reads" in warnings[0].getMessage()
    src.close()
    assert fake.release_count == 1


def test_read_exception_surfaces_in_consumer() -> None:
    fake = FakeCapture([make_image(1), RuntimeError("decoder exploded")])
    src = stream_source(fake)
    it = iter(src)
    assert int(next(it).image[0, 0, 0]) == 1
    with pytest.raises(RuntimeError, match="decoder exploded"):
        next(it)
    src.close()
    assert fake.release_count == 1


def test_shape_change_mid_stream_raises() -> None:
    fake = FakeCapture([make_image(1), make_image(2, width=8, height=6)], gated=True)
    src = camera_source(fake)
    it = iter(src)
    fake.allow(1)
    assert next(it).index == 0
    fake.allow(1)
    with pytest.raises(ValueError, match="shape"):
        next(it)
    assert src.dropped == 1
    shutdown(src, fake)


# --- lifecycle ----------------------------------------------------------------------------


def _close_after_normal_end(src: StreamSource, fake: FakeCapture) -> None:
    assert len(list(src)) == 1
    src.close()


def _close_during_iteration(src: StreamSource, fake: FakeCapture) -> None:
    it = iter(src)
    next(it)
    src.close()
    with pytest.raises(StopIteration):
        next(it)


def _close_twice(src: StreamSource, fake: FakeCapture) -> None:
    list(src)
    src.close()
    src.close()


def _close_without_iterating(src: StreamSource, fake: FakeCapture) -> None:
    src.close()
    src.close()


def _context_manager(src: StreamSource, fake: FakeCapture) -> None:
    with src as entered:
        assert entered is src
        next(iter(src))


@pytest.mark.parametrize(
    "scenario",
    [
        _close_after_normal_end,
        _close_during_iteration,
        _close_twice,
        _close_without_iterating,
        _context_manager,
    ],
)
def test_capture_released_exactly_once(scenario: Any) -> None:
    fake = FakeCapture([make_image(1)])
    src = stream_source(fake, max_read_failures=2)
    scenario(src, fake)
    assert fake.release_count == 1
    assert capture_threads() == []
    # Closing again later changes nothing.
    src.close()
    assert fake.release_count == 1


def test_iteration_after_close_yields_nothing() -> None:
    fake = FakeCapture([make_image(1)])
    src = stream_source(fake)
    src.close()
    assert list(src) == []
    assert fake.reads_entered == 0


def test_second_concurrent_iteration_raises() -> None:
    fake = FakeCapture()
    src = stream_source(fake, max_read_failures=10**9)
    it = iter(src)
    with pytest.raises(RuntimeError, match="already being iterated"):
        iter(src)
    it.close()  # type: ignore[attr-defined]
    second = iter(src)  # Allowed once the first one is done.
    src.close()
    assert list(second) == []


@pytest.mark.parametrize("factory", [stream_source, camera_source])
def test_live_sources_satisfy_source_protocol(factory: Any) -> None:
    fake = FakeCapture()
    src = factory(fake)
    assert isinstance(src, Source)
    assert src.info.frame_count is None
    assert src.info.seekable is False
    assert src.info.live is True
    src.close()


@pytest.mark.parametrize("factory", [stream_source, camera_source])
def test_invalid_max_read_failures_rejected_before_opening(factory: Any) -> None:
    opened: list[Any] = []

    def capture_factory(arg: Any) -> FakeCapture:
        opened.append(arg)
        return FakeCapture()

    with pytest.raises(ValueError, match="max_read_failures"):
        if factory is stream_source:
            StreamSource("rtsp://x/y", max_read_failures=0, capture_factory=capture_factory)
        else:
            CameraSource(max_read_failures=0, capture_factory=capture_factory)
    assert opened == []


# --- CameraSource -------------------------------------------------------------------------


def test_camera_requests_and_reads_back_settings() -> None:
    fake = FakeCapture(width=640, height=480, fps=30.0)
    indices: list[int] = []

    def factory(index: int) -> FakeCapture:
        indices.append(index)
        return fake

    src = CameraSource(2, width=1280, height=720, fps=60.0, capture_factory=factory)

    assert indices == [2]
    assert fake.set_calls == [
        (cv2.CAP_PROP_FRAME_WIDTH, 1280),
        (cv2.CAP_PROP_FRAME_HEIGHT, 720),
        (cv2.CAP_PROP_FPS, 60.0),
    ]
    assert (src.info.width, src.info.height, src.info.fps) == (1280, 720, 60.0)
    assert src.info.uri == "camera:2"
    src.close()


def test_camera_info_reflects_driver_ignoring_request() -> None:
    fake = FakeCapture(width=640, height=480, fps=30.0, honor_set=False)
    src = camera_source(fake, width=1920, height=1080, fps=120.0)
    assert len(fake.set_calls) == 3
    assert (src.info.width, src.info.height, src.info.fps) == (640, 480, 30.0)
    src.close()


def test_camera_sets_nothing_when_not_requested() -> None:
    fake = FakeCapture()
    src = camera_source(fake)
    assert fake.set_calls == []
    src.close()


@pytest.mark.parametrize(
    ("reported", "requested", "expected"),
    [
        (0.0, 15.0, 15.0),
        (0.0, None, 30.0),
        (float("nan"), None, 30.0),
        (float("inf"), 24.0, 24.0),
        (25.0, 15.0, 25.0),
    ],
)
def test_camera_fps_fallback(reported: float, requested: float | None, expected: float) -> None:
    fake = FakeCapture(fps=reported, honor_set=False)
    src = camera_source(fake, fps=requested)
    assert src.info.fps == expected
    src.close()


def test_camera_open_failure_raises_and_releases() -> None:
    fake = FakeCapture(opened=False)
    with pytest.raises(OSError, match="Cannot open camera 3"):
        CameraSource(3, capture_factory=lambda _index: fake)
    assert fake.release_count == 1


# --- StreamSource -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("rtsp://user:secret@host:554/path?token=abc", "rtsp://***@host:554/path"),
        ("rtsps://user:secret@host/path", "rtsps://***@host/path"),
        ("http://user@cam.local/video.mjpg", "http://***@cam.local/video.mjpg"),
        ("https://u:p@cam.local:8443/stream?auth=xyz#frag", "https://***@cam.local:8443/stream"),
        ("rtsp://host:554/path", "rtsp://host:554/path"),
        ("http://192.168.1.20:81/stream", "http://192.168.1.20:81/stream"),
        ("https://cam.local/stream?token=abc", "https://cam.local/stream"),
        ("rtsp://host", "rtsp://host"),
        ("rtsp://user:p@ss@host/path", "rtsp://***@host/path"),
        ("not a url", "not a url"),
        ("/dev/video0", "/dev/video0"),
        ("", ""),
    ],
)
def test_redact_url(url: str, expected: str) -> None:
    assert redact_url(url) == expected


def test_stream_never_exposes_credentials(caplog: pytest.LogCaptureFixture) -> None:
    received: list[str] = []
    fake = FakeCapture([make_image(1)])

    def factory(url: str) -> FakeCapture:
        received.append(url)
        return fake

    with caplog.at_level(logging.DEBUG):
        src = StreamSource(SECRET_URL, max_read_failures=2, capture_factory=factory)
        frames = list(src)
        src.close()

    assert received == [SECRET_URL]  # The real URL reaches the capture factory only.
    assert src.info.uri == "rtsp://***@cam.example:554/live/ch1"
    public = {k: v for k, v in vars(src).items() if not k.startswith("_")}
    exposed = [repr(src.info), repr(public), repr(frames[0].source), caplog.text]
    exposed += [r.getMessage() for r in caplog.records]
    assert caplog.records  # The info line and the end-of-stream warning were logged.
    for text in exposed:
        assert SECRET not in text
        assert TOKEN not in text


def test_stream_open_failure_message_is_redacted(caplog: pytest.LogCaptureFixture) -> None:
    fake = FakeCapture(opened=False)
    with caplog.at_level(logging.DEBUG), pytest.raises(OSError) as excinfo:
        StreamSource(SECRET_URL, capture_factory=lambda _url: fake)

    message = str(excinfo.value)
    assert "rtsp://***@cam.example:554/live/ch1" in message
    for text in (message, repr(excinfo.value), caplog.text):
        assert SECRET not in text
        assert TOKEN not in text
    assert fake.release_count == 1


@pytest.mark.parametrize("reported", [0.0, float("nan"), float("inf"), -1.0])
def test_stream_fps_fallback(reported: float) -> None:
    fake = FakeCapture(fps=reported)
    src = stream_source(fake)
    assert src.info.fps == 25.0
    src.close()


def test_stream_uses_reported_fps_and_size() -> None:
    fake = FakeCapture(width=800, height=600, fps=12.5)
    src = stream_source(fake)
    assert (src.info.width, src.info.height, src.info.fps) == (800, 600, 12.5)
    src.close()
