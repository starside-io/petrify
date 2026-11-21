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
import random
import re
import threading
import time
import traceback
from collections import Counter
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
    with pytest.raises(OSError, match="RuntimeError"):
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
        ("rtsp://user:secret@host:554/path?token=abc", "rtsp://***@host:554"),
        ("rtsps://user:secret@host/path", "rtsps://***@host"),
        ("http://user@cam.local/video.mjpg", "http://***@cam.local"),
        ("https://u:p@cam.local:8443/stream?auth=xyz#frag", "https://***@cam.local:8443"),
        ("rtsp://host:554/path", "rtsp://host:554"),
        ("http://192.168.1.20:81/stream", "http://192.168.1.20:81"),
        ("https://cam.local/stream?token=abc", "https://cam.local"),
        ("rtsp://host", "rtsp://host"),
        ("rtsp://user:p@ss@host/path", "rtsp://***@host"),
        ("not a url", "not a url"),
        ("/dev/video0", "/dev/video0"),
        ("", ""),
    ],
)
def test_redact_url(url: str, expected: str) -> None:
    assert redact_url(url) == expected


REDACTED = "<redacted>"
SCHEME_REDACTED = "rtsp://<redacted>"


# (url, expected, secret substrings that must not survive)
_REDACTION_TABLE: list[tuple[str, str, tuple[str, ...]]] = [
    # S1 confirmed leaks.
    ("rtsp://admin:pa/ss@cam/live", SCHEME_REDACTED, ("pa/ss", "admin", "cam")),
    ("rtsp://admin:pa?ss@cam", SCHEME_REDACTED, ("pa?ss", "admin", "cam")),
    ("rtsp://admin:pa#ss@cam", SCHEME_REDACTED, ("pa#ss", "admin", "cam")),
    ("admin:secret@cam:554/live", REDACTED, ("secret", "admin")),
    ("//admin:secret@cam/live", REDACTED, ("secret", "admin")),
    ("rtsp:/admin:secret@cam", REDACTED, ("secret", "admin")),
    ("rtsp:admin:secret@cam", REDACTED, ("secret", "admin")),
    ("rtsp://cam/live;user=admin;password=secret", "rtsp://cam", ("secret", "admin")),
    ("http://cam/stream/tok_SECRET/video", "http://cam", ("tok_SECRET",)),
    # An "@" at or after the first delimiter is ambiguous: the host is hidden, not guessed.
    ("http://cam/a@tok_SECRET/video", "http://<redacted>", ("tok_SECRET",)),
    ("rtsp://admin:pw@cam/live?mail=a@b.com", SCHEME_REDACTED, ("admin", "pw", "b.com")),
    ("http://cam/live#frag@evil", "http://<redacted>", ("frag", "evil")),
    ("rtsp://cam\\x@secret", SCHEME_REDACTED, ("secret",)),
    # Cases that already worked.
    ("rtsp://user:p@ss@host/path", "rtsp://***@host", ("p@ss", "user")),
    ("rtsp://us%40er:pa%2Fss%3A@cam.local:554/live", "rtsp://***@cam.local:554", ("pa%2Fss",)),
    ("RTSP://admin:secret@Cam.Local:554/live", "rtsp://***@Cam.Local:554", ("secret",)),
    ("  rtsp://admin:secret@cam:554/live \n", "rtsp://***@cam:554", ("secret",)),
    ("rtsp://admin:secret@[fe80::1]:554/live", "rtsp://***@[fe80::1]:554", ("secret",)),
    ("rtsp://admin:secret@[2001:db8::7]/live", "rtsp://***@[2001:db8::7]", ("secret",)),
    ("rtsp://[::1]:8554/live?token=secret", "rtsp://[::1]:8554", ("secret",)),
    ("rtsp://[::1]", "rtsp://[::1]", ()),
    ("rtsp://cam.local:554", "rtsp://cam.local:554", ()),
    ("rtsp://cam.local", "rtsp://cam.local", ()),
    ("http://cam.local/?token=secret#frag", "http://cam.local", ("secret",)),
    # Cannot be parsed confidently: reveal nothing.
    ("rtsp://admin:secret", REDACTED, ("secret",)),
    ("rtsp://admin:secret@", REDACTED, ("secret",)),
    ("rtsp://", REDACTED, ()),
    ("bad scheme://admin:secret@cam", REDACTED, ("secret",)),
    ("rtsp://cam;password=secret", REDACTED, ("secret",)),
    # No scheme separator and no "@": returned unchanged.
    ("/dev/video0", "/dev/video0", ()),
    ("clip.mp4", "clip.mp4", ()),
    ("", "", ()),
]


@pytest.mark.parametrize(("url", "expected", "secrets"), _REDACTION_TABLE)
def test_redact_url_s1_table(url: str, expected: str, secrets: tuple[str, ...]) -> None:
    result = redact_url(url)
    assert result == expected
    for secret in secrets:
        assert secret not in result


@pytest.mark.parametrize("value", [None, 42, b"rtsp://admin:secret@cam"])
def test_redact_url_s1_never_raises(value: Any) -> None:
    assert redact_url(value) == REDACTED


_STRICT_HOST = r"(?:\[[0-9A-Fa-f:.]+(?:%[A-Za-z0-9._~%-]+)?\]|[A-Za-z0-9._~%-]+)"
_HOST_SHAPE = re.compile(
    r"(?P<scheme>[a-z][a-z0-9+.-]*)://"
    r"(?:<redacted>|(?:\*\*\*@)?" + _STRICT_HOST + r"(?::[0-9]{1,5})?)"
)


def _redaction_shape(url: str, result: str) -> str:
    """Name which of exactly four shapes ``result`` has; fail when it has none."""
    if result == url and "://" not in url and "@" not in url:
        return "unchanged"
    if result == REDACTED:
        return "redacted"
    match = _HOST_SHAPE.fullmatch(result)
    assert match is not None, (url, result)
    assert match["scheme"] == url.strip().partition("://")[0].lower(), (url, result)
    return "scheme_redacted" if result.endswith("://<redacted>") else "host"


def test_redact_url_s1_randomized_no_marker_survives() -> None:
    rng = random.Random(20260811)
    # Every secret-bearing part may hold any printable ASCII, "@" and delimiters included.
    any_chars = [chr(c) for c in range(0x20, 0x7F)]
    schemes = ["rtsp", "rtsps", "http", "https", "RTSP", "rtmp", "srt", "bad scheme", "1rtsp"]
    hosts = ["cam", "cam.example", "192.168.1.20", "[::1]", "[fe80::1]", "Host-7.local"]
    shapes: Counter[str] = Counter()

    def secret(mark: str) -> str:
        def chunk() -> str:
            return "".join(rng.choice(any_chars) for _ in range(rng.randint(0, 6)))

        return chunk() + mark + chunk()

    for i in range(1000):
        marks = [f"MK{i:04d}{k}X" for k in "upamqf"]
        user, password, path, params, query, fragment = marks
        url = rng.choice(schemes) + "://"
        kind = rng.randrange(3)  # 0: user and password, 1: user only, 2: no userinfo.
        if kind == 0:
            url += f"{secret(user)}:{secret(password)}@"
        elif kind == 1:
            url += secret(user) + "@"
        url += rng.choice(hosts)
        if rng.random() < 0.6:
            url += f":{rng.randint(1, 65535)}"
        if rng.random() < 0.7:
            url += "/" + secret(path)
            if rng.random() < 0.5:
                url += ";" + secret(params)
        if rng.random() < 0.5:
            url += "?" + secret(query)
        if rng.random() < 0.4:
            url += "#" + secret(fragment)

        result = redact_url(url)
        for mark in marks:
            assert mark not in result, (url, result)
        shape = _redaction_shape(url, result)
        if shape == "host":
            shape += " with userinfo" if "***@" in result else " without userinfo"
        shapes[shape] += 1

    # Every reachable shape was exercised, so the assertions above are not vacuous.
    expected = {"redacted", "scheme_redacted", "host with userinfo", "host without userinfo"}
    assert set(shapes) == expected, shapes
    assert min(shapes.values()) >= 20, shapes


@pytest.mark.parametrize(("url", "expected", "secrets"), _REDACTION_TABLE)
def test_redact_url_s1_table_shape(url: str, expected: str, secrets: tuple[str, ...]) -> None:
    _redaction_shape(url, redact_url(url))


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
    assert src.info.uri == "rtsp://***@cam.example:554"
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
    assert "rtsp://***@cam.example:554" in message
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


# --- S2: exception text is redacted -------------------------------------------------------


class _OpenedRaises(FakeCapture):
    def isOpened(self) -> bool:  # noqa: N802 (mirrors the OpenCV API)
        raise RuntimeError(f"probe failed for {SECRET_URL}")


def _assert_no_secret(exc: BaseException, caplog: pytest.LogCaptureFixture) -> None:
    assert exc.__cause__ is None
    assert exc.__context__ is None
    formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    for text in (str(exc), repr(exc), formatted, caplog.text):
        assert SECRET not in text
        assert TOKEN not in text


def test_s2_stream_factory_error_is_redacted(caplog: pytest.LogCaptureFixture) -> None:
    def factory(url: str) -> FakeCapture:
        raise RuntimeError(f"could not connect to {url}")

    with caplog.at_level(logging.DEBUG), pytest.raises(OSError) as excinfo:
        StreamSource(SECRET_URL, capture_factory=factory)
    assert type(excinfo.value) is OSError
    assert "rtsp://***@cam.example:554" in str(excinfo.value)
    assert "RuntimeError" in str(excinfo.value)
    _assert_no_secret(excinfo.value, caplog)


def test_s2_stream_is_opened_error_is_redacted_and_released(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = _OpenedRaises()
    with caplog.at_level(logging.DEBUG), pytest.raises(OSError) as excinfo:
        StreamSource(SECRET_URL, capture_factory=lambda _url: fake)
    assert "RuntimeError" in str(excinfo.value)
    _assert_no_secret(excinfo.value, caplog)
    assert fake.release_count == 1


def test_s2_stream_read_error_is_redacted(caplog: pytest.LogCaptureFixture) -> None:
    fake = FakeCapture([make_image(1), ValueError(f"demux error on {SECRET_URL}")])
    with caplog.at_level(logging.DEBUG):
        src = StreamSource(SECRET_URL, capture_factory=lambda _url: fake)
        it = iter(src)
        next(it)
        with pytest.raises(OSError) as excinfo:
            next(it)
        src.close()
    assert type(excinfo.value) is OSError
    assert "rtsp://***@cam.example:554" in str(excinfo.value)
    assert "ValueError" in str(excinfo.value)
    _assert_no_secret(excinfo.value, caplog)
    assert fake.release_count == 1


def test_s2_camera_errors_are_wrapped() -> None:
    def factory(index: int) -> FakeCapture:
        raise RuntimeError("driver exploded")

    with pytest.raises(OSError, match=r"Cannot open camera 0.*RuntimeError") as excinfo:
        CameraSource(capture_factory=factory)
    assert excinfo.value.__context__ is None
    assert "driver exploded" not in str(excinfo.value)

    fake = FakeCapture([RuntimeError("driver exploded")])
    src = camera_source(fake)
    with pytest.raises(OSError, match=r"camera:0.*RuntimeError") as read_info:
        list(src)
    assert read_info.value.__context__ is None
    assert "driver exploded" not in str(read_info.value)
    src.close()


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
def test_s2_interrupts_pass_through_untouched(exc_type: type[BaseException]) -> None:
    raised = exc_type("stop")

    def factory(url: str) -> FakeCapture:
        raise raised

    with pytest.raises(exc_type) as excinfo:
        StreamSource(SECRET_URL, capture_factory=factory)
    assert excinfo.value is raised

    fake = FakeCapture([raised])
    src = stream_source(fake)
    with pytest.raises(exc_type) as read_info:
        list(src)
    assert read_info.value is raised
    src.close()


# --- S3: close() wakes a consumer while read() is stuck -----------------------------------


class _StuckCapture(FakeCapture):
    """A capture whose ``read`` blocks until ``unblock`` is set, like a dead RTSP stream."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.unblock = threading.Event()
        self.reading = False
        self.released_while_reading = False

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.reading = True
        self.entered.set()
        self.unblock.wait()
        self.reading = False
        return False, None

    def release(self) -> None:
        if self.reading:
            self.released_while_reading = True
        super().release()


@pytest.mark.parametrize("factory", [stream_source, camera_source])
def test_s3_close_ends_consumer_while_read_is_stuck(factory: Any) -> None:
    fake = _StuckCapture()
    src = factory(fake, max_read_failures=10**9, _join_timeout=0.01)
    result: list[Any] = []
    consumer = threading.Thread(target=lambda: result.append(list(src)))
    try:
        consumer.start()
        assert fake.entered.wait(WAIT)
        (capture_thread,) = capture_threads()

        started = time.monotonic()
        src.close()  # Returns after the join timeout even though read() never returns.
        assert time.monotonic() - started < WAIT
        consumer.join(WAIT)
        assert not consumer.is_alive()
        assert result == [[]]

        assert capture_thread.is_alive()
        assert fake.release_count == 0  # Not released while read() is still running.
    finally:
        fake.unblock.set()
        consumer.join(WAIT)
    capture_thread.join(WAIT)
    assert not capture_thread.is_alive()
    assert fake.release_count == 1
    assert not fake.released_while_reading
    src.close()
    assert fake.release_count == 1


# --- S4: device reporting 0x0 until the first frame ---------------------------------------


@pytest.mark.parametrize("factory", [stream_source, camera_source])
@pytest.mark.parametrize(("width", "height"), [(0, 0), (0, HEIGHT), (WIDTH, 0), (-1, -1)])
def test_s4_size_probed_from_first_frame(factory: Any, width: float, height: float) -> None:
    script = [None, make_image(1, 8, 6), make_image(2, 8, 6)]
    fake = FakeCapture(script, width=width, height=height, gated=True)
    fake.allow(2)
    src = factory(fake, max_read_failures=3)
    assert (src.info.width, src.info.height) == (8, 6)
    assert fake.reads_entered == 2
    it = iter(src)
    fake.allow(1)
    frame = next(it)
    assert int(frame.image[0, 0, 0]) == 2  # The probe frame was discarded.
    assert frame.index == 0
    shutdown(src, fake)
    assert fake.release_count == 1


@pytest.mark.parametrize("factory", [stream_source, camera_source])
def test_s4_probe_failure_raises_and_releases(factory: Any) -> None:
    fake = FakeCapture(width=0, height=0)  # Every read fails.
    with pytest.raises(OSError, match="frame size"):
        factory(fake, max_read_failures=3)
    assert fake.reads_entered == 3
    assert fake.release_count == 1
    assert capture_threads() == []


def test_s4_probe_failure_message_is_redacted(caplog: pytest.LogCaptureFixture) -> None:
    fake = FakeCapture([RuntimeError(f"bad {SECRET_URL}")], width=0, height=0)
    with caplog.at_level(logging.DEBUG), pytest.raises(OSError) as excinfo:
        StreamSource(SECRET_URL, capture_factory=lambda _url: fake)
    assert "rtsp://***@cam.example:554" in str(excinfo.value)
    _assert_no_secret(excinfo.value, caplog)
    assert fake.release_count == 1


def test_s4_positive_size_is_trusted_without_probing() -> None:
    fake = FakeCapture([make_image(1)], width=WIDTH, height=HEIGHT)
    src = stream_source(fake)
    assert fake.reads_entered == 0
    src.close()


def test_s4_probed_non_color_frame_still_raises_on_iteration() -> None:
    gray = np.zeros((6, 8), dtype=np.uint8)
    fake = FakeCapture([gray, gray.copy()], width=0, height=0, gated=True)
    fake.allow(1)
    src = stream_source(fake, max_read_failures=3)
    assert (src.info.width, src.info.height) == (8, 6)
    it = iter(src)
    fake.allow(1)
    with pytest.raises(ValueError, match="shape"):
        next(it)
    shutdown(src, fake)


# --- release() that raises ----------------------------------------------------------------


class _ReleaseRaises(FakeCapture):
    def release(self) -> None:
        super().release()
        raise RuntimeError(f"release failed for {SECRET_URL}")


class _StuckReleaseRaises(_StuckCapture):
    def release(self) -> None:
        super().release()
        raise RuntimeError(f"release failed for {SECRET_URL}")


def _assert_release_warning(caplog: pytest.LogCaptureFixture) -> None:
    warnings = [r.getMessage() for r in caplog.records if "releasing the capture" in r.getMessage()]
    assert len(warnings) == 1
    assert "RuntimeError" in warnings[0]
    assert SECRET not in caplog.text
    assert TOKEN not in caplog.text


@pytest.mark.parametrize("iterate", [False, True])
def test_release_error_does_not_escape_close(
    caplog: pytest.LogCaptureFixture, iterate: bool
) -> None:
    fake = _ReleaseRaises([make_image(1)])
    with caplog.at_level(logging.WARNING):
        src = StreamSource(SECRET_URL, max_read_failures=2, capture_factory=lambda _url: fake)
        if iterate:
            assert len(list(src)) == 1
        src.close()
        src.close()
    assert fake.release_count == 1
    _assert_release_warning(caplog)


def test_release_error_does_not_escape_capture_thread(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    escaped: list[Any] = []
    monkeypatch.setattr(threading, "excepthook", escaped.append)
    fake = _StuckReleaseRaises()
    with caplog.at_level(logging.WARNING):
        src = StreamSource(
            SECRET_URL,
            max_read_failures=10**9,
            capture_factory=lambda _url: fake,
            _join_timeout=0.01,
        )
        it = iter(src)
        try:
            assert fake.entered.wait(WAIT)
            (capture_thread,) = capture_threads()
            src.close()  # Gives up joining; the thread releases the capture on exit.
            assert fake.release_count == 0
        finally:
            fake.unblock.set()
        capture_thread.join(WAIT)
        assert not capture_thread.is_alive()
    assert list(it) == []
    assert escaped == []
    assert fake.release_count == 1
    _assert_release_warning(caplog)
