"""Tests for the local web page server (petrify.server.app) and its static page.

Everything runs on loopback: FastAPI's test client in-process, plus one real
uvicorn server on 127.0.0.1 with an ephemeral port. No browser, no device.
"""

from __future__ import annotations

import html.parser
import json
import logging
import re
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient  # noqa: E402

from petrify.core import FrameMeta, PetrifyConfig, SourceInfo  # noqa: E402
from petrify.core.plugin import OptionSpec  # noqa: E402
from petrify.core.registry import ModeEntry  # noqa: E402
from petrify.server import app as server_app  # noqa: E402
from petrify.server.app import (  # noqa: E402
    MAX_BODY_BYTES,
    WEB_EXTRA_MESSAGE,
    create_app,
    require_web,
    serve_in_thread,
)
from petrify.server.control import (  # noqa: E402
    RUNTIME_FIELDS,
    RunControl,
    RunSnapshot,
    config_to_dict,
)
from petrify.sinks.mjpeg import MULTIPART_CONTENT_TYPE, MjpegFrame, MjpegSink  # noqa: E402

PORT = 8765
TOKEN = "TOKEN-zq7Xw3pLk9Vb2Rn5Ty8Mf4Hd6Jc1Gs0Ae"
W, H = 320, 256
INFO = SourceInfo(fps=25.0, width=W, height=H, frame_count=None, seekable=False, live=True,
                  uri="camera:0")
STATIC_DIR = Path(server_app.__file__).parent / "static"
ORIGIN = f"http://127.0.0.1:{PORT}"


class ScriptedFeed:
    """Duck-typed MjpegSink reader side: hands out the given frames, then nothing."""

    def __init__(self, frames: list[MjpegFrame]) -> None:
        self.frames = list(frames)
        self.calls: list[int] = []

    @property
    def closed(self) -> bool:
        return not self.frames

    def wait_next(self, after_sequence: int, timeout: float | None = None) -> MjpegFrame | None:
        self.calls.append(after_sequence)
        while self.frames:
            frame = self.frames.pop(0)
            if frame.sequence > after_sequence:
                return frame
        return None


def _frame(seq: int, payload: bytes = b"") -> MjpegFrame:
    jpeg = b"\xff\xd8" + payload + bytes([seq]) + b"\xff\xd9"
    return MjpegFrame(jpeg=jpeg, sequence=seq, index=seq, timestamp=seq / 25.0, width=W, height=H)


def _ended(control: RunControl) -> None:
    control.publish(RunSnapshot(config=config_to_dict(control.applied), ended=True))


@pytest.fixture
def control() -> RunControl:
    return RunControl(PetrifyConfig(), (W, H))


def _client(app: Any) -> TestClient:
    return TestClient(app, base_url=f"http://127.0.0.1:{PORT}")


@pytest.fixture
def client(control: RunControl) -> Iterator[TestClient]:
    _ended(control)  # Streams end at once, so the test client can read them whole.
    app = create_app(control, ScriptedFeed([]), token=TOKEN, port=PORT)  # type: ignore[arg-type]
    with _client(app) as c:
        yield c


AUTH = {"Authorization": f"Bearer {TOKEN}"}
JSON = {"Content-Type": "application/json"}


def _request(client: TestClient, method: str, path: str, **kwargs: Any) -> httpx.Response:
    if method == "PATCH":
        kwargs.setdefault("content", b'{"engine": {"lookahead": 5}}')
        kwargs["headers"] = {**JSON, **kwargs.get("headers", {})}
    return client.request(method, path, **kwargs)


PROTECTED = [
    ("GET", "/api/state"),
    ("PATCH", "/api/config"),
    ("POST", "/api/stop"),
    ("GET", "/stream.mjpg"),
]
PUBLIC = [("GET", "/"), ("GET", "/static/app.js"), ("GET", "/static/app.css")]


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("method", "path"), PUBLIC)
def test_page_and_assets_need_no_token(client: TestClient, method: str, path: str) -> None:
    response = _request(client, method, path)
    assert response.status_code == 200
    assert response.content


@pytest.mark.parametrize(("method", "path"), PROTECTED)
def test_protected_routes_with_the_token(client: TestClient, method: str, path: str) -> None:
    assert _request(client, method, path, headers=AUTH).status_code == 200


@pytest.mark.parametrize(("method", "path"), PROTECTED)
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong-token-wrong-token-wrong"},
        {"Authorization": f"Bearer {TOKEN}x"},
        {"Authorization": f"Bearer {TOKEN[:-1]}"},
        {"Authorization": f"Basic {TOKEN}"},
        {"Authorization": TOKEN},
        {"Authorization": "Bearer "},
        {"Authorization": b"Bearer \xe9t\xe9"},
    ],
    ids=["none", "wrong", "longer", "shorter", "basic", "no-scheme", "empty", "non-ascii"],
)
def test_protected_routes_without_a_valid_token(
    client: TestClient, method: str, path: str, headers: dict[str, Any]
) -> None:
    response = _request(client, method, path, headers=headers)
    assert response.status_code == 401
    assert response.json() == {"error": "Unauthorized"}  # No hint about what was wrong.


@pytest.mark.parametrize(("method", "path"), PROTECTED[:3])
def test_token_in_the_query_is_rejected_on_api_routes(
    client: TestClient, method: str, path: str
) -> None:
    response = _request(client, method, f"{path}?token={TOKEN}")
    assert response.status_code == 401


def test_token_in_the_query_is_accepted_on_the_stream_route(client: TestClient) -> None:
    assert client.get(f"/stream.mjpg?token={TOKEN}").status_code == 200
    assert client.get("/stream.mjpg?token=wrong").status_code == 401


def test_raw_route_does_not_exist_unless_enabled(client: TestClient) -> None:
    for headers, url in ((AUTH, "/raw.mjpg"), ({}, f"/raw.mjpg?token={TOKEN}"), ({}, "/raw.mjpg")):
        response = client.get(url, headers=headers)
        assert response.status_code == 404
        assert response.json() == {"error": "Not found"}


def test_no_other_routes(client: TestClient) -> None:
    for path in ("/docs", "/redoc", "/openapi.json", "/static/index.html",
                 "/static/../app.py", "/static/%2e%2e/app.py", "/api/nothing"):
        assert client.get(path, headers=AUTH).status_code == 404, path
    assert client.delete("/api/config", headers=AUTH).status_code == 405


# ---------------------------------------------------------------------------
# Host and Origin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host", [f"127.0.0.1:{PORT}"])
def test_accepted_hosts(client: TestClient, host: str) -> None:
    assert client.get("/", headers={"host": host}).status_code == 200
    assert client.get("/api/state", headers={"host": host, **AUTH}).status_code == 200


@pytest.mark.parametrize(
    "host",
    [
        "evil.example",
        f"evil.example:{PORT}",
        f"localhost.evil.example:{PORT}",
        f"localhost:{PORT}",
        f"LOCALHOST:{PORT}",
        f"127.0.0.1.nip.io:{PORT}",
        "127.0.0.1",
        "localhost",
        f"127.0.0.1:{PORT + 1}",
        f"localhost:{PORT + 1}",
        f"[::1]:{PORT}",
        f"0.0.0.0:{PORT}",
        f"127.0.0.2:{PORT}",
        "",
    ],
)
@pytest.mark.parametrize(("method", "path"), PUBLIC[:1] + PROTECTED)
def test_rejected_hosts(client: TestClient, host: str, method: str, path: str) -> None:
    response = _request(client, method, path, headers={"host": host, **AUTH})
    assert response.status_code == 400
    assert response.json() == {"error": "Bad request"}


@pytest.mark.parametrize(("method", "path"), [("PATCH", "/api/config"), ("POST", "/api/stop")])
@pytest.mark.parametrize(
    ("origin", "status"),
    [
        (None, 200),
        (f"http://127.0.0.1:{PORT}", 200),
        (f"http://localhost:{PORT}", 403),
        ("http://evil.example", 403),
        (f"http://evil.example:{PORT}", 403),
        (f"https://127.0.0.1:{PORT}", 403),
        (f"http://127.0.0.1:{PORT + 1}", 403),
        ("http://127.0.0.1", 403),
        ("null", 403),
        ("", 403),
    ],
)
def test_origin_matrix(
    control: RunControl, method: str, path: str, origin: str | None, status: int
) -> None:
    app = create_app(control, ScriptedFeed([]), token=TOKEN, port=PORT)  # type: ignore[arg-type]
    headers = dict(AUTH)
    if origin is not None:
        headers["Origin"] = origin
    with _client(app) as c:
        response = _request(c, method, path, headers=headers)
    assert response.status_code == status
    if status == 403:
        assert response.json() == {"error": "Forbidden"}
        assert control.take_pending() is None
        assert control.stop_requested is False


# ---------------------------------------------------------------------------
# PATCH content type, size, validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content_type",
    ["text/plain", "application/x-www-form-urlencoded", "multipart/form-data", "",
     "application/jsonx", "text/json"],
)
def test_patch_needs_json_content_type(
    client: TestClient, control: RunControl, content_type: str
) -> None:
    response = client.patch("/api/config", content=b'{"engine": {"lookahead": 5}}',
                            headers={**AUTH, "Content-Type": content_type})
    assert response.status_code == 415
    assert control.take_pending() is None


def test_patch_json_with_charset_is_fine(client: TestClient) -> None:
    response = client.patch("/api/config", content=b'{"detector": "both"}',
                            headers={**AUTH, "Content-Type": "application/json; charset=utf-8"})
    assert response.status_code == 200


def test_patch_body_size_limit(client: TestClient, control: RunControl) -> None:
    padding = " " * (MAX_BODY_BYTES + 1)
    big = ('{"detector": "both"' + padding + "}").encode()
    response = client.patch("/api/config", content=big, headers={**AUTH, **JSON})
    assert response.status_code == 413
    assert response.json() == {"error": "Request body too large"}

    def chunks() -> Iterator[bytes]:  # No Content-Length: the limit is enforced while reading.
        yield b'{"detector": "both"'
        for _ in range(10):
            yield b" " * 1000
        yield b"}"

    response = client.patch("/api/config", content=chunks(), headers={**AUTH, **JSON})
    assert response.status_code == 413
    assert control.take_pending() is None
    ok = ('{"detector": "both"' + " " * (MAX_BODY_BYTES - 40) + "}").encode()
    assert client.patch("/api/config", content=ok, headers={**AUTH, **JSON}).status_code == 200


def test_patch_success_returns_the_accepted_config(client: TestClient, control: RunControl) -> None:
    response = client.patch("/api/config", json={"engine": {"lookahead": 42}}, headers=AUTH)
    assert response.status_code == 200
    assert response.json()["config"]["engine"]["lookahead"] == 42
    pending = control.take_pending()
    assert pending is not None and pending.engine.lookahead == 42


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b'{"engine": {"lookahead": 9999}}', "engine.lookahead must be"),
        (b'{"encode": {"crf": 1}}', "cannot be changed"),
        (b'{"<b>x</b>": 1}', "does not exist"),
        (b"[1, 2]", "must be a JSON object"),
        (b"{not json", "not valid JSON"),
        (b"\xff\xfe", "not valid JSON"),
    ],
)
def test_patch_validation_failure(
    client: TestClient, control: RunControl, body: bytes, message: str
) -> None:
    response = client.patch("/api/config", content=body, headers={**AUTH, **JSON})
    assert response.status_code == 422
    payload = response.json()
    assert set(payload) == {"error"}
    assert message in payload["error"]
    assert "<b>" not in response.text and "9999" not in response.text
    assert control.take_pending() is None


def test_stop(client: TestClient, control: RunControl) -> None:
    response = client.post("/api/stop", headers=AUTH)
    assert response.status_code == 200
    assert response.json() == {"stopping": True}
    assert control.stop_requested is True


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def test_state_shape(control: RunControl) -> None:
    control.publish(RunSnapshot(config=config_to_dict(control.applied), running=True,
                                roi=(1, 2, 3, 4)))
    app = create_app(control, ScriptedFeed([]), token=TOKEN, port=PORT)  # type: ignore[arg-type]
    with _client(app) as c:
        data = c.get("/api/state", headers=AUTH).json()
    assert set(data) == {"config", "stats", "timings", "source", "background", "roi",
                         "running", "ended", "last_error", "weakened", "raw_preview"}
    assert set(data["stats"]) == {"frames", "frozen", "passed_through", "failed_closed",
                                  "over_budget", "sink_failures", "reconfigurations"}
    assert set(data["timings"]) == {"detect_mean_ms", "detect_last_ms", "render_mean_ms",
                                    "render_last_ms", "write_mean_ms", "write_last_ms", "fps"}
    assert set(data["source"]) == {"uri", "width", "height", "fps", "live", "dropped"}
    assert set(data["background"]) == {"strategy", "ready", "rejected"}
    assert data["running"] is True and data["ended"] is False
    assert data["roi"] == [1, 2, 3, 4]
    assert data["raw_preview"] is False
    assert data["weakened"] == []
    assert data["config"] == config_to_dict(control.applied)


def test_state_reports_raw_preview_when_enabled(control: RunControl) -> None:
    _ended(control)
    app = create_app(control, ScriptedFeed([]), token=TOKEN,  # type: ignore[arg-type]
                     raw=ScriptedFeed([]), port=PORT)  # type: ignore[arg-type]
    with _client(app) as c:
        assert c.get("/api/state", headers=AUTH).json()["raw_preview"] is True


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------

_PART = re.compile(
    rb"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: (\d+)\r\n\r\n"
)


def _parts(body: bytes) -> list[bytes]:
    parts = []
    pos = 0
    while pos < len(body):
        match = _PART.match(body, pos)
        assert match is not None, body[pos : pos + 80]
        length = int(match.group(1))
        start = match.end()
        parts.append(body[start : start + length])
        assert body[start + length : start + length + 2] == b"\r\n"
        pos = start + length + 2
    return parts


@pytest.mark.parametrize("route", ["/stream.mjpg", "/raw.mjpg"])
def test_stream_yields_well_formed_parts_and_ends_with_the_run(
    control: RunControl, route: str
) -> None:
    frames = [_frame(1, b"a" * 10), _frame(2, b"b" * 300), _frame(3)]
    feed = ScriptedFeed(frames)
    other = ScriptedFeed([_frame(9)])
    _ended(control)
    processed, raw = (feed, other) if route == "/stream.mjpg" else (other, feed)
    app = create_app(control, processed, token=TOKEN, raw=raw, port=PORT)  # type: ignore[arg-type]
    with _client(app) as c:
        response = c.get(f"{route}?token={TOKEN}")
    assert response.status_code == 200
    assert response.headers["content-type"] == MULTIPART_CONTENT_TYPE
    assert _parts(response.content) == [f.jpeg for f in frames]
    assert feed.calls[:3] == [0, 1, 2]  # Each wait asks for a newer frame.


def test_stream_ends_when_the_sink_closes_and_the_run_ends(control: RunControl) -> None:
    sink = MjpegSink(max_fps=None)
    sink.open(INFO)
    image = np.full((H, W, 3), 90, dtype=np.uint8)
    sink.write(image, FrameMeta(index=0, timestamp=0.0, source=INFO))
    app = create_app(control, sink, token=TOKEN, port=PORT)

    def finish() -> None:
        time.sleep(0.3)
        sink.close()
        _ended(control)

    closer = threading.Thread(target=finish)
    closer.start()
    try:
        with _client(app) as c:
            response = c.get("/stream.mjpg", headers=AUTH)
    finally:
        closer.join()
    parts = _parts(response.content)
    assert len(parts) == 1 and parts[0][:2] == b"\xff\xd8"


# ---------------------------------------------------------------------------
# Headers on every response
# ---------------------------------------------------------------------------


def _all_kinds_of_responses(c: TestClient) -> list[httpx.Response]:
    return [
        c.get("/"),
        c.get("/static/app.js"),
        c.get("/static/app.css"),
        c.get("/static/missing.js"),
        c.get("/api/state", headers=AUTH),
        c.get("/api/state"),
        c.get("/api/state", headers={"host": "evil.example", **AUTH}),
        c.get(f"/stream.mjpg?token={TOKEN}"),
        c.get("/raw.mjpg", headers=AUTH),
        c.patch("/api/config", json={"engine": {"lookahead": 1}}, headers=AUTH),
        c.patch("/api/config", json={"engine": {"lookahead": -1}}, headers=AUTH),
        c.patch("/api/config", content=b"x", headers={**AUTH, "Content-Type": "text/plain"}),
        c.patch("/api/config", json={}, headers={**AUTH, "Origin": "http://evil.example"}),
        c.patch("/api/config", content=b"{" + b" " * 9000 + b"}", headers={**AUTH, **JSON}),
        c.post("/api/stop", headers=AUTH),
        c.delete("/api/stop", headers=AUTH),
        c.options("/api/config", headers={"Origin": "http://evil.example",
                                          "Access-Control-Request-Method": "PATCH"}),
    ]


# Literal values (T1): a header removed from, or weakened in, SECURITY_HEADERS fails here.
EXPECTED_SECURITY_HEADERS = {
    "cache-control": "no-store",
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "x-frame-options": "DENY",
    "content-security-policy": (
        "default-src 'self'; img-src 'self'; frame-ancestors 'none'; "
        "base-uri 'none'; form-action 'none'"
    ),
    "cross-origin-resource-policy": "same-origin",
    "cross-origin-opener-policy": "same-origin",
}


def test_security_headers_on_every_response_and_no_cors(client: TestClient) -> None:
    responses = _all_kinds_of_responses(client)
    assert {r.status_code for r in responses} >= {200, 400, 401, 403, 404, 405, 413, 415, 422}
    for response in responses:
        for name, value in EXPECTED_SECURITY_HEADERS.items():
            assert response.headers.get(name) == value, (response.request.url, name)
            assert len(response.headers.get_list(name)) == 1, (response.request.url, name)
        assert not [k for k in response.headers if k.lower().startswith("access-control-")]
        assert "server" not in response.headers


def test_errors_are_minimal_json(client: TestClient) -> None:
    for response in _all_kinds_of_responses(client):
        if response.status_code >= 400:
            payload = response.json()
            assert set(payload) == {"error"}, response.text
            assert "Traceback" not in response.text and "petrify/" not in response.text


def test_unexpected_error_is_a_minimal_500(control: RunControl, all_logs: Any) -> None:
    app = create_app(control, ScriptedFeed([]), token=TOKEN, port=PORT)  # type: ignore[arg-type]

    def boom() -> RunSnapshot:
        raise RuntimeError(f"secret detail {TOKEN} /internal/path")

    control.snapshot = boom  # type: ignore[method-assign]
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}", raise_server_exceptions=False) as c:
        response = c.get("/api/state", headers=AUTH)
    assert response.status_code == 500
    assert response.json() == {"error": "Internal error"}
    assert TOKEN not in response.text and "internal" not in response.text.lower().replace(
        "internal error", "")
    for name, value in EXPECTED_SECURITY_HEADERS.items():
        assert response.headers.get(name) == value
    logged = [r for r in _server_records(all_logs) if r.name == "petrify.server.app"]
    assert [r.getMessage() for r in logged] == ["Web request failed: RuntimeError"]
    assert all(TOKEN not in _record_text(r) for r in _server_records(all_logs))


# ---------------------------------------------------------------------------
# Token hygiene
# ---------------------------------------------------------------------------


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        if not any(r is record for r in self.records):  # Attached at several levels.
            self.records.append(record)


# The test's own HTTP client (httpx, httpcore) logs the URLs it requests; that is
# the test harness, not the server, so its records are left out.
_CLIENT_LOGGERS = ("httpx", "httpcore")


def _server_records(collect: _Collect) -> list[logging.LogRecord]:
    records = [r for r in collect.records if not r.name.startswith(_CLIENT_LOGGERS)]
    return records


def _record_text(record: logging.LogRecord) -> str:
    text = record.getMessage()
    if record.exc_info:
        text += logging.Formatter().formatException(record.exc_info)
    return text


@pytest.fixture
def all_logs() -> Iterator[_Collect]:
    """Capture every record at every level, from the root and from uvicorn's loggers."""
    collect = _Collect()
    names = ["", "uvicorn", "uvicorn.error", "uvicorn.access", "fastapi", "starlette",
             "petrify"]
    loggers = [logging.getLogger(n) for n in names]
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.addHandler(collect)
    logging.getLogger().setLevel(logging.DEBUG)
    try:
        yield collect
    finally:
        for lg, level in zip(loggers, levels):
            lg.removeHandler(collect)
            lg.setLevel(level)


def test_token_never_logged_or_echoed_in_process(control: RunControl, all_logs: _Collect) -> None:
    _ended(control)
    app = create_app(control, ScriptedFeed([_frame(1)]), token=TOKEN, port=PORT)  # type: ignore[arg-type]
    with _client(app) as c:
        responses = _all_kinds_of_responses(c)
        responses.append(c.get(f"/api/state?token={TOKEN}"))
        responses.append(c.get(f"/nothing?token={TOKEN}", headers=AUTH))
    for response in responses:
        assert TOKEN not in response.text
        assert all(TOKEN not in v for v in response.headers.values())
    assert all(TOKEN not in _record_text(r) for r in _server_records(all_logs))


# ---------------------------------------------------------------------------
# Real server on loopback
# ---------------------------------------------------------------------------


def _fake_loop(sink: MjpegSink, stop: threading.Event) -> None:
    sink.open(INFO)
    i = 0
    while not stop.is_set():
        image = np.full((H, W, 3), i % 256, dtype=np.uint8)
        sink.write(image, FrameMeta(index=i, timestamp=i / 25.0, source=INFO))
        i += 1
        stop.wait(0.02)
    sink.close()


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def test_real_server_on_loopback(control: RunControl, all_logs: _Collect) -> None:
    sink = MjpegSink(max_fps=None)
    app = create_app(control, sink, token=TOKEN, port=0)
    handle = serve_in_thread(app, 0)
    stop = threading.Event()
    loop = threading.Thread(target=_fake_loop, args=(sink, stop))
    try:
        assert handle.host == "127.0.0.1"
        assert handle.port > 0
        assert handle.bound_addresses() == [("127.0.0.1", handle.port)]
        assert handle.thread.daemon and handle.thread.is_alive()
        base = f"http://127.0.0.1:{handle.port}"
        loop.start()
        with httpx.Client(base_url=base, timeout=10.0) as http:
            page = http.get("/")
            assert page.status_code == 200 and b"/static/app.js" in page.content
            state = http.get("/api/state", headers=AUTH)
            assert state.status_code == 200 and state.json()["ended"] is False
            assert http.get("/api/state").status_code == 401
            other_port = http.get("/", headers={"host": f"127.0.0.1:{handle.port + 1}"})
            assert other_port.status_code == 400
            with http.stream("GET", f"/stream.mjpg?token={TOKEN}") as response:
                assert response.status_code == 200
                assert response.headers["content-type"] == MULTIPART_CONTENT_TYPE
                assert response.headers["x-frame-options"] == "DENY"
                buffer = b""
                parts: list[bytes] = []
                for chunk in response.iter_bytes():
                    buffer += chunk
                    while True:
                        match = _PART.search(buffer)
                        if match is None:
                            break
                        length = int(match.group(1))
                        end = match.end() + length
                        if len(buffer) < end + 2:
                            break
                        parts.append(buffer[match.end() : end])
                        buffer = buffer[end + 2 :]
                    if len(parts) >= 2:
                        break
            assert len(parts) >= 2
            assert all(p[:2] == b"\xff\xd8" and p[-2:] == b"\xff\xd9" for p in parts)
    finally:
        stop.set()
        if loop.is_alive() or loop.ident is not None:
            loop.join(5)
        handle.stop()
    assert not handle.thread.is_alive()
    assert not _port_open(handle.port)
    handle.stop()  # Idempotent.
    assert all(TOKEN not in _record_text(r) for r in _server_records(all_logs))


def test_real_server_reports_a_busy_port(control: RunControl) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = busy.getsockname()[1]
        app = create_app(control, MjpegSink(), token=TOKEN, port=port)
        with pytest.raises(OSError):
            serve_in_thread(app, port)


def test_serve_rejects_an_app_made_for_another_port(control: RunControl) -> None:
    app = create_app(control, MjpegSink(), token=TOKEN, port=1)
    with pytest.raises(ValueError, match="port"):
        serve_in_thread(app, 0)


def test_short_token_is_refused(control: RunControl) -> None:
    with pytest.raises(ValueError, match="token"):
        create_app(control, MjpegSink(), token="short", port=PORT)


# ---------------------------------------------------------------------------
# Missing extra
# ---------------------------------------------------------------------------


def test_missing_fastapi_names_the_extra(monkeypatch: pytest.MonkeyPatch,
                                        control: RunControl) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "fastapi", None)
    with pytest.raises(RuntimeError, match=re.escape("petrify[web]")):
        require_web()
    with pytest.raises(RuntimeError, match=re.escape(WEB_EXTRA_MESSAGE)):
        create_app(control, MjpegSink(), token=TOKEN, port=PORT)
    monkeypatch.setitem(sys.modules, "fastapi", sys.modules.get("starlette"))
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    with pytest.raises(RuntimeError, match=re.escape("petrify[web]")):
        require_web()


# ---------------------------------------------------------------------------
# Static page (no browser)
# ---------------------------------------------------------------------------


class _PageParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.script_text: list[str] = []
        self._in_script = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))
        self._in_script = tag == "script"

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._in_script = False

    def handle_data(self, data: str) -> None:
        if self._in_script and data.strip():
            self.script_text.append(data)


def _page() -> _PageParser:
    parser = _PageParser()
    parser.feed((STATIC_DIR / "index.html").read_text(encoding="utf-8"))
    return parser


def test_page_has_no_inline_script_or_style() -> None:
    page = _page()
    assert page.script_text == []
    tags = [t for t, _ in page.tags]
    assert "style" not in tags
    for tag, attrs in page.tags:
        assert "style" not in attrs, tag
        assert not [a for a in attrs if a.startswith("on")], (tag, attrs)
        if tag == "script":
            assert attrs.get("src", "").startswith("/static/")


def test_page_references_only_static_assets() -> None:
    refs = [
        value
        for _, attrs in _page().tags
        for name, value in attrs.items()
        if name in ("src", "href", "action", "srcset", "poster", "data") and value
    ]
    assert sorted(refs) == ["/static/app.css", "/static/app.js"]


@pytest.mark.parametrize("name", ["index.html", "app.js", "app.css"])
def test_static_files_have_no_external_url(name: str) -> None:
    text = (STATIC_DIR / name).read_text(encoding="utf-8")
    for needle in ("http://", "https://", "//", "@import", "url("):
        assert needle not in text, (name, needle)


@pytest.mark.parametrize("name", ["index.html", "app.js", "app.css"])
def test_static_files_have_no_em_or_en_dashes(name: str) -> None:
    text = (STATIC_DIR / name).read_text(encoding="utf-8")
    assert "\u2014" not in text and "\u2013" not in text


def test_every_control_maps_to_a_runtime_field() -> None:
    page = _page()
    fields = [attrs["data-field"] for _, attrs in page.tags if "data-field" in attrs]
    assert sorted(fields) == sorted(RUNTIME_FIELDS)
    ids = {attrs.get("id") for _, attrs in page.tags}
    for tag, attrs in page.tags:
        element_id = attrs.get("id") or ""
        if element_id.startswith("ctl-"):
            assert attrs.get("data-field") in RUNTIME_FIELDS, element_id
            expected = "ctl-" + attrs["data-field"].replace(".", "-").replace("_", "-")
            assert element_id == expected
    # Every form control has a label tied to it.
    labelled = {attrs.get("for") for tag, attrs in page.tags if tag == "label"}
    for tag, attrs in page.tags:
        if tag in ("input", "select"):
            assert attrs.get("id") in labelled, attrs
    assert labelled <= ids


def test_app_js_uses_the_fragment_token_and_bearer_header() -> None:
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert "window.location.hash" in js
    assert '"Authorization": "Bearer " + token' in js
    assert "/stream.mjpg?token=" in js and "/raw.mjpg?token=" in js
    assert "localStorage" not in js and "sessionStorage" not in js and "eval(" not in js


def test_server_modules_have_no_em_or_en_dashes() -> None:
    root = Path(server_app.__file__).parent
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "\u2014" not in text and "\u2013" not in text, path


def test_page_json_contract_matches_the_snapshot() -> None:
    """The fields the page reads exist in the JSON the state route returns."""
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    data = RunSnapshot(config=config_to_dict(PetrifyConfig())).to_dict()
    for key in ("stats", "timings", "source", "background"):
        for field in re.findall(rf"\b{key}\.(\w+)", js.replace("state.", "")):
            if key == "source" and field == "live":
                continue
            assert field in data[{"bg": "background"}.get(key, key)], (key, field)
    json.dumps(data)


# ---------------------------------------------------------------------------
# N1: only 127.0.0.1 in Host and Origin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host", [f"localhost:{PORT}", f"LocalHost:{PORT}"])
def test_n1_localhost_host_is_rejected(client: TestClient, host: str) -> None:
    for path, headers in (("/", {}), ("/api/state", AUTH), (f"/stream.mjpg?token={TOKEN}", {})):
        response = client.get(path, headers={"host": host, **headers})
        assert response.status_code == 400, path
        assert response.json() == {"error": "Bad request"}


@pytest.mark.parametrize(("method", "path"), [("PATCH", "/api/config"), ("POST", "/api/stop")])
def test_n1_localhost_origin_is_rejected(control: RunControl, method: str, path: str) -> None:
    app = create_app(control, ScriptedFeed([]), token=TOKEN, port=PORT)  # type: ignore[arg-type]
    with _client(app) as c:
        response = _request(c, method, path, headers={**AUTH, "Origin": f"http://localhost:{PORT}"})
    assert response.status_code == 403
    assert control.take_pending() is None and control.stop_requested is False


# ---------------------------------------------------------------------------
# N2: no redirects
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "headers"),
    [
        (f"/stream.mjpg/?token={TOKEN}", {}),
        ("/api/state/", AUTH),
        ("/static/app.js/", {}),
        ("/api/stop/", AUTH),
    ],
)
def test_n2_trailing_slash_is_404_without_a_redirect(
    client: TestClient, path: str, headers: dict[str, str]
) -> None:
    method = "POST" if path.startswith("/api/stop") else "GET"
    response = client.request(method, path, headers=headers, follow_redirects=False)
    assert response.status_code == 404
    assert "location" not in response.headers
    assert response.json() == {"error": "Not found"}
    assert TOKEN not in response.text


# ---------------------------------------------------------------------------
# N4 and T5: uvicorn settings, no access log
# ---------------------------------------------------------------------------


def test_n4_t5_uvicorn_config(control: RunControl) -> None:
    app = create_app(control, MjpegSink(), token=TOKEN, port=0)
    config = server_app._uvicorn_config(app, 1)
    assert config.access_log is False
    assert config.host == "127.0.0.1"
    assert config.limit_concurrency == 64
    assert config.timeout_keep_alive == 5
    assert config.proxy_headers is False and config.server_header is False


def test_t5_real_server_logs_no_token_at_debug(control: RunControl) -> None:
    secret = "DISTINCTIVE-Qm8vR2xT6yH4kL0pN3wZ"
    _ended(control)  # The stream ends at once.
    app = create_app(control, MjpegSink(max_fps=None), token=secret, port=0)
    access = logging.getLogger("uvicorn.access")
    saved_access = (access.propagate, list(access.handlers))
    # An earlier server in this process may have cut the access logger off the
    # root already; start from uvicorn's defaults so only this config decides.
    access.propagate = True
    collect = _Collect()
    root = logging.getLogger()
    root.addHandler(collect)
    manager = logging.root.manager
    names = ["", "uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi", *manager.loggerDict]
    loggers = [logging.getLogger(n) for n in dict.fromkeys(names)]
    saved = [(lg, lg.level, lg.disabled) for lg in loggers]
    handle = None
    try:
        handle = serve_in_thread(app, 0)
        # After the server started: uvicorn sets its own logger levels when configured.
        for lg in loggers:
            lg.setLevel(logging.DEBUG)
            lg.disabled = False
        base = f"http://127.0.0.1:{handle.port}"
        with httpx.Client(base_url=base, timeout=10.0) as http:
            assert http.get(f"/stream.mjpg?token={secret}").status_code == 200
            assert http.get(f"/nothing/here?token={secret}").status_code == 404
            assert http.get(f"/stream.mjpg/?token={secret}").status_code == 404
            assert http.get("/api/state", headers={"Authorization": f"Bearer {secret}"}
                            ).status_code == 200
    finally:
        if handle is not None:
            handle.stop()
        root.removeHandler(collect)
        for lg, level, disabled in saved:
            lg.setLevel(level)
            lg.disabled = disabled
        access.propagate, access.handlers = saved_access[0], saved_access[1]
    assert not handle.thread.is_alive()
    records = _server_records(collect)
    leaked = [(r.name, _record_text(r)) for r in records
              if secret in _record_text(r) or "nothing/here" in _record_text(r)]
    assert leaked == []  # No token and no access line.


# ---------------------------------------------------------------------------
# W1, W2 through the API
# ---------------------------------------------------------------------------


def test_w1_full_frame_roi_is_rejected(client: TestClient, control: RunControl) -> None:
    response = client.patch("/api/config", json={"roi": [0, 0, W, H]}, headers=AUTH)
    assert response.status_code == 422
    assert "at most 15 percent" in response.json()["error"]
    assert "shown live" in response.json()["error"]
    assert control.take_pending() is None


def test_w2_threshold_over_the_bound_is_rejected(client: TestClient, control: RunControl) -> None:
    response = client.patch("/api/config", json={"motion": {"threshold": 10_000_000}},
                            headers=AUTH)
    assert response.status_code == 422
    assert "from 0 to 5120" in response.json()["error"]  # 160 x 128 downsampled, 25 %.
    assert control.take_pending() is None


def test_w2_state_lists_weakened_protections(control: RunControl) -> None:
    accepted = control.submit({"background_diff": {"enabled": False}})
    control.take_pending()
    control.report_applied(accepted)
    app = create_app(control, ScriptedFeed([]), token=TOKEN,  # type: ignore[arg-type]
                     raw=ScriptedFeed([]), port=PORT)  # type: ignore[arg-type]
    with _client(app) as c:
        data = c.get("/api/state", headers=AUTH).json()
    assert data["weakened"] == [
        "The background differ is off: a person who stops moving is no longer detected "
        "and can be shown.",
        "The raw preview is on: the unprocessed feed, people included, is shown on this page.",
    ]


def test_w2_page_shows_weakened_protections_and_passed_through() -> None:
    text = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    tags = {attrs.get("id"): (tag, attrs) for tag, attrs in _page().tags if attrs.get("id")}
    tag, attrs = tags["weakened"]
    assert attrs.get("role") == "alert"
    assert "hidden" in attrs
    assert attrs.get("aria-labelledby") == "weakened-title"
    assert re.search(r'<h2 id="weakened-title">Weakened protections</h2>', text)
    assert tags["weakened-list"][0] == "ul"
    # At the top: before the video and the status.
    assert text.index('id="weakened"') < text.index('class="feeds"')
    assert "bg-diff-warning" not in text  # Replaced by the list.
    assert tags["st-passed"][0] == "dd"
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert "state.weakened" in js
    assert 'text("st-passed", String(stats.passed_through))' in js
    assert "bg-diff-warning" not in js


# ---------------------------------------------------------------------------
# T3, T4: static checks on the page script
# ---------------------------------------------------------------------------


def _js_code() -> str:
    """app.js without its comments, so a commented-out statement does not count."""
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    js = re.sub(r"/\*.*?\*/", "", js, flags=re.S)
    return re.sub(r"^\s*//.*$", "", js, flags=re.M)


@pytest.mark.parametrize(
    "sink",
    ["innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function",
     "createContextualFragment", "DOMParser", "srcdoc", "javascript:", "`", ".style",
     'setAttribute("style"', "setAttribute(\"on", "Function(",
     "setHTMLUnsafe", "parseHTMLUnsafe"],
)
def test_t3_app_js_has_no_html_or_code_sinks(sink: str) -> None:
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert sink not in js


def test_t3_app_js_timers_take_functions_not_strings() -> None:
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert not re.search(r"set(Timeout|Interval)\(\s*[\"'`]", js)
    for call in re.finditer(r"set(?:Timeout|Interval)\(\s*([^,)\s]+)", js):
        assert call.group(1) in ("function", "poll"), call.group(0)


def test_t3_index_html_has_no_event_handler_attributes() -> None:
    text = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert not re.search(r"<[^>]*\son[a-z]+\s*=", text, flags=re.I)


def test_t4_fragment_is_stripped_from_the_address_bar() -> None:
    """The token leaves the URL: replaceState with the path and query, no fragment."""
    statement = re.compile(
        r"window\.history\.replaceState\(\s*null\s*,\s*\"\"\s*,\s*"
        r"window\.location\.pathname\s*\+\s*window\.location\.search\s*\)\s*;"
    )
    code = _js_code()
    match = statement.search(code)
    assert match is not None
    # It runs inside readToken, after the fragment is read and before the token is returned.
    body = code[code.index("function readToken()") : code.index("return value;")]
    assert statement.search(body)
    assert body.index("window.location.hash") < statement.search(body).start()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# T13: the mode and its options through the API and on the page
# ---------------------------------------------------------------------------

MODE_SPECS = (
    OptionSpec("ratio", "float", 0.5, "How much <b>ratio</b>.", minimum=0.0, maximum=2.0,
               runtime=True),
    OptionSpec("shape", "choice", "box", "Fixed shape.", choices=("box", "ellipse")),
)


class _Factory:
    name = "fake"
    summary = "A fake mode."
    options = MODE_SPECS

    def weakened(self, options: Any) -> tuple[str, ...]:
        return ("The ratio is high.",) if options["ratio"] > 1 else ()


def _mode_control() -> RunControl:
    config = PetrifyConfig(mode="fake", mode_options=(("ratio", 0.5), ("shape", "box")))
    modes = (
        ModeEntry("erase", "petrify.modes.erase:ERASE", None, None, True, None),
        ModeEntry("fake", "fake_mod:F", "petrify-fake", "1.0", False, None),
    )
    return RunControl(config, (W, H), modes=modes, factory=_Factory())  # type: ignore[arg-type]


@pytest.fixture
def mode_client() -> Iterator[tuple[TestClient, RunControl]]:
    control = _mode_control()
    _ended(control)
    app = create_app(control, ScriptedFeed([]), token=TOKEN, port=PORT)  # type: ignore[arg-type]
    with _client(app) as c:
        yield c, control


MODE_PATCH = b'{"mode_options": {"ratio": 2}}'


def test_t13_state_returns_the_mode_and_the_modes(
    mode_client: tuple[TestClient, RunControl],
) -> None:
    c, _ = mode_client
    data = c.get("/api/state", headers=AUTH).json()
    assert data["mode"]["name"] == "fake"
    assert data["mode"]["origin"] == "petrify-fake 1.0"
    assert data["mode"]["builtin"] is False
    assert data["mode"]["options"][0] == {
        "name": "ratio", "kind": "float", "value": 0.5, "default": 0.5, "choices": [],
        "minimum": 0.0, "maximum": 2.0, "help": "How much <b>ratio</b>.", "runtime": True,
    }
    assert data["mode"]["options"][1]["runtime"] is False
    assert [m["name"] for m in data["modes"]["listed"]] == ["erase", "fake"]
    assert [m["active"] for m in data["modes"]["listed"]] == [False, True]
    assert data["modes"]["omitted"] == 0
    # The default control (no factory) keeps the old shape: see test_state_shape.


def test_t13_mode_options_patch_round_trip(mode_client: tuple[TestClient, RunControl]) -> None:
    c, control = mode_client
    response = c.patch("/api/config", content=MODE_PATCH,
                       headers={**AUTH, **JSON, "Origin": ORIGIN})
    assert response.status_code == 200
    options = response.json()["config"]["mode_options"]
    assert options == {"ratio": 2.0, "shape": "box"}
    assert "2.0" in response.text  # Stored as a float, not the int that was sent.
    pending = control.take_pending()
    assert pending is not None and pending.mode_options == (("ratio", 2.0), ("shape", "box"))
    assert type(pending.mode_options_dict()["ratio"]) is float
    control.report_applied(pending)
    # The loop publishes the applied config after the next frame.
    control.publish(RunSnapshot(config=config_to_dict(control.applied), ended=True))
    data = c.get("/api/state", headers=AUTH).json()
    assert data["mode"]["options"][0]["value"] == 2.0
    assert data["config"]["mode_options"] == {"ratio": 2.0, "shape": "box"}
    assert data["weakened"] == ["The ratio is high."]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b'{"mode_options": {"ratio": 3}}', "mode_options.ratio must be a number from 0 to 2"),
        (b'{"mode_options": {"ratio": true}}', "mode_options.ratio must be a number"),
        (b'{"mode_options": {"shape": "ellipse"}}', "mode_options.shape cannot be changed"),
        (b'{"mode_options": {"<b>x</b>": 1}}', "mode option that does not exist"),
        (b'{"mode_options": {"__proto__": 1}}', "mode option that does not exist"),
        (b'{"mode_options": [1]}', "mode_options must be an object"),
        (b'{"mode": "erase"}', "The mode cannot be changed"),
    ],
)
def test_t13_invalid_mode_options_patch_is_a_422_like_other_patches(
    mode_client: tuple[TestClient, RunControl], body: bytes, message: str
) -> None:
    c, control = mode_client
    response = c.patch("/api/config", content=body, headers={**AUTH, **JSON})
    assert response.status_code == 422
    payload = response.json()
    assert set(payload) == {"error"}
    assert message in payload["error"]
    assert "<b>" not in response.text
    assert control.take_pending() is None


def test_t13_mode_options_patch_without_a_factory_is_a_422(client: TestClient,
                                                          control: RunControl) -> None:
    response = client.patch("/api/config", content=MODE_PATCH, headers={**AUTH, **JSON})
    assert response.status_code == 422
    assert "does not report its mode" in response.json()["error"]
    assert control.take_pending() is None


@pytest.mark.parametrize(
    ("headers", "status", "error"),
    [
        ({}, 401, "Unauthorized"),
        ({"Authorization": "Bearer wrong-token-wrong-token-wrong"}, 401, "Unauthorized"),
        ({**AUTH, "host": "evil.example"}, 400, "Bad request"),
        ({**AUTH, "host": f"localhost:{PORT}"}, 400, "Bad request"),
        ({**AUTH, "Origin": "http://evil.example"}, 403, "Forbidden"),
        ({**AUTH, "Origin": f"http://localhost:{PORT}"}, 403, "Forbidden"),
        ({**AUTH, "Origin": "null"}, 403, "Forbidden"),
    ],
    ids=["no-token", "wrong-token", "wrong-host", "localhost-host", "foreign-origin",
         "localhost-origin", "null-origin"],
)
def test_t13_mode_options_patch_is_refused_like_other_patches(
    mode_client: tuple[TestClient, RunControl], headers: dict[str, str], status: int, error: str
) -> None:
    c, control = mode_client
    response = c.patch("/api/config", content=MODE_PATCH, headers={**JSON, **headers})
    assert response.status_code == status
    assert response.json() == {"error": error}
    assert control.take_pending() is None
    # The same refusal as an ordinary patch.
    other = c.patch("/api/config", content=b'{"engine": {"lookahead": 5}}',
                    headers={**JSON, **headers})
    assert (other.status_code, other.json()) == (status, {"error": error})
    assert control.take_pending() is None


def test_t13_served_static_files_and_csp_are_unchanged(
    mode_client: tuple[TestClient, RunControl],
) -> None:
    c, _ = mode_client
    for path, name in (("/", "index.html"), ("/static/app.js", "app.js"),
                       ("/static/app.css", "app.css")):
        response = c.get(path)
        assert response.status_code == 200
        assert response.content == (STATIC_DIR / name).read_bytes()
        assert response.headers["content-security-policy"] == (
            "default-src 'self'; img-src 'self'; frame-ancestors 'none'; "
            "base-uri 'none'; form-action 'none'"
        )


def test_t13_page_has_the_mode_section() -> None:
    text = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    tags = {attrs.get("id"): (tag, attrs) for tag, attrs in _page().tags if attrs.get("id")}
    assert "hidden" in tags["mode-section"][1]  # Shown only when the state has a mode.
    assert "hidden" in tags["mode-plugin-notice"][1]
    for needle in ("receives the unprocessed frames", "The mode cannot be changed during a run",
                   "This mode has no options.", "Installed modes"):
        assert needle in text, needle
    assert tags["mode-list"][0] == "ul" and tags["mode-options"][0] == "div"
    # The settings controls are unchanged: the mode controls are built by the script.
    assert text.index('id="mode-section"') < text.index('id="settings"')


def test_t13_page_reads_only_fields_the_snapshot_has() -> None:
    js = _js_code()
    data = _mode_control().snapshot().to_dict()
    mode, modes = data["mode"], data["modes"]
    checks = {
        r"\bmode\.(\w+)": set(mode),
        r"\bmodes\.(\w+)": set(modes),
        r"\bopt\.(\w+)": set(mode["options"][0]),
        r"\bitem\.(\w+)": set(modes["listed"][0]),
    }
    for pattern, keys in checks.items():
        found = set(re.findall(pattern, js))
        assert found, pattern
        assert found <= keys, (pattern, found - keys)
    assert "state.mode, state.modes" in js


def test_t13_app_js_treats_option_names_as_data() -> None:
    js = _js_code()
    mode_code = js[js.index("var modeShape") : js.index("function render(state)")]
    # No bracket access keyed by a name, no selector or id built from one.
    assert not re.search(r"[\w)\]]\[[^\]]*\bname\b", js)
    assert not re.search(r"querySelector(All)?\([^)]*\+", js)
    assert not re.search(r"\$\([^)]*name", js)
    assert not re.search(r"\.id\s*=\s*[^;]*name", js)
    # The patch object has no prototype, and the name is defined as an own property.
    assert "Object.create(null)" in mode_code
    assert "Object.defineProperty(values, String(entry.opt.name)" in mode_code
    # Untrusted text goes through textContent only.
    assert "textContent" in mode_code
    assert not re.search(r"\.(innerHTML|outerHTML|innerText)\b", js)


def test_t13b_css_contains_untrusted_text_in_its_own_box() -> None:
    css = (STATIC_DIR / "app.css").read_text(encoding="utf-8")
    rule = re.search(r"((?:[#.\w\- >]+,\s*)+[#.\w\- >]+)\{\s*overflow: hidden;\s*contain: paint;",
                     css)
    assert rule is not None
    selectors = {s.strip() for s in rule.group(1).split(",")}
    assert {"#mode-section", "#mode-list li", "#mode-options .field", "#mode-options fieldset",
            "#weakened-list li"} <= selectors
    assert "unicode-bidi: isolate;" in css


def test_t13b_a_slow_weakened_call_does_not_stall_the_routes() -> None:
    entered, release = threading.Event(), threading.Event()

    class Slow(_Factory):
        def weakened(self, options: Any) -> tuple[str, ...]:
            if options["ratio"] > 1:
                entered.set()
                assert release.wait(10)
            return ()

    config = PetrifyConfig(mode="fake", mode_options=(("ratio", 0.5), ("shape", "box")))
    control = RunControl(config, (W, H), factory=Slow())  # type: ignore[arg-type]
    _ended(control)
    app = create_app(control, ScriptedFeed([]), token=TOKEN, port=PORT)  # type: ignore[arg-type]
    applying = threading.Thread(
        target=control.report_applied,
        args=(PetrifyConfig(mode="fake", mode_options=(("ratio", 2.0), ("shape", "box"))),),
        daemon=True,
    )
    try:
        with _client(app) as c:
            applying.start()
            assert entered.wait(5)
            started = time.monotonic()
            assert c.get("/api/state", headers=AUTH).status_code == 200
            assert c.patch("/api/config", content=b'{"mode_options": {"ratio": 1}}',
                           headers={**AUTH, **JSON}).status_code == 200
            assert c.get(f"/stream.mjpg?token={TOKEN}").status_code == 200
            assert time.monotonic() - started < 2
    finally:
        release.set()
        applying.join(5)


def test_t13b_app_js_builds_no_id_attribute_link_or_navigation_from_data() -> None:
    js = _js_code()
    # Ids are set only from the option's index: "mode-opt-" + index, then suffixes.
    for match in re.finditer(r"\.id\s*=\s*([^;]+);", js):
        assert match.group(1).strip() in ('"mode-opt-" + index', "id", 'id + "-" + j',
                                          'id + "-help"', 'id + "-error"'), match.group(0)
    assert 'var id = "mode-opt-" + index;' in js
    # A label points only at an id the script built.
    for match in re.finditer(r"\.htmlFor\s*=\s*([^;]+);", js):
        assert match.group(1).strip() in ("box.id", "id"), match.group(0)
    # Attribute names, class names and element tags are literals, never data.
    for match in re.finditer(r"setAttribute\(\s*([^,]+),", js):
        assert re.fullmatch(r'"[a-z-]+"', match.group(1).strip()), match.group(0)
    for match in re.finditer(r"\.className\s*=\s*([^;]+);", js):
        assert match.group(1).strip() == "className", match.group(0)
    for match in re.finditer(r"(?<!function )\belement\(([^,)]+)(?:,\s*([^,)]+))?", js):
        for arg in (match.group(1), match.group(2)):
            if arg is not None:
                assert re.fullmatch(r'"[a-z]*"|"[a-z -]+"|kind === "choices" \? "fieldset" : "div"'
                                    r'|kind === "bool" \? "field field-check" : "field"|tag',
                                    arg.strip()), match.group(0)
    for forbidden in (".href", "dataset", "window.open", "location.href", "location.assign",
                      "location.replace", "location =", "createElement(tag", "src ="):
        if forbidden == "src =":
            assert not re.search(r'\.src\s*=\s*(?!\s|"/(?:stream|raw)\.mjpg\?token=")', js)
            continue
        if forbidden == "createElement(tag":
            # Tags are literals, or the tag parameter of element(), whose callers pass literals.
            for call in re.finditer(r"document\.createElement\(([^)]*)\)", js):
                assert re.fullmatch(r'"[a-z]+"|tag', call.group(1).strip()), call.group(0)
            continue
        assert forbidden not in js, forbidden


# ---------------------------------------------------------------------------
# T13b: the page script's behavior under Node with a stub DOM
# ---------------------------------------------------------------------------

NODE = shutil.which("node")
HARNESS = Path(__file__).parent / "page_harness.js"


def _option(name: str, kind: str, value: Any, *, runtime: bool = True,
            choices: list[str] | None = None, minimum: Any = None, maximum: Any = None) -> dict:
    return {"name": name, "kind": kind, "value": value, "default": value,
            "choices": choices or [], "minimum": minimum, "maximum": maximum,
            "help": f"Help for {name}.", "runtime": runtime}


def _entry_state(name: str, origin: str, *, builtin: bool = False, active: bool = False,
                 problem: str | None = None) -> dict:
    return {"name": name, "origin": origin, "builtin": builtin, "selectable": problem is None,
            "problem": problem, "active": active}


def _state(mode: dict | None, modes: dict | None, **overrides: Any) -> dict:
    data = RunSnapshot(config=config_to_dict(PetrifyConfig()), running=True).to_dict()
    data["raw_preview"] = False
    if mode is not None:
        data["mode"], data["modes"] = mode, modes
    data.update(overrides)
    return data


PLUGIN_MODE = {
    "name": "paint", "origin": "built in", "builtin": False, "summary": "<b>Paints</b>.",
    "options": [
        _option("count", "int", 3, minimum=1, maximum=10),
        _option("fixed_count", "int", 2, runtime=False, minimum=0, maximum=5),
        _option("flag", "bool", False),
        _option("shape", "choice", "box", choices=["box", "zz_choice"]),
        _option("fixed_shape", "choice", "box", runtime=False, choices=["box", "ellipse"]),
        _option("parts", "choices", ["face"], choices=["face", "qq_choice"]),
        _option("ratio", "float", 0.5, minimum=0, maximum=2),
    ],
}
PLUGIN_MODES = {
    "listed": [
        _entry_state("erase", "built in", builtin=True),
        _entry_state("paint", "built in", active=True),  # A distribution named "built in".
        _entry_state("paint", "other-dist 2.0"),  # Same name, not the active one.
        _entry_state("fake (built in): the mode of this run;", "evil 1.0",
                     problem="invalid mode name"),
    ],
    "omitted": 2,
}
WEAKENED = ["First weakened sentence.", "Second <i>weakened</i> sentence.", "Third one."]


def _run_page(tmp_path: Path, state: dict, *, actions: list[dict] | None = None,
              patch_response: dict | None = None) -> dict:
    scenario = tmp_path / "scenario.json"
    scenario.write_text(json.dumps({"state": state, "actions": actions or [],
                                    "patch_response": patch_response}), encoding="utf-8")
    result = subprocess.run([NODE, str(HARNESS), str(STATIC_DIR), str(scenario)],
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


needs_node = pytest.mark.skipif(
    NODE is None, reason="node is not on PATH: the page behavior tests run app.js under Node"
)


@needs_node
def test_t13b_page_shows_a_plugin_mode_its_notice_and_every_weakened_sentence(
    tmp_path: Path,
) -> None:
    report = _run_page(tmp_path, _state(PLUGIN_MODE, PLUGIN_MODES, weakened=WEAKENED))
    hidden, text = report["hidden"], report["text"]
    assert hidden["mode-section"] is False
    # The builtin flag decides, not the origin text "built in".
    assert hidden["mode-plugin-notice"] is False
    assert text["mode-origin"] == "Plugin built in"
    assert text["mode-summary"] == "<b>Paints</b>."  # Markup stays text.
    assert hidden["weakened"] is False
    assert report["weakened"] == WEAKENED
    assert hidden["mode-no-options"] is True and hidden["mode-fixed-note"] is False
    assert text["mode-list-omitted"] == "2 more installed modes are not listed."


@needs_node
def test_t13b_page_lists_modes_status_first_and_uses_the_active_flag(tmp_path: Path) -> None:
    report = _run_page(tmp_path, _state(PLUGIN_MODE, PLUGIN_MODES))
    rows = report["modeList"]
    assert [[c["className"] for c in row["children"]] for row in rows] == [
        ["mode-status", "mode-name", "mode-origin"],
        ["mode-status", "mode-name", "mode-origin"],
        ["mode-status", "mode-name", "mode-origin"],
        ["mode-status", "mode-name", "mode-origin", "mode-problem"],
    ]
    statuses = [row["children"][0]["text"] for row in rows]
    assert statuses == ["Installed, selectable for a new run:", "Active in this run:",
                        "Installed, selectable for a new run:", "Cannot be selected:"]
    names = [row["children"][1]["text"] for row in rows]
    assert names[3] == "fake (built in): the mode of this run;"  # Only ever in the name element.
    origins = [row["children"][2]["text"] for row in rows]
    assert origins == ["built in", "plugin built in", "plugin other-dist 2.0", "plugin evil 1.0"]
    assert rows[3]["children"][3]["text"] == "Reason: invalid mode name"


@needs_node
def test_t13b_page_builds_one_control_per_option_and_disables_fixed_ones(tmp_path: Path) -> None:
    report = _run_page(tmp_path, _state(PLUGIN_MODE, PLUGIN_MODES))
    inputs = report["inputs"]
    assert [(i["tag"], i["type"], i["disabled"]) for i in inputs] == [
        ("INPUT", "number", False),  # count
        ("INPUT", "number", True),  # fixed_count
        ("INPUT", "checkbox", False),  # flag
        ("SELECT", "", False),  # shape
        ("SELECT", "", True),  # fixed_shape
        ("INPUT", "checkbox", False),  # parts: face
        ("INPUT", "checkbox", False),  # parts: qq_choice
        ("INPUT", "number", False),  # ratio
    ]
    assert (inputs[0]["min"], inputs[0]["max"], inputs[0]["step"]) == ("1", "10", "1")
    assert (inputs[7]["min"], inputs[7]["max"], inputs[7]["step"]) == ("0", "2", "any")
    assert inputs[0]["value"] == "3" and inputs[5]["checked"] is True
    labels = [label["text"] for label in report["labels"]]
    assert "fixed_count (fixed for this run) (0 to 5)" in labels
    record = report["record"]
    # Ids come from positions only, never from option or choice names.
    assert record["ids"] and all(re.fullmatch(r"mode-opt-\d+(-\d+|-help|-error)?", i)
                                 for i in record["ids"])
    assert not any("choice" in i or "count" in i for i in record["ids"])
    assert set(record["attributes"]) <= {"data-mode-option", "aria-describedby", "aria-live"}
    assert record["hrefs"] == [] and record["dataset"] == []
    assert record["opened"] == [] and record["locations"] == []


@needs_node
def test_t13b_page_sends_only_runtime_changes_and_shows_the_server_error(tmp_path: Path) -> None:
    actions = [
        {"input": 0, "value": "7"},  # count
        {"input": 1, "value": "4"},  # fixed_count: disabled, never sent
        {"input": 6, "checked": True},  # parts: qq_choice
    ]
    report = _run_page(tmp_path, _state(PLUGIN_MODE, PLUGIN_MODES), actions=actions,
                       patch_response={"status": 422,
                                       "body": {"error": "mode_options.count must be <x>"}})
    assert report["sent"] == [{"mode_options": {"count": 7}},
                              {"mode_options": {"parts": ["face", "qq_choice"]}}]
    assert report["errors"].count("Rejected: mode_options.count must be <x>") == 2


@needs_node
def test_t13b_page_for_erase_and_a_mode_without_a_background(tmp_path: Path) -> None:
    erase = {"name": "erase", "origin": "built in", "builtin": True, "summary": "Erase.",
             "options": []}
    modes = {"listed": [_entry_state("erase", "built in", builtin=True, active=True)],
             "omitted": 0}
    report = _run_page(tmp_path, _state(erase, modes, background={
        "strategy": "locked", "ready": True, "rejected": 0}))
    assert report["hidden"]["mode-plugin-notice"] is True
    assert report["hidden"]["mode-no-options"] is False
    assert report["hidden"]["weakened"] is True
    assert report["text"]["mode-origin"] == "Built in"
    assert report["text"]["st-background"] == "locked: ready"
    background = {"strategy": "running", "ready": False, "rejected": 0, "in_use": False}
    report = _run_page(tmp_path, _state(PLUGIN_MODE, PLUGIN_MODES, background=background))
    assert report["text"]["st-background"] == "Not used: this mode does not use a background"


@needs_node
def test_t13b_page_without_mode_fields_keeps_the_section_hidden(tmp_path: Path) -> None:
    report = _run_page(tmp_path, _state(None, None, background={
        "strategy": "locked", "ready": False, "rejected": 0}))
    assert report["hidden"]["mode-section"] is True
    assert report["text"]["st-background"] == "locked: still learning"


def test_t13b_node_is_present_for_the_page_behavior_tests() -> None:
    if NODE is None:
        pytest.skip("node is not on PATH")
    version = subprocess.run([NODE, "--version"], capture_output=True, text=True, check=True)
    assert version.stdout.startswith("v")
