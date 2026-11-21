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

"""Local web page for a running ``petrify run``: watch the processed feed, change settings.

FastAPI and uvicorn are optional (``pip install 'petrify[web]'``) and imported
only when :func:`create_app` or :func:`serve_in_thread` is called.

Security model: this is a local control surface for a privacy tool, and any
other page open in the user's browser is treated as hostile. The server binds
to ``127.0.0.1`` only; every request must carry a ``Host`` of
``127.0.0.1:<port>`` or ``localhost:<port>`` (defeats DNS rebinding); every
route except the page and its assets needs the per-run token (a Bearer header,
or a ``token`` query parameter on the two stream routes only); ``PATCH`` and
``POST`` reject a foreign ``Origin``; ``PATCH`` needs a JSON content type and a
small body. No CORS headers are sent, and every response carries no-store,
nosniff, no-referrer, frame-deny and a strict Content-Security-Policy. The
token is never logged, and uvicorn's access log is off.
"""

import json
import logging
import secrets
import socket
import threading
import time
from dataclasses import dataclass, field
from importlib import resources
from typing import TYPE_CHECKING, Any

from petrify.server.control import RunControl, config_to_dict
from petrify.sinks.mjpeg import MULTIPART_CONTENT_TYPE, MjpegSink, multipart_chunk

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
MAX_BODY_BYTES = 4096
STREAM_WAIT_SECONDS = 0.5

WEB_EXTRA_MESSAGE = (
    "The web page needs FastAPI and uvicorn, which are not installed. Install the "
    "optional extra: pip install 'petrify[web]'"
)

SECURITY_HEADERS: dict[str, str] = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self'; frame-ancestors 'none'; "
        "base-uri 'none'; form-action 'none'"
    ),
}

_STATIC_TYPES = {
    "app.js": "text/javascript; charset=utf-8",
    "app.css": "text/css; charset=utf-8",
}

_STATUS_MESSAGES = {
    400: "Bad request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not found",
    405: "Method not allowed",
    413: "Request body too large",
    415: "Content-Type must be application/json",
    422: "Invalid request",
    500: "Internal error",
}


def require_web() -> None:
    """Raise ``RuntimeError`` naming the ``petrify[web]`` extra when FastAPI or uvicorn is missing."""
    try:
        import fastapi  # noqa: F401, PLC0415
        import uvicorn  # noqa: F401, PLC0415
    except ImportError:
        raise RuntimeError(WEB_EXTRA_MESSAGE) from None


def _static_bytes(name: str) -> bytes:
    return resources.files("petrify.server").joinpath("static", name).read_bytes()


@dataclass
class _ServerState:
    """Per-app settings shared by the middleware and the routes."""

    token: str
    port: int
    closing: threading.Event = field(default_factory=threading.Event)

    def allowed_hosts(self) -> tuple[str, str]:
        return (f"127.0.0.1:{self.port}", f"localhost:{self.port}")

    def allowed_origins(self) -> tuple[str, str]:
        return (f"http://127.0.0.1:{self.port}", f"http://localhost:{self.port}")


class _SecurityMiddleware:
    """Pure ASGI middleware: Host check, security headers, and a minimal 500 on any error."""

    def __init__(self, app: Any, state: _ServerState) -> None:
        self.app = app
        self.state = state

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
                return
            await self.app(scope, receive, send)
            return
        started = False

        async def send_secured(message: dict[str, Any]) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                headers = [
                    (k, v)
                    for k, v in message.get("headers", [])
                    if not k.lower().startswith(b"access-control-")
                    and k.lower().decode("latin-1") not in _LOWER_SECURITY
                ]
                headers.extend(
                    (k.lower().encode("latin-1"), v.encode("latin-1"))
                    for k, v in SECURITY_HEADERS.items()
                )
                message = {**message, "headers": headers}
            await send(message)

        host = None
        for key, value in scope.get("headers", []):
            if key == b"host":
                host = value.decode("latin-1").lower()
                break
        if host not in self.state.allowed_hosts():
            await _send_json(send_secured, 400, {"error": _STATUS_MESSAGES[400]})
            return
        try:
            await self.app(scope, receive, send_secured)
        except Exception as exc:  # noqa: BLE001
            # Never a traceback, a path or request content: only the error type.
            logger.error("Web request failed: %s", type(exc).__name__)
            if not started:
                await _send_json(send_secured, 500, {"error": _STATUS_MESSAGES[500]})


_LOWER_SECURITY = {k.lower() for k in SECURITY_HEADERS}


async def _send_json(send: Any, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def create_app(
    control: RunControl,
    processed: MjpegSink,
    *,
    token: str,
    raw: MjpegSink | None = None,
    port: int,
) -> "FastAPI":
    """Build the web app for one run.

    ``processed`` is the processed feed and ``raw`` the opt-in unprocessed one
    (``/raw.mjpg`` does not exist when it is None). ``port`` is the port the
    browser uses, checked in the ``Host`` and ``Origin`` headers; with 0,
    :func:`serve_in_thread` fills in the bound port. Raises ``RuntimeError`` when
    FastAPI or uvicorn is missing.
    """
    require_web()
    import anyio  # noqa: PLC0415
    from fastapi import FastAPI, Request  # noqa: PLC0415
    from fastapi.exceptions import RequestValidationError  # noqa: PLC0415
    from fastapi.responses import JSONResponse, Response, StreamingResponse  # noqa: PLC0415
    from starlette.exceptions import HTTPException  # noqa: PLC0415

    if not isinstance(token, str) or len(token) < 16:
        raise ValueError("The session token must be a string of at least 16 characters")
    state = _ServerState(token=token, port=int(port))
    index_html = _static_bytes("index.html")
    static_files = {name: _static_bytes(name) for name in _STATIC_TYPES}
    token_bytes = token.encode("utf-8")

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.state.petrify = state
    app.add_middleware(_SecurityMiddleware, state=state)

    def error(status: int, message: str | None = None) -> JSONResponse:
        return JSONResponse({"error": message or _STATUS_MESSAGES.get(status, "Error")}, status)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return error(exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error(422)

    def authorized(request: Request, *, allow_query: bool) -> bool:
        supplied: str | None = None
        header = request.headers.get("authorization")
        if header is not None:
            scheme, _, value = header.partition(" ")
            if scheme.lower() == "bearer":
                supplied = value.strip()
        if supplied is None and allow_query:
            supplied = request.query_params.get("token")
        if not supplied:
            return False
        return secrets.compare_digest(supplied.encode("utf-8"), token_bytes)

    def origin_ok(request: Request) -> bool:
        origin = request.headers.get("origin")
        return origin is None or origin in state.allowed_origins()

    async def mjpeg(feed: MjpegSink, request: Request) -> Any:
        last = 0
        while not state.closing.is_set():
            if await request.is_disconnected():
                return
            frame = await anyio.to_thread.run_sync(feed.wait_next, last, STREAM_WAIT_SECONDS)
            if frame is None:
                if control.snapshot().ended or state.closing.is_set():
                    return
                if feed.closed:
                    await anyio.sleep(0.1)  # Not open yet: the run is still starting.
                continue
            last = frame.sequence
            yield multipart_chunk(frame)

    def stream_route(feed: MjpegSink) -> Any:
        async def handler(request: Request) -> Response:
            if not authorized(request, allow_query=True):
                return error(401)
            return StreamingResponse(mjpeg(feed, request), media_type=MULTIPART_CONTENT_TYPE)

        return handler

    @app.get("/")
    async def page() -> Response:
        return Response(index_html, media_type="text/html; charset=utf-8")

    @app.get("/static/{name}")
    async def static(name: str) -> Response:
        body = static_files.get(name)
        if body is None:
            return error(404)
        return Response(body, media_type=_STATIC_TYPES[name])

    app.add_api_route("/stream.mjpg", stream_route(processed), methods=["GET"])
    if raw is not None:
        app.add_api_route("/raw.mjpg", stream_route(raw), methods=["GET"])

    @app.get("/api/state")
    async def api_state(request: Request) -> Response:
        if not authorized(request, allow_query=False):
            return error(401)
        payload = control.snapshot().to_dict()
        payload["raw_preview"] = raw is not None
        return JSONResponse(payload)

    @app.patch("/api/config")
    async def api_config(request: Request) -> Response:
        if not authorized(request, allow_query=False):
            return error(401)
        if not origin_ok(request):
            return error(403)
        content_type = request.headers.get("content-type", "")
        if content_type.split(";", 1)[0].strip().lower() != "application/json":
            return error(415)
        length = request.headers.get("content-length")
        if length is not None:
            try:
                if int(length) > MAX_BODY_BYTES:
                    return error(413)
            except ValueError:
                return error(400)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_BODY_BYTES:
                return error(413)
        try:
            patch = json.loads(bytes(body).decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return error(422, "The request body is not valid JSON")
        try:
            accepted = control.submit(patch)
        except ValueError as exc:
            return error(422, str(exc))
        return JSONResponse({"config": config_to_dict(accepted)})

    @app.post("/api/stop")
    async def api_stop(request: Request) -> Response:
        if not authorized(request, allow_query=False):
            return error(401)
        if not origin_ok(request):
            return error(403)
        control.request_stop()
        return JSONResponse({"stopping": True})

    return app


@dataclass
class ServerHandle:
    """A running server: :attr:`port` is the bound port, :meth:`stop` shuts it down."""

    host: str
    port: int
    thread: threading.Thread
    _server: Any
    _socket: socket.socket
    _state: _ServerState | None

    def bound_addresses(self) -> list[tuple[str, int]]:
        """The (host, port) of every socket the server listens on."""
        return [
            tuple(sock.getsockname()[:2])  # type: ignore[misc]
            for server in getattr(self._server, "servers", [])
            for sock in server.sockets
        ]

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the server, join its thread and close the listening socket. Idempotent."""
        if self._state is not None:
            self._state.closing.set()
        self._server.should_exit = True
        if self.thread.is_alive() and self.thread is not threading.current_thread():
            self.thread.join(timeout)
        try:
            self._socket.close()
        except OSError:
            pass


def serve_in_thread(app: "FastAPI", port: int, *, start_timeout: float = 10.0) -> ServerHandle:
    """Serve ``app`` with uvicorn on ``127.0.0.1:port`` from a daemon thread.

    The socket is bound here (port 0 picks a free port) before the thread starts,
    so :attr:`ServerHandle.port` is known at once and the app's ``Host`` check
    uses it. No signal handlers are installed (uvicorn only installs them on the
    main thread). Raises ``OSError`` when the port cannot be bound and
    ``RuntimeError`` when the server does not start.
    """
    require_web()
    import uvicorn  # noqa: PLC0415

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((HOST, int(port)))
    except OSError:
        sock.close()
        raise
    bound_host, bound_port = sock.getsockname()[:2]
    state: _ServerState | None = getattr(app.state, "petrify", None)
    if state is not None:
        if state.port not in (0, bound_port):
            sock.close()
            raise ValueError(f"The app was created for port {state.port}, not {bound_port}")
        state.port = bound_port

    config = uvicorn.Config(
        app,
        host=HOST,
        port=bound_port,
        log_config=None,
        log_level="warning",
        access_log=False,
        lifespan="off",
        ws="none",
        proxy_headers=False,
        server_header=False,
        date_header=False,
        timeout_graceful_shutdown=2,
    )
    server = uvicorn.Server(config)
    failure: list[BaseException] = []

    def run() -> None:
        try:
            server.run(sockets=[sock])
        except BaseException as exc:  # noqa: BLE001 (uvicorn exits with SystemExit on failure)
            failure.append(exc)

    thread = threading.Thread(target=run, name="petrify-web", daemon=True)
    handle = ServerHandle(bound_host, bound_port, thread, server, sock, state)
    thread.start()
    deadline = time.monotonic() + start_timeout
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            handle.stop()
            raise RuntimeError("The web server could not start on 127.0.0.1")
        time.sleep(0.01)
    return handle
