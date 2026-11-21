"""Tests for petrify.sinks.preview.PreviewSink with a fake GUI (no window is ever opened)."""

from __future__ import annotations

from typing import Any, ClassVar

import cv2
import numpy as np
import pytest

from petrify.core import (
    DetectionKind,
    Detections,
    Engine,
    EngineConfig,
    Frame,
    FrameMeta,
    Sink,
    SourceInfo,
)
from petrify.sinks import PreviewSink
from petrify.sinks.preview import HEADLESS_MESSAGE

H, W = 4, 6
INFO = SourceInfo(fps=10.0, width=W, height=H, frame_count=None, seekable=False, live=True)

GUI_BUILD = "General configuration\n  GUI:                           COCOA\n    Cocoa:   YES\n"
HEADLESS_BUILD = "General configuration\n  GUI:                           NONE\n  Media I/O:\n"


def _image(i: int) -> np.ndarray:
    return np.full((H, W, 3), 100 + i, dtype=np.uint8)


def _meta(i: int) -> FrameMeta:
    return FrameMeta(index=i, timestamp=i / INFO.fps, source=INFO)


class FakeGui:
    """Records GUI calls; mirrors the parts of the cv2 module the sink uses."""

    error = cv2.error
    WND_PROP_VISIBLE = cv2.WND_PROP_VISIBLE

    def __init__(
        self,
        *,
        build: str | None = GUI_BUILD,
        keys: dict[int, int] | None = None,
        closed_at: int | None = None,
        imshow_error_at: int | None = None,
        destroy_error: bool = False,
    ) -> None:
        self.build = build
        self.keys = keys or {}
        self.closed_at = closed_at
        self.imshow_error_at = imshow_error_at
        self.destroy_error = destroy_error
        self.shown: list[tuple[str, np.ndarray]] = []
        self.destroyed: list[str] = []
        self.wait_calls = 0
        self.property_queries: list[tuple[str, int]] = []

    def getBuildInformation(self) -> str:  # noqa: N802 (cv2 API name)
        if self.build is None:
            raise AttributeError("no build info")
        return self.build

    def imshow(self, title: str, image: np.ndarray) -> None:
        if self.imshow_error_at is not None and len(self.shown) == self.imshow_error_at:
            raise cv2.error("The function is not implemented. Rebuild the library with GTK+")
        self.shown.append((title, image))

    def waitKey(self, delay: int) -> int:  # noqa: N802
        self.wait_calls += 1
        return self.keys.get(len(self.shown) - 1, -1)

    def getWindowProperty(self, title: str, prop: int) -> float:  # noqa: N802
        self.property_queries.append((title, prop))
        if self.closed_at is not None and len(self.shown) - 1 >= self.closed_at:
            return -1.0
        return 1.0

    def destroyWindow(self, title: str) -> None:  # noqa: N802
        self.destroyed.append(title)
        if self.destroy_error:
            raise cv2.error("NULL window")


def _run(sink: PreviewSink, n: int) -> None:
    sink.open(INFO)
    for i in range(n):
        sink.write(_image(i), _meta(i))


def test_satisfies_sink_and_is_not_critical() -> None:
    sink = PreviewSink(gui=FakeGui())
    assert isinstance(sink, Sink)
    assert sink.critical is False
    assert PreviewSink.critical is False


def test_headless_build_detected_in_open() -> None:
    gui = FakeGui(build=HEADLESS_BUILD)
    sink = PreviewSink(gui=gui)
    with pytest.raises(RuntimeError) as info:
        sink.open(INFO)
    message = str(info.value)
    assert "opencv-python-headless" in message
    assert "pip install opencv-python" in message
    assert gui.shown == [] and gui.wait_calls == 0
    sink.abort()
    assert gui.destroyed == []


def test_real_cv2_build_info_has_a_gui_line() -> None:
    # Guards the parser against the real format; does not call any GUI function.
    from petrify.sinks.preview import _GUI_LINE

    assert _GUI_LINE.search(cv2.getBuildInformation()) is not None


def test_cv2_error_on_first_use_is_translated() -> None:
    gui = FakeGui(build=None, imshow_error_at=0)
    sink = PreviewSink(gui=gui)
    sink.open(INFO)  # No usable build info: the first GUI call decides.
    with pytest.raises(RuntimeError, match="opencv-python-headless") as info:
        sink.write(_image(0), _meta(0))
    assert info.value.args[0] == HEADLESS_MESSAGE
    assert isinstance(info.value.__cause__, cv2.error)
    sink.abort()
    assert gui.destroyed == []


def test_cv2_error_after_first_frame_is_not_translated() -> None:
    gui = FakeGui(imshow_error_at=2)
    sink = PreviewSink(gui=gui)
    sink.open(INFO)
    sink.write(_image(0), _meta(0))
    sink.write(_image(1), _meta(1))
    with pytest.raises(cv2.error):
        sink.write(_image(2), _meta(2))


def test_frames_shown_in_order_with_title() -> None:
    gui = FakeGui()
    sink = PreviewSink("cam 1", gui=gui)
    _run(sink, 4)
    assert [title for title, _ in gui.shown] == ["cam 1"] * 4
    for i, (_, image) in enumerate(gui.shown):
        assert np.array_equal(image, _image(i))
    assert gui.wait_calls == 4
    assert gui.property_queries[0] == ("cam 1", cv2.WND_PROP_VISIBLE)
    assert sink.stop_requested is False
    sink.close()
    assert gui.destroyed == ["cam 1"]


@pytest.mark.parametrize("key", [ord("q"), 27, 0x100000 | ord("q")])
def test_quit_key_requests_stop(key: int) -> None:
    gui = FakeGui(keys={1: key})
    sink = PreviewSink(gui=gui)
    _run(sink, 5)
    assert sink.stop_requested is True
    assert len(gui.shown) == 2  # Nothing shown after the key press.
    assert gui.destroyed == ["petrify"]
    sink.close()
    sink.abort()
    assert gui.destroyed == ["petrify"]


def test_other_keys_do_not_stop() -> None:
    gui = FakeGui(keys={0: ord("a"), 1: 32})
    sink = PreviewSink(gui=gui)
    _run(sink, 3)
    assert sink.stop_requested is False
    assert len(gui.shown) == 3


def test_closed_window_requests_stop() -> None:
    gui = FakeGui(closed_at=2)
    sink = PreviewSink(gui=gui)
    _run(sink, 6)
    assert sink.stop_requested is True
    assert len(gui.shown) == 3


def test_window_property_unavailable_is_ignored() -> None:
    class NoProperty(FakeGui):
        getWindowProperty = None  # type: ignore[assignment]

    gui = NoProperty()
    sink = PreviewSink(gui=gui)
    _run(sink, 3)
    assert sink.stop_requested is False
    assert len(gui.shown) == 3


def test_close_and_abort_destroy_once() -> None:
    gui = FakeGui()
    sink = PreviewSink(gui=gui)
    _run(sink, 2)
    sink.close()
    sink.close()
    sink.abort()
    assert gui.destroyed == ["petrify"]


def test_abort_destroys_once() -> None:
    gui = FakeGui()
    sink = PreviewSink(gui=gui)
    _run(sink, 2)
    sink.abort()
    sink.abort()
    sink.close()
    assert gui.destroyed == ["petrify"]


def test_destroy_errors_are_tolerated() -> None:
    gui = FakeGui(destroy_error=True)
    sink = PreviewSink(gui=gui)
    _run(sink, 1)
    sink.close()
    sink.abort()
    assert gui.destroyed == ["petrify"]


def test_abort_before_open_is_safe() -> None:
    gui = FakeGui()
    sink = PreviewSink(gui=gui)
    sink.abort()
    sink.close()
    assert gui.destroyed == []


def test_write_before_open_raises() -> None:
    sink = PreviewSink(gui=FakeGui())
    with pytest.raises(RuntimeError, match="not open"):
        sink.write(_image(0), _meta(0))


def test_reopen_resets_stop_flag() -> None:
    gui = FakeGui(keys={0: ord("q")})
    sink = PreviewSink(gui=gui)
    _run(sink, 1)
    assert sink.stop_requested is True
    sink.close()
    gui.keys = {}
    sink.open(INFO)
    assert sink.stop_requested is False


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------


class PassThroughMode:
    name = "pass"
    requires: ClassVar[frozenset[DetectionKind]] = frozenset()

    def render(self, frame: Frame, detections: Detections, subject_present: bool) -> np.ndarray:
        return frame.image


class CollectingSink:
    critical = True

    def __init__(self) -> None:
        self.indices: list[int] = []
        self.events: list[str] = []

    def open(self, info: SourceInfo) -> None:
        self.events.append("open")

    def write(self, image: np.ndarray, meta: FrameMeta) -> None:
        self.indices.append(meta.index)

    def close(self) -> None:
        self.events.append("close")

    def abort(self) -> None:
        self.events.append("abort")


class FailingGui(FakeGui):
    def imshow(self, title: str, image: np.ndarray) -> Any:
        if len(self.shown) == 2:
            raise cv2.error("window backend crashed")
        return super().imshow(title, image)


def test_engine_drops_failing_preview_and_keeps_recording() -> None:
    collecting = CollectingSink()
    gui = FailingGui()
    preview = PreviewSink(gui=gui)
    engine = Engine([], PassThroughMode(), [], [collecting, preview], EngineConfig(lookahead=0, cold_start_frames=0))
    engine.open(INFO)
    for i in range(6):
        engine.step(Frame(image=_image(i), index=i, timestamp=i / INFO.fps, source=INFO))
    stats = engine.close()

    assert collecting.indices == list(range(6))
    assert collecting.events == ["open", "close"]
    assert stats.sink_failures == 1
    assert len(gui.shown) == 2
    assert gui.destroyed == ["petrify"]  # Aborted once when dropped.
