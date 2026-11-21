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

"""Preview sink: show the processed feed in a local OpenCV window."""

from __future__ import annotations

import logging
import re
from typing import Any

import cv2
import numpy as np

from petrify.core.frame import FrameMeta, SourceInfo

logger = logging.getLogger(__name__)

_ESCAPE = 27
_GUI_LINE = re.compile(r"^\s*GUI:\s*(\S+)", re.MULTILINE)

HEADLESS_MESSAGE = (
    "The preview window needs an OpenCV build with GUI support, but the installed "
    "OpenCV has none (opencv-python-headless). Replace it with the GUI build: "
    "pip uninstall opencv-python-headless && pip install opencv-python"
)


class PreviewSink:
    """Show each output image in a window titled ``title`` via ``cv2.imshow``.

    Pressing ``q`` or Escape, or closing the window, sets :attr:`stop_requested`
    and stops showing frames. The sink never stops the engine itself: the caller
    polls the flag.

    GUI calls must run on the main thread on macOS, so this sink must not be
    wrapped in ``ThreadedSink``. It shows only the processed image: like every
    sink, it never receives the raw frame.

    ``open`` raises ``RuntimeError`` when OpenCV was built without a GUI (the
    project's default ``opencv-python-headless``). ``close`` and ``abort`` destroy
    the window, are idempotent, and never raise if the window is already gone.
    ``gui`` replaces the ``cv2`` module, for tests.
    """

    critical = False

    def __init__(self, title: str = "petrify", *, gui: Any = cv2) -> None:
        self.title = title
        self.stop_requested = False
        self._gui = gui
        self._open = False
        self._window = False  # A window was created and not destroyed yet.

    def open(self, info: SourceInfo) -> None:
        """Check that OpenCV has a GUI backend; no window is created yet."""
        if not _has_gui(self._gui):
            raise RuntimeError(HEADLESS_MESSAGE)
        self.stop_requested = False
        self._window = False
        self._open = True

    def write(self, image: np.ndarray, meta: FrameMeta) -> None:
        """Show one image and poll the keyboard and the window state."""
        if not self._open:
            raise RuntimeError("PreviewSink is not open")
        if self.stop_requested:
            return
        error_type = getattr(self._gui, "error", cv2.error)
        try:
            self._gui.imshow(self.title, image)
            key = self._gui.waitKey(1)
        except error_type as exc:
            if self._window:
                raise
            raise RuntimeError(HEADLESS_MESSAGE) from exc
        self._window = True

        key = -1 if key is None else int(key)
        if key != -1 and (key & 0xFF) in (ord("q"), ord("Q"), _ESCAPE):
            logger.info("Preview stop requested from the keyboard")
            self._request_stop()
        elif self._window_closed():
            logger.info("Preview window closed")
            self._request_stop()

    def close(self) -> None:
        """Destroy the window. Idempotent."""
        self._open = False
        self._destroy()

    def abort(self) -> None:
        """Destroy the window. Idempotent and safe before ``open``."""
        self._open = False
        self._destroy()

    def _request_stop(self) -> None:
        self.stop_requested = True
        self._destroy()

    def _window_closed(self) -> bool:
        get_property = getattr(self._gui, "getWindowProperty", None)
        if get_property is None:
            return False
        prop = getattr(self._gui, "WND_PROP_VISIBLE", cv2.WND_PROP_VISIBLE)
        try:
            return float(get_property(self.title, prop)) < 1
        except Exception:
            # A window the user closed may no longer be queryable at all.
            return True

    def _destroy(self) -> None:
        if not self._window:
            return
        self._window = False
        try:
            self._gui.destroyWindow(self.title)
            # Lets the GUI event loop process the destroy (needed on macOS).
            self._gui.waitKey(1)
        except Exception as exc:
            logger.debug("Preview window already gone: %r", exc)


def _has_gui(gui: Any) -> bool:
    """Return False when the build information says OpenCV has no GUI backend.

    Returns True when the build information is unavailable or has no GUI line; a
    missing backend then surfaces as a ``cv2.error`` on the first frame.
    """
    build_info = getattr(gui, "getBuildInformation", None)
    if build_info is None:
        return True
    try:
        match = _GUI_LINE.search(str(build_info()))
    except Exception:
        return True
    return match is None or match.group(1).upper() != "NONE"
