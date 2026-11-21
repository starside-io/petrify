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

"""Virtual camera sink: publish the processed feed as a webcam via pyvirtualcam."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from petrify.core.frame import FrameMeta, SourceInfo

logger = logging.getLogger(__name__)

MISSING_MESSAGE = (
    "The virtual camera needs the optional pyvirtualcam package: "
    "pip install petrify[virtualcam]. On macOS it also needs OBS 30 or newer "
    "installed (its virtual camera is the device pyvirtualcam drives)."
)


class VirtualCamSink:
    """Send each output image to a virtual camera, so video calls see the processed feed.

    ``pyvirtualcam`` is imported lazily in ``open``. ``camera_factory`` replaces
    ``pyvirtualcam.Camera`` (for tests); it receives the same keyword arguments. If
    the factory has a ``PixelFormat`` attribute with a ``BGR`` member (as the
    ``pyvirtualcam`` module does), frames are sent as BGR without conversion;
    otherwise the camera's default RGB format is used and each frame is converted.

    ``critical`` defaults to True: in a video call the virtual camera is the output
    that matters. ``close`` and ``abort`` close the camera once, are idempotent,
    and are safe before ``open``.
    """

    def __init__(
        self,
        *,
        device: str | None = None,
        critical: bool = True,
        camera_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.critical = critical
        self._device = device
        self._camera_factory = camera_factory
        self._cam: Any = None
        self._bgr = False
        self._shape: tuple[int, int, int] | None = None

    def open(self, info: SourceInfo) -> None:
        """Create the virtual camera with the source's size and frame rate."""
        if self._cam is not None:
            raise RuntimeError("VirtualCamSink is already open")
        if not math.isfinite(info.fps) or info.fps <= 0:
            raise ValueError(f"Frame rate must be a positive number, got {info.fps!r}")
        fps = round(info.fps, 2) or 0.01  # 29.97002997 -> 29.97; never round to 0.
        factory = self._camera_factory or _load_camera_class()
        bgr_format = getattr(getattr(factory, "PixelFormat", None), "BGR", None)

        kwargs: dict[str, Any] = {"width": info.width, "height": info.height, "fps": fps}
        if bgr_format is not None:
            kwargs["fmt"] = bgr_format
        if self._device is not None:
            kwargs["device"] = self._device
        cam = factory(**kwargs)
        self._cam = cam
        self._bgr = bgr_format is not None
        self._shape = (info.height, info.width, 3)
        logger.info(
            "Virtual camera started: %s (%dx%d at %g fps)",
            getattr(cam, "device", "unknown device"),
            info.width,
            info.height,
            fps,
        )

    def write(self, image: np.ndarray, meta: FrameMeta) -> None:
        """Send one BGR uint8 image to the virtual camera."""
        cam = self._cam
        if cam is None or self._shape is None:
            raise RuntimeError("VirtualCamSink is not open")
        if image.shape != self._shape or image.dtype != np.uint8:
            raise ValueError(
                f"Expected a uint8 image of shape {self._shape}, "
                f"got {image.dtype} {image.shape}"
            )
        frame = image if self._bgr else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # No cam.sleep_until_next_frame(): pacing is the source's job (a live source
        # already paces), and sleeping here would stall the engine.
        cam.send(np.ascontiguousarray(frame))

    def close(self) -> None:
        """Close the camera. Idempotent."""
        cam, self._cam = self._cam, None
        self._shape = None
        if cam is not None:
            cam.close()

    def abort(self) -> None:
        """Close the camera without raising. Idempotent and safe before ``open``."""
        cam, self._cam = self._cam, None
        self._shape = None
        if cam is not None:
            try:
                cam.close()
            except Exception as exc:
                logger.warning("Could not close the virtual camera: %r", exc)


def _load_camera_class() -> Any:
    """Import ``pyvirtualcam`` and return its ``Camera`` class with ``PixelFormat`` reachable."""
    try:
        import pyvirtualcam
    except ImportError as exc:
        raise RuntimeError(MISSING_MESSAGE) from exc
    return _PyVirtualCamFactory(pyvirtualcam)


class _PyVirtualCamFactory:
    """Callable that builds ``pyvirtualcam.Camera`` and exposes ``pyvirtualcam.PixelFormat``."""

    def __init__(self, module: Any) -> None:
        self._module = module
        self.PixelFormat = getattr(module, "PixelFormat", None)

    def __call__(self, **kwargs: Any) -> Any:
        return self._module.Camera(**kwargs)
