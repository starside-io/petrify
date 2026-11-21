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

"""Thread-safe hand-off between the processing loop and the local web page.

The processing loop owns the engine and runs on the main thread; the web server
runs on another thread. They share one :class:`RunControl`: the server submits
validated config changes and stop requests, the loop takes the latest change
between two frames and publishes a :class:`RunSnapshot` after every frame. This
module has no web dependencies.
"""

from __future__ import annotations

import dataclasses
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from petrify.core.config import BackgroundDiffConfig, PetrifyConfig, RoiConfig

# Every setting the page may change while a run is in progress, as dotted paths
# into the config. The page has one control per entry.
RUNTIME_FIELDS: tuple[str, ...] = (
    "detector",
    "motion.threshold",
    "motion.var_threshold",
    "motion.detect_shadows",
    "person.min_confidence",
    "person.min_box_area",
    "person.scale",
    "background_diff.enabled",
    "background_diff.pixel_threshold",
    "background_diff.area_threshold",
    "engine.lookahead",
    "engine.frame_budget_ms",
    "roi",
)

DETECTORS = ("motion", "person", "both")
MAX_LOOKAHEAD = 600
MAX_AREA = 10_000_000
MAX_FRAME_BUDGET_MS = 60_000.0

# Top-level config keys that exist but are fixed for the whole run.
_FIXED_SECTIONS = {
    "background": "the background strategy and its settings",
    "encode": "the output encoding",
    "source": "the source",
    "outputs": "the outputs",
    "output": "the outputs",
    "mode": "the mode",
}


def _fail(message: str) -> None:
    raise ValueError(message)


def _int_in(path: str, value: Any, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        _fail(f"{path} must be a whole number from {low} to {high}")
    return int(value)


def _float_in(
    path: str, value: Any, low: float, high: float, *, low_open: bool = False
) -> float:
    ok = (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and (low < value if low_open else low <= value)
        and value <= high
    )
    if not ok:
        bound = f"greater than {low:g}" if low_open else f"at least {low:g}"
        _fail(f"{path} must be a number {bound} and at most {high:g}")
    return float(value)


def _bool(path: str, value: Any) -> bool:
    if not isinstance(value, bool):
        _fail(f"{path} must be true or false")
    return bool(value)


def _section(
    path: str, value: Any, allowed: tuple[str, ...], fixed: tuple[str, ...]
) -> dict[str, Any]:
    """Return ``value`` as a dict after checking every key is runtime-changeable."""
    if not isinstance(value, dict):
        _fail(f"{path} must be an object")
    for key in value:
        if not isinstance(key, str):
            _fail("The update contains a setting that does not exist")
        if key not in allowed:
            if key in fixed:
                _fail(f"{path}.{key} cannot be changed while the run is in progress")
            _fail("The update contains a setting that does not exist")
    return value


# Config fields that exist but are fixed while running, per section.
_FIXED_FIELDS = {
    "motion": ("history", "downsample"),
    "person": ("downsample", "win_stride"),
    "background_diff": ("downsample",),
    "engine": ("cold_start_frames", "fail_closed"),
}


def apply_patch(
    config: PetrifyConfig,
    patch: Any,
    frame_size: tuple[int, int],
    differ_defaults: BackgroundDiffConfig,
) -> PetrifyConfig:
    """Return ``config`` with the partial update ``patch`` applied, or raise ``ValueError``.

    ``patch`` mirrors the config's JSON shape, restricted to :data:`RUNTIME_FIELDS`::

        {"detector": "both",
         "motion": {"threshold": 300},
         "background_diff": {"enabled": true, "pixel_threshold": 30},
         "engine": {"lookahead": 20, "frame_budget_ms": null},
         "roi": "disabled"}          # or [x, y, w, h]

    Nothing is applied unless every entry is valid. ``differ_defaults`` supplies
    the thresholds when the background differ is turned on without them.
    """
    if not isinstance(patch, dict):
        _fail("The update must be a JSON object")
    if not patch:
        _fail("The update is empty")
    changes: dict[str, Any] = {}
    for key in patch:
        if not isinstance(key, str):
            _fail("The update contains a setting that does not exist")
        if key in _FIXED_SECTIONS:
            _fail(f"{_FIXED_SECTIONS[key].capitalize()} cannot be changed while the run is in progress")
        if key not in ("detector", "motion", "person", "background_diff", "engine", "roi"):
            _fail("The update contains a setting that does not exist")

    if "detector" in patch:
        value = patch["detector"]
        if not isinstance(value, str) or value not in DETECTORS:
            _fail("detector must be one of: motion, person, both")
        changes["detector"] = value

    if "motion" in patch:
        section = _section(
            "motion", patch["motion"], ("threshold", "var_threshold", "detect_shadows"),
            _FIXED_FIELDS["motion"],
        )
        motion: dict[str, Any] = {}
        if "threshold" in section:
            motion["threshold"] = _int_in("motion.threshold", section["threshold"], 0, MAX_AREA)
        if "var_threshold" in section:
            motion["var_threshold"] = _float_in(
                "motion.var_threshold", section["var_threshold"], 0.0, 1000.0, low_open=True
            )
        if "detect_shadows" in section:
            motion["detect_shadows"] = _bool("motion.detect_shadows", section["detect_shadows"])
        changes["motion"] = dataclasses.replace(config.motion, **motion)

    if "person" in patch:
        section = _section(
            "person", patch["person"], ("min_confidence", "min_box_area", "scale"),
            _FIXED_FIELDS["person"],
        )
        person: dict[str, Any] = {}
        if "min_confidence" in section:
            person["min_confidence"] = _float_in(
                "person.min_confidence", section["min_confidence"], 0.0, 10.0
            )
        if "min_box_area" in section:
            person["min_box_area"] = _int_in(
                "person.min_box_area", section["min_box_area"], 0, MAX_AREA
            )
        if "scale" in section:
            person["scale"] = _float_in("person.scale", section["scale"], 1.0, 2.0, low_open=True)
        changes["person"] = dataclasses.replace(config.person, **person)

    if "background_diff" in patch:
        section = _section(
            "background_diff", patch["background_diff"],
            ("enabled", "pixel_threshold", "area_threshold"), _FIXED_FIELDS["background_diff"],
        )
        current = config.background_diff
        enabled = current is not None
        if "enabled" in section:
            enabled = _bool("background_diff.enabled", section["enabled"])
        thresholds: dict[str, Any] = {}
        if "pixel_threshold" in section:
            thresholds["pixel_threshold"] = _int_in(
                "background_diff.pixel_threshold", section["pixel_threshold"], 0, 255
            )
        if "area_threshold" in section:
            thresholds["area_threshold"] = _int_in(
                "background_diff.area_threshold", section["area_threshold"], 0, MAX_AREA
            )
        if not enabled:
            if thresholds:
                _fail(
                    "The background differ is off; turn it on to change its thresholds"
                )
            changes["background_diff"] = None
        else:
            base = current if current is not None else differ_defaults
            changes["background_diff"] = dataclasses.replace(base, **thresholds)

    if "engine" in patch:
        section = _section(
            "engine", patch["engine"], ("lookahead", "frame_budget_ms"), _FIXED_FIELDS["engine"]
        )
        engine: dict[str, Any] = {}
        if "lookahead" in section:
            engine["lookahead"] = _int_in(
                "engine.lookahead", section["lookahead"], 0, MAX_LOOKAHEAD
            )
        if "frame_budget_ms" in section:
            budget = section["frame_budget_ms"]
            engine["frame_budget_ms"] = (
                None
                if budget is None
                else _float_in(
                    "engine.frame_budget_ms", budget, 0.0, MAX_FRAME_BUDGET_MS, low_open=True
                )
            )
        changes["engine"] = dataclasses.replace(config.engine, **engine)

    if "roi" in patch:
        changes["roi"] = _roi(config.roi, patch["roi"], frame_size)

    return dataclasses.replace(config, **changes)


def _roi(current: RoiConfig, value: Any, frame_size: tuple[int, int]) -> RoiConfig:
    width, height = frame_size
    if value == "disabled":
        return dataclasses.replace(current, mode="disabled", box=None)
    message = (
        f'roi must be "disabled" or a box [x, y, w, h] of whole numbers inside the '
        f"{width}x{height} frame, with w and h at least 1"
    )
    if not isinstance(value, list) or len(value) != 4:
        _fail(message)
    if any(isinstance(v, bool) or not isinstance(v, int) for v in value):
        _fail(message)
    x, y, w, h = (int(v) for v in value)
    if x < 0 or y < 0 or w < 1 or h < 1 or x + w > width or y + h > height:
        _fail(message)
    return dataclasses.replace(current, mode="manual", box=(x, y, w, h))


def config_to_dict(config: PetrifyConfig) -> dict[str, Any]:
    """Return ``config`` as plain JSON-safe values (tuples become lists)."""
    return _plain(dataclasses.asdict(config))


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _frozen(value: Any) -> Any:
    """Return a read-only view of plain JSON data (mappings and tuples)."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: _frozen(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(v) for v in value)
    return value


@dataclass(frozen=True)
class RunStats:
    """Engine counters, as in :class:`petrify.core.engine.EngineStats`."""

    frames: int = 0
    frozen: int = 0
    passed_through: int = 0
    failed_closed: int = 0
    over_budget: int = 0
    sink_failures: int = 0
    reconfigurations: int = 0


@dataclass(frozen=True)
class RunTimings:
    """Rolling per-frame timings in milliseconds and the output frame rate (None before a sample)."""

    detect_mean_ms: float | None = None
    detect_last_ms: float | None = None
    render_mean_ms: float | None = None
    render_last_ms: float | None = None
    write_mean_ms: float | None = None
    write_last_ms: float | None = None
    fps: float | None = None


@dataclass(frozen=True)
class SourceFacts:
    """What the page may know about the source; ``uri`` is already redacted."""

    uri: str | None = None
    width: int = 0
    height: int = 0
    fps: float = 0.0
    live: bool = False
    dropped: int = 0


@dataclass(frozen=True)
class BackgroundState:
    """The background strategy, whether it is ready, and how many samples it rejected."""

    strategy: str = ""
    ready: bool = False
    rejected: int = 0


@dataclass(frozen=True)
class RunSnapshot:
    """Immutable state of a run, published by the loop and read by the server.

    ``config`` is the applied config as a read-only mapping (see :meth:`to_dict`
    for plain JSON). ``running`` is True while frames are processed; ``ended`` is
    True once the run is over for any reason (both False while the run starts).
    ``last_error`` is the message of the last rejected or failed reconfiguration
    (None until one happens); it stays until another one replaces it.
    """

    config: Mapping[str, Any]
    stats: RunStats = RunStats()
    timings: RunTimings = RunTimings()
    source: SourceFacts = SourceFacts()
    background: BackgroundState = BackgroundState()
    roi: tuple[int, int, int, int] | None = None
    running: bool = False
    ended: bool = False
    last_error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.config, MappingProxyType):
            object.__setattr__(self, "config", _frozen(self.config))
        if self.roi is not None:
            object.__setattr__(self, "roi", tuple(int(v) for v in self.roi))

    def to_dict(self) -> dict[str, Any]:
        """Return the snapshot as plain JSON-safe values."""
        return {
            "config": _plain(self.config),
            "stats": dataclasses.asdict(self.stats),
            "timings": _plain(dataclasses.asdict(self.timings)),
            "source": _plain(dataclasses.asdict(self.source)),
            "background": dataclasses.asdict(self.background),
            "roi": None if self.roi is None else list(self.roi),
            "running": self.running,
            "ended": self.ended,
            "last_error": self.last_error,
        }


class RunControl:
    """Thread-safe link between the processing loop and the web server.

    Server side: :meth:`submit`, :meth:`request_stop`, :meth:`snapshot`. Loop side:
    :meth:`take_pending`, :meth:`report_applied`, :meth:`report_rejected`,
    :meth:`publish`, :attr:`stop_requested`. Only the loop touches the engine.

    ``frame_size`` (width, height) bounds a manual ROI box.
    """

    def __init__(self, config: PetrifyConfig, frame_size: tuple[int, int]) -> None:
        width, height = frame_size
        if width < 1 or height < 1:
            raise ValueError(f"frame_size must be positive, got {frame_size}")
        self._frame_size = (int(width), int(height))
        self._lock = threading.Lock()
        self._applied = config
        self._target = config  # Latest accepted: the pending config or the applied one.
        self._pending: PetrifyConfig | None = None
        self._differ_defaults = config.background_diff or BackgroundDiffConfig(
            downsample=config.motion.downsample
        )
        self._last_error: str | None = None
        self._stop = threading.Event()
        self._snapshot = RunSnapshot(config=config_to_dict(config))

    # -- server side ----------------------------------------------------

    def submit(self, patch: Any) -> PetrifyConfig:
        """Validate ``patch`` against the latest accepted config and queue the result.

        Returns the resulting config. Raises ``ValueError`` with a message for the
        user when anything in ``patch`` is invalid; nothing is queued then, and the
        message becomes :attr:`RunSnapshot.last_error`.
        """
        with self._lock:
            try:
                new = apply_patch(self._target, patch, self._frame_size, self._differ_defaults)
            except ValueError as exc:
                self._last_error = str(exc)
                raise
            self._target = new
            self._pending = new
            return new

    def request_stop(self) -> None:
        """Ask the loop to end the run after the current frame."""
        self._stop.set()

    def snapshot(self) -> RunSnapshot:
        """Return the latest published snapshot, with the current ``last_error``."""
        with self._lock:
            snap = self._snapshot
            error = self._last_error
        if snap.last_error == error:
            return snap
        return dataclasses.replace(snap, last_error=error)

    # -- loop side ------------------------------------------------------

    @property
    def stop_requested(self) -> bool:
        """True once :meth:`request_stop` was called."""
        return self._stop.is_set()

    @property
    def applied(self) -> PetrifyConfig:
        """The config the loop is running with."""
        with self._lock:
            return self._applied

    def take_pending(self) -> PetrifyConfig | None:
        """Return the most recent accepted config since the last call, or None."""
        with self._lock:
            pending, self._pending = self._pending, None
            return pending

    def report_applied(self, config: PetrifyConfig) -> None:
        """Record that the loop now runs with ``config``."""
        with self._lock:
            self._applied = config
            if self._pending is None:
                self._target = config
            if config.background_diff is not None:
                self._differ_defaults = config.background_diff

    def report_rejected(self, message: str) -> None:
        """Record that the loop could not apply a config; the applied one stays."""
        with self._lock:
            self._last_error = message
            if self._pending is None:
                self._target = self._applied

    def publish(self, snapshot: RunSnapshot) -> None:
        """Store the latest snapshot (its ``last_error`` is replaced by the control's)."""
        with self._lock:
            self._snapshot = snapshot
