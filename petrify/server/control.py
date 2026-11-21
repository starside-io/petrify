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

The runtime ranges are narrower than the command line's: they keep every
detector able to fire (see :func:`apply_patch`), and a manual ROI box may cover
at most the run's ``roi.max_area_cap`` of the frame, because the ROI region is
shown live. Settings that still weaken the protection are listed in
:attr:`RunSnapshot.weakened` (see :func:`weakened_protections`).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from petrify.core.config import BackgroundDiffConfig, PetrifyConfig, RoiConfig

logger = logging.getLogger(__name__)

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
MAX_FRAME_BUDGET_MS = 60_000.0
# Runtime bounds that keep each detector able to fire. Area thresholds are capped
# at this share of the downsampled frame the detector works on.
MAX_AREA_SHARE = 0.25
MAX_PIXEL_THRESHOLD = 128
MAX_VAR_THRESHOLD = 100.0
MAX_MIN_CONFIDENCE = 2.0
MAX_SCALE = 1.5
# A manual ROI larger than this share of the frame is listed as weakened.
ROI_WEAKENED_SHARE = 0.05

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


def _downsampled(frame_size: tuple[int, int], downsample: float) -> tuple[int, int]:
    """The frame size a detector works on, computed as the detectors resize."""
    width, height = frame_size
    return max(1, int(width * downsample)), max(1, int(height * downsample))


def _area_in(path: str, value: Any, frame_size: tuple[int, int], downsample: float) -> int:
    """An area threshold: at most :data:`MAX_AREA_SHARE` of the downsampled frame."""
    width, height = _downsampled(frame_size, downsample)
    high = int(width * height * MAX_AREA_SHARE)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= high:
        _fail(
            f"{path} must be a whole number from 0 to {high}, {MAX_AREA_SHARE * 100:g} percent "
            f"of the {width}x{height} frame that detector works on"
        )
    return int(value)


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

    The ranges keep each detector meaningful: an area threshold is at most
    :data:`MAX_AREA_SHARE` of the downsampled ``frame_size`` its detector works
    on, ``background_diff.pixel_threshold`` at most :data:`MAX_PIXEL_THRESHOLD`,
    ``motion.var_threshold`` at most :data:`MAX_VAR_THRESHOLD`,
    ``person.min_confidence`` at most :data:`MAX_MIN_CONFIDENCE` and
    ``person.scale`` at most :data:`MAX_SCALE`. A manual ROI box covers at most
    ``config.roi.max_area_cap`` of the frame. The command line is not bounded
    this way: a run may start with any value it accepts.
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
            motion["threshold"] = _area_in(
                "motion.threshold", section["threshold"], frame_size, config.motion.downsample
            )
        if "var_threshold" in section:
            motion["var_threshold"] = _float_in(
                "motion.var_threshold", section["var_threshold"], 0.0, MAX_VAR_THRESHOLD,
                low_open=True,
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
                "person.min_confidence", section["min_confidence"], 0.0, MAX_MIN_CONFIDENCE
            )
        if "min_box_area" in section:
            person["min_box_area"] = _area_in(
                "person.min_box_area", section["min_box_area"], frame_size,
                config.person.downsample,
            )
        if "scale" in section:
            person["scale"] = _float_in(
                "person.scale", section["scale"], 1.0, MAX_SCALE, low_open=True
            )
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
        base = current if current is not None else differ_defaults
        thresholds: dict[str, Any] = {}
        if "pixel_threshold" in section:
            thresholds["pixel_threshold"] = _int_in(
                "background_diff.pixel_threshold", section["pixel_threshold"], 0,
                MAX_PIXEL_THRESHOLD,
            )
        if "area_threshold" in section:
            thresholds["area_threshold"] = _area_in(
                "background_diff.area_threshold", section["area_threshold"], frame_size,
                base.downsample,
            )
        if not enabled:
            if thresholds:
                _fail(
                    "The background differ is off; turn it on to change its thresholds"
                )
            changes["background_diff"] = None
        else:
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
    if w * h > current.max_area_cap * width * height:
        _fail(
            f"roi box may cover at most {current.max_area_cap * 100:g} percent of the "
            f"{width}x{height} frame, because the ROI region is shown live (not anonymized)"
        )
    return dataclasses.replace(current, mode="manual", box=(x, y, w, h))


# ---------------------------------------------------------------------------
# Weakened protections
# ---------------------------------------------------------------------------

# Thresholds whose higher values make a detector less sensitive, compared with
# their startup values by :func:`weakened_protections`.
_THRESHOLDS: tuple[tuple[str, str], ...] = (
    ("motion", "threshold"),
    ("motion", "var_threshold"),
    ("person", "min_confidence"),
    ("person", "min_box_area"),
    ("background_diff", "pixel_threshold"),
    ("background_diff", "area_threshold"),
)


def _num(value: float) -> str:
    return str(value) if isinstance(value, int) else f"{value:g}"


def weakened_protections(
    config: PetrifyConfig,
    startup: PetrifyConfig,
    frame_size: tuple[int, int],
    *,
    raw_preview: bool = False,
    startup_differ: BackgroundDiffConfig | None = None,
) -> list[tuple[str, str]]:
    """Return the protections ``config`` weakens, as (key, sentence) pairs.

    Listed whenever they hold, startup included: the background differ off; the
    person detector alone with the differ off; ``engine.lookahead`` 0; a manual
    ROI over :data:`ROI_WEAKENED_SHARE` of the frame; the raw preview. Listed
    relative to ``startup`` (the config the run started with): a threshold more
    than twice its startup value, and a lookahead below its startup value. The
    differ thresholds compare with ``startup_differ`` (the startup differ, or the
    settings it is turned on with when the run started without it). ``key``
    names the protection, so a value change keeps the same key.
    """
    found: list[tuple[str, str]] = []
    if config.background_diff is None:
        found.append((
            "differ_off",
            "The background differ is off: a person who stops moving is no longer "
            "detected and can be shown.",
        ))
        if config.detector == "person":
            found.append((
                "person_only",
                "The detector is person only and the background differ is off: nothing "
                "detects a subject the person detector misses.",
            ))
    for section, key in _THRESHOLDS:
        current = getattr(config, section)
        start = getattr(startup, section)
        if section == "background_diff":
            start = startup_differ or start
        if current is None or start is None:
            continue
        value, before = getattr(current, key), getattr(start, key)
        if value > 2 * before:
            field = f"{section}.{key}"
            found.append((
                f"threshold:{field}",
                f"{field} is {_num(value)}, more than twice its startup value of "
                f"{_num(before)}: detection is less sensitive and can miss a person.",
            ))
    lookahead, start_lookahead = config.engine.lookahead, startup.engine.lookahead
    if lookahead < start_lookahead:
        found.append((
            "lookahead_below",
            f"engine.lookahead is {lookahead}, below its startup value of {start_lookahead}: "
            "fewer frames before a detection are held back and hidden.",
        ))
    if lookahead == 0:
        found.append((
            "lookahead_zero",
            "engine.lookahead is 0: no frame is held back, so the frames just before a "
            "detection can show a person.",
        ))
    roi = config.roi
    if roi.mode == "manual" and roi.box is not None:
        width, height = frame_size
        share = roi.box[2] * roi.box[3] / (width * height)
        if share > ROI_WEAKENED_SHARE:
            found.append((
                "roi",
                f"The manual ROI covers {share * 100:.1f} percent of the frame, and that "
                "region is shown live (not anonymized).",
            ))
    if raw_preview:
        found.append((
            "raw_preview",
            "The raw preview is on: the unprocessed feed, people included, is shown on "
            "this page.",
        ))
    return found


def config_to_dict(config: PetrifyConfig) -> dict[str, Any]:
    """Return ``config`` as plain JSON-safe values (tuples become lists).

    ``mode_options`` becomes an object from option name to value.
    """
    data = dataclasses.asdict(config)
    data["mode_options"] = config.mode_options_dict()
    return _plain(data)


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
    ``weakened`` lists, as plain sentences, the protections the applied config
    weakens (see :func:`weakened_protections`); :class:`RunControl` fills it in.
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
    weakened: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.config, MappingProxyType):
            object.__setattr__(self, "config", _frozen(self.config))
        object.__setattr__(self, "weakened", tuple(str(w) for w in self.weakened))
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
            "weakened": list(self.weakened),
        }


class RunControl:
    """Thread-safe link between the processing loop and the web server.

    Server side: :meth:`submit`, :meth:`request_stop`, :meth:`snapshot`. Loop side:
    :meth:`take_pending`, :meth:`report_applied`, :meth:`report_rejected`,
    :meth:`publish`, :attr:`stop_requested`. Only the loop touches the engine.

    ``frame_size`` (width, height) bounds a manual ROI box and the area
    thresholds. The config the run started with is kept: the weakened
    protections are computed against it, and each one that a change newly
    weakens is logged once, at warning level, when that change is applied.
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
        self._startup = config
        self._startup_differ = self._differ_defaults
        self._raw_preview = False
        self._weakened = self._weakened_for(config)
        self._last_error: str | None = None
        self._stop = threading.Event()
        self._snapshot = RunSnapshot(config=config_to_dict(config))

    def _weakened_for(self, config: PetrifyConfig) -> list[tuple[str, str]]:
        return weakened_protections(
            config, self._startup, self._frame_size, raw_preview=self._raw_preview,
            startup_differ=self._startup_differ,
        )

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

    def mark_raw_preview(self) -> None:
        """Record that the unprocessed feed is served: it is listed as weakened."""
        with self._lock:
            self._raw_preview = True
            self._weakened = self._weakened_for(self._applied)

    def snapshot(self) -> RunSnapshot:
        """Return the latest published snapshot, with the current errors and weakened list."""
        with self._lock:
            snap = self._snapshot
            error = self._last_error
            weakened = tuple(sentence for _, sentence in self._weakened)
        if snap.last_error == error and snap.weakened == weakened:
            return snap
        return dataclasses.replace(snap, last_error=error, weakened=weakened)

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
        """Record that the loop now runs with ``config``; log each newly weakened protection."""
        with self._lock:
            self._applied = config
            if self._pending is None:
                self._target = config
            if config.background_diff is not None:
                self._differ_defaults = config.background_diff
            before = {key for key, _ in self._weakened}
            self._weakened = self._weakened_for(config)
            new = [sentence for key, sentence in self._weakened if key not in before]
        for sentence in new:
            logger.warning("Weakened protection: %s", sentence)

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
