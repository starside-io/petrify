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

When the control is given the run's mode factory, the page also sees the
installed modes and the active mode's options (:attr:`RunSnapshot.mode` and
:attr:`RunSnapshot.modes`), and a patch may change the options the mode marks
``runtime``. Every string that comes from a plugin or from package metadata is
untrusted: it is reduced to plain text and capped (see :func:`untrusted_text`).
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
import threading
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from petrify.core.config import (
    BackgroundDiffConfig,
    OptionValue,
    PetrifyConfig,
    RoiConfig,
    mode_options_from,
)
from petrify.core.plugin import OptionSpec, _safe_text, check_option_specs
from petrify.core.registry import BUILTIN_MODES

if TYPE_CHECKING:
    from petrify.core.plugin import ModeFactory
    from petrify.core.registry import ModeEntry

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

# Length caps for untrusted text (from a plugin or from package metadata).
NAME_TEXT_LIMIT = 64
ORIGIN_TEXT_LIMIT = 120
SUMMARY_TEXT_LIMIT = 200
HELP_TEXT_LIMIT = 300
PROBLEM_TEXT_LIMIT = 300
WEAKENED_TEXT_LIMIT = 300
# At most this many installed modes are listed, and at most this many sentences
# are taken from the mode's own weakened list.
MAX_LISTED_MODES = 50
MAX_MODE_WEAKENED = 10
# Shown, as a weakened protection, when the mode cannot report its own list.
MODE_WEAKENED_UNKNOWN = (
    "The mode {name} could not report which protections its current options weaken; "
    "treat its options as possibly weakening the protection."
)
# Added when the mode reported more weakened protections than are listed.
MODE_WEAKENED_MORE = (
    "The mode {name} reported {count} more weakened protections that are not listed here."
)

# Shown, as a weakened protection, for each kind the mode requires that the tracker
# is on for but cannot hold, because no detector in use provides it.
TRACKER_UNTRACKED_WEAKENED = (
    "The tracker is on but does not hold {kind}: no detector in use provides them, "
    "so {kind} are not held between detections."
)

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
    *,
    mode_specs: tuple[OptionSpec, ...] | None = None,
) -> PetrifyConfig:
    """Return ``config`` with the partial update ``patch`` applied, or raise ``ValueError``.

    ``patch`` mirrors the config's JSON shape, restricted to :data:`RUNTIME_FIELDS`
    and the mode's runtime options::

        {"detector": "both",
         "motion": {"threshold": 300},
         "background_diff": {"enabled": true, "pixel_threshold": 30},
         "engine": {"lookahead": 20, "frame_budget_ms": null},
         "roi": "disabled",          # or [x, y, w, h]
         "mode_options": {"strength": 4}}

    Nothing is applied unless every entry is valid. ``differ_defaults`` supplies
    the thresholds when the background differ is turned on without them.

    ``mode_specs`` are the active mode's option specs; ``mode_options`` is
    rejected when it is None (the control does not know the mode). Only an
    option declared there with ``runtime`` set is accepted, and its value is
    stored as the spec's ``validate`` returns it (see :func:`_mode_options_patch`).

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
        if key not in (
            "detector", "motion", "person", "background_diff", "engine", "roi", "mode_options"
        ):
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
        if config.person_dnn is not None:
            _fail(
                "The HOG person settings cannot be changed: this run uses the DNN person "
                "detector, which they do not affect"
            )
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

    if "mode_options" in patch:
        changes["mode_options"] = _mode_options_patch(config, patch["mode_options"], mode_specs)

    return dataclasses.replace(config, **changes)


def _expected(spec: OptionSpec) -> str:
    """What a valid value of ``spec`` is, for an error message (never the given value)."""
    if spec.kind == "bool":
        return "true or false"
    if spec.kind in ("int", "float"):
        what = "a whole number" if spec.kind == "int" else "a number"
        low, high = spec.minimum, spec.maximum
        if low is not None and high is not None:
            return f"{what} from {_num(low)} to {_num(high)}"
        if low is not None:
            return f"{what} of at least {_num(low)}"
        if high is not None:
            return f"{what} of at most {_num(high)}"
        return what
    if spec.kind == "choice":
        return "one of: " + ", ".join(spec.choices)
    return "a list of distinct entries from: " + ", ".join(spec.choices)


def _mode_options_patch(
    config: PetrifyConfig, value: Any, specs: tuple[OptionSpec, ...] | None
) -> tuple[tuple[str, OptionValue], ...]:
    """Return ``config.mode_options`` with the runtime option changes in ``value``.

    Every option already in the config is kept; the changed ones get the value
    their spec's ``validate`` returns (so ``1`` for a float option is ``1.0``).
    An unknown option name is not echoed; a known one is (it matches the name
    pattern). The given value is never echoed.
    """
    if specs is None:
        _fail("The mode options cannot be changed: this run does not report its mode")
    if not isinstance(value, dict):
        _fail("mode_options must be an object")
    by_name = {spec.name: spec for spec in specs}
    changed: dict[str, OptionValue] = {}
    for name, given in value.items():
        spec = by_name.get(name) if isinstance(name, str) else None
        if spec is None:
            _fail("The update contains a mode option that does not exist")
        if not spec.runtime:
            _fail(f"mode_options.{spec.name} cannot be changed while the run is in progress")
        try:
            changed[spec.name] = spec.validate(given)
        except ValueError:
            _fail(f"mode_options.{spec.name} must be {_expected(spec)}")
    options = config.mode_options_dict()
    options.update(changed)
    return mode_options_from(options)


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
        if section == "person" and config.person_dnn is not None:
            continue  # The HOG settings are not used while the DNN finds the person boxes.
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


# ---------------------------------------------------------------------------
# The mode, for the page
# ---------------------------------------------------------------------------


# Letters and symbols that are printable but render as blank space, so they
# could hide text or fake a gap (Hangul fillers, Khmer inherent vowels, braille blank).
_BLANK_RENDERING = frozenset("\u115f\u1160\u17b4\u17b5\u2800\u3164\uffa0")
# Raw text read before cleaning, as a multiple of the cap: enough for text
# padded with characters that cleaning removes, and still bounded.
_RAW_FACTOR = 16


def _squeeze(text: str, limit: int) -> str:
    """Collapse whitespace runs to one space and cut ``text`` to ``limit`` with ``...``."""
    cleaned = " ".join(text.split())
    if len(cleaned) > limit:
        cleaned = cleaned[: max(0, limit - 3)].rstrip() + "..."
    return cleaned


def untrusted_text(value: object, limit: int) -> str:
    """Return ``value`` as plain text of at most ``limit`` characters; never raises.

    For any string the loaded plugin supplies (summary, option help, weakened
    sentences, error messages). Letters of every script are kept. Combining
    marks and variation selectors are removed (stacked marks can paint over
    neighboring text). Characters that are not printable (control characters,
    line breaks, bidirectional overrides and other format characters) and
    characters that render blank become spaces, runs of whitespace become one
    space, and text over ``limit`` is cut and ends with ``...``. A value whose
    ``str`` fails becomes its type name, as in :func:`petrify.core.plugin._safe_text`.
    """
    raw = _safe_text(value, use_repr=False, limit=_RAW_FACTOR * limit)
    kept = (ch for ch in raw if not unicodedata.category(ch).startswith("M"))
    text = "".join(
        ch if ch.isprintable() and ch not in _BLANK_RENDERING else " " for ch in kept
    )
    return _squeeze(text, limit)


def metadata_text(value: object, limit: int) -> str:
    """Return ``value`` as printable ASCII of at most ``limit`` characters; never raises.

    For strings that come from package metadata of installed distributions,
    selected or not (mode names, distribution names, versions, and the problem
    texts built from them). Whitespace becomes a space; every other character
    outside printable ASCII (letters of other scripts, look-alike letters,
    combining marks, right-to-left letters, blank-rendering characters)
    becomes a visible ``?``, so such text cannot pass for another mode's name.
    """
    raw = _safe_text(value, use_repr=False, limit=_RAW_FACTOR * limit)
    text = "".join(
        ch if " " <= ch <= "~" else (" " if ch.isspace() else "?") for ch in raw
    )
    return _squeeze(text, limit)


def _option_value(value: OptionValue) -> Any:
    """An option value as JSON data: a choices tuple becomes a list of plain text."""
    if isinstance(value, tuple):
        return [untrusted_text(item, NAME_TEXT_LIMIT) for item in value]
    if isinstance(value, str):
        return untrusted_text(value, NAME_TEXT_LIMIT)
    return value


def _own_spec(spec: OptionSpec) -> OptionSpec:
    """Return ``spec`` as a plain :class:`OptionSpec` (a subclass is copied field by field)."""
    if type(spec) is OptionSpec:
        return spec
    return OptionSpec(
        name=spec.name, kind=spec.kind, default=spec.default, help=spec.help,
        choices=spec.choices, minimum=spec.minimum, maximum=spec.maximum, runtime=spec.runtime,
    )


def _option_info(spec: OptionSpec, value: OptionValue) -> dict[str, Any]:
    """What the page needs to show one option: kind, value, default, bounds, help, runtime."""
    return {
        "name": untrusted_text(spec.name, NAME_TEXT_LIMIT),
        "kind": spec.kind,
        "value": _option_value(value),
        "default": _option_value(spec.default),
        "choices": [untrusted_text(choice, NAME_TEXT_LIMIT) for choice in spec.choices],
        "minimum": spec.minimum,
        "maximum": spec.maximum,
        "help": untrusted_text(spec.help, HELP_TEXT_LIMIT),
        "runtime": spec.runtime,
    }


def _entry_info(entry: ModeEntry, active: bool) -> dict[str, Any]:
    """One listed mode, from its entry point metadata only (nothing is imported)."""
    problem = entry.problem
    return {
        "name": metadata_text(entry.name, NAME_TEXT_LIMIT),
        "origin": metadata_text(entry.origin, ORIGIN_TEXT_LIMIT),
        "builtin": bool(entry.builtin),
        "selectable": problem is None,
        "problem": None if problem is None else metadata_text(problem, PROBLEM_TEXT_LIMIT),
        "active": active,
    }


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
    detectors: tuple[tuple[str, float | None, float | None], ...] = ()
    """``(name, mean_ms, last_ms)`` per detector and refiner, in run order.

    The names come from class names of petrify's own components, but are cleaned
    like any other text the page shows, so a component class cannot put markup or
    control characters on the page.
    """

    def __post_init__(self) -> None:
        object.__setattr__(self, "detectors", tuple(
            (untrusted_text(name, NAME_TEXT_LIMIT), mean, last)
            for name, mean, last in self.detectors
        ))

    def to_dict(self) -> dict[str, Any]:
        """Return the timings as plain JSON values; ``detectors`` is a list of objects."""
        data = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        data["detectors"] = [
            {"name": name, "mean_ms": mean, "last_ms": last}
            for name, mean, last in self.detectors
        ]
        return _plain(data)


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
    """The background strategy, whether it is ready, and how many samples it rejected.

    ``in_use`` is False when the mode uses no background at all; ``ready`` and
    ``rejected`` then mean nothing. :meth:`to_dict` adds ``"in_use": false``
    only in that case, so the state of a mode with a background keeps its shape.
    """

    strategy: str = ""
    ready: bool = False
    rejected: int = 0
    in_use: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return the state as plain JSON values."""
        data: dict[str, Any] = {
            "strategy": self.strategy, "ready": self.ready, "rejected": self.rejected,
        }
        if not self.in_use:
            data["in_use"] = False
        return data


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

    ``mode`` (the active mode and its options, with the applied values) and
    ``modes`` (the installed modes) are filled in by a :class:`RunControl` that
    was given the run's mode factory; both stay None otherwise, and
    :meth:`to_dict` then leaves them out, so the state keeps its earlier shape.
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
    mode: Mapping[str, Any] | None = None
    modes: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.config, MappingProxyType):
            object.__setattr__(self, "config", _frozen(self.config))
        object.__setattr__(self, "weakened", tuple(str(w) for w in self.weakened))
        if self.roi is not None:
            object.__setattr__(self, "roi", tuple(int(v) for v in self.roi))
        for name in ("mode", "modes"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, MappingProxyType):
                object.__setattr__(self, name, _frozen(value))

    def to_dict(self) -> dict[str, Any]:
        """Return the snapshot as plain JSON-safe values."""
        data = {
            "config": _plain(self.config),
            "stats": dataclasses.asdict(self.stats),
            "timings": self.timings.to_dict(),
            "source": _plain(dataclasses.asdict(self.source)),
            "background": self.background.to_dict(),
            "roi": None if self.roi is None else list(self.roi),
            "running": self.running,
            "ended": self.ended,
            "last_error": self.last_error,
            "weakened": list(self.weakened),
            "person_detector": "dnn" if self.config.get("person_dnn") is not None else "hog",
        }
        if self.mode is not None:
            data["mode"] = _plain(self.mode)
            data["modes"] = _plain(self.modes) if self.modes is not None else None
        return data


class RunControl:
    """Thread-safe link between the processing loop and the web server.

    Server side: :meth:`submit`, :meth:`request_stop`, :meth:`snapshot`. Loop side:
    :meth:`take_pending`, :meth:`report_applied`, :meth:`report_rejected`,
    :meth:`publish`, :attr:`stop_requested`. Only the loop touches the engine.

    ``frame_size`` (width, height) bounds a manual ROI box and the area
    thresholds. The config the run started with is kept: the weakened
    protections are computed against it, and each one that a change newly
    weakens is logged once, at warning level, when that change is applied.

    ``factory`` is the run's already loaded mode factory and ``modes`` the
    listed modes (:func:`petrify.core.registry.list_modes`, metadata only). With
    a factory, the snapshot describes the modes and the active mode's options,
    a patch may change the options marked ``runtime``, and the sentences of
    ``factory.weakened`` join the weakened list. Without one (the default) the
    control behaves as before and rejects every ``mode_options`` patch. No other
    plugin is imported: ``modes`` is read as given.
    """

    def __init__(
        self,
        config: PetrifyConfig,
        frame_size: tuple[int, int],
        *,
        modes: Sequence[ModeEntry] = (),
        factory: ModeFactory | None = None,
    ) -> None:
        width, height = frame_size
        if width < 1 or height < 1:
            raise ValueError(f"frame_size must be positive, got {frame_size}")
        self._frame_size = (int(width), int(height))
        self._factory = factory
        self._specs: tuple[OptionSpec, ...] | None = None
        self._mode_head: dict[str, Any] = {}
        self._modes: Mapping[str, Any] | None = None
        if factory is not None:
            self._init_mode(config, factory, tuple(modes))
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
        # Patches accepted since the loop last took a pending config, so the
        # pending change can be rebuilt on the applied config after a rejection.
        self._pending_patches: list[Any] = []
        # The weakened list is the built-in part plus the mode's part. The mode's
        # part comes from plugin code, which never runs under the lock: each
        # applied config gets a generation, and only the result computed for the
        # latest generation is installed (see report_applied).
        self._generation = 0
        self._builtin_part = self._builtin_weakened(config)
        self._mode_part = self._mode_part_for(config) if factory is not None else []
        self._weakened = self._builtin_part + self._mode_part
        self._announced = {key for key, _ in self._weakened}
        self._mode = self._mode_for(config)
        self._last_error: str | None = None
        self._stop = threading.Event()
        self._snapshot = RunSnapshot(config=config_to_dict(config))

    def _init_mode(
        self, config: PetrifyConfig, factory: ModeFactory, modes: tuple[ModeEntry, ...]
    ) -> None:
        """Read what the page shows about the modes once; only ``factory`` is consulted."""
        name = factory.name
        if type(name) is not str or name != config.mode:
            raise ValueError(
                f"the factory is named {_safe_text(name)}, but the run's mode is {config.mode!r}"
            )
        # Plain OptionSpec copies: a plugin's subclass could override validate or
        # make its fields properties, and the specs are used under the lock.
        self._specs = tuple(_own_spec(spec) for spec in check_option_specs(factory.options))
        try:
            summary = factory.summary
        except Exception:  # noqa: BLE001 (a plugin property; the page shows no summary)
            summary = ""
        active = next((e for e in modes if e.name == name and e.problem is None), None)
        builtin = active.builtin if active is not None else name in BUILTIN_MODES
        if active is not None:
            origin = metadata_text(active.origin, ORIGIN_TEXT_LIMIT)
        else:
            origin = "built in" if builtin else "unknown origin"
        self._mode_head = {
            "name": metadata_text(name, NAME_TEXT_LIMIT),
            "origin": origin,
            "builtin": bool(builtin),
            "summary": untrusted_text(summary, SUMMARY_TEXT_LIMIT),
        }
        listed = [_entry_info(e, e is active) for e in modes[:MAX_LISTED_MODES]]
        self._modes = _frozen(
            {"listed": listed, "omitted": max(0, len(modes) - MAX_LISTED_MODES)}
        )

    def _options_for(self, config: PetrifyConfig) -> dict[str, OptionValue]:
        """Every declared option: the config's value, else the default."""
        given = config.mode_options_dict()
        return {spec.name: given.get(spec.name, spec.default) for spec in self._specs or ()}

    def _mode_for(self, config: PetrifyConfig) -> Mapping[str, Any] | None:
        """The active mode and its options with the values of ``config``, read-only."""
        if self._factory is None:
            return None
        options = self._options_for(config)
        return _frozen({
            **self._mode_head,
            "options": [_option_info(spec, options[spec.name]) for spec in self._specs or ()],
        })

    def _builtin_weakened(self, config: PetrifyConfig) -> list[tuple[str, str]]:
        """The protections ``config`` weakens, from petrify's own rules (no plugin code)."""
        return weakened_protections(
            config, self._startup, self._frame_size, raw_preview=self._raw_preview,
            startup_differ=self._startup_differ,
        )

    def _mode_part_for(self, config: PetrifyConfig) -> list[tuple[str, str]]:
        """The part of the weakened list that runs plugin code; never under the lock."""
        return self._mode_weakened(config) + self._tracker_weakened(config)

    def _tracker_weakened(self, config: PetrifyConfig) -> list[tuple[str, str]]:
        """One sentence per kind the tracker cannot hold; never raises.

        What the mode requires comes from ``factory.requires`` (plugin code: never
        call it with the lock held). When it fails or returns something else than
        a frozenset of kinds, nothing is listed here: the run itself refuses such
        a mode. The kinds come from the pipeline's own rule, so the page and the
        startup warning name the same ones.
        """
        if config.tracker is None:
            return []
        factory = self._factory
        assert factory is not None
        # Imported here: the pipeline loads OpenCV, which this module does not need otherwise.
        from petrify.core.detections import DetectionKind  # noqa: PLC0415
        from petrify.pipeline import untracked_kinds  # noqa: PLC0415

        try:
            requires = factory.requires(MappingProxyType(self._options_for(config)))
        except (Exception, SystemExit):  # noqa: BLE001 (plugin code; the run reports it)
            return []
        if not isinstance(requires, frozenset) or not all(
            isinstance(kind, DetectionKind) for kind in requires
        ):
            return []
        return [
            (f"untracked:{kind.value}", TRACKER_UNTRACKED_WEAKENED.format(kind=kind.value))
            for kind in untracked_kinds(config, requires)
        ]

    def _mode_weakened(self, config: PetrifyConfig) -> list[tuple[str, str]]:
        """The sentences of ``factory.weakened`` as (key, sentence) pairs; never raises.

        Runs plugin code: never call it with the lock held. When the call raises,
        or returns anything but a tuple of strings, or a sentence is empty once
        cleaned, the list holds the fixed sentence :data:`MODE_WEAKENED_UNKNOWN`
        (once): an empty or shorter list would read as "nothing is weakened",
        which is not known then. At most :data:`MAX_MODE_WEAKENED` sentences are
        listed; when the mode reported more, :data:`MODE_WEAKENED_MORE` says how
        many are not shown.
        """
        factory = self._factory
        assert factory is not None
        name = self._mode_head["name"]
        unknown = ("mode_weakened_unknown", MODE_WEAKENED_UNKNOWN.format(name=name))
        options = MappingProxyType(self._options_for(config))
        sentences: list[str] = []
        try:
            result = factory.weakened(options)
            ok = isinstance(result, tuple) and all(isinstance(s, str) for s in result)
            if ok:
                sentences = list(result)
        except (Exception, SystemExit) as exc:
            logger.warning(
                "The mode %s could not report its weakened protections: %s",
                name, untrusted_text(type(exc).__name__, NAME_TEXT_LIMIT),
            )
            ok = False
        if not ok:
            return [unknown]
        found: list[tuple[str, str]] = []
        blank = False
        for sentence in sentences[:MAX_MODE_WEAKENED]:
            text = untrusted_text(sentence, WEAKENED_TEXT_LIMIT)
            if not text:
                blank = True
            elif all(text != seen for _, seen in found):
                found.append((f"mode:{text}", text))
        if blank:
            found.append(unknown)
        more = len(sentences) - MAX_MODE_WEAKENED
        if more > 0:
            found.append(("mode_weakened_more", MODE_WEAKENED_MORE.format(name=name, count=more)))
        return found

    def _announce(self) -> list[str]:
        """Return the listed sentences not announced yet, and mark every one announced."""
        new = [sentence for key, sentence in self._weakened if key not in self._announced]
        self._announced = {key for key, _ in self._weakened}
        return new

    # -- server side ----------------------------------------------------

    def submit(self, patch: Any) -> PetrifyConfig:
        """Validate ``patch`` against the latest accepted config and queue the result.

        Returns the resulting config. Raises ``ValueError`` with a message for the
        user when anything in ``patch`` is invalid; nothing is queued then, and the
        message becomes :attr:`RunSnapshot.last_error`.
        """
        with self._lock:
            try:
                new = apply_patch(
                    self._target, patch, self._frame_size, self._differ_defaults,
                    mode_specs=self._specs,
                )
            except ValueError as exc:
                self._last_error = str(exc)
                raise
            self._target = new
            self._pending = new
            self._pending_patches.append(copy.deepcopy(patch))
            return new

    def request_stop(self) -> None:
        """Ask the loop to end the run after the current frame."""
        self._stop.set()

    def mark_raw_preview(self) -> None:
        """Record that the unprocessed feed is served: it is listed as weakened."""
        with self._lock:
            self._raw_preview = True
            self._builtin_part = self._builtin_weakened(self._applied)
            self._weakened = self._builtin_part + self._mode_part
            self._announced = {key for key, _ in self._weakened}

    def snapshot(self) -> RunSnapshot:
        """Return the latest published snapshot, with the current errors and weakened list.

        With a mode factory, ``mode`` and ``modes`` describe the modes, with the
        option values of the applied config.
        """
        with self._lock:
            snap = self._snapshot
            error = self._last_error
            weakened = tuple(sentence for _, sentence in self._weakened)
            mode = self._mode
        if (
            snap.last_error == error
            and snap.weakened == weakened
            and snap.mode is mode
            and snap.modes is self._modes
        ):
            return snap
        return dataclasses.replace(
            snap, last_error=error, weakened=weakened, mode=mode, modes=self._modes
        )

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
            self._pending_patches = []
            return pending

    def report_applied(self, config: PetrifyConfig) -> None:
        """Record that the loop now runs with ``config``; log each newly weakened protection.

        The built-in part of the weakened list is updated at once. The mode's
        part is computed by plugin code outside the lock and then installed only
        if no later :meth:`report_applied` started meanwhile: the newest applied
        config always wins, and a slower result for an older one is discarded.
        Until it is installed, the mode's previous sentences stay listed.
        """
        new: list[str] = []
        with self._lock:
            self._applied = config
            if self._pending is None:
                self._target = config
            if config.background_diff is not None:
                self._differ_defaults = config.background_diff
            self._generation += 1
            generation = self._generation
            self._builtin_part = self._builtin_weakened(config)
            self._weakened = self._builtin_part + self._mode_part
            self._mode = self._mode_for(config)
            if self._factory is None:
                new = self._announce()
        if self._factory is not None:
            part = self._mode_part_for(config)  # Plugin code: outside the lock.
            with self._lock:
                if generation != self._generation:
                    return  # A newer config was applied; its own call installs its list.
                self._mode_part = part
                self._weakened = self._builtin_part + part
                new = self._announce()
        for sentence in new:
            logger.warning("Weakened protection: %s", sentence)

    def report_rejected(self, message: str) -> None:
        """Record that the loop could not apply a config; the applied one stays.

        ``message`` may come from plugin code: it is cleaned and capped. A change
        accepted after the rejected config was taken was built on it; it is
        rebuilt on the applied config from its own patches, so it stands on its
        own and is applied next. When one of those patches no longer applies
        without the rejected change, that change is dropped too and the error
        says so.
        """
        text = untrusted_text(message, PROBLEM_TEXT_LIMIT)
        with self._lock:
            if self._pending is None:
                self._target = self._applied
                self._last_error = text
                return
            rebuilt = self._applied
            try:
                for patch in self._pending_patches:
                    rebuilt = apply_patch(
                        rebuilt, patch, self._frame_size, self._differ_defaults,
                        mode_specs=self._specs,
                    )
            except ValueError as exc:
                self._pending = None
                self._pending_patches = []
                self._target = self._applied
                self._last_error = untrusted_text(
                    f"{text} A later change was dropped too, because without the rejected "
                    f"one it is no longer valid: {exc}",
                    PROBLEM_TEXT_LIMIT,
                )
                return
            self._pending = rebuilt
            self._target = rebuilt
            self._last_error = text

    def publish(self, snapshot: RunSnapshot) -> None:
        """Store the latest snapshot (its ``last_error`` is replaced by the control's)."""
        with self._lock:
            self._snapshot = snapshot
