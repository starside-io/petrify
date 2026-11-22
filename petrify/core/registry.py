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

"""The mode registry: lists the available modes and loads the selected one.

Built-in modes are declared in :data:`BUILTIN_MODES`, in code, so they work from
a source checkout and installed metadata can neither remove nor override them.
Installed distributions add modes through the ``petrify.modes`` entry point
group; each entry point names a :class:`~petrify.core.plugin.ModeFactory` as
``module:attr``.

Trust model: a mode plugin is ordinary Python code. It runs in the petrify
process with petrify's permissions and receives the raw camera frames, people
included. Installing one means trusting it as much as petrify itself. The
registry does not sandbox or vet plugin code; it only guarantees that a plugin
that is not selected is never imported: :func:`list_modes` reads entry point
metadata alone, and :func:`load_mode` imports only the selected mode's target.
The checks :func:`load_mode` and :func:`build_mode` apply are a guard against
mistakes in a plugin, not against a hostile one.

Discovery never raises: a distribution whose metadata cannot be read is skipped
with a logged warning, and the built-in modes load without discovery at all.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import EntryPoint
from types import MappingProxyType

from petrify.core.config import _NAME_PATTERN, mode_options_from
from petrify.core.detections import DetectionKind
from petrify.core.plugin import (
    Mode,
    ModeContext,
    ModeFactory,
    OptionSpec,
    _safe_text,
    check_option_specs,
    describe_error,
)

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "petrify.modes"

BUILTIN_MODES: Mapping[str, str] = MappingProxyType({"erase": "petrify.modes.erase:ERASE"})
"""Built-in mode names and their ``module:attr`` targets (read-only)."""

_DETECT_POLICIES: tuple[str, ...] = ("first_hit", "all")
# Kinds that come from a detector of their own, grouped by that detector. Under the
# "first_hit" policy the engine skips the later detectors on a frame where an
# earlier one fired, so a mode that needs one of these kinds together with a kind
# from another detector would get None in that field on most frames.
_OWN_DETECTOR_KINDS: tuple[frozenset[DetectionKind], ...] = (
    frozenset({DetectionKind.FACES, DetectionKind.LANDMARKS}),
    frozenset({DetectionKind.MASK}),
)
# Longest name or value copied into a problem string, which a web page may show.
_PROBLEM_TEXT_LIMIT = 80
# Most registrations named in one ambiguity message.
_AMBIGUOUS_LISTED = 5


@dataclass(frozen=True)
class ModeEntry:
    """One listed mode: where it comes from and whether it can be selected."""

    name: str
    target: str  # "module:attr"
    distribution: str | None  # None for built-ins.
    version: str | None
    builtin: bool
    problem: str | None  # Why the mode cannot be selected, or None.

    @property
    def origin(self) -> str:
        """The mode's origin for messages, for example ``petrify-shadow 1.0``."""
        if self.builtin:
            return "built in"
        if self.distribution is None:
            return "unknown distribution"
        name = _safe_text(self.distribution, use_repr=False, limit=_PROBLEM_TEXT_LIMIT)
        if self.version is None:
            return name
        version = _safe_text(self.version, use_repr=False, limit=_PROBLEM_TEXT_LIMIT)
        return f"{name} {version}"


class ModeLoadError(RuntimeError):
    """A mode cannot be loaded or built; the message says why, for the user."""


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _normalized(name: str) -> str:
    """The distribution name as ``importlib.metadata`` compares it (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _distribution_name(dist: object) -> str | None:
    """Return the distribution's name, or None when its metadata cannot be read."""
    try:
        name = dist.name  # type: ignore[attr-defined]
    except Exception:
        return None
    return name if type(name) is str and name else None


def _discover() -> list[EntryPoint]:
    """Return the installed ``petrify.modes`` entry points; never raises.

    Walks the distributions in ``sys.path`` order and keeps only the first of
    each normalized name, as ``importlib.metadata.entry_points`` does. A
    distribution whose entry points cannot be read is skipped with a warning.
    """
    found: list[EntryPoint] = []
    seen: set[str] = set()
    try:
        dists = iter(importlib.metadata.distributions())
    except Exception as exc:
        logger.warning("Mode plugin discovery failed: %s", describe_error(exc))
        return found
    while True:
        try:
            dist = next(dists)
        except StopIteration:
            break
        except Exception as exc:
            logger.warning("Mode plugin discovery stopped early: %s", describe_error(exc))
            break
        name = _distribution_name(dist)
        if name is not None:
            key = _normalized(name)
            if key in seen:
                continue
            seen.add(key)
        try:
            entries = [ep for ep in dist.entry_points if ep.group == ENTRY_POINT_GROUP]
        except Exception as exc:
            shown = (
                _safe_text(name, use_repr=False, limit=_PROBLEM_TEXT_LIMIT)
                if name is not None
                else "with unreadable metadata"
            )
            logger.warning(
                "Skipping distribution %s while looking for modes: its entry points "
                "cannot be read (%s)",
                shown,
                describe_error(exc),
            )
            continue
        found.extend(entries)
    return found


def _check_dotted(module: str, attr: str) -> tuple[str, str] | None:
    if not module or not attr:
        return None
    parts = module.split(".") + attr.split(".")
    if not all(part.isidentifier() for part in parts):
        return None
    return module, attr


def _split_target(target: str) -> tuple[str, str] | None:
    """Return ``(module, attr)`` for a stored ``module:attr`` target, or None."""
    module, sep, attr = target.partition(":")
    if not sep:
        return None
    return _check_dotted(module.strip(), attr.strip())


def _parse_entry_point(entry_point: EntryPoint) -> tuple[str, str] | None:
    """Return the entry point's own parsed ``(module, attr)``, or None when malformed.

    Extras (``module:attr [extra]``) are allowed and ignored.
    """
    try:
        module = entry_point.module
        attr = entry_point.attr
    except Exception:  # A value the entry point pattern does not match.
        return None
    if type(module) is not str or type(attr) is not str:
        return None
    return _check_dotted(module, attr)


def _distribution_of(entry_point: EntryPoint) -> tuple[str | None, str | None]:
    """Return the distribution name and version, from metadata only."""
    try:
        dist = getattr(entry_point, "dist", None)
    except Exception:
        return None, None
    if dist is None:
        return None, None
    name = _distribution_name(dist)
    try:
        version = dist.version
    except Exception:
        version = None
    return name, version if type(version) is str else None


def _short(value: str) -> str:
    return _safe_text(value, limit=_PROBLEM_TEXT_LIMIT)


def list_modes(entry_points: Iterable[EntryPoint] | None = None) -> tuple[ModeEntry, ...]:
    """Return every built-in and installed mode, selectable or not; never raises.

    ``entry_points`` defaults to the installed ``petrify.modes`` entry points;
    entries of any other group are ignored. Nothing is imported: only entry point
    names, values and their distribution's name and version are read.

    An entry point gets a ``problem`` (and can never be loaded) when its name does
    not match ``[a-z][a-z0-9_]{0,31}``, when its value is not ``module:attr``
    (optionally followed by extras), when its name is a built-in mode's, or when
    another distribution (or the same distribution with another target)
    registers the same name, in which case every one of them is ambiguous.
    Order: built-ins first, then by name, then by distribution.
    """
    builtins = tuple(
        ModeEntry(name, target, None, None, True, None) for name, target in BUILTIN_MODES.items()
    )
    if entry_points is None:
        entry_points = _discover()

    candidates: list[ModeEntry] = []
    seen: set[tuple[str, str, str | None]] = set()
    try:
        for entry_point in entry_points:
            try:
                if entry_point.group != ENTRY_POINT_GROUP:
                    continue
                name = entry_point.name
                value = entry_point.value
            except Exception:
                continue
            if type(name) is not str or type(value) is not str:
                continue
            distribution, version = _distribution_of(entry_point)
            parsed = _parse_entry_point(entry_point)
            target = f"{parsed[0]}:{parsed[1]}" if parsed is not None else value
            key = (name, target, distribution)
            if key in seen:
                continue  # The same registration seen twice.
            seen.add(key)
            problem: str | None = None
            if not _NAME_PATTERN.fullmatch(name):
                problem = (
                    f"invalid mode name {_short(name)}: it must match [a-z][a-z0-9_]{{0,31}}"
                )
            elif parsed is None:
                problem = f"invalid entry point value {_short(value)}: it must be module:attr"
            elif name in BUILTIN_MODES:
                problem = f"shadowed by the built-in mode {name!r}; it can never be selected"
            candidates.append(ModeEntry(name, target, distribution, version, False, problem))
    except Exception as exc:
        logger.warning("Listing mode plugins stopped early: %s", describe_error(exc))

    by_name: dict[str, list[ModeEntry]] = {}
    for entry in candidates:
        by_name.setdefault(entry.name, []).append(entry)
    plugins: list[ModeEntry] = []
    for name, group in by_name.items():
        if len(group) > 1 and name not in BUILTIN_MODES:
            problem = _ambiguous(name, group)
            group = [
                entry
                if entry.problem is not None
                else ModeEntry(
                    entry.name, entry.target, entry.distribution, entry.version, False, problem
                )
                for entry in group
            ]
        plugins.extend(group)
    plugins.sort(key=lambda e: (e.name, e.distribution or "", e.version or "", e.target))
    return builtins + tuple(plugins)


def _ambiguous(name: str, group: list[ModeEntry]) -> str:
    """The problem of a mode name registered more than once, naming each registration."""
    registrations = sorted(f"{entry.origin} as {_short(entry.target)}" for entry in group)
    listed = ", ".join(registrations[:_AMBIGUOUS_LISTED])
    if len(registrations) > _AMBIGUOUS_LISTED:
        listed += f" and {len(registrations) - _AMBIGUOUS_LISTED} more"
    return (
        f"ambiguous: the mode name {_short(name)} is registered more than once "
        f"({listed}); uninstall all but one"
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _type_name(obj: object) -> str:
    try:
        return _safe_text(type(obj).__name__, use_repr=False, limit=_PROBLEM_TEXT_LIMIT)
    except Exception:
        return "object"


def _import_target(entry: ModeEntry) -> object:
    """Import ``entry.target`` and return the object it names, or raise ModeLoadError."""
    label = f"mode {_short(entry.name)} ({entry.origin})"
    split = _split_target(entry.target)
    if split is None:
        raise ModeLoadError(f"{label} has an invalid target {_short(entry.target)}")
    module_name, attr = split
    try:
        obj: object = importlib.import_module(module_name)
    except (Exception, SystemExit) as exc:
        raise ModeLoadError(
            f"{label} could not be imported from {_short(module_name)}: {describe_error(exc)}"
        ) from exc
    for part in attr.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise ModeLoadError(
                f"{label}: {_short(module_name)} has no attribute {_short(attr)}"
            ) from exc
        except (Exception, SystemExit) as exc:
            raise ModeLoadError(
                f"{label}: reading {_short(entry.target)} failed: {describe_error(exc)}"
            ) from exc
    return obj


def _check_factory(entry: ModeEntry, obj: object) -> ModeFactory:
    """Return ``obj`` when it is a valid factory for ``entry``, or raise ModeLoadError."""
    label = f"mode {_short(entry.name)} ({entry.origin})"
    target = _short(entry.target)
    if isinstance(obj, type):
        raise ModeLoadError(f"{label}: {target} is a class; it must be a ModeFactory instance")
    if not isinstance(obj, ModeFactory):
        raise ModeLoadError(
            f"{label}: {target} is a {_type_name(obj)}, not a ModeFactory "
            "(it needs name, summary, options, needs_background, roi_overlay, requires, "
            "mandatory, create and weakened)"
        )
    name = obj.name
    if type(name) is not str or name != entry.name:
        raise ModeLoadError(
            f"{label}: the factory is named {_safe_text(name)}, which does not match the "
            f"registered name {_short(entry.name)}"
        )
    if not isinstance(obj.summary, str):
        raise ModeLoadError(f"{label}: summary must be a string, got {_safe_text(obj.summary)}")
    try:
        check_option_specs(obj.options)
    except ValueError as exc:
        raise ModeLoadError(f"{label}: invalid options: {_safe_text(exc, use_repr=False)}") from exc
    for flag in ("needs_background", "roi_overlay"):
        value = getattr(obj, flag)
        if type(value) is not bool:
            raise ModeLoadError(f"{label}: {flag} must be True or False, got {_safe_text(value)}")
    for method in ("requires", "mandatory", "create", "weakened"):
        if not callable(getattr(obj, method)):
            raise ModeLoadError(f"{label}: {method} must be callable")
    return obj


def _load_entry(entry: ModeEntry) -> ModeFactory:
    obj = _import_target(entry)
    try:
        return _check_factory(entry, obj)
    except ModeLoadError:
        raise
    except (Exception, SystemExit) as exc:
        raise ModeLoadError(
            f"mode {_short(entry.name)} ({entry.origin}): checking the factory failed: "
            f"{describe_error(exc)}"
        ) from exc


def load_mode(name: str, entries: Sequence[ModeEntry] | None = None) -> ModeFactory:
    """Import and return the factory of the mode ``name``.

    A built-in name loads the built-in factory directly, without plugin
    discovery, so it works whatever is installed. Otherwise ``entries`` defaults
    to :func:`list_modes`, and only the selected mode's target is imported.
    Raises :class:`ModeLoadError` with a reason for the user when the name is
    unknown or not selectable, when the import fails, and when the object is not
    a valid :class:`~petrify.core.plugin.ModeFactory` for that name. Any
    exception raised by the plugin's code while loading, ``SystemExit``
    included, becomes a :class:`ModeLoadError`.
    """
    if type(name) is not str:
        raise ModeLoadError(f"a mode name must be a string, got {_safe_text(name)}")
    builtin_target = BUILTIN_MODES.get(name)
    if builtin_target is not None:
        return _load_entry(ModeEntry(name, builtin_target, None, None, True, None))
    if entries is None:
        entries = list_modes()
    matching = [entry for entry in entries if entry.name == name]
    selectable = [entry for entry in matching if entry.problem is None]
    if not matching:
        names = sorted({entry.name for entry in entries if entry.problem is None})
        raise ModeLoadError(
            f"unknown mode {_short(name)}; available modes: "
            f"{', '.join(names) if names else 'none'}"
        )
    if not selectable:
        reasons = "; ".join(f"{entry.origin}: {entry.problem}" for entry in matching[:5])
        raise ModeLoadError(f"mode {_short(name)} cannot be selected: {reasons}")
    if len(selectable) > 1:
        origins = ", ".join(entry.origin for entry in selectable[:5])
        raise ModeLoadError(f"mode {_short(name)} is registered more than once ({origins})")
    return _load_entry(selectable[0])


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def _is_builtin_factory(factory: ModeFactory) -> bool:
    """True when ``factory`` is the object a built-in mode's target names."""
    target = BUILTIN_MODES.get(factory.name)
    if target is None:
        return False
    module_name, attr = target.split(":")
    return getattr(importlib.import_module(module_name), attr) is factory


def _is_kind_set(value: object) -> bool:
    """True for a frozenset holding only DetectionKind members; never raises."""
    try:
        return isinstance(value, frozenset) and all(
            isinstance(kind, DetectionKind) for kind in value
        )
    except Exception:
        return False


def _kinds(label: str, what: str, value: object) -> frozenset[DetectionKind]:
    if not _is_kind_set(value):
        raise ModeLoadError(
            f"{label}: {what} must return a frozenset of DetectionKind, got {_safe_text(value)}"
        )
    return value  # type: ignore[return-value]


def _call(label: str, factory: ModeFactory, method: str, *args: object) -> object:
    """Call ``factory.<method>(*args)``.

    Only a ``ValueError`` raised by the call itself propagates as such; anything
    else, reading the method included, becomes a ModeLoadError.
    """
    try:
        bound = getattr(factory, method)
    except Exception as exc:
        raise ModeLoadError(f"{label}: reading {method} failed: {describe_error(exc)}") from exc
    try:
        return bound(*args)
    except ValueError:
        raise
    except Exception as exc:
        raise ModeLoadError(f"{label}: {method} failed: {describe_error(exc)}") from exc


def _check_context(
    label: str,
    name: str,
    specs: tuple[OptionSpec, ...],
    needs_background: bool,
    ctx: ModeContext,
) -> None:
    """Raise ValueError unless ``ctx`` is consistent for this factory (a caller bug)."""
    if ctx.config.mode != name:
        raise ValueError(
            f"{label}: the context's config selects the mode {ctx.config.mode!r}"
        )
    declared = {spec.name for spec in specs}
    if set(ctx.options) != declared:
        raise ValueError(
            f"{label}: the context options {_safe_text(sorted(ctx.options))} do not match "
            f"the declared options {_safe_text(sorted(declared))}"
        )
    for spec in specs:
        value = ctx.options[spec.name]
        canonical = spec.validate(value)
        if type(canonical) is not type(value) or canonical != value:
            raise ValueError(
                f"{label}: option {spec.name!r} is {_safe_text(value)}, not in its declared type"
            )
    expected = mode_options_from(ctx.options)
    actual = ctx.config.mode_options

    def typed(pairs: tuple[tuple[str, object], ...]) -> list[tuple[str, type, object]]:
        return [(key, type(value), value) for key, value in pairs]

    if typed(actual) != typed(expected):
        raise ValueError(
            f"{label}: the context's config.mode_options {_safe_text(actual)} do not match "
            f"its options {_safe_text(expected)}"
        )
    if needs_background and ctx.background is None:
        raise ValueError(f"{label} needs a background, and the context has none")
    if not needs_background and ctx.background is not None:
        raise ValueError(f"{label} does not use a background, and the context has one")


def build_mode(factory: ModeFactory, ctx: ModeContext, previous: Mode | None = None) -> Mode:
    """Build the mode with ``factory.create(ctx, previous)`` and check the result.

    The context must fit the factory, else ``ValueError`` (a caller bug):
    ``ctx.config.mode`` is the factory's name, ``ctx.options`` holds exactly the
    factory's options, each in its declared type, ``ctx.config.mode_options``
    holds the same values (``mode_options_from(ctx.options)``), and
    ``ctx.background`` is set exactly when the factory sets ``needs_background``.

    The mode must satisfy :class:`~petrify.core.plugin.Mode`, carry the factory's
    name, and have as ``requires`` and effective ``mandatory`` (missing means
    empty) the frozensets the factory reports for ``ctx.options``. A mode that is
    not built in must declare ``detect_policy`` explicitly, as ``"first_hit"`` or
    ``"all"``: left out, the engine would apply ``"first_hit"`` and the mode
    would receive incomplete regions.

    Errors: a ``ValueError`` raised by the factory's own ``requires``,
    ``mandatory`` or ``create`` call propagates as ``ValueError``; it means these
    settings are not acceptable, and the pipeline rejects the settings change and
    keeps running. Any other exception from the factory, and every broken
    contract on what it returned, raises :class:`ModeLoadError`: the plugin
    itself is faulty.
    """
    if not isinstance(ctx, ModeContext):
        raise ValueError(f"build_mode needs a ModeContext, got {_type_name(ctx)}")
    try:
        name = factory.name
        if type(name) is not str:
            raise TypeError(f"name must be a string, got {_safe_text(name)}")
        specs = check_option_specs(factory.options)
        needs_background = factory.needs_background
        if type(needs_background) is not bool:
            raise TypeError(
                f"needs_background must be True or False, got {_safe_text(needs_background)}"
            )
        builtin = _is_builtin_factory(factory)
    except Exception as exc:
        raise ModeLoadError(f"the factory is not usable: {describe_error(exc)}") from exc
    label = f"mode {_short(name)}"

    _check_context(label, name, specs, needs_background, ctx)

    expected_requires = _kinds(label, "requires", _call(label, factory, "requires", ctx.options))
    expected_mandatory = _kinds(
        label, "mandatory", _call(label, factory, "mandatory", ctx.options)
    )
    mode = _call(label, factory, "create", ctx, previous)

    try:
        _check_mode(label, name, mode, expected_requires, expected_mandatory, builtin)
    except ModeLoadError:
        raise
    except Exception as exc:
        raise ModeLoadError(
            f"{label}: checking the built mode failed: {describe_error(exc)}"
        ) from exc
    return mode  # type: ignore[return-value]


def _check_mode(
    label: str,
    name: str,
    mode: object,
    requires: frozenset[DetectionKind],
    mandatory: frozenset[DetectionKind],
    builtin: bool,
) -> None:
    if isinstance(mode, type) or not isinstance(mode, Mode):
        raise ModeLoadError(
            f"{label}: create returned a {_type_name(mode)}, which is not a mode instance "
            "(it needs name, requires and render)"
        )
    mode_name = mode.name
    if type(mode_name) is not str or mode_name != name:
        raise ModeLoadError(f"{label}: create returned a mode named {_safe_text(mode_name)}")
    actual_requires = mode.requires
    if not _is_kind_set(actual_requires):
        raise ModeLoadError(
            f"{label}: the mode's requires must be a frozenset of DetectionKind, "
            f"got {_safe_text(actual_requires)}"
        )
    if actual_requires != requires:
        raise ModeLoadError(
            f"{label}: the mode requires {_names(actual_requires)}, but the factory "
            f"reported {_names(requires)}"
        )
    actual_mandatory = getattr(mode, "mandatory", frozenset())
    if not _is_kind_set(actual_mandatory):
        raise ModeLoadError(
            f"{label}: the mode's mandatory must be a frozenset of DetectionKind, "
            f"got {_safe_text(actual_mandatory)}"
        )
    if actual_mandatory != mandatory:
        raise ModeLoadError(
            f"{label}: the mode's mandatory kinds are {_names(actual_mandatory)}, but the "
            f"factory reported {_names(mandatory)}"
        )
    if hasattr(mode, "detect_policy"):
        policy = mode.detect_policy
        if type(policy) is not str or policy not in _DETECT_POLICIES:
            raise ModeLoadError(
                f"{label}: detect_policy must be 'first_hit' or 'all', got {_safe_text(policy)}"
            )
        if policy == "first_hit":
            _check_first_hit(label, requires)
    elif not builtin:
        raise ModeLoadError(
            f"{label}: the mode must declare detect_policy ('first_hit' or 'all'); "
            "without it the engine stops at the first detector that fires and the mode "
            "would receive incomplete regions"
        )


def _check_first_hit(label: str, requires: frozenset[DetectionKind]) -> None:
    """Refuse a "first_hit" mode that needs faces, landmarks or a mask with another kind."""
    for group in _OWN_DETECTOR_KINDS:
        if requires & group and not requires <= group:
            raise ModeLoadError(
                f"{label}: requires {_names(requires)} with the 'first_hit' detection "
                f"policy, which skips the detector of {_names(requires & group)} on frames "
                "where another detector fired; the mode must declare "
                "detect_policy = \"all\""
            )


def _names(kinds: frozenset[DetectionKind]) -> str:
    """Return detection kinds as ``{a, b}`` for messages, or ``nothing`` when empty."""
    names = sorted(kind.value for kind in kinds)
    return "{" + ", ".join(names) + "}" if names else "nothing"
