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
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import EntryPoint
from importlib.metadata import entry_points as _installed_entry_points
from types import MappingProxyType

from petrify.core.config import _NAME_PATTERN
from petrify.core.detections import DetectionKind
from petrify.core.plugin import Mode, ModeContext, ModeFactory, check_option_specs

ENTRY_POINT_GROUP = "petrify.modes"

BUILTIN_MODES: Mapping[str, str] = MappingProxyType({"erase": "petrify.modes.erase:ERASE"})
"""Built-in mode names and their ``module:attr`` targets (read-only)."""

_DETECT_POLICIES: tuple[str, ...] = ("first_hit", "all")


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
        if self.version is None:
            return self.distribution
        return f"{self.distribution} {self.version}"


class ModeLoadError(RuntimeError):
    """A mode cannot be loaded or built; the message says why, for the user."""


def _split_target(target: str) -> tuple[str, str] | None:
    """Return ``(module, attr)`` for a ``module:attr`` target, or None when malformed."""
    module, sep, attr = target.partition(":")
    module = module.strip()
    attr = attr.strip()
    if not sep or not module or not attr:
        return None
    parts = module.split(".") + attr.split(".")
    if not all(part.isidentifier() for part in parts):
        return None
    return module, attr


def _distribution_of(entry_point: EntryPoint) -> tuple[str | None, str | None]:
    """Return the distribution name and version, from metadata only."""
    dist = getattr(entry_point, "dist", None)
    if dist is None:
        return None, None
    try:
        name = dist.name
    except Exception:
        name = None
    try:
        version = dist.version
    except Exception:
        version = None
    return name, version


def list_modes(entry_points: Iterable[EntryPoint] | None = None) -> tuple[ModeEntry, ...]:
    """Return every built-in and installed mode, selectable or not.

    ``entry_points`` defaults to the installed ``petrify.modes`` entry points;
    entries of any other group are ignored. Nothing is imported: only entry point
    names, values and their distribution's name and version are read.

    An entry point gets a ``problem`` (and can never be loaded) when its name does
    not match ``[a-z][a-z0-9_]{0,31}``, when its value is not ``module:attr``,
    when its name is a built-in mode's, or when another distribution (or the
    same distribution with another target) registers the same name, in which case
    every one of them is ambiguous. Order: built-ins first, then by name, then by
    distribution.
    """
    if entry_points is None:
        entry_points = _installed_entry_points(group=ENTRY_POINT_GROUP)

    builtins = tuple(
        ModeEntry(name, target, None, None, True, None) for name, target in BUILTIN_MODES.items()
    )

    candidates: list[ModeEntry] = []
    seen: set[tuple[str, str, str | None]] = set()
    for entry_point in entry_points:
        if entry_point.group != ENTRY_POINT_GROUP:
            continue
        name = entry_point.name
        target = entry_point.value
        distribution, version = _distribution_of(entry_point)
        key = (name, target, distribution)
        if key in seen:
            continue  # The same registration seen twice, for example a duplicate path.
        seen.add(key)
        problem: str | None = None
        if not _NAME_PATTERN.fullmatch(name):
            problem = f"invalid mode name {name!r}: it must match [a-z][a-z0-9_]{{0,31}}"
        elif _split_target(target) is None:
            problem = f"invalid entry point value {target!r}: it must be module:attr"
        elif name in BUILTIN_MODES:
            problem = f"shadowed by the built-in mode {name!r}; it can never be selected"
        candidates.append(ModeEntry(name, target, distribution, version, False, problem))

    by_name: dict[str, list[ModeEntry]] = {}
    for entry in candidates:
        by_name.setdefault(entry.name, []).append(entry)
    plugins: list[ModeEntry] = []
    for name, group in by_name.items():
        if len(group) > 1 and name not in BUILTIN_MODES:
            origins = ", ".join(sorted(entry.origin for entry in group))
            group = [
                entry
                if entry.problem is not None
                else ModeEntry(
                    entry.name,
                    entry.target,
                    entry.distribution,
                    entry.version,
                    False,
                    f"ambiguous: the mode name {name!r} is registered more than once "
                    f"({origins}); uninstall all but one",
                )
                for entry in group
            ]
        plugins.extend(group)
    plugins.sort(key=lambda e: (e.name, e.distribution or "", e.version or "", e.target))
    return builtins + tuple(plugins)


def _import_target(entry: ModeEntry) -> object:
    """Import ``entry.target`` and return the object it names, or raise ModeLoadError."""
    split = _split_target(entry.target)
    if split is None:
        raise ModeLoadError(
            f"mode {entry.name!r} ({entry.origin}) has an invalid target {entry.target!r}"
        )
    module_name, attr = split
    try:
        obj: object = importlib.import_module(module_name)
    except Exception as exc:
        raise ModeLoadError(
            f"mode {entry.name!r} ({entry.origin}) could not be imported from "
            f"{module_name!r}: {type(exc).__name__}: {exc}"
        ) from exc
    for part in attr.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise ModeLoadError(
                f"mode {entry.name!r} ({entry.origin}): {module_name!r} has no attribute "
                f"{attr!r}"
            ) from exc
        except Exception as exc:
            raise ModeLoadError(
                f"mode {entry.name!r} ({entry.origin}): reading {entry.target!r} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    return obj


def _check_factory(entry: ModeEntry, obj: object) -> ModeFactory:
    """Return ``obj`` when it is a valid factory for ``entry``, or raise ModeLoadError."""
    label = f"mode {entry.name!r} ({entry.origin})"
    if isinstance(obj, type):
        raise ModeLoadError(
            f"{label}: {entry.target!r} is a class; it must be a ModeFactory instance"
        )
    if not isinstance(obj, ModeFactory):
        raise ModeLoadError(
            f"{label}: {entry.target!r} is a {type(obj).__name__}, not a ModeFactory "
            "(it needs name, summary, options, needs_background, roi_overlay, requires, "
            "mandatory, create and weakened)"
        )
    name = obj.name
    if type(name) is not str or name != entry.name:
        raise ModeLoadError(
            f"{label}: the factory is named {name!r}, which does not match the "
            f"registered name {entry.name!r}"
        )
    if not isinstance(obj.summary, str):
        raise ModeLoadError(f"{label}: summary must be a string, got {obj.summary!r}")
    try:
        check_option_specs(obj.options)
    except ValueError as exc:
        raise ModeLoadError(f"{label}: invalid options: {exc}") from exc
    for flag in ("needs_background", "roi_overlay"):
        value = getattr(obj, flag)
        if type(value) is not bool:
            raise ModeLoadError(f"{label}: {flag} must be True or False, got {value!r}")
    for method in ("requires", "mandatory", "create", "weakened"):
        if not callable(getattr(obj, method)):
            raise ModeLoadError(f"{label}: {method} must be callable")
    return obj


def load_mode(name: str, entries: Sequence[ModeEntry] | None = None) -> ModeFactory:
    """Import and return the factory of the mode ``name``.

    ``entries`` defaults to :func:`list_modes`. Only the selected mode's target is
    imported. Raises :class:`ModeLoadError` with a reason for the user when the
    name is unknown or not selectable, when the import fails, and when the object
    is not a valid :class:`~petrify.core.plugin.ModeFactory` for that name. Any
    exception raised by the plugin's code while loading becomes a
    :class:`ModeLoadError`.
    """
    if entries is None:
        entries = list_modes()
    matching = [entry for entry in entries if entry.name == name]
    selectable = [entry for entry in matching if entry.problem is None]
    if not matching:
        names = sorted({entry.name for entry in entries if entry.problem is None})
        raise ModeLoadError(
            f"unknown mode {name!r}; available modes: {', '.join(names) if names else 'none'}"
        )
    if not selectable:
        reasons = "; ".join(f"{entry.origin}: {entry.problem}" for entry in matching)
        raise ModeLoadError(f"mode {name!r} cannot be selected: {reasons}")
    if len(selectable) > 1:
        origins = ", ".join(entry.origin for entry in selectable)
        raise ModeLoadError(f"mode {name!r} is registered more than once ({origins})")
    entry = selectable[0]
    obj = _import_target(entry)
    try:
        return _check_factory(entry, obj)
    except ModeLoadError:
        raise
    except Exception as exc:
        raise ModeLoadError(
            f"mode {entry.name!r} ({entry.origin}): checking the factory failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _is_builtin_factory(factory: ModeFactory) -> bool:
    """True when ``factory`` is the object a built-in mode's target names."""
    target = BUILTIN_MODES.get(factory.name)
    if target is None:
        return False
    module_name, attr = target.split(":")
    return getattr(importlib.import_module(module_name), attr) is factory


def _kinds(label: str, what: str, value: object) -> frozenset[DetectionKind]:
    if not isinstance(value, frozenset) or not all(
        isinstance(kind, DetectionKind) for kind in value
    ):
        raise ModeLoadError(f"{label}: {what} must return a frozenset of DetectionKind, got {value!r}")
    return value


def build_mode(factory: ModeFactory, ctx: ModeContext, previous: Mode | None = None) -> Mode:
    """Build the mode with ``factory.create(ctx, previous)`` and check the result.

    ``ctx.options`` must hold exactly the factory's options, each in its declared
    type, and ``ctx.background`` must be set exactly when the factory sets
    ``needs_background``; otherwise ``ValueError``.

    The mode must satisfy :class:`~petrify.core.plugin.Mode`, carry the factory's
    name, and have the ``requires`` and effective ``mandatory`` (missing means
    empty) the factory reports for ``ctx.options``. A mode that is not built in
    must declare ``detect_policy`` explicitly, as ``"first_hit"`` or ``"all"``:
    left out, the engine would apply ``"first_hit"`` and the mode would receive
    incomplete regions.

    Errors: a ``ValueError`` raised by the factory's ``requires``, ``mandatory``
    or ``create`` propagates as ``ValueError``; it means these settings are not
    acceptable, and the pipeline rejects the settings change and keeps running.
    Any other exception from the factory, and every broken contract above, raises
    :class:`ModeLoadError`: the plugin itself is faulty.
    """
    if not isinstance(ctx, ModeContext):
        raise ValueError(f"build_mode needs a ModeContext, got {type(ctx).__name__}")
    try:
        name = factory.name
        specs = check_option_specs(factory.options)
        needs_background = factory.needs_background
        builtin = _is_builtin_factory(factory)
    except Exception as exc:
        raise ModeLoadError(
            f"the factory is not usable: {type(exc).__name__}: {exc}"
        ) from exc
    label = f"mode {name!r}"

    declared = {spec.name for spec in specs}
    if set(ctx.options) != declared:
        raise ValueError(
            f"{label}: the context options {sorted(ctx.options)} do not match the declared "
            f"options {sorted(declared)}"
        )
    for spec in specs:
        value = ctx.options[spec.name]
        canonical = spec.validate(value)
        if type(canonical) is not type(value) or canonical != value:
            raise ValueError(
                f"{label}: option {spec.name!r} is {value!r}, not in its declared type"
            )
    if needs_background and ctx.background is None:
        raise ValueError(f"{label} needs a background, and the context has none")
    if not needs_background and ctx.background is not None:
        raise ValueError(f"{label} does not use a background, and the context has one")

    try:
        expected_requires = _kinds(label, "requires", factory.requires(ctx.options))
        expected_mandatory = _kinds(label, "mandatory", factory.mandatory(ctx.options))
        mode = factory.create(ctx, previous)
    except (ValueError, ModeLoadError):
        raise
    except Exception as exc:
        raise ModeLoadError(
            f"{label}: building the mode failed: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        _check_mode(label, name, mode, expected_requires, expected_mandatory, builtin)
    except ModeLoadError:
        raise
    except Exception as exc:
        raise ModeLoadError(
            f"{label}: checking the built mode failed: {type(exc).__name__}: {exc}"
        ) from exc
    return mode


def _check_mode(
    label: str,
    name: str,
    mode: object,
    requires: frozenset[DetectionKind],
    mandatory: frozenset[DetectionKind],
    builtin: bool,
) -> None:
    if not isinstance(mode, Mode) or isinstance(mode, type):
        raise ModeLoadError(
            f"{label}: create returned a {type(mode).__name__}, which is not a mode "
            "(it needs name, requires and render)"
        )
    if mode.name != name:
        raise ModeLoadError(f"{label}: create returned a mode named {mode.name!r}")
    if mode.requires != requires:
        raise ModeLoadError(
            f"{label}: the mode requires {_names(mode.requires)}, but the factory "
            f"reported {_names(requires)}"
        )
    actual_mandatory = getattr(mode, "mandatory", frozenset())
    if actual_mandatory != mandatory:
        raise ModeLoadError(
            f"{label}: the mode's mandatory kinds are {_names(actual_mandatory)}, but the "
            f"factory reported {_names(mandatory)}"
        )
    if hasattr(mode, "detect_policy"):
        policy = mode.detect_policy
        if not isinstance(policy, str) or policy not in _DETECT_POLICIES:
            raise ModeLoadError(
                f"{label}: detect_policy must be 'first_hit' or 'all', got {policy!r}"
            )
    elif not builtin:
        raise ModeLoadError(
            f"{label}: the mode must declare detect_policy ('first_hit' or 'all'); "
            "without it the engine stops at the first detector that fires and the mode "
            "would receive incomplete regions"
        )


def _names(kinds: object) -> str:
    """Return detection kinds as ``{a, b}`` for messages, or ``nothing`` when empty."""
    try:
        names = sorted(
            kind.value if isinstance(kind, DetectionKind) else repr(kind)
            for kind in kinds  # type: ignore[attr-defined]
        )
    except Exception:
        return repr(kinds)
    return "{" + ", ".join(names) + "}" if names else "nothing"
