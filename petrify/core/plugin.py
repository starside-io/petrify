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

"""Protocols for the pluggable engine components."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from petrify.core.config import _NAME_PATTERN, OptionValue, PetrifyConfig
from petrify.core.detections import DetectionKind, Detections
from petrify.core.frame import Frame, FrameMeta, SourceInfo

DetectPolicy = Literal["first_hit", "all"]
"""How the engine runs a mode's stateless detectors on each frame (see :class:`Mode`)."""


class BackgroundNotReady(RuntimeError):
    """Raised when a background model has no image yet; the engine fails the frame closed."""


@runtime_checkable
class BackgroundModel(Protocol):
    """An online estimate of the clean background.

    ``current`` returns an array the caller must not mutate. The model never mutates
    an array it has already returned: it replaces it and increases ``version``.
    """

    # Increases every time current() would return a different image.
    version: int

    def current(self) -> np.ndarray | None:
        """Return the current background image, or None when not ready yet."""
        ...

    def update(self, frame: Frame, subject_present: bool) -> None:
        """Feed one released frame and the engine's final decision for it."""
        ...


@runtime_checkable
class Detector(Protocol):
    """Produces part of a frame's detections."""

    provides: frozenset[DetectionKind]
    # True when the detector must see every frame in order to keep its internal
    # model correct (for example MOG2).
    stateful: bool

    def detect(self, frame: Frame) -> Detections:
        """Return detections with only the fields in ``provides`` set."""
        ...

    def reset(self) -> None:
        """Drop any internal state."""
        ...


@runtime_checkable
class Refiner(Protocol):
    """Derives further detections from a frame and the detections found so far.

    A refiner (for example a tracker that holds a box for a few frames after a
    miss) runs after every active detector, in the order given to the engine, and
    receives the union of the detectors' results and of earlier refiners' results.
    It is active when ``provides`` intersects the mode's ``requires``. Every kind
    in ``consumes`` must be provided by an active detector or an earlier active
    refiner, and refiners require the "all" detection policy, so they see the
    output of every detector on every frame that did not fail.
    """

    consumes: frozenset[DetectionKind]
    provides: frozenset[DetectionKind]

    def refine(self, frame: Frame, detections: Detections) -> Detections:
        """Return ADDITIONAL detections for this frame; the engine unions them in."""
        ...

    def reset(self) -> None:
        """Drop any internal state."""
        ...


@runtime_checkable
class Mode(Protocol):
    """Turns a frame and its detections into the output image.

    A mode may also carry two optional attributes. They are not protocol members,
    so a mode without them still satisfies ``isinstance(obj, Mode)``:

    - ``detect_policy`` (:data:`DetectPolicy`, default ``"first_hit"``): with
      ``"first_hit"`` the engine stops running stateless detectors on a frame once
      one of them has found a subject (stateful detectors still run on every
      frame). That suits a mode that only needs a yes or no. With ``"all"`` every
      active detector runs on every frame, so a mode that draws on boxes, faces,
      or a mask receives every region. Refiners require ``"all"``.
    - ``mandatory`` (``frozenset[DetectionKind]``, default empty): every kind in
      it must be provided by an active detector or refiner, or the engine raises
      ``ValueError`` at construction and in ``reconfigure``. A non-empty
      ``mandatory`` requires ``detect_policy = "all"``, since ``"first_hit"`` can
      skip the detectors that provide it.

    Output rule: :meth:`render` must return a ``numpy.ndarray`` with the input
    frame image's shape and dtype. When ``subject_present`` is True the returned
    array must not share memory with ``frame.image`` (the engine checks with
    ``numpy.may_share_memory``, so even a view of a disjoint slice is rejected).
    Otherwise the frame fails closed. When ``subject_present`` is False the mode
    may return ``frame.image`` itself.
    """

    name: str
    requires: frozenset[DetectionKind]

    def render(self, frame: Frame, detections: Detections, subject_present: bool) -> np.ndarray:
        """Return the output image.

        ``subject_present`` is the engine's decision after lookahead smoothing and
        cold start, not the raw detection.
        """
        ...


@runtime_checkable
class Stage(Protocol):
    """Post-processing step run after the mode on every frame.

    Output rule: :meth:`apply` must return a ``numpy.ndarray`` with the input
    frame image's shape and dtype, and on a frame whose final decision is subject
    present the result must not share memory with ``frame.image`` (checked with
    ``numpy.may_share_memory``); otherwise the frame fails closed. Returning
    ``image`` unchanged is always allowed when the mode's image satisfied the rule.
    """

    def apply(self, frame: Frame, image: np.ndarray) -> np.ndarray:
        """Return the processed image."""
        ...


@runtime_checkable
class Sink(Protocol):
    """Consumes output images.

    A sink only ever receives the processed output image and the frame's metadata,
    never the input frame. ``abort`` must be safe after a failed ``open``.
    """

    # True when a failure of this sink must abort the whole run (for example a file
    # recording). A non-critical sink (for example a preview) that fails is aborted
    # and dropped while the run continues.
    critical: bool

    def open(self, info: SourceInfo) -> None: ...

    def write(self, image: np.ndarray, meta: FrameMeta) -> None: ...

    def close(self) -> None: ...

    def abort(self) -> None:
        """Discard any partial output; called instead of close when a run fails."""
        ...


@runtime_checkable
class Source(Protocol):
    """Yields frames in order."""

    info: SourceInfo

    def __iter__(self) -> Iterator[Frame]: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Mode construction contract
# ---------------------------------------------------------------------------

OptionKind = Literal["bool", "int", "float", "choice", "choices"]
"""The value kinds a mode option may have (see :class:`OptionSpec`)."""

_OPTION_KINDS: tuple[str, ...] = ("bool", "int", "float", "choice", "choices")
_TRUE_TEXT = frozenset({"true", "yes", "on", "1"})
_FALSE_TEXT = frozenset({"false", "no", "off", "0"})
_INT_TEXT = re.compile(r"[+-]?[0-9]+")


def _is_real_number(value: object) -> bool:
    """True for an int or a finite float, never a bool."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True  # math.isfinite would overflow on a huge int.
    return isinstance(value, float) and math.isfinite(value)


@dataclass(frozen=True)
class OptionSpec:
    """One option a mode accepts, with its kind, default and bounds.

    ``kind`` decides the single Python type every value of the option has after
    :meth:`validate` or :meth:`parse`: ``bool`` for ``"bool"``, ``int`` for
    ``"int"``, ``float`` for ``"float"``, ``str`` for ``"choice"`` (one of
    ``choices``) and ``tuple[str, ...]`` for ``"choices"`` (distinct entries of
    ``choices``, in the order given, possibly empty). The single type matters
    because ``1``, ``True`` and ``1.0`` compare equal, and the pipeline detects a
    settings change with ``!=``: a value must reach ``PetrifyConfig.mode_options``
    already in its declared type.

    ``choices`` names must match the option name pattern, so a list of them can be
    written as comma-separated text. ``minimum`` and ``maximum`` (inclusive) apply
    only to ``"int"`` and ``"float"``. ``runtime`` marks an option that may be
    changed while a run is live.

    The spec checks itself on construction and raises ``ValueError``; a default
    given in another accepted form (for example ``1`` for a float option or a list
    for a ``"choices"`` option) is stored in the declared type.
    """

    name: str
    kind: OptionKind
    default: OptionValue
    help: str
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    runtime: bool = False

    def __post_init__(self) -> None:
        if type(self.name) is not str or not _NAME_PATTERN.fullmatch(self.name):
            raise ValueError(
                f"option name must match [a-z][a-z0-9_]{{0,31}}, got {self.name!r}"
            )
        label = f"option {self.name!r}"
        if type(self.kind) is not str or self.kind not in _OPTION_KINDS:
            raise ValueError(
                f"{label}: kind must be one of {', '.join(_OPTION_KINDS)}, got {self.kind!r}"
            )
        if not isinstance(self.help, str):
            raise ValueError(f"{label}: help must be a string, got {type(self.help).__name__}")
        if type(self.runtime) is not bool:
            raise ValueError(f"{label}: runtime must be True or False, got {self.runtime!r}")
        if not isinstance(self.choices, tuple):
            raise ValueError(
                f"{label}: choices must be a tuple, got {type(self.choices).__name__}"
            )
        if self.kind in ("choice", "choices"):
            if not self.choices:
                raise ValueError(f"{label}: a {self.kind!r} option needs at least one choice")
            for choice in self.choices:
                if type(choice) is not str or not _NAME_PATTERN.fullmatch(choice):
                    raise ValueError(
                        f"{label}: choice names must match [a-z][a-z0-9_]{{0,31}}, "
                        f"got {choice!r}"
                    )
            if len(set(self.choices)) != len(self.choices):
                raise ValueError(f"{label}: choices must be distinct, got {self.choices!r}")
        elif self.choices:
            raise ValueError(f"{label}: only 'choice' and 'choices' options take choices")
        for bound_name in ("minimum", "maximum"):
            bound = getattr(self, bound_name)
            if bound is None:
                continue
            if self.kind not in ("int", "float"):
                raise ValueError(f"{label}: only 'int' and 'float' options take a {bound_name}")
            if not _is_real_number(bound):
                raise ValueError(f"{label}: {bound_name} must be a finite number, got {bound!r}")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError(
                f"{label}: minimum {self.minimum} is greater than maximum {self.maximum}"
            )
        try:
            default = self.validate(self.default)
        except ValueError as exc:
            raise ValueError(f"{label}: invalid default: {exc}") from exc
        object.__setattr__(self, "default", default)

    def validate(self, value: object) -> OptionValue:
        """Return ``value`` (from JSON or Python) in the option's declared type.

        Raises ``ValueError`` when the value has the wrong type or is out of range.
        A bool option takes only ``True`` or ``False``. An int option takes an int
        or a float with no fractional part, never a bool. A float option takes an
        int or a float, never a bool, NaN or infinity. A choices option takes a
        list or tuple.
        """
        label = f"option {self.name!r}"
        kind = self.kind
        if kind == "bool":
            if type(value) is not bool:
                raise ValueError(f"{label} must be true or false, got {value!r}")
            return value
        if kind in ("int", "float"):
            if not _is_real_number(value):
                raise ValueError(f"{label} must be a finite number, got {value!r}")
            number: int | float
            if kind == "int":
                if isinstance(value, float) and not value.is_integer():
                    raise ValueError(f"{label} must be a whole number, got {value!r}")
                number = int(value)  # type: ignore[call-overload]
            else:
                try:
                    number = float(value)  # type: ignore[arg-type]
                except OverflowError:
                    raise ValueError(f"{label} is too large, got {value!r}") from None
            if self.minimum is not None and number < self.minimum:
                raise ValueError(f"{label} must be at least {self.minimum}, got {number!r}")
            if self.maximum is not None and number > self.maximum:
                raise ValueError(f"{label} must be at most {self.maximum}, got {number!r}")
            return number
        if kind == "choice":
            return self._choice(label, value)
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{label} must be a list of choices, got {value!r}")
        picked = tuple(self._choice(label, item) for item in value)
        if len(set(picked)) != len(picked):
            raise ValueError(f"{label} must not repeat a choice, got {list(picked)!r}")
        return picked

    def parse(self, text: str) -> OptionValue:
        """Return the value written as command line ``text``, in the declared type.

        Accepted forms: ``true``, ``false``, ``yes``, ``no``, ``on``, ``off``,
        ``1`` or ``0`` for a bool (any case); a whole number such as ``3`` for an
        int; a number such as ``0.5`` for a float; a choice name for a choice; and
        comma-separated choice names such as ``a,b`` for choices (empty text is
        the empty list). Surrounding spaces are ignored. Raises ``ValueError``.
        """
        label = f"option {self.name!r}"
        if not isinstance(text, str):
            raise ValueError(f"{label}: expected text, got {type(text).__name__}")
        stripped = text.strip()
        kind = self.kind
        if kind == "bool":
            lowered = stripped.lower()
            if lowered in _TRUE_TEXT:
                return True
            if lowered in _FALSE_TEXT:
                return False
            raise ValueError(f"{label} must be true or false, got {text!r}")
        if kind == "int":
            if not _INT_TEXT.fullmatch(stripped):
                raise ValueError(f"{label} must be a whole number, got {text!r}")
            return self.validate(int(stripped))
        if kind == "float":
            try:
                number = float(stripped)
            except ValueError:
                raise ValueError(f"{label} must be a number, got {text!r}") from None
            return self.validate(number)
        if kind == "choice":
            return self.validate(stripped)
        if not stripped:
            return ()
        return self.validate([item.strip() for item in stripped.split(",")])

    def _choice(self, label: str, value: object) -> str:
        if isinstance(value, str):
            for choice in self.choices:
                if value == choice:
                    return choice  # The spec's own plain str, never a subclass.
        raise ValueError(f"{label} must be one of {', '.join(self.choices)}, got {value!r}")


def check_option_specs(specs: object) -> tuple[OptionSpec, ...]:
    """Return ``specs`` when it is a tuple of :class:`OptionSpec` with distinct names.

    Raises ``ValueError`` otherwise.
    """
    if not isinstance(specs, tuple):
        raise ValueError(f"options must be a tuple of OptionSpec, got {type(specs).__name__}")
    seen: set[str] = set()
    for spec in specs:
        if not isinstance(spec, OptionSpec):
            raise ValueError(f"options must hold only OptionSpec, got {type(spec).__name__}")
        if spec.name in seen:
            raise ValueError(f"options declare {spec.name!r} more than once")
        seen.add(spec.name)
    return specs


def resolve_options(
    specs: tuple[OptionSpec, ...], given: Mapping[str, object]
) -> dict[str, OptionValue]:
    """Return every option of ``specs``: the validated ``given`` value, else the default.

    Raises ``ValueError`` for a name no spec declares and for an invalid value.
    The result holds each value in its declared type, so it can go straight into
    :func:`petrify.core.config.mode_options_from`.
    """
    check_option_specs(specs)
    if not isinstance(given, Mapping):
        raise ValueError(f"mode options must be a mapping, got {type(given).__name__}")
    by_name = {spec.name: spec for spec in specs}
    unknown = sorted(repr(name) for name in given if name not in by_name)
    if unknown:
        known = ", ".join(by_name) if by_name else "none"
        raise ValueError(f"unknown mode option {', '.join(unknown)}; known options: {known}")
    return {
        spec.name: spec.validate(given[spec.name]) if spec.name in given else spec.default
        for spec in specs
    }


def resolve_option_text(
    specs: tuple[OptionSpec, ...], assignments: Iterable[str]
) -> dict[str, OptionValue]:
    """Like :func:`resolve_options`, from command line ``KEY=VALUE`` texts.

    Each value is read with :meth:`OptionSpec.parse`. Raises ``ValueError`` for a
    text without ``=``, a name given twice, an unknown name, or an invalid value.
    """
    check_option_specs(specs)
    by_name = {spec.name: spec for spec in specs}
    given: dict[str, object] = {}
    for assignment in assignments:
        if not isinstance(assignment, str) or "=" not in assignment:
            raise ValueError(f"mode options must be written KEY=VALUE, got {assignment!r}")
        key, _, text = assignment.partition("=")
        key = key.strip()
        if key in given:
            raise ValueError(f"mode option {key!r} is given more than once")
        spec = by_name.get(key)
        if spec is None:
            known = ", ".join(by_name) if by_name else "none"
            raise ValueError(f"unknown mode option {key!r}; known options: {known}")
        given[key] = spec.parse(text)
    return resolve_options(specs, given)


@dataclass(frozen=True, eq=False)
class ModeContext:
    """What a :class:`ModeFactory` receives to build its mode.

    ``options`` holds every option the factory declares, defaults filled in and
    validated (see :func:`resolve_options`); it is stored as a read-only mapping.
    ``background`` is the locked background image or the online background model,
    provided only when the factory sets ``needs_background``, None otherwise.
    """

    source: SourceInfo
    config: PetrifyConfig
    options: Mapping[str, OptionValue]
    background: np.ndarray | BackgroundModel | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceInfo):
            raise TypeError(
                f"ModeContext.source must be a SourceInfo, got {type(self.source).__name__}"
            )
        if not isinstance(self.config, PetrifyConfig):
            raise TypeError(
                f"ModeContext.config must be a PetrifyConfig, got {type(self.config).__name__}"
            )
        if not isinstance(self.options, Mapping):
            raise TypeError(
                f"ModeContext.options must be a mapping, got {type(self.options).__name__}"
            )
        background = self.background
        if background is not None and not isinstance(background, (np.ndarray, BackgroundModel)):
            raise TypeError(
                "ModeContext.background must be None, an image or a BackgroundModel, "
                f"got {type(background).__name__}"
            )
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


@runtime_checkable
class ModeFactory(Protocol):
    """Describes a mode and builds it; what a mode plugin registers.

    A factory is a plain object (an instance, not a class) published under the
    ``petrify.modes`` entry point group. ``name`` must match the entry point name
    and the mode's own ``name``. ``options`` declares every option the mode
    accepts, with distinct names.

    The methods taking ``options`` receive the resolved options (every declared
    option, validated, defaults filled in) and must not keep or mutate them.
    """

    name: str
    # One line describing the mode, for listings.
    summary: str
    options: tuple[OptionSpec, ...]
    # True when the mode needs the clean background (the locked image or the online
    # model); the pipeline then builds it and passes it in ModeContext.background.
    needs_background: bool
    # Whether the live timestamp ROI stage (petrify.stages.roi_overlay) may run
    # after this mode. That stage pastes the LIVE pixels of the timestamp region
    # over the mode's output. That is right for erase, which shows a frozen clean
    # background with a live clock, and wrong for a mode that obscures regions
    # (blur, pixelate, silhouette): a face inside the ROI would be pasted back
    # un-obscured. Such a mode must set it to False.
    roi_overlay: bool

    def requires(self, options: Mapping[str, OptionValue]) -> frozenset[DetectionKind]:
        """Return the ``requires`` the mode built with ``options`` will have."""
        ...

    def mandatory(self, options: Mapping[str, OptionValue]) -> frozenset[DetectionKind]:
        """Return the effective ``mandatory`` of the mode built with ``options``."""
        ...

    def create(self, ctx: ModeContext, previous: Mode | None = None) -> Mode:
        """Return the mode for ``ctx``.

        ``previous`` is the mode instance currently in use, or None. When nothing
        the mode depends on changed, return ``previous`` itself, so a settings
        change that does not concern the mode keeps the same object. Raise
        ``ValueError`` for options or a context the mode cannot work with; the
        pipeline then rejects the settings change and keeps running.
        """
        ...

    def weakened(self, options: Mapping[str, OptionValue]) -> tuple[str, ...]:
        """Return one human-readable sentence per option value that reduces protection.

        Shown in the web page's list of weakened protections; empty when none.
        """
        ...
