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

"""The shadow mode factory, the object the ``petrify.modes`` entry point names."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping

from petrify.core.config import OptionValue
from petrify.core.detections import DetectionKind
from petrify.core.plugin import Mode, ModeContext, OptionSpec
from petrify_shadow.mode import ShadowMode, mandatory_for, requires_for
from petrify_shadow.options import (
    BODY_STYLES,
    DEFAULT_DILATE,
    DEFAULT_PADDING,
    FACE_STYLES,
    MASK_STYLES,
    OPTIONS,
    STYLE_NAMES,
    parse_params,
)

logger = logging.getLogger(__name__)

BODYBOX_NEEDS_PERSONS = (
    "the bodybox style needs person boxes: use --detector person or both, with "
    "--person-detector dnn for the recommended detector"
)

WEAK_NO_BODY = (
    "Only faces are covered: bodies, clothing and posture stay visible, and a face the "
    "face detector misses is shown when another face is found on the same frame, "
    "unless the person detector finds that person."
)
# Worded to hold with and without a body style: a person the person detector or
# the segmenter misses keeps only the face cover.
WEAK_REVERSIBLE = (
    "The gaussian and pixelate styles can be partially reversed by machine learning "
    "models, and a face no body style covers is protected only by them; add noise or "
    "facebox for a cover that cannot be undone."
)
WEAK_STRENGTH = (
    "strength is {v}: the blur or pixelation is weaker than the default and easier "
    "to reverse."
)
WEAK_EYEBAR_ONLY = (
    "The eyebar style covers only the eyes: a face no body style covers stays visible "
    "apart from the eyes and can be recognized."
)
WEAK_PADDING = (
    "padding is {v}, below the default of 0.3: hair, ears and chin around the face "
    "box can be shown."
)
WEAK_DILATE = (
    "dilate is {v}, below the default of 0.01: the edge of the person can show "
    "around the silhouette."
)
WEAK_SEGMENT_MISS = (
    "The body of a person the segmenter misses is shown when another person is "
    "segmented on the same frame, unless the person detector finds that person; add "
    "bodybox to cover every person box."
)

WARN_MASK_WITHOUT_PERSONS = (
    "shadow: the mask styles run without person boxes (--detector motion). A person the "
    "segmenter misses is shown whenever the segmenter marks anything else on the frame, "
    "and the orphan rule that would cover that person needs person boxes. Use --detector "
    "person or both, with --person-detector dnn."
)
"""Logged once per newly built mode with a mask style, no bodybox, and --detector motion."""

# The styles that replace every pixel of the padded face box, which makes the
# blur strength irrelevant and the face cover irreversible.
_OPAQUE_FACE_STYLES = frozenset({"noise", "facebox"})
# The mask styles that cover the body; outline draws only an edge.
_SILHOUETTE_STYLES = frozenset({"silhouette", "silhouette_color"})

class ShadowFactory:
    """The :class:`~petrify.core.plugin.ModeFactory` of the shadow mode (see :data:`SHADOW`)."""

    name = "shadow"
    summary = "Obscure faces and bodies while a subject is present."
    options: tuple[OptionSpec, ...] = OPTIONS
    needs_background = False
    # A region mode: the live ROI overlay would paste a face inside the timestamp
    # region back over its cover.
    roi_overlay = False

    def requires(self, options: Mapping[str, OptionValue]) -> frozenset[DetectionKind]:
        """Return motion and persons plus the kinds the selected styles need.

        Raises ``ValueError`` for options :func:`parse_params` refuses, so an
        invalid style list is reported as such and not as a missing detector.
        """
        return requires_for(parse_params(options).styles)

    def mandatory(self, options: Mapping[str, OptionValue]) -> frozenset[DetectionKind]:
        """Return the kinds the selected styles cannot work without; raises ``ValueError``."""
        return mandatory_for(parse_params(options).styles)

    def weakened(self, options: Mapping[str, OptionValue]) -> tuple[str, ...]:
        """Return one sentence per selected setting that reduces protection.

        Never raises, also for invalid options: each value is read on its own and
        a value that cannot be read counts as its default, so the other settings
        are still reported.
        """
        styles = _read_styles(options)
        padding = _read_number(options, "padding", DEFAULT_PADDING)
        dilate = _read_number(options, "dilate", DEFAULT_DILATE)
        strength = _read_strength(options)
        faces = styles & FACE_STYLES
        has_body = bool(styles & BODY_STYLES)
        weak_blur = bool(styles & {"gaussian", "pixelate"}) and not styles & _OPAQUE_FACE_STYLES
        silhouette = bool(styles & _SILHOUETTE_STYLES)
        sentences: list[str] = []
        if not has_body:
            sentences.append(WEAK_NO_BODY)
        # A body style does not lift these two: a person it misses while another is
        # found keeps only the face cover.
        if weak_blur:
            sentences.append(WEAK_REVERSIBLE)
        if weak_blur and strength != "high":
            sentences.append(WEAK_STRENGTH.format(v=strength))
        if faces == {"eyebar"}:
            sentences.append(WEAK_EYEBAR_ONLY)
        if faces and padding < DEFAULT_PADDING:
            sentences.append(WEAK_PADDING.format(v=f"{padding:g}"))
        if silhouette and dilate < DEFAULT_DILATE:
            sentences.append(WEAK_DILATE.format(v=f"{dilate:g}"))
        # With bodybox every person box is filled; without it the silhouette is the
        # body cover and only the orphan rule catches a person the segmenter misses.
        if silhouette and "bodybox" not in styles:
            sentences.append(WEAK_SEGMENT_MISS)
        return tuple(sentences)

    def create(self, ctx: ModeContext, previous: Mode | None = None) -> Mode:
        """Return a :class:`ShadowMode` for ``ctx.options``, or ``previous`` if equal.

        Raises ``ValueError`` for invalid options, and for bodybox with
        ``--detector motion``: that check comes before the engine's terser
        mandatory error and also rejects a live switch to the motion detector.

        A new mode that replaces a previous :class:`ShadowMode` takes over its
        held-box memory. Each newly built mode with a mask style, no bodybox and
        ``--detector motion`` logs :data:`WARN_MASK_WITHOUT_PERSONS` once.
        """
        params = parse_params(ctx.options)
        if "bodybox" in params.styles and ctx.config.detector == "motion":
            raise ValueError(BODYBOX_NEEDS_PERSONS)
        # Equal params mean an equal mode; keeping the instance also keeps its
        # noise generator running instead of starting a new one.
        if isinstance(previous, ShadowMode) and previous.params == params:
            return previous
        if (
            params.styles & MASK_STYLES
            and "bodybox" not in params.styles
            and ctx.config.detector == "motion"
        ):
            logger.warning(WARN_MASK_WITHOUT_PERSONS)
        # The held-box memory carries over, so a held cover does not shrink on
        # the frame the options change.
        held_from = previous if isinstance(previous, ShadowMode) else None
        return ShadowMode(params, held_from=held_from)


_UNREADABLE = object()


def _get(options: object, name: str) -> object:
    """Return ``options[name]``, or a marker when it is missing or reading it fails.

    ``weakened`` must never raise, and a hostile mapping can raise from ``get``.
    """
    try:
        if isinstance(options, Mapping):
            return options.get(name, _UNREADABLE)
    except Exception:
        pass
    return _UNREADABLE


def _read_styles(options: object) -> frozenset[str]:
    """Return the known style names in ``options["styles"]``; the default when unreadable."""
    value = _get(options, "styles")
    if not isinstance(value, (list, tuple, set, frozenset)):
        return frozenset({"facebox"})
    try:
        return frozenset(item for item in value if type(item) is str and item in STYLE_NAMES)
    except Exception:
        return frozenset({"facebox"})


def _read_number(options: object, name: str, default: float) -> float:
    """Return ``options[name]`` as a float when it is a finite number, else ``default``."""
    value = _get(options, name)
    if type(value) not in (int, float):
        return default
    try:
        number = float(value)  # type: ignore[arg-type]
    except OverflowError:
        return default
    if not math.isfinite(number):
        return default
    return number + 0.0  # Turns -0.0 into 0.0, so it is never printed as "-0".


def _read_strength(options: object) -> str:
    """Return ``options["strength"]`` when it is a known strength, else ``"high"``."""
    value = _get(options, "strength")
    if type(value) is str and value in ("high", "medium", "low"):
        return value
    return "high"


SHADOW = ShadowFactory()
"""The shadow mode factory, registered as ``shadow``."""
