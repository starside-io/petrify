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

"""The options of the shadow mode, its style groups, and the parsed parameters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from petrify.core.config import OptionValue
from petrify.core.plugin import OptionSpec, resolve_options

FACE_STYLES: frozenset[str] = frozenset({"gaussian", "pixelate", "noise", "facebox", "eyebar"})
"""Styles drawn on padded face boxes; they need the face detector."""

MASK_STYLES: frozenset[str] = frozenset({"silhouette", "silhouette_color", "outline"})
"""Styles drawn from the person segmentation mask; they need the segmenter."""

BODY_STYLES: frozenset[str] = frozenset({"silhouette", "silhouette_color", "bodybox"})
"""Styles that cover the whole body. ``outline`` is not one: it draws only an edge."""

STYLE_NAMES: tuple[str, ...] = (
    "gaussian",
    "pixelate",
    "noise",
    "facebox",
    "eyebar",
    "silhouette",
    "silhouette_color",
    "outline",
    "bodybox",
)
"""Every style, in the order the option lists them."""

APPLY_ORDER: tuple[str, ...] = (
    "silhouette",
    "silhouette_color",
    "bodybox",
    "outline",
    "gaussian",
    "pixelate",
    "noise",
    "facebox",
    "eyebar",
)
"""The fixed order styles are drawn in, whatever order they were given in.

Large opaque body fills come first, the outline is drawn on top of them, then the
face covers, and the eyebar last so it sits on top of any face cover. The orphan
black fills are not a style; the mode draws them after the body fills.
"""

COLORS: Mapping[str, tuple[int, int, int]] = MappingProxyType({
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
})
"""Color names to BGR, the channel order OpenCV images use."""

DEFAULT_PADDING = 0.3
DEFAULT_DILATE = 0.01

OPTIONS: tuple[OptionSpec, ...] = (
    OptionSpec(
        name="styles",
        kind="choices",
        default=("facebox",),
        help=(
            "Styles to apply, one or more. Face styles: gaussian, pixelate, noise, "
            "facebox, eyebar (need --faces). Mask styles: silhouette, silhouette_color, "
            "outline (need --segment). bodybox needs --detector person or both. Styles "
            "are always drawn in a fixed order, and outline is refused on its own."
        ),
        choices=STYLE_NAMES,
        runtime=True,
    ),
    OptionSpec(
        name="padding",
        kind="float",
        default=DEFAULT_PADDING,
        help=(
            "Margin added around each face box, as a share of the box size, so hair, "
            "ears and chin are covered too."
        ),
        minimum=0.0,
        maximum=1.0,
        runtime=True,
    ),
    OptionSpec(
        name="strength",
        kind="choice",
        default="high",
        help="Strength of the gaussian and pixelate styles; lower values are easier to reverse.",
        choices=("high", "medium", "low"),
        runtime=True,
    ),
    OptionSpec(
        name="color",
        kind="choice",
        default="black",
        help=(
            "Fill color of the facebox, eyebar, bodybox, silhouette_color and outline "
            "styles. The silhouette style is always black."
        ),
        choices=tuple(COLORS),
        runtime=True,
    ),
    OptionSpec(
        name="dilate",
        kind="float",
        default=DEFAULT_DILATE,
        help=(
            "How far the mask styles grow the person mask, as a share of the frame "
            "height, so the edge of the person is covered too."
        ),
        minimum=0.0,
        maximum=0.1,
        runtime=True,
    ),
)
"""The five options of the shadow mode; all may change while a run is live."""


@dataclass(frozen=True)
class ShadowParams:
    """The parsed options a shadow mode is built from.

    ``styles`` is a frozenset because the drawing order is fixed (:data:`APPLY_ORDER`):
    the web page sends styles in option order and the command line in typed
    order, and both must give equal parameters for the same selection.
    """

    styles: frozenset[str]
    padding: float
    strength: str
    color: tuple[int, int, int]
    dilate: float


def parse_params(options: Mapping[str, OptionValue]) -> ShadowParams:
    """Return the parameters for ``options``, missing options taking their defaults.

    Raises ``ValueError`` for an unknown option or an invalid value, for an empty
    style list, and for ``outline`` alone, which hides no one.
    """
    # Validated again here, not only by the registry, so a direct caller cannot
    # build a mode from values the option specs would refuse.
    resolved = resolve_options(OPTIONS, options)
    styles = frozenset(resolved["styles"])  # type: ignore[arg-type]
    if not styles:
        raise ValueError("the shadow mode needs at least one style in 'styles'")
    if styles == {"outline"}:
        raise ValueError(
            "the outline style does not hide anyone on its own: add another style to 'styles'"
        )
    # Adding 0.0 turns -0.0 into 0.0: the two compare equal but print differently,
    # and -0 would otherwise reach the weakened sentences and the page.
    return ShadowParams(
        styles=styles,
        padding=resolved["padding"] + 0.0,  # type: ignore[operator]
        strength=resolved["strength"],  # type: ignore[arg-type]
        color=COLORS[resolved["color"]],  # type: ignore[index]
        dilate=resolved["dilate"] + 0.0,  # type: ignore[operator]
    )
