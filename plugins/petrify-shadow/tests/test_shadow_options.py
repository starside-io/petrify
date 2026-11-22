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

"""Options, parameters, requires, mandatory, weakened and create of the shadow mode."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import pytest
from conftest import ShadowEntryPoint
from petrify_shadow import SHADOW
from petrify_shadow.factory import (
    BODYBOX_NEEDS_PERSONS,
    WEAK_DILATE,
    WEAK_EYEBAR_ONLY,
    WEAK_NO_BODY,
    WEAK_PADDING,
    WEAK_REVERSIBLE,
    WEAK_SEGMENT_MISS,
    WEAK_STRENGTH,
)
from petrify_shadow.mode import ShadowMode
from petrify_shadow.options import (
    APPLY_ORDER,
    BODY_STYLES,
    COLORS,
    FACE_STYLES,
    MASK_STYLES,
    OPTIONS,
    STYLE_NAMES,
    ShadowParams,
    parse_params,
)

from petrify.core import PetrifyConfig
from petrify.core.config import mode_options_from
from petrify.core.detections import DetectionKind
from petrify.core.frame import SourceInfo
from petrify.core.plugin import (
    ModeContext,
    check_option_specs,
    resolve_option_text,
    resolve_options,
)
from petrify.core.registry import build_mode, load_mode

MOTION = DetectionKind.MOTION
PERSONS = DetectionKind.PERSONS
FACES = DetectionKind.FACES
LANDMARKS = DetectionKind.LANDMARKS
MASK = DetectionKind.MASK

INFO = SourceInfo(fps=10.0, width=64, height=48, frame_count=None, seekable=False,
                  live=True, uri="test")


def _opts(**given: Any) -> dict[str, Any]:
    """Resolved options, as the registry hands them to the factory."""
    if "styles" in given and isinstance(given["styles"], str):
        given["styles"] = tuple(given["styles"].split(","))
    return resolve_options(OPTIONS, given)


def _ctx(detector: str = "person", **given: Any) -> ModeContext:
    options = _opts(**given)
    config = PetrifyConfig(
        mode="shadow", detector=detector, mode_options=mode_options_from(options)  # type: ignore[arg-type]
    )
    return ModeContext(INFO, config, options)


# ---------------------------------------------------------------------------
# Constants and option specs
# ---------------------------------------------------------------------------


def test_style_groups_partition_the_style_names() -> None:
    assert len(STYLE_NAMES) == 9
    assert set(STYLE_NAMES) == FACE_STYLES | MASK_STYLES | {"bodybox"}
    assert FACE_STYLES == {"gaussian", "pixelate", "noise", "facebox", "eyebar"}
    assert MASK_STYLES == {"silhouette", "silhouette_color", "outline"}
    assert BODY_STYLES == {"silhouette", "silhouette_color", "bodybox"}
    assert not FACE_STYLES & MASK_STYLES


def test_apply_order_is_the_design_order() -> None:
    assert APPLY_ORDER == (
        "silhouette", "silhouette_color", "bodybox", "outline",
        "gaussian", "pixelate", "noise", "facebox", "eyebar",
    )
    assert sorted(APPLY_ORDER) == sorted(STYLE_NAMES)


def test_colors_are_bgr() -> None:
    assert dict(COLORS) == {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (128, 128, 128),
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
    }
    with pytest.raises(TypeError):
        COLORS["black"] = (1, 1, 1)  # type: ignore[index]


def test_option_specs_pass_the_core_check() -> None:
    assert check_option_specs(OPTIONS) is OPTIONS
    assert [s.name for s in OPTIONS] == ["styles", "padding", "strength", "color", "dilate"]


def test_option_specs_match_the_design_table() -> None:
    by_name = {s.name: s for s in OPTIONS}
    styles = by_name["styles"]
    assert (styles.kind, styles.default, styles.choices) == ("choices", ("facebox",), STYLE_NAMES)
    padding = by_name["padding"]
    assert (padding.kind, padding.default, padding.minimum, padding.maximum) == (
        "float", 0.3, 0.0, 1.0)
    strength = by_name["strength"]
    assert (strength.kind, strength.default, strength.choices) == (
        "choice", "high", ("high", "medium", "low"))
    color = by_name["color"]
    assert (color.kind, color.default, color.choices) == (
        "choice", "black", ("black", "white", "gray", "red", "green", "blue"))
    dilate = by_name["dilate"]
    assert (dilate.kind, dilate.default, dilate.minimum, dilate.maximum) == (
        "float", 0.01, 0.0, 0.1)


def test_every_option_is_runtime_with_help_and_bounded_numbers() -> None:
    for spec in OPTIONS:
        assert spec.runtime is True, spec.name
        assert spec.help.strip(), spec.name
        if spec.kind in ("int", "float"):
            assert spec.minimum is not None and spec.maximum is not None, spec.name


def test_no_text_holds_an_em_or_en_dash() -> None:
    texts = [spec.help for spec in OPTIONS] + [
        WEAK_NO_BODY, WEAK_REVERSIBLE, WEAK_STRENGTH, WEAK_EYEBAR_ONLY, WEAK_PADDING,
        WEAK_DILATE, WEAK_SEGMENT_MISS, BODYBOX_NEEDS_PERSONS, SHADOW.summary,
    ]
    for text in texts:
        assert chr(0x2014) not in text and chr(0x2013) not in text, text


@pytest.mark.parametrize(
    ("assignments", "expected"),
    [
        (["styles=silhouette,facebox"], {"styles": ("silhouette", "facebox")}),
        (["styles= noise , eyebar "], {"styles": ("noise", "eyebar")}),
        (["padding=0.5", "strength=low"], {"padding": 0.5, "strength": "low"}),
        (["color=red", "dilate=0.02"], {"color": "red", "dilate": 0.02}),
    ],
)
def test_resolve_option_text_round_trips_cli_input(
    assignments: list[str], expected: dict[str, Any]
) -> None:
    resolved = resolve_option_text(OPTIONS, assignments)
    assert resolved == _opts(**expected)
    assert resolve_options(OPTIONS, resolved) == resolved


def test_resolve_option_text_refuses_unknown_styles_and_out_of_range() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        resolve_option_text(OPTIONS, ["styles=blur"])
    with pytest.raises(ValueError, match="at most"):
        resolve_option_text(OPTIONS, ["padding=1.5"])
    with pytest.raises(ValueError, match="at most"):
        resolve_option_text(OPTIONS, ["dilate=0.2"])


# ---------------------------------------------------------------------------
# parse_params
# ---------------------------------------------------------------------------


def test_parse_params_defaults() -> None:
    expected = ShadowParams(
        styles=frozenset({"facebox"}), padding=0.3, strength="high", color=(0, 0, 0), dilate=0.01
    )
    assert parse_params({}) == expected
    assert parse_params(_opts()) == expected


def test_parse_params_reads_each_option() -> None:
    params = parse_params(
        _opts(styles="noise,silhouette", padding=0.6, strength="medium", color="blue",
              dilate=0.05)
    )
    assert params == ShadowParams(
        styles=frozenset({"noise", "silhouette"}), padding=0.6, strength="medium",
        color=(255, 0, 0), dilate=0.05,
    )
    assert type(params.padding) is float and type(params.dilate) is float


@pytest.mark.parametrize("name", list(COLORS))
def test_parse_params_maps_color_names_to_bgr(name: str) -> None:
    assert parse_params(_opts(color=name)).color == COLORS[name]


def test_parse_params_ignores_style_order() -> None:
    a = parse_params(_opts(styles="silhouette,facebox,eyebar"))
    b = parse_params(_opts(styles="eyebar,facebox,silhouette"))
    assert a == b
    assert hash(a) == hash(b)


def test_parse_params_is_frozen() -> None:
    params = parse_params({})
    with pytest.raises(AttributeError):
        params.padding = 0.0  # type: ignore[misc]


def test_parse_params_refuses_an_empty_style_list() -> None:
    with pytest.raises(ValueError, match=r"^the shadow mode needs at least one style in 'styles'$"):
        parse_params(_opts(styles=()))


def test_parse_params_refuses_outline_alone() -> None:
    with pytest.raises(
        ValueError,
        match=r"^the outline style does not hide anyone on its own: add another style to "
        r"'styles'$",
    ):
        parse_params(_opts(styles="outline"))


def test_parse_params_accepts_outline_with_another_style() -> None:
    assert parse_params(_opts(styles="outline,facebox")).styles == {"outline", "facebox"}


@pytest.mark.parametrize(
    "given",
    [{"padding": 2.0}, {"dilate": -0.1}, {"strength": "max"}, {"color": "pink"},
     {"styles": ("facebox", "facebox")}, {"styles": "facebox"}, {"unknown": 1}],
)
def test_parse_params_refuses_invalid_values_with_value_error(given: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        parse_params(given)


# ---------------------------------------------------------------------------
# requires and mandatory
# ---------------------------------------------------------------------------

BASE = {MOTION, PERSONS}
FACE = {FACES, LANDMARKS}

REQUIRES_TABLE: list[tuple[str, set[DetectionKind], set[DetectionKind]]] = [
    # Every single style.
    ("gaussian", BASE | FACE, FACE),
    ("pixelate", BASE | FACE, FACE),
    ("noise", BASE | FACE, FACE),
    ("facebox", BASE | FACE, FACE),
    ("eyebar", BASE | FACE, FACE),
    ("silhouette", BASE | {MASK}, {MASK}),
    ("silhouette_color", BASE | {MASK}, {MASK}),
    ("bodybox", BASE, {PERSONS}),
    # outline alone is refused, so it appears only in stacks.
    ("outline,facebox", BASE | FACE | {MASK}, FACE | {MASK}),
    # Representative stacks.
    ("facebox,noise,eyebar", BASE | FACE, FACE),
    ("silhouette,outline", BASE | {MASK}, {MASK}),
    ("silhouette,facebox", BASE | FACE | {MASK}, FACE | {MASK}),
    ("bodybox,facebox", BASE | FACE, FACE | {PERSONS}),
    ("bodybox,silhouette_color", BASE | {MASK}, {MASK, PERSONS}),
    (",".join(STYLE_NAMES), BASE | FACE | {MASK}, FACE | {MASK, PERSONS}),
]


@pytest.mark.parametrize(("styles", "requires", "mandatory"), REQUIRES_TABLE)
def test_requires_and_mandatory_table(
    styles: str, requires: set[DetectionKind], mandatory: set[DetectionKind]
) -> None:
    options = _opts(styles=styles)
    assert SHADOW.requires(options) == frozenset(requires)
    assert SHADOW.mandatory(options) == frozenset(mandatory)
    # The built mode carries the same sets, which build_mode checks.
    mode = SHADOW.create(_ctx(styles=styles))
    assert isinstance(mode, ShadowMode)
    assert mode.requires == frozenset(requires)
    assert mode.mandatory == frozenset(mandatory)
    # Never the background differ: shadow has no clean background.
    assert DetectionKind.BACKGROUND_DIFF not in mode.requires


def test_table_covers_every_style_alone() -> None:
    singles = {row[0] for row in REQUIRES_TABLE if "," not in row[0]}
    assert singles == set(STYLE_NAMES) - {"outline"}


def test_mandatory_is_always_a_subset_of_requires() -> None:
    for styles, requires, mandatory in REQUIRES_TABLE:
        assert mandatory <= requires, styles


def test_requires_and_mandatory_refuse_invalid_styles() -> None:
    for styles in [(), ("outline",)]:
        options = _opts(styles=styles)
        with pytest.raises(ValueError):
            SHADOW.requires(options)
        with pytest.raises(ValueError):
            SHADOW.mandatory(options)


# ---------------------------------------------------------------------------
# weakened
# ---------------------------------------------------------------------------


def _weak(**given: Any) -> tuple[str, ...]:
    return SHADOW.weakened(_opts(**given))


def test_weakened_default_lists_only_the_face_only_sentence() -> None:
    assert _weak() == (WEAK_NO_BODY,)


def test_weakened_no_body_style() -> None:
    assert WEAK_NO_BODY in _weak(styles="facebox")
    assert WEAK_NO_BODY in _weak(styles="outline,facebox")  # outline is no body cover.
    for body in sorted(BODY_STYLES):
        assert WEAK_NO_BODY not in _weak(styles=f"{body},facebox"), body


def test_weakened_reversible_blur() -> None:
    for style in ("gaussian", "pixelate"):
        assert WEAK_REVERSIBLE in _weak(styles=style)
        assert WEAK_REVERSIBLE in _weak(styles=f"{style},eyebar")
        # A body style does not lift it: a person the body style misses keeps only
        # the blur on the face.
        for body in ("silhouette", "silhouette_color", "bodybox", "outline"):
            assert WEAK_REVERSIBLE in _weak(styles=f"{style},{body}"), body
        for cover in ("noise", "facebox"):
            assert WEAK_REVERSIBLE not in _weak(styles=f"{style},{cover}"), cover
            assert WEAK_REVERSIBLE not in _weak(styles=f"{style},{cover},bodybox"), cover
    assert WEAK_REVERSIBLE not in _weak(styles="facebox")
    assert WEAK_REVERSIBLE not in _weak(styles="bodybox")


def test_weakened_strength() -> None:
    for strength in ("medium", "low"):
        sentence = WEAK_STRENGTH.format(v=strength)
        assert sentence in _weak(styles="gaussian", strength=strength)
        assert sentence in _weak(styles="pixelate,bodybox", strength=strength)
        assert sentence not in _weak(styles="facebox,noise", strength=strength)
        # noise and facebox replace the blurred box, so the strength does not matter.
        assert sentence not in _weak(styles="gaussian,noise", strength=strength)
        assert sentence not in _weak(styles="pixelate,facebox", strength=strength)
        assert sentence not in _weak(styles="bodybox", strength=strength)
    assert not any("strength is" in s for s in _weak(styles="gaussian", strength="high"))
    assert "strength is low: the blur or pixelation is weaker than the default and easier " \
        "to reverse." in _weak(styles="gaussian", strength="low")


def test_weakened_eyebar_only() -> None:
    assert WEAK_EYEBAR_ONLY in _weak(styles="eyebar")
    assert WEAK_EYEBAR_ONLY in _weak(styles="eyebar,outline")
    # A body style does not lift it: a person the body style misses keeps only the bar.
    for body in ("bodybox", "silhouette", "silhouette_color"):
        assert WEAK_EYEBAR_ONLY in _weak(styles=f"eyebar,{body}"), body
    for cover in ("facebox", "noise", "gaussian", "pixelate"):
        assert WEAK_EYEBAR_ONLY not in _weak(styles=f"eyebar,{cover}"), cover


def test_weakened_padding_boundary() -> None:
    assert not any(s.startswith("padding is") for s in _weak(padding=0.3))
    assert not any(s.startswith("padding is") for s in _weak(padding=0.8))
    assert WEAK_PADDING.format(v="0.29") in _weak(padding=0.29)
    assert WEAK_PADDING.format(v="0") in _weak(padding=0.0)
    assert (
        "padding is 0.1, below the default of 0.3: hair, ears and chin around the face box "
        "can be shown." in _weak(padding=0.1)
    )
    # No face style, so padding does not apply.
    assert not any(s.startswith("padding is") for s in _weak(styles="bodybox", padding=0.0))


def test_weakened_dilate_boundary() -> None:
    assert not any(s.startswith("dilate is") for s in _weak(styles="silhouette", dilate=0.01))
    assert not any(s.startswith("dilate is") for s in _weak(styles="silhouette", dilate=0.05))
    assert WEAK_DILATE.format(v="0.009") in _weak(styles="silhouette", dilate=0.009)
    assert WEAK_DILATE.format(v="0") in _weak(styles="silhouette_color", dilate=0.0)
    assert WEAK_DILATE.format(v="0") in _weak(styles="silhouette,outline", dilate=0.0)
    # The sentence speaks of the silhouette: outline alone in a stack has none.
    assert not any(s.startswith("dilate is") for s in _weak(styles="outline,bodybox", dilate=0.0))
    assert not any(s.startswith("dilate is") for s in _weak(styles="outline,facebox", dilate=0.0))
    # No mask style, so dilate does not apply.
    assert not any(s.startswith("dilate is") for s in _weak(styles="facebox", dilate=0.0))


def test_weakened_segmenter_miss() -> None:
    for styles in ("silhouette", "silhouette_color", "silhouette,facebox", "silhouette,noise",
                   "silhouette_color,outline,eyebar"):
        assert WEAK_SEGMENT_MISS in _weak(styles=styles), styles
    # bodybox fills every person box; outline covers no body, which WEAK_NO_BODY says.
    for styles in ("silhouette,bodybox", "silhouette_color,bodybox,facebox", "outline,facebox",
                   "facebox", "bodybox"):
        assert WEAK_SEGMENT_MISS not in _weak(styles=styles), styles


@pytest.mark.parametrize("name", ["padding", "dilate"])
@pytest.mark.parametrize("text", ["-0", "-0.0"])
def test_negative_zero_is_normalized(name: str, text: str) -> None:
    options = resolve_option_text(OPTIONS, [f"{name}={text}", "styles=silhouette,facebox"])
    value = getattr(parse_params(options), name)
    assert value == 0.0 and math.copysign(1.0, value) == 1.0
    sentences = SHADOW.weakened(options)
    assert f"{name} is 0," in " ".join(sentences)
    assert "-0" not in " ".join(sentences)


def test_weakened_combination_lists_all_in_order() -> None:
    assert _weak(styles="gaussian,eyebar,outline", strength="low", padding=0.1, dilate=0.0) == (
        WEAK_NO_BODY,
        WEAK_REVERSIBLE,
        WEAK_STRENGTH.format(v="low"),
        WEAK_PADDING.format(v="0.1"),
    )
    assert _weak(styles="pixelate,silhouette", strength="medium", padding=0.0,
                 dilate=0.0) == (
        WEAK_REVERSIBLE,
        WEAK_STRENGTH.format(v="medium"),
        WEAK_PADDING.format(v="0"),
        WEAK_DILATE.format(v="0"),
        WEAK_SEGMENT_MISS,
    )
    assert _weak(styles="eyebar", padding=0.0) == (
        WEAK_NO_BODY, WEAK_EYEBAR_ONLY, WEAK_PADDING.format(v="0"),
    )


def test_weakened_strongest_settings_list_nothing() -> None:
    assert _weak(styles="silhouette,bodybox,noise") == ()
    assert _weak(styles="bodybox,facebox") == ()
    assert _weak(styles=",".join(STYLE_NAMES)) == ()


class _Hostile(Mapping[str, Any]):
    def __getitem__(self, key: str) -> Any:
        raise RuntimeError("no")

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("no")

    def __len__(self) -> int:
        raise RuntimeError("no")

    def get(self, key: str, default: Any = None) -> Any:
        raise RuntimeError("no")


class _BadIter(tuple):  # type: ignore[type-arg]
    def __iter__(self) -> Iterator[Any]:
        raise RuntimeError("no")


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"styles": ()},
        {"styles": ("outline",)},
        {"styles": None},
        {"styles": "facebox"},
        {"styles": (1, None, "nope", "gaussian")},
        {"styles": _BadIter(("gaussian",))},
        {"padding": "x", "dilate": None, "strength": 3, "color": object()},
        {"padding": float("nan"), "dilate": float("inf")},
        {"padding": 10**400, "dilate": True},
        {"unknown": 1},
        _Hostile(),
        None,
        42,
    ],
)
def test_weakened_never_raises(options: Any) -> None:
    result = SHADOW.weakened(options)
    assert isinstance(result, tuple)
    assert all(isinstance(s, str) and s for s in result)


def test_weakened_on_invalid_styles_still_reports_the_other_settings() -> None:
    assert SHADOW.weakened({"styles": (), "padding": 0.0}) == (WEAK_NO_BODY,)
    assert SHADOW.weakened({"styles": ("outline",), "dilate": 0.0}) == (WEAK_NO_BODY,)
    assert SHADOW.weakened({"styles": ("silhouette", 7), "dilate": 0.0}) == (
        WEAK_DILATE.format(v="0"), WEAK_SEGMENT_MISS,
    )


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


def test_create_returns_a_new_mode_with_its_params() -> None:
    mode = SHADOW.create(_ctx(styles="noise", padding=0.5))
    assert isinstance(mode, ShadowMode)
    assert mode.params == parse_params(_opts(styles="noise", padding=0.5))
    assert isinstance(mode._rng, np.random.Generator)


def test_create_returns_previous_for_equal_params() -> None:
    first = SHADOW.create(_ctx(styles="silhouette,facebox"))
    assert SHADOW.create(_ctx(styles="silhouette,facebox"), first) is first
    # The same set in another order is the same mode.
    assert SHADOW.create(_ctx(styles="facebox,silhouette"), first) is first


@pytest.mark.parametrize(
    "change",
    [{"styles": "noise"}, {"padding": 0.4}, {"strength": "low"}, {"color": "white"},
     {"dilate": 0.02}],
)
def test_create_returns_a_new_mode_when_params_differ(change: dict[str, Any]) -> None:
    first = SHADOW.create(_ctx())
    second = SHADOW.create(_ctx(**change), first)
    assert second is not first
    assert isinstance(second, ShadowMode)
    assert second.params != first.params


def test_create_does_not_reuse_a_foreign_previous_mode() -> None:
    class Other:
        name = "shadow"
        params = parse_params({})

    mode = SHADOW.create(_ctx(), Other())  # type: ignore[arg-type]
    assert isinstance(mode, ShadowMode)


def test_create_refuses_bodybox_with_the_motion_detector() -> None:
    for styles in ("bodybox", "bodybox,facebox"):
        with pytest.raises(ValueError) as info:
            SHADOW.create(_ctx(detector="motion", styles=styles))
        assert str(info.value) == (
            "the bodybox style needs person boxes: use --detector person or both, with "
            "--person-detector dnn for the recommended detector"
        )


@pytest.mark.parametrize("detector", ["person", "both"])
def test_create_accepts_bodybox_with_person_boxes(detector: str) -> None:
    assert isinstance(SHADOW.create(_ctx(detector=detector, styles="bodybox")), ShadowMode)


def test_create_accepts_other_styles_with_the_motion_detector() -> None:
    assert isinstance(SHADOW.create(_ctx(detector="motion", styles="facebox")), ShadowMode)


def test_create_refuses_invalid_styles() -> None:
    for styles in [(), ("outline",)]:
        with pytest.raises(ValueError):
            SHADOW.create(_ctx(styles=styles))


def test_mode_rng_is_unseeded_by_default_and_injectable(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []
    real = np.random.default_rng

    def record(*args: Any, **kwargs: Any) -> np.random.Generator:
        calls.append(args + tuple(kwargs.values()))
        return real(*args, **kwargs)

    monkeypatch.setattr(np.random, "default_rng", record)
    SHADOW.create(_ctx(styles="noise"))
    assert calls == [()]
    seeded = real(7)
    assert ShadowMode(parse_params({}), rng=seeded)._rng is seeded


# ---------------------------------------------------------------------------
# Through the real registry
# ---------------------------------------------------------------------------

VALID_STYLE_SETS = [row[0] for row in REQUIRES_TABLE]


@pytest.mark.parametrize("styles", VALID_STYLE_SETS)
def test_build_mode_accepts_valid_option_sets(
    shadow_entry_point: ShadowEntryPoint, styles: str
) -> None:
    factory = load_mode("shadow")
    ctx = _ctx(styles=styles, padding=0.5, strength="low", color="gray", dilate=0.05)
    mode = build_mode(factory, ctx)
    assert isinstance(mode, ShadowMode)
    assert build_mode(factory, ctx, mode) is mode


@pytest.mark.parametrize(
    ("styles", "message"),
    [((), "at least one style"), (("outline",), "does not hide anyone")],
)
def test_build_mode_rejects_invalid_styles_as_settings(
    shadow_entry_point: ShadowEntryPoint, styles: tuple[str, ...], message: str
) -> None:
    # ModeLoadError is a RuntimeError, so it would escape this check: the settings
    # must be rejected, the plugin must not be reported as faulty.
    with pytest.raises(ValueError, match=message):
        build_mode(load_mode("shadow"), _ctx(styles=styles))


def test_build_mode_rejects_bodybox_with_motion_as_settings(
    shadow_entry_point: ShadowEntryPoint,
) -> None:
    with pytest.raises(ValueError, match="bodybox style needs person boxes"):
        build_mode(load_mode("shadow"), _ctx(detector="motion", styles="bodybox"))
