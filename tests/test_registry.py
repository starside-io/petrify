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

"""Tests for the mode construction contract and the mode registry."""

from __future__ import annotations

import sys
import textwrap
from collections.abc import Iterator
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import pytest

from petrify.core import (
    BUILTIN_MODES,
    ENTRY_POINT_GROUP,
    ModeContext,
    ModeEntry,
    ModeFactory,
    ModeLoadError,
    OptionSpec,
    PetrifyConfig,
    SourceInfo,
    build_mode,
    list_modes,
    load_mode,
    mode_options_from,
    registry,
    resolve_option_text,
    resolve_options,
)
from petrify.modes import ERASE

INFO = SourceInfo(fps=25.0, width=64, height=48, frame_count=None, seekable=False, live=False)


def _ep(name: str, value: str = "somewhere:FACTORY", group: str = ENTRY_POINT_GROUP) -> EntryPoint:
    return EntryPoint(name=name, value=value, group=group)


# ----------------------------------------------------------------------
# OptionSpec construction
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "Bad", "kind": "bool", "default": True},
        {"name": "1bad", "kind": "bool", "default": True},
        {"name": "a" * 33, "kind": "bool", "default": True},
        {"name": "ok", "kind": "string", "default": "x"},
        {"name": "ok", "kind": "bool", "default": 1},
        {"name": "ok", "kind": "int", "default": 1.5},
        {"name": "ok", "kind": "int", "default": True},
        {"name": "ok", "kind": "float", "default": float("nan")},
        {"name": "ok", "kind": "choice", "default": "a"},
        {"name": "ok", "kind": "choice", "default": "a", "choices": ("a", "a")},
        {"name": "ok", "kind": "choice", "default": "c", "choices": ("a", "b")},
        {"name": "ok", "kind": "choice", "default": "a", "choices": ["a", "b"]},
        {"name": "ok", "kind": "choice", "default": "a", "choices": ("a", "b,c")},
        {"name": "ok", "kind": "choices", "default": (), "choices": ()},
        {"name": "ok", "kind": "choices", "default": ("a", "a"), "choices": ("a",)},
        {"name": "ok", "kind": "bool", "default": True, "choices": ("a",)},
        {"name": "ok", "kind": "bool", "default": True, "minimum": 0},
        {"name": "ok", "kind": "int", "default": 5, "minimum": 6, "maximum": 4},
        {"name": "ok", "kind": "int", "default": 5, "minimum": float("-inf")},
        {"name": "ok", "kind": "float", "default": 5.0, "maximum": float("nan")},
        {"name": "ok", "kind": "float", "default": 5.0, "maximum": True},
        {"name": "ok", "kind": "int", "default": 0, "minimum": 1},
        {"name": "ok", "kind": "int", "default": 11, "maximum": 10},
        {"name": "ok", "kind": "bool", "default": True, "runtime": 1},
        {"name": "ok", "kind": "bool", "default": True, "help": 3},
    ],
)
def test_option_spec_rejects_invalid_construction(kwargs: dict[str, Any]) -> None:
    kwargs.setdefault("help", "text")
    with pytest.raises(ValueError):
        OptionSpec(**kwargs)


def test_option_spec_stores_default_in_declared_type() -> None:
    spec = OptionSpec("scale", "float", 1, "scale")
    assert spec.default == 1.0 and type(spec.default) is float
    styles = OptionSpec("styles", "choices", ["b", "a"], "styles", choices=("a", "b"))
    assert styles.default == ("b", "a")
    hash(styles)  # A list default would make the frozen spec unhashable.


def test_option_spec_reports_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="minimum 6 is greater than maximum 4"):
        OptionSpec("n", "int", 5, "n", minimum=6, maximum=4)


def test_option_spec_accepts_equal_bounds() -> None:
    spec = OptionSpec("n", "int", 3, "n", minimum=3, maximum=3)
    assert spec.validate(3) == 3


# ----------------------------------------------------------------------
# validate and parse: one canonical type per kind
# ----------------------------------------------------------------------

BOOL = OptionSpec("flag", "bool", False, "a flag")
INT = OptionSpec("count", "int", 2, "a count", minimum=1, maximum=10)
FLOAT = OptionSpec("ratio", "float", 0.5, "a ratio", minimum=0, maximum=2)
CHOICE = OptionSpec("style", "choice", "blur", "a style", choices=("blur", "pixelate"))
CHOICES = OptionSpec("kinds", "choices", (), "kinds", choices=("a", "b", "c"))


@pytest.mark.parametrize(
    ("spec", "value", "expected"),
    [
        (BOOL, True, True),
        (BOOL, False, False),
        (INT, 1, 1),
        (INT, 1.0, 1),
        (FLOAT, 1, 1.0),
        (FLOAT, 1.0, 1.0),
        (CHOICE, "pixelate", "pixelate"),
        (CHOICES, ["c", "a"], ("c", "a")),
        (CHOICES, ("b",), ("b",)),
        (CHOICES, [], ()),
    ],
)
def test_validate_returns_canonical_type(spec: OptionSpec, value: object, expected: object) -> None:
    result = spec.validate(value)
    assert result == expected
    assert type(result) is type(expected)


@pytest.mark.parametrize(
    ("spec", "value"),
    [
        (BOOL, 1),
        (BOOL, 0),
        (BOOL, 1.0),
        (BOOL, "1"),
        (BOOL, "true"),
        (INT, True),
        (INT, "1"),
        (INT, 1.5),
        (INT, float("inf")),
        (INT, 0),  # Below the minimum.
        (INT, 11),  # Above the maximum.
        (FLOAT, True),
        (FLOAT, "1"),
        (FLOAT, float("nan")),
        (FLOAT, float("inf")),
        (FLOAT, 10**400),
        (FLOAT, -0.1),
        (FLOAT, 2.5),
        (CHOICE, 1),
        (CHOICE, True),
        (CHOICE, 1.0),
        (CHOICE, "1"),
        (CHOICE, "zoom"),
        (CHOICES, "a"),
        (CHOICES, 1),
        (CHOICES, True),
        (CHOICES, ["a", "a"]),
        (CHOICES, ["d"]),
        (CHOICES, [1]),
    ],
)
def test_validate_rejects(spec: OptionSpec, value: object) -> None:
    with pytest.raises(ValueError):
        spec.validate(value)


class _LyingStr(str):
    pass


def test_validate_choice_returns_plain_str_for_subclass() -> None:
    result = CHOICE.validate(_LyingStr("blur"))
    assert type(result) is str
    assert type(CHOICES.validate([_LyingStr("a")])[0]) is str


@pytest.mark.parametrize(
    ("spec", "text", "expected"),
    [
        (BOOL, "1", True),
        (BOOL, "true", True),
        (BOOL, " TRUE ", True),
        (BOOL, "yes", True),
        (BOOL, "on", True),
        (BOOL, "0", False),
        (BOOL, "false", False),
        (BOOL, "no", False),
        (BOOL, "off", False),
        (INT, "1", 1),
        (INT, "+3", 3),
        (FLOAT, "1", 1.0),
        (FLOAT, "0.5", 0.5),
        (CHOICE, "blur", "blur"),
        (CHOICES, "a,b", ("a", "b")),
        (CHOICES, " c , a ", ("c", "a")),
        (CHOICES, "", ()),
    ],
)
def test_parse_returns_canonical_type(spec: OptionSpec, text: str, expected: object) -> None:
    result = spec.parse(text)
    assert result == expected
    assert type(result) is type(expected)


@pytest.mark.parametrize(
    ("spec", "text"),
    [
        (BOOL, "2"),
        (BOOL, ""),
        (BOOL, "1.0"),
        (INT, "1.0"),
        (INT, "1.5"),
        (INT, "true"),
        (INT, "0"),
        (INT, ""),
        (FLOAT, "nan"),
        (FLOAT, "inf"),
        (FLOAT, "1e400"),
        (FLOAT, "true"),
        (FLOAT, "3"),  # Above the maximum.
        (CHOICE, "1"),
        (CHOICE, "zoom"),
        (CHOICES, "a,a"),
        (CHOICES, "a,,b"),
        (CHOICES, "1"),
    ],
)
def test_parse_rejects(spec: OptionSpec, text: str) -> None:
    with pytest.raises(ValueError):
        spec.parse(text)


def test_parse_rejects_non_text() -> None:
    with pytest.raises(ValueError):
        INT.parse(3)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# resolve_options and resolve_option_text
# ----------------------------------------------------------------------

SPECS = (BOOL, INT, FLOAT, CHOICE, CHOICES)


def test_resolve_options_fills_defaults() -> None:
    assert resolve_options(SPECS, {}) == {
        "flag": False,
        "count": 2,
        "ratio": 0.5,
        "style": "blur",
        "kinds": (),
    }


def test_resolve_options_coerces_given_values() -> None:
    resolved = resolve_options(SPECS, {"count": 3.0, "ratio": 1, "kinds": ["b"]})
    assert type(resolved["count"]) is int and resolved["count"] == 3
    assert type(resolved["ratio"]) is float and resolved["ratio"] == 1.0
    assert resolved["kinds"] == ("b",)


def test_resolve_options_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown mode option 'nope'"):
        resolve_options(SPECS, {"nope": 1})


def test_resolve_options_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="count"):
        resolve_options(SPECS, {"count": True})


def test_resolve_options_rejects_bad_specs() -> None:
    with pytest.raises(ValueError, match="more than once"):
        resolve_options((BOOL, BOOL), {})
    with pytest.raises(ValueError, match="tuple"):
        resolve_options([BOOL], {})  # type: ignore[arg-type]


def test_resolve_options_result_goes_into_mode_options_from() -> None:
    resolved = resolve_options(SPECS, {"kinds": ["c", "a"], "ratio": 2})
    pairs = mode_options_from(resolved)
    config = PetrifyConfig(mode="fake", mode_options=pairs)
    assert config.mode_options_dict() == resolved


def test_canonical_values_make_configs_differ() -> None:
    # 1, True and 1.0 compare equal; coerced to the declared type, a change of
    # type never hides behind ==.
    float_spec = OptionSpec("level", "float", 0.0, "level")
    a = mode_options_from(resolve_options((float_spec,), {"level": 1}))
    b = mode_options_from(resolve_options((float_spec,), {"level": 1.0}))
    assert a == b
    assert type(a[0][1]) is float


def test_resolve_option_text() -> None:
    resolved = resolve_option_text(SPECS, ["flag=true", "count=4", "kinds=a,c"])
    assert resolved == {
        "flag": True,
        "count": 4,
        "ratio": 0.5,
        "style": "blur",
        "kinds": ("a", "c"),
    }


@pytest.mark.parametrize(
    "items",
    [["flag"], ["flag=true", "flag=false"], ["nope=1"], ["count=x"]],
)
def test_resolve_option_text_rejects(items: list[str]) -> None:
    with pytest.raises(ValueError):
        resolve_option_text(SPECS, items)


# ----------------------------------------------------------------------
# ModeContext
# ----------------------------------------------------------------------


def test_mode_context_options_are_read_only() -> None:
    given = {"a": 1}
    ctx = ModeContext(INFO, PetrifyConfig(), given)
    given["a"] = 2
    assert ctx.options["a"] == 1
    with pytest.raises(TypeError):
        ctx.options["a"] = 3  # type: ignore[index]


def test_mode_context_rejects_wrong_types() -> None:
    with pytest.raises(TypeError):
        ModeContext("info", PetrifyConfig(), {})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ModeContext(INFO, PetrifyConfig(), {}, background="image")  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# list_modes with synthetic entry points
# ----------------------------------------------------------------------


def test_builtins_listed_without_entry_points() -> None:
    entries = list_modes([])
    assert entries == (
        ModeEntry("erase", "petrify.modes.erase:ERASE", None, None, True, None),
    )
    assert dict(BUILTIN_MODES) == {"erase": "petrify.modes.erase:ERASE"}
    with pytest.raises(TypeError):
        BUILTIN_MODES["blur"] = "x:Y"  # type: ignore[index]


def test_valid_entry_point_is_selectable_and_not_imported() -> None:
    entries = list_modes([_ep("blur", "petrify_no_such_module_xyz:FACTORY")])
    blur = [e for e in entries if e.name == "blur"]
    assert blur == [
        ModeEntry("blur", "petrify_no_such_module_xyz:FACTORY", None, None, False, None)
    ]
    assert "petrify_no_such_module_xyz" not in sys.modules


def test_shadowed_entry_point_has_problem() -> None:
    entries = list_modes([_ep("erase", "evil:FACTORY")])
    assert entries[0].builtin and entries[0].problem is None
    shadow = entries[1]
    assert (shadow.name, shadow.builtin) == ("erase", False)
    assert shadow.problem is not None and "shadowed" in shadow.problem


def test_duplicate_name_is_ambiguous() -> None:
    entries = list_modes([_ep("blur", "one:F"), _ep("blur", "two:F")])
    blur = [e for e in entries if e.name == "blur"]
    assert len(blur) == 2
    assert all(e.problem is not None and "ambiguous" in e.problem for e in blur)


def test_identical_registration_is_listed_once() -> None:
    entries = list_modes([_ep("blur", "one:F"), _ep("blur", "one:F")])
    assert [e.problem for e in entries if e.name == "blur"] == [None]


@pytest.mark.parametrize("name", ["Blur", "1blur", "blur-x", "", "b" * 33])
def test_invalid_name_has_problem(name: str) -> None:
    entries = list_modes([_ep(name)])
    entry = entries[1]
    assert entry.name == name
    assert entry.problem is not None and "invalid mode name" in entry.problem


@pytest.mark.parametrize("value", ["module_only", "a:", ":b", "a b:c", "a:b-c"])
def test_invalid_target_has_problem(value: str) -> None:
    entry = list_modes([_ep("blur", value)])[1]
    assert entry.problem is not None and "module:attr" in entry.problem


def test_other_groups_are_ignored() -> None:
    assert list_modes([_ep("blur", group="something.else")]) == list_modes([])


def test_order_is_builtins_then_name() -> None:
    entries = list_modes([_ep("zeta", "z:F"), _ep("alpha", "a:F"), _ep("mid", "m:F")])
    assert [e.name for e in entries] == ["erase", "alpha", "mid", "zeta"]


# ----------------------------------------------------------------------
# load_mode with synthetic entries
# ----------------------------------------------------------------------


def test_load_erase_returns_erase_factory() -> None:
    factory = load_mode("erase", list_modes([]))
    assert factory is ERASE
    assert isinstance(factory, ModeFactory)
    assert load_mode("erase") is ERASE


def test_load_unknown_lists_selectable_names() -> None:
    entries = list_modes([_ep("blur", "b:F"), _ep("Bad", "b:F")])
    with pytest.raises(ModeLoadError, match=r"unknown mode 'nope'; available modes: blur, erase"):
        load_mode("nope", entries)


def test_load_entry_with_problem_is_refused() -> None:
    entries = list_modes([_ep("blur", "one:F"), _ep("blur", "two:F")])
    with pytest.raises(ModeLoadError, match="ambiguous"):
        load_mode("blur", entries)


def test_load_shadowed_name_gives_builtin() -> None:
    entries = list_modes([_ep("erase", "petrify_no_such_module_xyz:F")])
    assert load_mode("erase", entries) is ERASE
    assert "petrify_no_such_module_xyz" not in sys.modules


def test_load_missing_module() -> None:
    entries = list_modes([_ep("blur", "petrify_no_such_module_xyz:FACTORY")])
    with pytest.raises(ModeLoadError, match="could not be imported"):
        load_mode("blur", entries)


# ----------------------------------------------------------------------
# A fake installed distribution
# ----------------------------------------------------------------------

_FAKE_BASE = '''
from petrify.core.detections import DetectionKind
from petrify.core.plugin import OptionSpec

MARKER = {marker!r}
open(MARKER + "/imported_" + __name__, "w").close()


class _LyingStr(str):
    pass


class FakeMode:
    name = "{mode}"
    requires = frozenset({{DetectionKind.PERSONS}})
    detect_policy = "all"
    mandatory = frozenset()

    def render(self, frame, detections, subject_present):
        return frame.image.copy()


class Factory:
    summary = "A fake mode for tests."
    options = (OptionSpec("strength", "int", 3, "strength", minimum=1, maximum=10),)
    needs_background = False
    roi_overlay = False

    def __init__(self, name="{mode}", mode_cls=FakeMode):
        self.name = name
        self.mode_cls = mode_cls

    def requires(self, options):
        return frozenset({{DetectionKind.PERSONS}})

    def mandatory(self, options):
        return frozenset()

    def create(self, ctx, previous=None):
        return self.mode_cls()

    def weakened(self, options):
        return ()
'''


class _Installer:
    """Writes fake distributions and modules under ``tmp_path`` and cleans up after."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.modules: set[str] = set()
        monkeypatch.syspath_prepend(str(root))

    def dist(
        self, dist: str, version: str, entries: dict[str, str], root: Path | None = None
    ) -> None:
        lines = "".join(f"{name} = {target}\n" for name, target in entries.items())
        self.raw_dist(dist, version, f"[{ENTRY_POINT_GROUP}]\n{lines}".encode(), root)

    def raw_dist(
        self, dist: str, version: str, entry_points: bytes, root: Path | None = None
    ) -> None:
        info = (root or self.root) / f"{dist}-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n"
        )
        (info / "entry_points.txt").write_bytes(entry_points)

    def module(self, name: str, mode: str = "fake", extra: str = "") -> None:
        source = _FAKE_BASE.format(marker=str(self.root), mode=mode)
        (self.root / f"{name}.py").write_text(source + textwrap.dedent(extra))
        self.modules.add(name)

    def raw_module(self, name: str, source: str) -> None:
        (self.root / f"{name}.py").write_text(textwrap.dedent(source))
        self.modules.add(name)

    def imported(self, name: str) -> bool:
        return name in sys.modules or (self.root / f"imported_{name}").exists()

    def cleanup(self) -> None:
        for name in self.modules:
            sys.modules.pop(name, None)


def _fake_modules_loaded() -> list[str]:
    return [name for name in sys.modules if name.endswith("mode_mod")]


@pytest.fixture
def install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Installer]:
    # A fake module left by an earlier test would be reused instead of imported.
    assert _fake_modules_loaded() == []
    installer = _Installer(tmp_path, monkeypatch)
    yield installer
    installer.cleanup()
    assert _fake_modules_loaded() == []


def test_installer_cleanup_removes_fake_modules(install: _Installer) -> None:
    install.dist("fakemode", "0.1", {"fake": "fakemode_mod:FACTORY"})
    install.module("fakemode_mod", extra="FACTORY = Factory()\n")
    load_mode("fake")
    assert "fakemode_mod" in sys.modules
    install.cleanup()
    assert "fakemode_mod" not in sys.modules


def _plugin_entries() -> list[ModeEntry]:
    return [e for e in list_modes() if not e.builtin]


def _ctx(factory: ModeFactory, background: Any = None, **given: object) -> ModeContext:
    """A context consistent with the factory, as the pipeline will build it."""
    options = resolve_options(factory.options, given)
    config = PetrifyConfig(mode=factory.name, mode_options=mode_options_from(options))
    return ModeContext(INFO, config, options, background)


def test_fake_plugin_is_listed_with_distribution_and_loads(install: _Installer) -> None:
    install.dist("fakemode", "0.1", {"fake": "fakemode_mod:FACTORY"})
    install.module("fakemode_mod", extra="FACTORY = Factory()\n")
    entries = list_modes()
    fake = [e for e in entries if e.name == "fake"]
    assert fake == [ModeEntry("fake", "fakemode_mod:FACTORY", "fakemode", "0.1", False, None)]
    assert not install.imported("fakemode_mod")
    factory = load_mode("fake", entries)
    assert isinstance(factory, ModeFactory)
    assert install.imported("fakemode_mod")
    mode = build_mode(factory, _ctx(factory, strength=5))
    assert mode.name == "fake"


def test_listing_imports_nothing_and_loading_imports_only_the_selected(
    install: _Installer,
) -> None:
    install.dist("fakemode", "0.1", {"fake": "fakemode_mod:FACTORY"})
    install.dist("othermode", "0.2", {"other": "othermode_mod:FACTORY"})
    install.module("fakemode_mod", extra="FACTORY = Factory()\n")
    install.module("othermode_mod", mode="other", extra="FACTORY = Factory()\n")
    entries = list_modes()
    assert {e.name for e in entries} >= {"fake", "other"}
    assert not install.imported("fakemode_mod")
    assert not install.imported("othermode_mod")
    load_mode("fake", entries)
    assert install.imported("fakemode_mod")
    assert not install.imported("othermode_mod")


def test_plugin_raising_import_error(install: _Installer) -> None:
    install.dist("fakemode", "0.1", {"fake": "fakemode_mod:FACTORY"})
    install.raw_module("fakemode_mod", "import petrify_missing_dependency_xyz\n")
    with pytest.raises(ModeLoadError) as info:
        load_mode("fake")
    message = str(info.value)
    assert "fakemode 0.1" in message
    assert "ModuleNotFoundError" in message
    assert "petrify_missing_dependency_xyz" in message


def test_plugin_raising_arbitrary_exception_at_import(install: _Installer) -> None:
    install.dist("fakemode", "0.1", {"fake": "fakemode_mod:FACTORY"})
    install.raw_module("fakemode_mod", "raise KeyError('boom at import')\n")
    with pytest.raises(ModeLoadError, match="KeyError.*boom at import"):
        load_mode("fake")


def test_plugin_missing_attribute(install: _Installer) -> None:
    install.dist("fakemode", "0.1", {"fake": "fakemode_mod:FACTORY"})
    install.module("fakemode_mod")
    with pytest.raises(ModeLoadError, match="has no attribute 'FACTORY'"):
        load_mode("fake")


def test_plugin_attribute_raising(install: _Installer) -> None:
    install.dist("fakemode", "0.1", {"fake": "fakemode_mod:holder.FACTORY"})
    install.module(
        "fakemode_mod",
        extra="""
        class _Holder:
            @property
            def FACTORY(self):
                raise RuntimeError("bad property")
        holder = _Holder()
        """,
    )
    with pytest.raises(ModeLoadError, match="bad property"):
        load_mode("fake")


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ("FACTORY = object()\n", "not a ModeFactory"),
        ("FACTORY = Factory\n", "is a class"),
        ("FACTORY = Factory(name='other')\n", "does not match the registered name"),
        (
            "FACTORY = Factory()\nFACTORY.options = [OptionSpec('a', 'bool', True, 'a')]\n",
            "invalid options",
        ),
        (
            "FACTORY = Factory()\nFACTORY.options = (OptionSpec('a', 'bool', True, 'a'),) * 2\n",
            "more than once",
        ),
        ("FACTORY = Factory()\nFACTORY.options = ('a',)\n", "invalid options"),
        ("FACTORY = Factory()\nFACTORY.needs_background = 1\n", "needs_background"),
        ("FACTORY = Factory()\nFACTORY.roi_overlay = None\n", "roi_overlay"),
        ("FACTORY = Factory()\nFACTORY.summary = 3\n", "summary"),
        ("FACTORY = Factory()\nFACTORY.create = 3\n", "create must be callable"),
    ],
)
def test_plugin_with_invalid_factory(install: _Installer, extra: str, match: str) -> None:
    install.dist("fakemode", "0.1", {"fake": "fakemode_mod:FACTORY"})
    install.module("fakemode_mod", extra=extra)
    with pytest.raises(ModeLoadError, match=match):
        load_mode("fake")


def test_plugin_factory_attribute_raising_becomes_mode_load_error(install: _Installer) -> None:
    install.dist("fakemode", "0.1", {"fake": "fakemode_mod:FACTORY"})
    install.module(
        "fakemode_mod",
        extra="""
        class Bad(Factory):
            @property
            def summary(self):
                raise ZeroDivisionError("summary broke")
        FACTORY = Bad()
        """,
    )
    with pytest.raises(ModeLoadError, match="summary broke"):
        load_mode("fake")


def test_plugin_named_erase_is_shadowed(install: _Installer) -> None:
    install.dist("evilmode", "0.1", {"erase": "evilmode_mod:FACTORY"})
    install.module("evilmode_mod", mode="erase", extra="FACTORY = Factory()\n")
    plugin = [e for e in _plugin_entries() if e.name == "erase"]
    assert len(plugin) == 1
    assert plugin[0].distribution == "evilmode"
    assert plugin[0].problem is not None and "shadowed" in plugin[0].problem
    assert load_mode("erase") is ERASE
    assert not install.imported("evilmode_mod")


def test_two_distributions_with_same_mode_name(install: _Installer) -> None:
    install.dist("fakemode", "0.1", {"dup": "fakemode_mod:FACTORY"})
    install.dist("othermode", "0.2", {"dup": "othermode_mod:FACTORY"})
    install.module("fakemode_mod", mode="dup", extra="FACTORY = Factory()\n")
    install.module("othermode_mod", mode="dup", extra="FACTORY = Factory()\n")
    dup = [e for e in _plugin_entries() if e.name == "dup"]
    assert [(e.distribution, e.version) for e in dup] == [("fakemode", "0.1"), ("othermode", "0.2")]
    assert all(e.problem is not None and "ambiguous" in e.problem for e in dup)
    with pytest.raises(ModeLoadError, match="ambiguous"):
        load_mode("dup")
    assert not install.imported("fakemode_mod")
    assert not install.imported("othermode_mod")


def test_installed_invalid_name(install: _Installer) -> None:
    install.dist("fakemode", "0.1", {"Fake_Mode": "fakemode_mod:FACTORY"})
    install.module("fakemode_mod", extra="FACTORY = Factory()\n")
    bad = [e for e in _plugin_entries() if e.name == "Fake_Mode"]
    assert len(bad) == 1 and bad[0].problem is not None
    with pytest.raises(ModeLoadError, match="invalid mode name"):
        load_mode("Fake_Mode")


# ----------------------------------------------------------------------
# build_mode with a fake plugin
# ----------------------------------------------------------------------


def _load_fake(install: _Installer, extra: str) -> ModeFactory:
    install.dist("fakemode", "0.1", {"fake": "fakemode_mod:FACTORY"})
    install.module("fakemode_mod", extra=extra)
    return load_mode("fake")


def test_build_rejects_mode_without_explicit_detect_policy(install: _Installer) -> None:
    factory = _load_fake(
        install,
        """
        class NoPolicy:
            name = "fake"
            requires = frozenset({DetectionKind.PERSONS})

            def render(self, frame, detections, subject_present):
                return frame.image.copy()

        FACTORY = Factory(mode_cls=NoPolicy)
        """,
    )
    with pytest.raises(ModeLoadError, match="must declare detect_policy"):
        build_mode(factory, _ctx(factory))


def test_build_rejects_invalid_detect_policy(install: _Installer) -> None:
    factory = _load_fake(
        install,
        """
        class Odd(FakeMode):
            detect_policy = "some"
        FACTORY = Factory(mode_cls=Odd)
        """,
    )
    with pytest.raises(ModeLoadError, match="detect_policy must be"):
        build_mode(factory, _ctx(factory))


def test_build_rejects_create_returning_non_mode(install: _Installer) -> None:
    factory = _load_fake(install, "FACTORY = Factory(mode_cls=object)\n")
    with pytest.raises(ModeLoadError, match="not a mode"):
        build_mode(factory, _ctx(factory))


def test_build_rejects_mode_name_mismatch(install: _Installer) -> None:
    factory = _load_fake(
        install,
        """
        class Renamed(FakeMode):
            name = "other"
        FACTORY = Factory(mode_cls=Renamed)
        """,
    )
    with pytest.raises(ModeLoadError, match="named 'other'"):
        build_mode(factory, _ctx(factory))


def test_build_rejects_requires_mismatch(install: _Installer) -> None:
    factory = _load_fake(
        install,
        """
        class Wider(FakeMode):
            requires = frozenset({DetectionKind.PERSONS, DetectionKind.FACES})
        FACTORY = Factory(mode_cls=Wider)
        """,
    )
    with pytest.raises(ModeLoadError, match="requires"):
        build_mode(factory, _ctx(factory))


def test_build_rejects_mandatory_mismatch(install: _Installer) -> None:
    factory = _load_fake(
        install,
        """
        class Strict(FakeMode):
            mandatory = frozenset({DetectionKind.PERSONS})
        FACTORY = Factory(mode_cls=Strict)
        """,
    )
    with pytest.raises(ModeLoadError, match="mandatory"):
        build_mode(factory, _ctx(factory))


def test_build_rejects_mandatory_mismatch_when_mode_omits_it(install: _Installer) -> None:
    factory = _load_fake(
        install,
        """
        class Loose:
            name = "fake"
            requires = frozenset({DetectionKind.PERSONS})
            detect_policy = "all"

            def render(self, frame, detections, subject_present):
                return frame.image.copy()

        class Claims(Factory):
            def mandatory(self, options):
                return frozenset({DetectionKind.PERSONS})

        FACTORY = Claims(mode_cls=Loose)
        """,
    )
    with pytest.raises(ModeLoadError, match="mandatory"):
        build_mode(factory, _ctx(factory))


def test_build_rejects_factory_requires_not_frozenset(install: _Installer) -> None:
    factory = _load_fake(
        install,
        """
        class Listy(Factory):
            def requires(self, options):
                return ["persons"]
        FACTORY = Listy()
        """,
    )
    with pytest.raises(ModeLoadError, match="frozenset of DetectionKind"):
        build_mode(factory, _ctx(factory))


def test_build_create_value_error_propagates(install: _Installer) -> None:
    factory = _load_fake(
        install,
        """
        class Picky(Factory):
            def create(self, ctx, previous=None):
                raise ValueError("strength 5 does not suit this source")
        FACTORY = Picky()
        """,
    )
    with pytest.raises(ValueError, match="does not suit") as info:
        build_mode(factory, _ctx(factory, strength=5))
    assert not isinstance(info.value, ModeLoadError)


def test_build_create_other_exception_becomes_mode_load_error(install: _Installer) -> None:
    factory = _load_fake(
        install,
        """
        class Broken(Factory):
            def create(self, ctx, previous=None):
                raise KeyError("internal")
        FACTORY = Broken()
        """,
    )
    with pytest.raises(ModeLoadError, match="KeyError"):
        build_mode(factory, _ctx(factory))


def test_build_checks_context_options(install: _Installer) -> None:
    factory = _load_fake(install, "FACTORY = Factory()\n")
    config = PetrifyConfig(mode="fake", mode_options=(("strength", 3),))
    with pytest.raises(ValueError, match="do not match the declared options"):
        build_mode(factory, ModeContext(INFO, config, {}))
    with pytest.raises(ValueError, match="declared type"):
        build_mode(factory, ModeContext(INFO, config, {"strength": 3.0}))


def test_build_accepts_consistent_context(install: _Installer) -> None:
    factory = _load_fake(install, "FACTORY = Factory()\n")
    config = PetrifyConfig(mode="fake", mode_options=(("strength", 3),))
    assert build_mode(factory, ModeContext(INFO, config, {"strength": 3})).name == "fake"


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (PetrifyConfig(mode="other", mode_options=(("strength", 3),)), "selects the mode"),
        (PetrifyConfig(mode="fake", mode_options=(("strength", 9),)), "mode_options"),
        (PetrifyConfig(mode="fake", mode_options=(("strength", 3.0),)), "mode_options"),
        (PetrifyConfig(mode="fake"), "mode_options"),
    ],
)
def test_build_requires_config_to_match_options(
    install: _Installer, config: PetrifyConfig, match: str
) -> None:
    factory = _load_fake(install, "FACTORY = Factory()\n")
    with pytest.raises(ValueError, match=match):
        build_mode(factory, ModeContext(INFO, config, {"strength": 3}))


def test_build_rejects_background_for_mode_without_one(install: _Installer) -> None:
    import numpy as np

    factory = _load_fake(install, "FACTORY = Factory()\n")
    with pytest.raises(ValueError, match="does not use a background"):
        build_mode(factory, _ctx(factory, np.zeros((48, 64, 3), dtype=np.uint8)))


def test_build_requires_background_when_needed(install: _Installer) -> None:
    factory = _load_fake(
        install,
        """
        class Needy(Factory):
            needs_background = True

            def create(self, ctx, previous=None):
                assert ctx.background is not None  # Never reached without one.
                return FakeMode()
        FACTORY = Needy()
        """,
    )
    with pytest.raises(ValueError, match="needs a background, and the context has none"):
        build_mode(factory, _ctx(factory))


def test_build_needs_a_mode_context() -> None:
    with pytest.raises(ValueError, match="needs a ModeContext"):
        build_mode(ERASE, object())  # type: ignore[arg-type]


def test_build_rejects_create_returning_a_class(install: _Installer) -> None:
    factory = _load_fake(
        install,
        """
        class ReturnsClass(Factory):
            def create(self, ctx, previous=None):
                return FakeMode
        FACTORY = ReturnsClass()
        """,
    )
    with pytest.raises(ModeLoadError, match="not a mode instance"):
        build_mode(factory, _ctx(factory))


@pytest.mark.parametrize(
    "policy", ["", "First_hit", "ALL", "all ", "first-hit", None, 1, "_LyingStr('all')"]
)
def test_build_rejects_near_miss_detect_policy(install: _Installer, policy: object) -> None:
    literal = policy if policy == "_LyingStr('all')" else repr(policy)
    factory = _load_fake(
        install,
        f"""
        class Near(FakeMode):
            detect_policy = {literal}
        FACTORY = Factory(mode_cls=Near)
        """,
    )
    with pytest.raises(ModeLoadError, match="detect_policy must be"):
        build_mode(factory, _ctx(factory))


@pytest.mark.parametrize("policy", ["first_hit", "all"])
def test_build_accepts_valid_detect_policy(install: _Installer, policy: str) -> None:
    factory = _load_fake(
        install,
        f"""
        class Valid(FakeMode):
            detect_policy = {policy!r}
        FACTORY = Factory(mode_cls=Valid)
        """,
    )
    assert build_mode(factory, _ctx(factory)).detect_policy == policy  # type: ignore[attr-defined]


def test_build_rejects_factory_kinds_that_are_plain_strings(install: _Installer) -> None:
    # DetectionKind is a str enum, so {"persons"} == {DetectionKind.PERSONS}.
    factory = _load_fake(
        install,
        """
        class Stringy(Factory):
            def requires(self, options):
                return frozenset({"persons"})
        FACTORY = Stringy()
        """,
    )
    with pytest.raises(ModeLoadError, match="frozenset of DetectionKind"):
        build_mode(factory, _ctx(factory))


def test_build_rejects_mode_kinds_that_are_plain_strings(install: _Installer) -> None:
    factory = _load_fake(
        install,
        """
        class Stringy(FakeMode):
            requires = frozenset({"persons"})
        FACTORY = Factory(mode_cls=Stringy)
        """,
    )
    with pytest.raises(ModeLoadError, match="frozenset of DetectionKind"):
        build_mode(factory, _ctx(factory))


@pytest.mark.parametrize("attr", ["requires", "mandatory"])
def test_build_rejects_mutable_set_on_mode(install: _Installer, attr: str) -> None:
    kinds = "{DetectionKind.PERSONS}" if attr == "requires" else "set()"
    factory = _load_fake(
        install,
        f"""
        class Mutable(FakeMode):
            {attr} = {kinds}
        FACTORY = Factory(mode_cls=Mutable)
        """,
    )
    with pytest.raises(ModeLoadError, match=f"{attr} must be a frozenset"):
        build_mode(factory, _ctx(factory))


def test_build_bad_repr_of_factory_result_is_mode_load_error(install: _Installer) -> None:
    # A ValueError raised while describing a faulty result must not pass for a
    # rejected setting.
    factory = _load_fake(
        install,
        """
        class BadRepr:
            def __repr__(self):
                raise ValueError("repr broke")

        class Odd(Factory):
            def requires(self, options):
                return BadRepr()
        FACTORY = Odd()
        """,
    )
    with pytest.raises(ModeLoadError, match="BadRepr object"):
        build_mode(factory, _ctx(factory))


@pytest.mark.parametrize("method", ["requires", "mandatory"])
def test_build_value_error_from_kind_methods_propagates(install: _Installer, method: str) -> None:
    factory = _load_fake(
        install,
        f"""
        class Picky(Factory):
            def {method}(self, options):
                raise ValueError("not with these options")
        FACTORY = Picky()
        """,
    )
    with pytest.raises(ValueError, match="not with these options") as info:
        build_mode(factory, _ctx(factory))
    assert not isinstance(info.value, ModeLoadError)


def test_load_rejects_str_subclass_factory_name(install: _Installer) -> None:
    install.dist("fakemode", "0.1", {"fake": "fakemode_mod:FACTORY"})
    install.module("fakemode_mod", extra="FACTORY = Factory(name=_LyingStr('fake'))\n")
    with pytest.raises(ModeLoadError, match="does not match the registered name"):
        load_mode("fake")


def test_load_strips_spaces_in_stored_target(install: _Installer) -> None:
    install.module("fakemode_mod", extra="FACTORY = Factory()\n")
    entry = ModeEntry("fake", " fakemode_mod : FACTORY ", "fakemode", "0.1", False, None)
    assert load_mode("fake", [entry]).name == "fake"


# ----------------------------------------------------------------------
# Load-time failures always become ModeLoadError
# ----------------------------------------------------------------------


def test_plugin_calling_sys_exit_at_import(install: _Installer) -> None:
    install.dist("fakemode", "0.1", {"fake": "fakemode_mod:FACTORY"})
    install.raw_module(
        "fakemode_mod",
        """
        import argparse
        argparse.ArgumentParser().parse_args(["--no-such-flag"])
        """,
    )
    with pytest.raises(ModeLoadError, match="SystemExit"):
        load_mode("fake")


def test_plugin_exception_with_raising_str(install: _Installer) -> None:
    install.dist("fakemode", "0.1", {"fake": "fakemode_mod:FACTORY"})
    install.raw_module(
        "fakemode_mod",
        """
        class Weird(Exception):
            def __str__(self):
                raise RuntimeError("str broke")
        raise Weird()
        """,
    )
    with pytest.raises(ModeLoadError, match="Weird: <Weird object>"):
        load_mode("fake")


def test_plugin_error_message_is_capped(install: _Installer) -> None:
    install.dist("fakemode", "0.1", {"fake": "fakemode_mod:FACTORY"})
    install.raw_module("fakemode_mod", "raise RuntimeError('x' * 100_000)\n")
    with pytest.raises(ModeLoadError) as info:
        load_mode("fake")
    assert len(str(info.value)) < 1000


def test_build_create_error_message_is_capped(install: _Installer) -> None:
    factory = _load_fake(
        install,
        """
        class Loud(Factory):
            def create(self, ctx, previous=None):
                raise RuntimeError("y" * 100_000)
        FACTORY = Loud()
        """,
    )
    with pytest.raises(ModeLoadError) as info:
        build_mode(factory, _ctx(factory))
    assert len(str(info.value)) < 1000


# ----------------------------------------------------------------------
# Discovery never raises
# ----------------------------------------------------------------------


def _malformed_neighbours(install: _Installer) -> None:
    install.dist("fakemode", "0.1", {"fake": "fakemode_mod:FACTORY"})
    install.module("fakemode_mod", extra="FACTORY = Factory()\n")
    install.raw_dist("unrelated", "1.0", b"[console_scripts]\nnoequals\n")
    install.raw_dist("badbytes", "1.0", b"[console_scripts]\nx = a:b\xff\n")


def test_malformed_distributions_do_not_break_listing(
    install: _Installer, caplog: pytest.LogCaptureFixture
) -> None:
    _malformed_neighbours(install)
    with caplog.at_level("WARNING", logger="petrify.core.registry"):
        entries = list_modes()
    names = [e.name for e in entries]
    assert names[0] == "erase" and "fake" in names
    assert load_mode("fake", entries).name == "fake"
    assert load_mode("erase") is ERASE
    warned = " ".join(record.getMessage() for record in caplog.records)
    assert "unrelated" in warned and "badbytes" in warned


def test_malformed_distributions_do_not_break_default_loading(install: _Installer) -> None:
    _malformed_neighbours(install)
    assert load_mode("fake").name == "fake"


def test_builtin_loads_when_discovery_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata

    def boom() -> Any:
        raise OSError("metadata unreadable")

    monkeypatch.setattr(importlib.metadata, "distributions", boom)
    assert list_modes() == list_modes([])

    def no_listing(*args: object) -> Any:
        raise AssertionError("list_modes must not run for a built-in")

    monkeypatch.setattr(registry, "list_modes", no_listing)
    assert load_mode("erase") is ERASE


def test_discovery_keeps_first_distribution_on_path(
    install: _Installer, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = install.root / "first"
    install.dist("fake_mode", "0.2", {"fake": "fakemode_mod:FACTORY"}, root=first)
    install.dist("Fake.Mode", "0.1", {"stale": "fakemode_mod:FACTORY"})
    monkeypatch.syspath_prepend(str(first))
    plugins = [e for e in _plugin_entries() if e.name in ("fake", "stale")]
    assert [(e.name, e.version) for e in plugins] == [("fake", "0.2")]


# ----------------------------------------------------------------------
# Listing details
# ----------------------------------------------------------------------


def test_extras_syntax_is_accepted() -> None:
    entry = list_modes([_ep("blur", "blurmod:FACTORY [gpu]")])[1]
    assert entry.problem is None
    assert entry.target == "blurmod:FACTORY"


def test_installed_extras_syntax_is_accepted(install: _Installer) -> None:
    install.dist("fakemode", "0.1", {"fake": "fakemode_mod:FACTORY [gpu]"})
    install.module("fakemode_mod", extra="FACTORY = Factory()\n")
    assert load_mode("fake").name == "fake"


def test_problem_strings_are_capped() -> None:
    entries = list_modes([_ep("x" * 200_000), _ep("blur", "y" * 200_000)])
    assert all(e.problem is not None and len(e.problem) < 300 for e in entries[1:])


def test_ambiguity_names_the_targets() -> None:
    entries = list_modes([_ep("blur", "one:F"), _ep("blur", "two:F")])
    problem = entries[1].problem
    assert problem is not None and "'one:F'" in problem and "'two:F'" in problem


def test_installed_ambiguity_within_one_distribution(install: _Installer) -> None:
    install.raw_dist(
        "fakemode", "0.1", f"[{ENTRY_POINT_GROUP}]\ndup = one:F\ndup = two:F\n".encode()
    )
    dup = [e for e in _plugin_entries() if e.name == "dup"]
    assert len(dup) == 2
    assert all(
        e.problem is not None and "fakemode 0.1 as 'one:F'" in e.problem for e in dup
    )


# ----------------------------------------------------------------------
# Guards in the option helpers and the context
# ----------------------------------------------------------------------


def test_resolve_option_text_strips_key() -> None:
    assert resolve_option_text(SPECS, [" count =4"])["count"] == 4


def test_resolve_option_text_rejects_non_text() -> None:
    with pytest.raises(ValueError, match="KEY=VALUE"):
        resolve_option_text(SPECS, [3])  # type: ignore[list-item]


def test_resolve_options_requires_mapping() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        resolve_options(SPECS, [("count", 3)])  # type: ignore[arg-type]


def test_mode_context_requires_mapping() -> None:
    with pytest.raises(TypeError, match="must be a mapping"):
        ModeContext(INFO, PetrifyConfig(), [("a", 1)])  # type: ignore[arg-type]


@pytest.mark.parametrize("key", [1, _LyingStr("a"), None])
def test_mode_context_rejects_non_str_keys(key: Any) -> None:
    with pytest.raises(TypeError, match="keys must be strings"):
        ModeContext(INFO, PetrifyConfig(), {key: 1, "b": 2})


@pytest.mark.parametrize("text", ["1_000", "١٢", "inf", "-inf", "nan", "0x10", "1e"])
def test_float_parse_is_ascii_strict(text: str) -> None:
    spec = OptionSpec("level", "float", 0.0, "level")
    with pytest.raises(ValueError, match="option 'level'"):
        spec.parse(text)


@pytest.mark.parametrize(("text", "expected"), [("1e2", 100.0), (".5", 0.5), ("-2.", -2.0)])
def test_float_parse_accepts_plain_forms(text: str, expected: float) -> None:
    assert OptionSpec("level", "float", 0.0, "level").parse(text) == expected


def test_int_parse_of_huge_text_names_the_option() -> None:
    with pytest.raises(ValueError, match="option 'count'"):
        INT.parse("9" * 10_000)


def test_float_parse_of_huge_text_names_the_option() -> None:
    with pytest.raises(ValueError, match="option 'ratio'"):
        FLOAT.parse("9" * 10_000)
