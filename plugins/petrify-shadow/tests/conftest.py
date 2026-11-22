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

"""Test harness for the petrify-shadow plugin, which runs without installing it.

This directory has no ``__init__.py`` on purpose: the root ``tests/`` directory is
a package named ``tests``, and a second package of that name would shadow it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from petrify.core.registry import ENTRY_POINT_GROUP

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = PLUGIN_ROOT / "pyproject.toml"

# Put the source tree on the path at import time, so unit tests can import
# petrify_shadow directly. The src layout keeps an uninstalled copy from being
# imported by accident anywhere else.
_SRC = str(PLUGIN_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "models: needs the model files in PETRIFY_MODEL_DIR (skipped without them)",
    )


@dataclass(frozen=True)
class ShadowEntryPoint:
    """What the ``shadow_entry_point`` fixture registered."""

    distribution: str
    version: str
    name: str  # The entry point name, the mode name.
    target: str  # "module:attr"
    dist_info: Path


def read_pyproject() -> dict[str, Any]:
    """Return the plugin's own pyproject.toml, parsed."""
    # tomllib is in the standard library from Python 3.11 only; petrify supports 3.10.
    tomllib = pytest.importorskip("tomllib")
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


@pytest.fixture
def shadow_entry_point(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ShadowEntryPoint:
    """Register the plugin's entry point through a hand-written dist-info directory.

    Python finds an entry point in a ``*.dist-info`` directory on ``sys.path`` the
    same way it finds one of an installed distribution, so the registry goes
    through its real discovery path. Name, version and entry point come from the
    plugin's pyproject.toml, so the test cannot drift from what would be installed.
    """
    project = read_pyproject()["project"]
    distribution = project["name"]
    version = project["version"]
    ((name, target),) = project["entry-points"][ENTRY_POINT_GROUP].items()
    info = tmp_path / f"{distribution.replace('-', '_')}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n"
    )
    (info / "entry_points.txt").write_text(f"[{ENTRY_POINT_GROUP}]\n{name} = {target}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    return ShadowEntryPoint(distribution, version, name, target, info)
