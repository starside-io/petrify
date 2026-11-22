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

from collections.abc import Mapping

from petrify.core.config import OptionValue
from petrify.core.detections import DetectionKind
from petrify.core.plugin import Mode, ModeContext, OptionSpec
from petrify_shadow.mode import BASE_REQUIRES, ShadowMode


class ShadowFactory:
    """The :class:`~petrify.core.plugin.ModeFactory` of the shadow mode (see :data:`SHADOW`)."""

    name = "shadow"
    summary = "Obscure faces and bodies while a subject is present."
    options: tuple[OptionSpec, ...] = ()
    needs_background = False
    # A region mode: the live ROI overlay would paste a face inside the timestamp
    # region back over its cover.
    roi_overlay = False

    def requires(self, options: Mapping[str, OptionValue]) -> frozenset[DetectionKind]:
        """Return the presence signals every shadow mode requires, motion and persons."""
        return BASE_REQUIRES

    def mandatory(self, options: Mapping[str, OptionValue]) -> frozenset[DetectionKind]:
        """Return the empty set: no style is selectable yet, so no kind is mandatory."""
        return frozenset()

    def weakened(self, options: Mapping[str, OptionValue]) -> tuple[str, ...]:
        """Return nothing: there is no option yet that could reduce protection."""
        return ()

    def create(self, ctx: ModeContext, previous: Mode | None = None) -> Mode:
        """Return a :class:`ShadowMode`, or ``previous`` when it already is one.

        With no options there is nothing that could differ between two shadow
        modes, so any previous shadow mode is reused as it is.
        """
        if isinstance(previous, ShadowMode):
            return previous
        return ShadowMode()


SHADOW = ShadowFactory()
"""The shadow mode factory, registered as ``shadow``."""
