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

"""Sample filters that keep people out of an online background model.

:func:`person_free` turns a person detector into a ``sample_filter`` for
:class:`~petrify.background.RunningBackground` and
:class:`~petrify.background.CalibratedBackground`: a candidate sample is accepted
only when the detector finds no person in it.

What this guarantees: a sample on which the detector fires is never stored, and a
frame the detector cannot analyze (for example one smaller than the HOG window,
for which :meth:`~petrify.detectors.person_hog.PersonDetector.is_person` raises
``ValueError``) is rejected too. The exception is not handled here: it reaches the
model, which rejects the sample and logs the first such error as a warning.

What it does not guarantee: that no person ends up in the background. The HOG
person detector finds upright, mostly full-body pedestrians. It often misses a
seated person framed from the chest up (the typical webcam view), partial bodies,
and people far from the camera, and every miss is accepted as a clean sample. A
stronger detector is planned.

:func:`any_person` combines several checks into one: a sample is flagged as
containing a person as soon as any check fires, so a person missed by one
detector can still be caught by another.

Cost: the filter runs on the thread that calls the model's ``update``, once per
candidate sample for ``RunningBackground`` and once per frame until calibration
completes for ``CalibratedBackground``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np


class PersonCheck(Protocol):
    """Anything with ``is_person(image) -> bool``, such as ``PersonDetector``."""

    def is_person(self, frame: np.ndarray) -> bool: ...


def person_free(detector: PersonCheck) -> Callable[[np.ndarray], bool]:
    """Return a sample filter that is True when ``detector`` finds no person.

    Exceptions raised by ``detector.is_person`` propagate, so that the model
    rejects the sample and reports the error.
    """

    def _filter(image: np.ndarray) -> bool:
        return not detector.is_person(image)

    return _filter


class _AnyPerson:
    """A :class:`PersonCheck` that fires when any of its checks fires."""

    def __init__(self, checks: Sequence[PersonCheck]) -> None:
        self._checks: tuple[PersonCheck, ...] = tuple(checks)
        if not self._checks:
            raise ValueError("any_person needs at least one check")

    def is_person(self, frame: np.ndarray) -> bool:
        for check in self._checks:
            if check.is_person(frame):
                return True
        return False

    def __repr__(self) -> str:
        inner = ", ".join(repr(check) for check in self._checks)
        return f"any_person([{inner}])"


def any_person(checks: Sequence[PersonCheck]) -> PersonCheck:
    """Return a check that finds a person when any of ``checks`` finds one.

    The checks run in the given order and stop at the first that returns True;
    the result is False only when every check returned False. An exception raised
    by a check before any check returned True propagates, so that a sample filter
    built on the result rejects the sample. ``checks`` is copied, so later changes
    to the caller's sequence have no effect.

    Raises ``ValueError`` when ``checks`` is empty, because a combinator with no
    checks would approve every sample.
    """
    return _AnyPerson(checks)
