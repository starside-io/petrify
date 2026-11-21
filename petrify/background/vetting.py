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

Cost: the filter runs on the thread that calls the model's ``update``, once per
candidate sample for ``RunningBackground`` and once per frame until calibration
completes for ``CalibratedBackground``.
"""

from __future__ import annotations

from collections.abc import Callable
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
