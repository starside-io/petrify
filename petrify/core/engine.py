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

"""Streaming engine primitives; for now only the lookahead buffer used for gap fill."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Decision(Generic[T]):
    """Final per-item verdict emitted by :class:`LookaheadBuffer`."""

    item: T
    index: int
    """0-based position in the stream."""
    raw: bool
    """The value pushed for this item."""
    filled: bool
    """``raw`` after gap fill: any raw True within +/- ``lookahead`` positions."""
    final: bool
    """``filled``, or ``index < cold_start_frames``."""


class LookaheadBuffer(Generic[T]):
    """Delay items by ``lookahead`` positions so gap fill can be decided in one pass.

    Item ``i`` is emitted by the push of item ``i + lookahead`` (or by :meth:`flush`
    at end of stream). The result matches ``_fill_gaps`` followed by
    ``_freeze_cold_start`` from the file pipeline, index for index.

    After :meth:`flush` the buffer is reset and may be reused for a new stream,
    whose first item gets index 0.
    """

    def __init__(self, lookahead: int, cold_start_frames: int = 0) -> None:
        if lookahead < 0:
            raise ValueError(f"lookahead must be >= 0, got {lookahead}")
        if cold_start_frames < 0:
            raise ValueError(f"cold_start_frames must be >= 0, got {cold_start_frames}")
        self._lookahead = lookahead
        self._cold_start = cold_start_frames
        self._pending: deque[tuple[T, bool]] = deque()
        self._next_index = 0
        self._last_true: int | None = None

    @property
    def lookahead(self) -> int:
        """Number of items of delay (equal to the gap fill)."""
        return self._lookahead

    @property
    def cold_start_frames(self) -> int:
        """Number of leading items forced to ``final=True``."""
        return self._cold_start

    @property
    def pending(self) -> int:
        """Number of items pushed but not yet emitted."""
        return len(self._pending)

    def push(self, item: T, raw: bool) -> list[Decision[T]]:
        """Add the next item; return the decisions this push made final, in order."""
        index = self._next_index
        self._next_index += 1
        if raw:
            self._last_true = index
        self._pending.append((item, raw))
        if len(self._pending) > self._lookahead:
            return [self._emit()]
        return []

    def flush(self) -> list[Decision[T]]:
        """Emit every pending item (end of stream) and reset for a new stream."""
        out = [self._emit() for _ in range(len(self._pending))]
        self._next_index = 0
        self._last_true = None
        return out

    def _emit(self) -> Decision[T]:
        """Pop and decide the oldest pending item.

        Every raw value up to ``index + lookahead`` (or end of stream) has been seen,
        and ``_last_true`` is the most recent True among them, so the item is filled
        exactly when that True is no more than ``lookahead`` positions before it.
        """
        item, raw = self._pending.popleft()
        index = self._next_index - 1 - len(self._pending)
        filled = self._last_true is not None and self._last_true >= index - self._lookahead
        final = filled or index < self._cold_start
        return Decision(item=item, index=index, raw=raw, filled=filled, final=final)
