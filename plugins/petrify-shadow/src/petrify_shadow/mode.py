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

"""The shadow mode: covers faces and bodies while a subject is present.

Each frame is decided in two steps. :func:`plan` is pure: it reads the
detections and returns a :class:`Plan` that says whether the frame passes
through, turns full black, or gets the selected styles drawn on its regions.
:meth:`ShadowMode.render` then carries the plan out on a copy of the frame.

The decision, with the primary group being the body group (silhouette,
silhouette_color, bodybox) when any body style is selected and the face group
otherwise:

- no subject: the input frame itself;
- a subject, and the primary group has no trusted region (no own region and no
  orphan person box that is fresh enough, see :func:`plan`): a full black
  frame, since the subject is not localized;
- otherwise: the styles drawn on their regions, orphan person boxes filled black.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import ClassVar, Literal

import numpy as np

from petrify.core.detections import Box, DetectionKind, Detections, Face
from petrify.core.frame import Frame
from petrify.core.plugin import DetectPolicy
from petrify_shadow.options import (
    BODY_STYLES,
    FACE_STYLES,
    MASK_STYLES,
    STYLE_NAMES,
    ShadowParams,
)
from petrify_shadow.regions import (
    HELD_REGION_FRAMES,
    PreparedMask,
    Primary,
    Rect,
    dilate_px,
    held_growth,
    orphan_indices,
    pad_face,
    pad_person,
    prepare_mask,
)
from petrify_shadow.styles.body import (
    apply_bodybox,
    apply_outline,
    apply_silhouette,
    outline_thickness,
)
from petrify_shadow.styles.face import (
    apply_eyebar,
    apply_fill,
    apply_gaussian,
    apply_noise,
    apply_pixelate,
)

BASE_REQUIRES: frozenset[DetectionKind] = frozenset(
    {DetectionKind.MOTION, DetectionKind.PERSONS}
)
"""Presence signals the mode always requires, so a missed face turns black, not visible."""


FACE_KINDS: frozenset[DetectionKind] = frozenset(
    {DetectionKind.FACES, DetectionKind.LANDMARKS}
)
"""What any face style adds. Landmarks always come with faces, so the tracker holds them too."""

BLACK: tuple[int, int, int] = (0, 0, 0)
"""The fill of the silhouette style and of orphan person boxes, whatever the color option."""

SILHOUETTE_STYLES: frozenset[str] = frozenset({"silhouette", "silhouette_color"})
"""The mask styles that cover the body. ``outline`` draws only an edge and is never primary."""

FACE_COVER_ORDER: tuple[str, ...] = ("gaussian", "pixelate", "noise", "facebox")
"""The face covers in drawing order; the eyebar is drawn after all of them."""

MIN_MASK_SHARE = Fraction(1, 1000)
"""Smallest share of the frame's pixels a mask must mark to count as a body region.

A few stray segmenter pixels would otherwise make a mask-primary frame "draw"
with only those pixels covered. A smaller mask is still drawn when the frame is
drawn for another region.
"""

Action = Literal["pass", "black", "draw"]


def requires_for(styles: frozenset[str]) -> frozenset[DetectionKind]:
    """Return the kinds a mode with ``styles`` requires: the base plus each group's kinds."""
    kinds = set(BASE_REQUIRES)
    if styles & FACE_STYLES:
        kinds |= FACE_KINDS
    if styles & MASK_STYLES:
        kinds.add(DetectionKind.MASK)
    return frozenset(kinds)


def mandatory_for(styles: frozenset[str]) -> frozenset[DetectionKind]:
    """Return the kinds a mode with ``styles`` cannot run without.

    Motion and persons stay optional in general: under the engine's lenient rule
    only the presence signals ``--detector`` builds run. Person boxes become
    mandatory only for bodybox, which has nothing to draw on without them.
    """
    kinds: set[DetectionKind] = set()
    if styles & FACE_STYLES:
        kinds |= FACE_KINDS
    if styles & MASK_STYLES:
        kinds.add(DetectionKind.MASK)
    if "bodybox" in styles:
        kinds.add(DetectionKind.PERSONS)
    return frozenset(kinds)


def primary_of(styles: frozenset[str]) -> Primary:
    """Return the orphan primary of ``styles``: ``"bodybox"``, ``"mask"`` or ``"face"``.

    The body group is primary when any body style is selected. Within it,
    bodybox decides the orphans (it covers every person box, so there are none);
    otherwise the silhouette does (``"mask"``). Without a body style the face
    group is primary. ``outline`` never counts.
    """
    if "bodybox" in styles:
        return "bodybox"
    if styles & BODY_STYLES:
        return "mask"
    return "face"


@dataclass(frozen=True)
class Plan:
    """What :meth:`ShadowMode.render` does with one frame.

    For ``"pass"`` and ``"black"`` every region field is empty. For ``"draw"``,
    every rect is grown by the held count of its box, and the fields hold every
    region of the selected styles, trusted or not (see :func:`plan`):

    - ``faces`` and ``face_rects``: aligned tuples, each face with a non-empty
      padded cover rect, only when a face style is selected;
    - ``mask``: the cropped and dilated mask when a mask style is selected and
      the mask has a True pixel, else None;
    - ``person_rects``: the padded person boxes when bodybox is selected;
    - ``orphans``: padded person boxes the primary group does not cover, filled
      black.
    """

    action: Action
    faces: tuple[Face, ...] = ()
    face_rects: tuple[Rect, ...] = ()
    mask: PreparedMask | None = None
    person_rects: tuple[Rect, ...] = ()
    orphans: tuple[Rect, ...] = ()


_PASS = Plan("pass")
_BLACK = Plan("black")


def _checked_styles(styles: object) -> frozenset[str]:
    """Return ``styles`` if it is a style set the mode can draw, or raise ValueError.

    :func:`~petrify_shadow.options.parse_params` already refuses these; checking
    again keeps a hand-built :class:`ShadowParams` from reaching a frame with no
    primary style, where the decision table has no meaning.
    """
    if not isinstance(styles, frozenset) or not styles:
        raise ValueError(f"styles must be a non-empty frozenset, got {styles!r}")
    unknown = styles - set(STYLE_NAMES)
    if unknown:
        raise ValueError(f"unknown styles {sorted(unknown)}")
    if styles == {"outline"}:
        raise ValueError("the outline style does not hide anyone on its own")
    return styles


def _frame_size(shape: Sequence[int]) -> tuple[int, int]:
    if len(shape) < 2:
        raise ValueError(f"shape must start with the frame height and width, got {shape!r}")
    return int(shape[0]), int(shape[1])


def _held_counts(held: Sequence[int] | None, n: int, what: str) -> tuple[int, ...]:
    if held is None:
        return (0,) * n
    counts = tuple(held)
    if len(counts) != n:
        raise ValueError(f"held counts must give one count per {what}: {len(counts)} for {n}")
    for count in counts:
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
            raise ValueError(f"held counts must be integers, got {count!r}")
        if count < 0:
            raise ValueError(f"held counts must not be negative, got {count}")
    return counts


def _trusted(count: int) -> bool:
    """True when a box held ``count`` frames still counts as a region."""
    return count < HELD_REGION_FRAMES


def _mask_is_region(mask: np.ndarray, h: int, w: int) -> bool:
    """True when ``mask`` marks at least :data:`MIN_MASK_SHARE` of the frame's pixels."""
    marked = int(np.count_nonzero(mask))
    return marked * MIN_MASK_SHARE.denominator >= MIN_MASK_SHARE.numerator * h * w


def plan(
    params: ShadowParams,
    detections: Detections,
    subject_present: bool,
    shape: Sequence[int],
    held: Sequence[int] | None = None,
    held_persons: Sequence[int] | None = None,
) -> Plan:
    """Return what to do with a frame of ``shape``; pure, draws nothing.

    ``held`` gives, per entry of ``detections.faces`` and in the same order, the
    number of consecutive earlier frames the box was seen unchanged (see
    :class:`ShadowMode`); ``held_persons`` does the same for
    ``detections.persons``. None means 0 for every box. Every face rect, bodybox
    rect and orphan rect is padded with ``grow=held_growth(count)`` of its box.

    Rules, in order:

    1. A None field counts as empty, and zero-area boxes yield no region.
    2. The primary group is the body group when any body style is selected,
       else the face group (:func:`primary_of`).
    3. Orphans come from :func:`~petrify_shadow.regions.orphan_indices` for the
       primary group, only when ``detections.persons`` is not None. Only faces
       with a held count of 0 may clear a person box: a held face no longer
       proves where the head is, so its person box is filled as an orphan while
       the held face still gets its own grown cover.
    4. A held box is trusted while its count is below
       :data:`~petrify_shadow.regions.HELD_REGION_FRAMES`, the count at which its
       growth reaches the cap. From then on it is drawn but is not a region.
    5. The primary group has regions when it has a trusted orphan, or a trusted
       own region: a face rect for the face group; for the body group a bodybox
       rect, or a prepared mask (with silhouette or silhouette_color selected)
       whose raw mask marks at least :data:`MIN_MASK_SHARE` of the frame.
    6. ``subject_present`` False, and the detections show no subject either:
       ``"pass"``. A subject, and the primary group has no regions: ``"black"``.
       Otherwise ``"draw"``.
    7. In ``"draw"``, every selected group draws everything it has, trusted or
       not: a secondary group with no region draws nothing.

    ``subject_present`` is the engine's decision after gap fill, which is never
    False while the frame's own detections show a subject. A caller that passes
    False anyway is not trusted: the frame is decided as if it were True, which
    only covers more.

    Left uncovered by design: when at least one trusted region exists, signals
    without a region (motion) are taken to belong to it. So a second person whom
    no detector localizes is shown, and with a mask primary a real mask
    anywhere makes the frame "draw" while a person the segmenter misses is shown
    unless a person box makes that person an orphan. The mask share minimum only
    stops a few stray pixels from doing this; it does not solve it.

    Raises ValueError for a style set the mode cannot draw, held counts of the
    wrong length or not non-negative integers, a mask whose shape is not the
    frame's, or a malformed box.
    """
    styles = _checked_styles(params.styles)
    h, w = _frame_size(shape)
    faces = detections.faces or ()
    persons = detections.persons
    face_counts = _held_counts(held, len(faces), "face")
    person_counts = _held_counts(held_persons, len(persons or ()), "person box")
    mask = detections.mask
    if mask is not None and tuple(mask.shape) != (h, w):
        raise ValueError(f"mask shape {mask.shape} does not match the frame {(h, w)}")
    primary = primary_of(styles)

    # Regions are computed before the decision, so a malformed detection raises
    # (and the engine fails the frame closed) on every frame, not only on drawn ones.
    covered: list[Face] = []
    rects: list[Rect] = []
    trusted_face = False
    if styles & FACE_STYLES:
        for face, count in zip(faces, face_counts, strict=True):
            rect = pad_face(face.box, params.padding, h, w, grow=held_growth(count))
            if rect is not None:
                covered.append(face)
                rects.append(rect)
                trusted_face = trusted_face or _trusted(count)
    prepared = prepare_mask(mask, dilate_px(h, params.dilate)) if styles & MASK_STYLES else None
    person_rects: list[Rect] = []
    trusted_person = False
    if "bodybox" in styles:
        for box, count in zip(persons or (), person_counts, strict=True):
            rect = pad_person(box, h, w, grow=held_growth(count))
            if rect is not None:
                person_rects.append(rect)
                trusted_person = trusted_person or _trusted(count)
    orphans: list[Rect] = []
    trusted_orphan = False
    if persons is not None:
        fresh = [face for face, count in zip(faces, face_counts, strict=True) if count == 0]
        for i in orphan_indices(persons, fresh, mask, primary, h, w):
            rect = pad_person(persons[i], h, w, grow=held_growth(person_counts[i]))
            if rect is not None:
                orphans.append(rect)
                trusted_orphan = trusted_orphan or _trusted(person_counts[i])

    if primary == "face":
        has_regions = trusted_face
    else:
        has_regions = trusted_person or (
            prepared is not None
            and bool(styles & SILHOUETTE_STYLES)
            and _mask_is_region(mask, h, w)  # type: ignore[arg-type]
        )
    has_regions = has_regions or trusted_orphan

    present = bool(subject_present) or detections.subject_present
    if not present:
        # No subject in the detections means no face, person box or mask pixel,
        # so there is no region to draw either.
        return _PASS
    if not has_regions:
        return _BLACK
    return Plan(
        action="draw",
        faces=tuple(covered),
        face_rects=tuple(rects),
        mask=prepared,
        person_rects=tuple(person_rects),
        orphans=tuple(orphans),
    )


def _checked_image(image: object) -> np.ndarray:
    """Return ``image`` if it is an HxWx3 uint8 array, else raise ValueError (fail closed)."""
    if not isinstance(image, np.ndarray):
        raise ValueError(f"frame image must be a numpy array, got {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(
            f"frame image must be an HxWx3 uint8 array, got shape {image.shape} "
            f"dtype {image.dtype}"
        )
    return image


def draw(out: np.ndarray, p: Plan, params: ShadowParams, rng: np.random.Generator) -> None:
    """Draw the ``"draw"`` plan ``p`` into ``out``, in the fixed style order.

    The order is the one of :data:`~petrify_shadow.options.APPLY_ORDER`, with the
    orphan fills after the body fills: silhouette (black), silhouette_color,
    bodybox, orphan fills (black), outline, then gaussian, pixelate, noise and
    facebox each over every face rect, then the eyebar. A later style wins where
    regions overlap.

    Each face cover runs over all faces before the next one starts, so an opaque
    cover (noise, facebox) is never followed by a blur or pixelation of another
    face that could reach into it (pixelate grows its rect to the block grid).

    Every style reads and writes ``out`` only, which must be a writable
    C-contiguous HxWx3 uint8 array.
    """
    styles = params.styles
    color = params.color
    pm = p.mask
    if pm is not None:
        if "silhouette" in styles:
            apply_silhouette(out, pm, BLACK)
        if "silhouette_color" in styles:
            apply_silhouette(out, pm, color)
    if "bodybox" in styles:
        apply_bodybox(out, p.person_rects, color)
    if p.orphans:
        apply_bodybox(out, p.orphans, BLACK)
    if pm is not None and "outline" in styles:
        apply_outline(out, pm, color, outline_thickness(out.shape[0]))
    for style in FACE_COVER_ORDER:
        if style not in styles:
            continue
        for rect in p.face_rects:
            if style == "gaussian":
                apply_gaussian(out, rect, params.strength)
            elif style == "pixelate":
                apply_pixelate(out, rect, params.strength)
            elif style == "noise":
                apply_noise(out, rect, rng)
            else:
                apply_fill(out, rect, color)
    if "eyebar" in styles:
        for face, rect in zip(p.faces, p.face_rects, strict=True):
            apply_eyebar(out, face, rect, color)


BoxKey = tuple[int, int, int, int]


def _advance(memory: dict[BoxKey, int], boxes: Sequence[Box]) -> tuple[dict[BoxKey, int], tuple[int, ...]]:
    """Return the new memory and the held count of each box, from the previous frame's memory."""
    keys = [(box.x, box.y, box.w, box.h) for box in boxes]
    counts = tuple(memory[key] + 1 if key in memory else 0 for key in keys)
    return dict(zip(keys, counts, strict=True)), counts


class ShadowMode:
    """Cover the selected regions on frames with a subject, as decided by :func:`plan`.

    Held boxes. With ``--track`` a face or person the detector misses keeps its
    last box unchanged, so a moving subject can leave it. The mode therefore
    remembers the face boxes and the person boxes of the previous frame it
    rendered, each keyed by ``(x, y, w, h)``. A box equal to one of the same kind
    on the previous frame is held: its count is that box's count plus one, and
    its padded rect is grown by :func:`~petrify_shadow.regions.held_growth` (10
    percent per held frame, at most 2 times). Any other box has count 0. Each
    memory holds only the previous frame's boxes, so a box that changes resets
    its count, and a frame without faces (or without person boxes) clears that
    memory. Both are updated on every rendered frame, whatever the action.

    How held boxes are trusted (see :func:`plan`): a held face never clears its
    person box from the orphan rule, and a held box whose count has reached
    :data:`~petrify_shadow.regions.HELD_REGION_FRAMES` (the growth cap) is drawn
    but does not count as a region, so a frame with nothing else turns black.

    Limits, accepted: a box that truly stays still is also seen as held, grows up
    to the cap and then stops counting as a region, which only covers more. The
    score is not part of the key, so a fresh detection with the exact same box as
    the previous frame counts as held too, with the same effect. A held cover
    below the cap can still miss a subject that moved faster than it grows.

    The memory belongs to the instance. A mode built to replace another one,
    after a live option change, takes it over through ``held_from``, so a held
    cover does not shrink on the switch frame.
    """

    name: ClassVar[str] = "shadow"
    # A region mode needs every detector on every frame; "first_hit" would hand it
    # incomplete regions.
    detect_policy: ClassVar[DetectPolicy] = "all"

    def __init__(
        self,
        params: ShadowParams,
        *,
        rng: np.random.Generator | None = None,
        held_from: ShadowMode | None = None,
    ) -> None:
        self.params = params
        # Per instance, since they follow the selected styles; the registry checks
        # that they equal what the factory reported for the same options.
        self.requires: frozenset[DetectionKind] = requires_for(params.styles)
        self.mandatory: frozenset[DetectionKind] = mandatory_for(params.styles)
        # No seed: the noise style must not be reproducible, so the generator draws
        # its seed from OS entropy. Only tests pass a seeded one.
        self._rng = rng if rng is not None else np.random.default_rng()
        self._held_faces: dict[BoxKey, int] = {}
        self._held_persons: dict[BoxKey, int] = {}
        if held_from is not None:
            if not isinstance(held_from, ShadowMode):
                raise TypeError(f"held_from must be a ShadowMode, got {type(held_from).__name__}")
            # Copies, so the two instances never share state.
            self._held_faces = dict(held_from._held_faces)
            self._held_persons = dict(held_from._held_persons)

    def render(self, frame: Frame, detections: Detections, subject_present: bool) -> np.ndarray:
        """Return ``frame.image`` itself, a new black image, or a covered copy.

        Raises ValueError for an image that is not HxWx3 uint8, and for anything
        :func:`plan` refuses; the engine then fails the frame closed.
        """
        image = _checked_image(frame.image)
        # Counted before planning, so a frame that then fails still advances the
        # memory: a held box keeps growing rather than starting over.
        self._held_faces, face_counts = _advance(
            self._held_faces, [face.box for face in detections.faces or ()]
        )
        self._held_persons, person_counts = _advance(self._held_persons, detections.persons or ())
        p = plan(self.params, detections, subject_present, image.shape, face_counts, person_counts)
        if p.action == "pass":
            return frame.image
        if p.action == "black":
            # np.zeros, not zeros_like: zeros_like keeps a Fortran-ordered input's
            # layout, and every output of this mode is C-contiguous.
            return np.zeros(image.shape, dtype=np.uint8)
        # A new C-contiguous array: the styles draw in place only into such an
        # array, and the engine fails the frame closed when the output shares
        # memory with the input.
        out = np.array(image, dtype=np.uint8, order="C", copy=True)
        draw(out, p, self.params, self._rng)
        return out
