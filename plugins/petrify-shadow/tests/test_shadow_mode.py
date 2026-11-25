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

"""The shadow mode: the per-frame decision table and the rendering on synthetic frames."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import logging
import time
from collections.abc import Sequence
from fractions import Fraction

import cv2
import numpy as np
import pytest
from petrify_shadow import SHADOW
from petrify_shadow.factory import WARN_MASK_WITHOUT_PERSONS
from petrify_shadow.mode import MIN_MASK_SHARE, Plan, ShadowMode, draw, plan, primary_of
from petrify_shadow.options import APPLY_ORDER, COLORS, OPTIONS, STYLE_NAMES, parse_params
from petrify_shadow.regions import (
    HELD_GROWTH_CAP,
    HELD_REGION_FRAMES,
    Rect,
    dilate_px,
    head_rect,
    held_growth,
    pad_face,
    pad_person,
    prepare_mask,
)
from petrify_shadow.styles.body import outline_thickness
from petrify_shadow.styles.face import _pixelate_block, eyebar_polygon

from petrify.core import Frame, PetrifyConfig
from petrify.core.config import EngineConfig, mode_options_from
from petrify.core.detections import Box, DetectionKind, Detections, Face
from petrify.core.engine import Engine, Output
from petrify.core.frame import SourceInfo
from petrify.core.plugin import ModeContext, resolve_options
from petrify.detectors.tracker import TrackerRefiner

H, W = 120, 160
SHAPE = (H, W, 3)
INFO = SourceInfo(fps=10.0, width=W, height=H, frame_count=None, seekable=False,
                  live=True, uri="test")
BLACK = COLORS["black"]
RED = COLORS["red"]

FACE = Face(Box(62, 24, 16, 18), landmarks=((66.0, 30.0), (74.0, 30.0)))
"""A face whose center (70, 33) lies in the top third of PERSON_FACE."""
ZERO_FACE = Face(Box(5, 100, 0, 6))
PERSON_FACE = Box(50, 10, 40, 100)
"""A person box holding FACE's center in its top third."""
PERSON_EMPTY = Box(110, 10, 30, 90)
"""A person box no face clears and no mask covers."""
ZERO_PERSON = Box(150, 0, 0, 0)


def _populated_mask() -> np.ndarray:
    # Covers about 80 percent of PERSON_FACE and its head zone, and nothing of
    # PERSON_EMPTY, so only PERSON_EMPTY is a mask orphan.
    mask = np.zeros((H, W), bool)
    mask[15:105, 52:88] = True
    mask.setflags(write=False)
    return mask


def _speck_mask(pixels: int = 1) -> np.ndarray:
    # Stray segmenter pixels in the top left corner, away from every person box.
    mask = np.zeros((H, W), bool)
    rows, cols = divmod(pixels, 8)
    mask[:rows, :8] = True
    mask[rows, :cols] = True
    mask.setflags(write=False)
    return mask


FACES = {"none": None, "empty": (), "one": (FACE,), "zero": (ZERO_FACE,)}
MASKS = {
    "none": None,
    "false": np.zeros((H, W), bool),
    "speck": _speck_mask(),
    "populated": _populated_mask(),
}
CAP = HELD_REGION_FRAMES
PERSONS = {
    "none": None,
    "empty": (),
    "zero": (ZERO_PERSON,),
    "with_face": (PERSON_FACE,),
    "without_face": (PERSON_EMPTY,),
    "both": (PERSON_FACE, PERSON_EMPTY),
}
GROUPS = {
    "face": ("facebox",),
    "mask": ("silhouette",),
    "bodybox": ("bodybox",),
    "face_mask": ("facebox", "silhouette"),
    "face_bodybox": ("facebox", "bodybox"),
    "outline_face": ("outline", "facebox"),
    "outline_bodybox": ("outline", "bodybox"),
    "all": STYLE_NAMES,
}


def _params(styles: Sequence[str], **options: object) -> object:
    return parse_params({"styles": tuple(styles), **options})


def _detections(faces: str, mask: str, persons: str) -> Detections:
    return Detections(faces=FACES[faces], mask=MASKS[mask], persons=PERSONS[persons])


def _image(seed: int = 0, shape: tuple[int, ...] = SHAPE) -> np.ndarray:
    # Values start at 1, so a black fill is always a visible change; read-only,
    # so any write to the input raises.
    image = np.random.default_rng(seed).integers(1, 256, shape, dtype=np.uint8)
    image.setflags(write=False)
    return image


def _frame(image: np.ndarray, index: int = 0) -> Frame:
    return Frame(image=image, index=index, timestamp=index / 10.0, source=INFO)


# ---------------------------------------------------------------------------
# Decision table
# ---------------------------------------------------------------------------


def _oracle(
    present: bool, faces: str, mask: str, persons: str, group: str,
    face_held: int = 0, person_held: int = 0, orphan: str = "box",
) -> tuple[str, tuple[Box, ...]]:
    """The decision table written out in plain terms, independent of mode.py.

    ``face_held`` and ``person_held`` are the held counts of every face and of
    every person box. ``orphan`` is the option value: ``"off"`` fills no orphan,
    ``"box"`` fills each orphan's whole padded box.
    """
    styles = set(GROUPS[group])
    one_face = faces == "one"
    populated = mask == "populated"  # A speck is drawn but is no region.
    boxes = PERSONS[persons]
    real = [b for b in boxes or () if b.w > 0 and b.h > 0]
    if "bodybox" in styles:
        primary = "bodybox"
    elif styles & {"silhouette", "silhouette_color"}:
        primary = "mask"
    else:
        primary = "face"
    if boxes is None or primary == "bodybox" or orphan == "off":
        orphans: list[Box] = []
    elif primary == "face":
        # Only a face that is not held clears its person box.
        orphans = [b for b in real if not (b is PERSON_FACE and one_face and face_held == 0)]
    else:
        orphans = [b for b in real if not (b is PERSON_FACE and populated)]
    # A held box at the growth cap is drawn but is no region.
    face_trusted = face_held < CAP
    person_trusted = person_held < CAP
    if primary == "face":
        own = one_face and face_trusted
    else:
        own = ("bodybox" in styles and bool(real) and person_trusted) or (
            bool(styles & {"silhouette", "silhouette_color"}) and populated
        )
    shows_subject = faces in ("one", "zero") or bool(boxes) or mask in ("speck", "populated")
    if not (present or shows_subject):
        return "pass", ()
    if not (own or (orphans and person_trusted)):
        return "black", ()
    return "draw", tuple(orphans)


def _check_plan(
    result: Plan, present: bool, faces: str, mask: str, persons: str, group: str,
    face_held: int = 0, person_held: int = 0, orphan: str = "box",
) -> None:
    params = _params(GROUPS[group], orphan=orphan)
    action, orphan_boxes = _oracle(
        present, faces, mask, persons, group, face_held, person_held, orphan
    )
    assert result.action == action
    pgrow = held_growth(person_held)
    assert result.orphans == tuple(pad_person(b, H, W, grow=pgrow) for b in orphan_boxes)
    styles = set(GROUPS[group])
    if action != "draw":
        assert result == Plan(action)  # type: ignore[arg-type]
        return
    wants_faces = bool(styles & {"gaussian", "pixelate", "noise", "facebox", "eyebar"})
    face = _held_face(FACE, face_held)
    assert result.faces == ((face,) if wants_faces and faces == "one" else ())
    grown = pad_face(FACE.box, params.padding, H, W, grow=held_growth(face_held))  # type: ignore[attr-defined]
    assert result.face_rects == ((grown,) if result.faces else ())
    wants_mask = bool(styles & {"silhouette", "silhouette_color", "outline"})
    assert (result.mask is not None) == (wants_mask and mask in ("speck", "populated"))
    boxes = [b for b in PERSONS[persons] or () if b.w > 0 and b.h > 0]
    expected_people = (
        tuple(pad_person(b, H, W, grow=pgrow) for b in boxes) if "bodybox" in styles else ()
    )
    assert result.person_rects == expected_people


def _held_box(box: Box, held: int) -> Box:
    return dataclasses.replace(box, held=held)


def _held_face(face: Face, held: int) -> Face:
    return dataclasses.replace(face, box=_held_box(face.box, held))


def _held_for(detections: Detections, face_held: int, person_held: int) -> Detections:
    """``detections`` with every face box held ``face_held`` and every person box ``person_held``."""
    faces = detections.faces
    persons = detections.persons
    return dataclasses.replace(
        detections,
        faces=None if faces is None else tuple(_held_face(f, face_held) for f in faces),
        persons=None if persons is None else tuple(_held_box(b, person_held) for b in persons),
    )


@pytest.mark.parametrize("orphan", ["off", "box"])
@pytest.mark.parametrize("group", list(GROUPS))
@pytest.mark.parametrize("persons", list(PERSONS))
@pytest.mark.parametrize("mask", list(MASKS))
@pytest.mark.parametrize("faces", list(FACES))
@pytest.mark.parametrize("present", [False, True])
def test_plan_follows_the_decision_table(
    present: bool, faces: str, mask: str, persons: str, group: str, orphan: str
) -> None:
    params = _params(GROUPS[group], orphan=orphan)
    result = plan(params, _detections(faces, mask, persons), present, SHAPE)
    _check_plan(result, present, faces, mask, persons, group, orphan=orphan)


@pytest.mark.parametrize("group", list(GROUPS))
@pytest.mark.parametrize("persons", ["none", "empty", "with_face", "without_face", "both"])
@pytest.mark.parametrize("mask", list(MASKS))
@pytest.mark.parametrize("faces", ["empty", "one"])
@pytest.mark.parametrize("person_held", [0, 1, CAP - 1, CAP, CAP + 3])
@pytest.mark.parametrize("face_held", [0, 1, CAP - 1, CAP, CAP + 3])
@pytest.mark.parametrize("orphan", ["off", "box"])
def test_plan_follows_the_decision_table_for_held_boxes(
    orphan: str, face_held: int, person_held: int, faces: str, mask: str, persons: str,
    group: str,
) -> None:
    detections = _held_for(_detections(faces, mask, persons), face_held, person_held)
    result = plan(_params(GROUPS[group], orphan=orphan), detections, True, SHAPE)
    _check_plan(result, True, faces, mask, persons, group, face_held, person_held, orphan)


E = "empty"
# (subject_present, faces, mask, persons, group, action, orphan boxes), written out.
TABLE = [
    # Face primary.
    (True, "none", "none", "none", "face", "black", ()),
    (True, "empty", "none", "none", "face", "black", ()),
    (True, "empty", "none", E, "face", "black", ()),
    (True, "empty", "none", "with_face", "face", "draw", (PERSON_FACE,)),
    (True, "empty", "none", "without_face", "face", "draw", (PERSON_EMPTY,)),
    (True, "empty", "none", "zero", "face", "black", ()),
    (True, "zero", "none", "none", "face", "black", ()),
    (True, "one", "none", "none", "face", "draw", ()),
    (True, "one", "none", "with_face", "face", "draw", ()),
    (True, "one", "none", "both", "face", "draw", (PERSON_EMPTY,)),
    (False, "none", "none", "none", "face", "pass", ()),
    (False, "empty", "false", E, "face", "pass", ()),
    # A False the detections contradict is not trusted.
    (False, "one", "none", "none", "face", "draw", ()),
    (False, "zero", "none", "none", "face", "black", ()),
    (False, "none", "none", "without_face", "face", "draw", (PERSON_EMPTY,)),
    # Mask primary.
    (True, "none", "false", "none", "mask", "black", ()),
    (True, "none", "none", "none", "mask", "black", ()),
    (True, "none", "false", "with_face", "mask", "draw", (PERSON_FACE,)),
    (True, "none", "false", "both", "mask", "draw", (PERSON_FACE, PERSON_EMPTY)),
    (True, "none", "populated", "none", "mask", "draw", ()),
    (True, "none", "populated", "both", "mask", "draw", (PERSON_EMPTY,)),
    # Faces found but the body primary has nothing: black.
    (True, "one", "false", "none", "face_mask", "black", ()),
    (True, "one", "none", E, "face_bodybox", "black", ()),
    (True, "one", "populated", "none", "face_mask", "draw", ()),
    # Bodybox primary: no orphans.
    (True, "none", "none", "both", "bodybox", "draw", ()),
    (True, "none", "none", "zero", "bodybox", "black", ()),
    (True, "one", "populated", "both", "all", "draw", ()),
    (True, "one", "populated", E, "all", "draw", ()),
    (True, "one", "false", E, "all", "black", ()),
    # Outline never counts as primary: the face group is.
    (True, "none", "populated", "none", "outline_face", "black", ()),
    (True, "none", "populated", "without_face", "outline_face", "draw", (PERSON_EMPTY,)),
    (True, "one", "false", "none", "outline_face", "draw", ()),
    # Nor with bodybox: a populated mask is no body region for outline alone.
    (True, "one", "populated", E, "outline_bodybox", "black", ()),
    (True, "none", "populated", "with_face", "outline_bodybox", "draw", ()),
    # A mask speck is drawn but is no region: it does not prevent black.
    (True, "none", "speck", "none", "mask", "black", ()),
    (True, "one", "speck", E, "face_mask", "black", ()),
    (True, "none", "speck", "with_face", "mask", "draw", (PERSON_FACE,)),
    (False, "none", "speck", "none", "mask", "black", ()),
]

# (face held, person held, faces, persons, group, action, orphan boxes), written out.
HELD_TABLE = [
    # A held face never clears its person box: the box is filled as an orphan.
    (1, 0, "one", "with_face", "face", "draw", (PERSON_FACE,)),
    (CAP - 1, 0, "one", "with_face", "face", "draw", (PERSON_FACE,)),
    (0, 0, "one", "with_face", "face", "draw", ()),
    # A held face at the cap is no region on its own.
    (CAP - 1, 0, "one", "none", "face", "draw", ()),
    (CAP, 0, "one", "none", "face", "black", ()),
    (CAP, 0, "one", E, "face", "black", ()),
    (CAP, 0, "one", "with_face", "face", "draw", (PERSON_FACE,)),
    # A held person box at the cap is no region either, orphan or bodybox.
    (CAP, CAP, "one", "with_face", "face", "black", ()),
    (CAP - 1, CAP - 1, "one", "with_face", "face", "draw", (PERSON_FACE,)),
    (0, CAP, "empty", "without_face", "face", "black", ()),
    (0, CAP, "empty", "both", "bodybox", "black", ()),
    (0, CAP - 1, "empty", "both", "bodybox", "draw", ()),
    # The fresh face is a region; the capped orphan is still drawn.
    (0, CAP, "one", "both", "face", "draw", (PERSON_EMPTY,)),
]


@pytest.mark.parametrize(
    ("face_held", "person_held", "faces", "persons", "group", "action", "orphans"), HELD_TABLE
)
def test_plan_written_out_held_cases(
    face_held: int, person_held: int, faces: str, persons: str, group: str, action: str,
    orphans: tuple[Box, ...],
) -> None:
    detections = _held_for(_detections(faces, "none", persons), face_held, person_held)
    result = plan(_params(GROUPS[group], orphan="box"), detections, True, SHAPE)
    assert result.action == action
    grow = held_growth(person_held)
    assert result.orphans == tuple(pad_person(b, H, W, grow=grow) for b in orphans)
    assert _oracle(True, faces, "none", persons, group, face_held, person_held) == (
        action, orphans
    )


@pytest.mark.parametrize(
    ("present", "faces", "mask", "persons", "group", "action", "orphans"), TABLE
)
def test_plan_written_out_cases(
    present: bool, faces: str, mask: str, persons: str, group: str, action: str,
    orphans: tuple[Box, ...],
) -> None:
    params = _params(GROUPS[group], orphan="box")
    result = plan(params, _detections(faces, mask, persons), present, SHAPE)
    assert result.action == action
    assert result.orphans == tuple(pad_person(b, H, W) for b in orphans)
    # The oracle agrees with every written-out row.
    assert _oracle(present, faces, mask, persons, group) == (action, orphans)


# (subject_present, faces, mask, persons, group, action), written out for orphan=off:
# no orphan is ever filled, and a person box alone is no region.
OFF_TABLE = [
    # Face primary: a person box no face clears is not filled, and alone it is no
    # region, so the frame is black exactly as with no person detector.
    (True, "empty", "none", "with_face", "face", "black"),
    (True, "empty", "none", "without_face", "face", "black"),
    (True, "empty", "none", "both", "face", "black"),
    (False, "none", "none", "without_face", "face", "black"),
    (True, "one", "none", "both", "face", "draw"),
    (True, "one", "none", "with_face", "face", "draw"),
    # Mask primary: the mask is the only region.
    (True, "none", "false", "both", "mask", "black"),
    (True, "none", "speck", "with_face", "mask", "black"),
    (True, "none", "populated", "both", "mask", "draw"),
    (True, "none", "populated", "without_face", "outline_face", "black"),
    # Bodybox primary: unaffected, it fills every person box itself.
    (True, "none", "none", "both", "bodybox", "draw"),
    (True, "none", "none", "zero", "bodybox", "black"),
    (False, "none", "none", "none", "face", "pass"),
]


@pytest.mark.parametrize(("present", "faces", "mask", "persons", "group", "action"), OFF_TABLE)
def test_plan_written_out_cases_with_orphan_off(
    present: bool, faces: str, mask: str, persons: str, group: str, action: str
) -> None:
    params = _params(GROUPS[group], orphan="off")
    result = plan(params, _detections(faces, mask, persons), present, SHAPE)
    assert result.action == action
    assert result.orphans == ()
    assert _oracle(present, faces, mask, persons, group, orphan="off") == (action, ())


@pytest.mark.parametrize("group", list(GROUPS))
@pytest.mark.parametrize("persons", list(PERSONS))
@pytest.mark.parametrize("mask", list(MASKS))
@pytest.mark.parametrize("faces", list(FACES))
def test_orphan_off_never_draws_a_frame_box_would_turn_black(
    faces: str, mask: str, persons: str, group: str
) -> None:
    # off only removes orphan fills: a frame black with box stays black, and a
    # frame drawn with off is drawn with box, with the same face, mask and
    # bodybox regions.
    detections = _detections(faces, mask, persons)
    off = plan(_params(GROUPS[group], orphan="off"), detections, True, SHAPE)
    box = plan(_params(GROUPS[group], orphan="box"), detections, True, SHAPE)
    if box.action == "black":
        assert off.action == "black"
    if off.action == "draw":
        assert box.action == "draw"
        assert (off.faces, off.face_rects, off.person_rects) == (
            box.faces, box.face_rects, box.person_rects)
        assert (off.mask is None) == (box.mask is None)
        if off.mask is not None:
            assert off.mask.rect == box.mask.rect  # type: ignore[union-attr]
            assert np.array_equal(off.mask.crop, box.mask.crop)  # type: ignore[union-attr]


def test_face_style_with_no_face_and_one_person_box_fills_that_box_black() -> None:
    params = _params(["facebox"], color="red", orphan="box")
    detections = Detections(faces=(), persons=(PERSON_EMPTY,))
    image = _image()
    out = ShadowMode(params).render(_frame(image), detections, True)
    rect = pad_person(PERSON_EMPTY, H, W)
    inside = np.zeros((H, W), bool)
    inside[rect.y0 : rect.y1, rect.x0 : rect.x1] = True
    assert not out[inside].any()
    assert np.array_equal(out[~inside], image[~inside])


@pytest.mark.parametrize("styles", [("facebox",), ("silhouette",), ("bodybox",)])
def test_subject_present_with_nothing_localized_is_full_black(styles: tuple[str, ...]) -> None:
    image = _image()
    out = ShadowMode(_params(styles)).render(_frame(image), Detections(motion=True), True)
    assert out.shape == image.shape and not out.any()


def test_primary_of_follows_the_body_group() -> None:
    assert primary_of(frozenset({"facebox"})) == "face"
    assert primary_of(frozenset({"outline", "facebox"})) == "face"
    assert primary_of(frozenset({"silhouette", "facebox"})) == "mask"
    assert primary_of(frozenset({"silhouette_color", "outline"})) == "mask"
    assert primary_of(frozenset({"silhouette", "bodybox"})) == "bodybox"
    assert primary_of(frozenset(STYLE_NAMES)) == "bodybox"


def test_secondary_face_styles_draw_on_every_face_with_a_body_primary() -> None:
    # The mask covers PERSON_FACE; a second face far away from it still gets its cover.
    far = Face(Box(120, 80, 14, 16), landmarks=((124.0, 85.0), (130.0, 85.0)))
    detections = Detections(faces=(FACE, far), mask=MASKS["populated"], persons=())
    result = plan(_params(["silhouette", "facebox"]), detections, True, SHAPE)
    assert result.action == "draw"
    assert result.faces == (FACE, far)
    assert len(result.face_rects) == 2


def test_plan_rejects_inputs_it_cannot_decide() -> None:
    params = _params(["facebox"])
    with pytest.raises(ValueError, match="outline"):
        plan(dataclasses.replace(params, styles=frozenset({"outline"})), Detections(), True, SHAPE)
    with pytest.raises(ValueError, match="non-empty"):
        plan(dataclasses.replace(params, styles=frozenset()), Detections(), True, SHAPE)
    with pytest.raises(ValueError, match="unknown"):
        plan(dataclasses.replace(params, styles=frozenset({"blur"})), Detections(), True, SHAPE)
    for bad in (-1, True, 1.5, None):
        with pytest.raises(ValueError, match="held"):
            plan(params, Detections(faces=(_held_face(FACE, bad),)), True, SHAPE)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="held"):
            plan(params, Detections(persons=(_held_box(PERSON_EMPTY, bad),)), True, SHAPE)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mask shape"):
        plan(_params(["silhouette"]), Detections(mask=np.zeros((H, W + 1), bool)), True, SHAPE)
    with pytest.raises(ValueError):
        plan(params, Detections(faces=(Face(Box(1.5, 2, 3, 4)),)), True, SHAPE)  # type: ignore[arg-type]


def test_plan_is_pure() -> None:
    params = _params(STYLE_NAMES)
    detections = _held_for(_detections("one", "populated", "both"), 3, 2)
    first = plan(params, detections, True, SHAPE)
    second = plan(params, detections, True, SHAPE)
    assert first.face_rects == second.face_rects
    assert first.orphans == second.orphans
    assert first.person_rects == second.person_rects
    assert np.array_equal(first.mask.crop, second.mask.crop)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Render properties
# ---------------------------------------------------------------------------


def _region(rect: Rect) -> np.ndarray:
    inside = np.zeros((H, W), bool)
    inside[rect.y0 : rect.y1, rect.x0 : rect.x1] = True
    return inside


def _crop_region(p: Plan) -> np.ndarray:
    inside = np.zeros((H, W), bool)
    if p.mask is not None:
        r = p.mask.rect
        inside[r.y0 : r.y1, r.x0 : r.x1] = p.mask.crop
    return inside


def _pixelate_region(rect: Rect, strength: str) -> np.ndarray:
    block = _pixelate_block(max(rect.width, rect.height), strength)
    x0, y0 = rect.x0 // block * block, rect.y0 // block * block
    x1 = min(-(-rect.x1 // block) * block, W)
    y1 = min(-(-rect.y1 // block) * block, H)
    return _region(Rect(x0, y0, x1, y1))


def _drawn(p: Plan, params: object) -> np.ndarray:
    """Every pixel the plan may write, for the byte-identity check outside it."""
    styles = params.styles  # type: ignore[attr-defined]
    drawn = _crop_region(p)
    for rect in (*p.face_rects, *p.person_rects, *p.orphans):
        drawn |= _region(rect)
    if "pixelate" in styles:
        for rect in p.face_rects:
            drawn |= _pixelate_region(rect, params.strength)  # type: ignore[attr-defined]
    if "eyebar" in styles:
        for face, rect in zip(p.faces, p.face_rects, strict=True):
            polygon = eyebar_polygon(face.landmarks, rect, face.box.w)
            if polygon is not None:
                canvas = np.zeros((H, W), np.uint8)
                cv2.fillConvexPoly(canvas, polygon, 1, lineType=cv2.LINE_8)
                drawn |= canvas.astype(bool)
    return drawn


def _check_output(out: np.ndarray, image: np.ndarray) -> None:
    assert type(out) is np.ndarray
    assert out.shape == image.shape and out.dtype == image.dtype
    assert out.flags.c_contiguous
    assert not np.shares_memory(out, image)
    assert not np.may_share_memory(out, image)


def test_pass_returns_the_input_itself() -> None:
    image = _image()
    before = image.copy()
    out = ShadowMode(_params(STYLE_NAMES)).render(_frame(image), Detections(motion=False), False)
    assert out is image
    assert np.array_equal(image, before)


def test_black_is_a_new_zero_image() -> None:
    image = _image()
    before = image.copy()
    out = ShadowMode(_params(["facebox"])).render(_frame(image), Detections(faces=()), True)
    _check_output(out, image)
    assert not out.any()
    assert np.array_equal(image, before)


@pytest.mark.parametrize("group", list(GROUPS))
@pytest.mark.parametrize("persons", ["none", "with_face", "both"])
@pytest.mark.parametrize("mask", ["none", "populated"])
def test_draw_changes_only_the_drawn_regions(group: str, persons: str, mask: str) -> None:
    params = _params(GROUPS[group], color="red")
    detections = _detections("one", mask, persons)
    p = plan(params, detections, True, SHAPE)
    image = _image(1)
    before = image.copy()
    out = ShadowMode(params, rng=np.random.default_rng(0)).render(_frame(image), detections, True)
    assert np.array_equal(image, before)
    if p.action == "black":
        _check_output(out, image)
        assert not out.any()
        return
    assert p.action == "draw"
    _check_output(out, image)
    outside = ~_drawn(p, params)
    assert np.array_equal(out[outside], image[outside])


def _all_color(out: np.ndarray, where: np.ndarray, color: tuple[int, int, int]) -> bool:
    return bool(where.any()) and bool((out[where] == np.array(color, np.uint8)).all())


@pytest.mark.parametrize(
    ("style", "color"),
    [("facebox", RED), ("silhouette", BLACK), ("silhouette_color", RED), ("bodybox", RED)],
)
def test_opaque_styles_fill_their_region_with_their_color(
    style: str, color: tuple[int, int, int]
) -> None:
    params = _params([style], color="red")
    detections = _detections("one", "populated", "with_face")
    p = plan(params, detections, True, SHAPE)
    assert p.action == "draw"
    out = ShadowMode(params).render(_frame(_image()), detections, True)
    region = _crop_region(p) if p.mask is not None else _drawn(p, params)
    assert _all_color(out, region, color)


def test_a_face_cover_wins_over_a_silhouette() -> None:
    params = _params(["silhouette", "facebox"], color="red")
    detections = _detections("one", "populated", "none")
    p = plan(params, detections, True, SHAPE)
    out = ShadowMode(params).render(_frame(_image()), detections, True)
    face = _region(p.face_rects[0])
    assert _all_color(out, face, RED)
    assert _all_color(out, _crop_region(p) & ~face, BLACK)


def test_silhouette_color_wins_over_silhouette() -> None:
    params = _params(["silhouette", "silhouette_color"], color="red")
    detections = _detections("none", "populated", "none")
    p = plan(params, detections, True, SHAPE)
    out = ShadowMode(params).render(_frame(_image()), detections, True)
    assert _all_color(out, _crop_region(p), RED)


def test_an_outline_is_drawn_over_a_silhouette() -> None:
    params = _params(["silhouette", "outline"], color="red")
    detections = _detections("none", "populated", "none")
    p = plan(params, detections, True, SHAPE)
    out = ShadowMode(params).render(_frame(_image()), detections, True)
    crop = _crop_region(p)
    red = np.all(out == np.array(RED, np.uint8), axis=2)
    black = ~out.any(axis=2)
    assert (red | black)[crop].all()
    assert red[crop].any() and black[crop].any()
    assert not red[~crop].any()
    # The outer edge of the crop is the outline color: the band is on top.
    edge = crop & ~cv2.erode(crop.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    assert red[edge].all()
    assert outline_thickness(H) == 2


def test_an_eyebar_is_drawn_over_a_face_cover() -> None:
    params = _params(["noise", "eyebar"], color="red")
    detections = _detections("one", "none", "none")
    p = plan(params, detections, True, SHAPE)
    out = ShadowMode(params, rng=np.random.default_rng(0)).render(
        _frame(_image()), detections, True
    )
    polygon = eyebar_polygon(FACE.landmarks, p.face_rects[0], FACE.box.w)
    assert polygon is not None
    canvas = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(canvas, polygon, 1, lineType=cv2.LINE_8)
    bar = canvas.astype(bool)
    assert _all_color(out, bar, RED)
    # The rest of the face rect is noise, not the bar color.
    rest = _region(p.face_rects[0]) & ~bar
    assert not _all_color(out, rest, RED)


def test_facebox_and_eyebar_fill_the_face_rect_with_the_color() -> None:
    params = _params(["facebox", "eyebar"], color="red")
    detections = _detections("one", "none", "none")
    p = plan(params, detections, True, SHAPE)
    out = ShadowMode(params).render(_frame(_image()), detections, True)
    assert _all_color(out, _region(p.face_rects[0]), RED)


def test_an_opaque_face_cover_is_never_touched_by_another_face_s_pixelation() -> None:
    # Pixelate grows its rect to the block grid, so it writes past its face rect.
    # Each face cover runs over all faces before the next, so the facebox of one
    # face is drawn after the pixelation of its neighbour and stays whole.
    params = _params(["pixelate", "facebox"], color="red")
    a, b = Face(Box(40, 30, 24, 28)), Face(Box(70, 34, 24, 28))
    detections = Detections(faces=(a, b))
    p = plan(params, detections, True, SHAPE)
    out = ShadowMode(params).render(_frame(_image()), detections, True)
    union = _region(p.face_rects[0]) | _region(p.face_rects[1])
    assert _all_color(out, union, RED)


def test_a_face_cover_wins_over_an_orphan_fill() -> None:
    # A face at belt height does not clear its person box, so the box is an
    # orphan filled black, and the face cover is drawn on top of it.
    low = Face(Box(62, 80, 16, 18))
    params = _params(["facebox"], color="red", orphan="box")
    detections = Detections(faces=(low,), persons=(PERSON_FACE,))
    p = plan(params, detections, True, SHAPE)
    assert p.orphans == (pad_person(PERSON_FACE, H, W),)
    out = ShadowMode(params).render(_frame(_image()), detections, True)
    face = _region(p.face_rects[0])
    assert _all_color(out, face, RED)
    assert _all_color(out, _region(p.orphans[0]) & ~face, BLACK)


def test_an_orphan_fill_wins_over_a_silhouette_color() -> None:
    # A sliver of mask inside PERSON_EMPTY leaves it an orphan; the black orphan
    # fill is drawn after the colored silhouette.
    mask = np.array(MASKS["populated"])
    mask[40:60, 108:116] = True
    params = _params(["silhouette_color"], color="red", dilate=0.0, orphan="box")
    detections = Detections(mask=mask, persons=PERSONS["both"])
    p = plan(params, detections, True, SHAPE)
    assert p.orphans == (pad_person(PERSON_EMPTY, H, W),)
    out = ShadowMode(params).render(_frame(_image()), detections, True)
    orphan = _region(p.orphans[0])
    assert (_crop_region(p) & orphan).any()
    assert _all_color(out, orphan, BLACK)
    assert _all_color(out, _crop_region(p) & ~orphan, RED)


def test_the_draw_order_matches_the_options_order() -> None:
    # draw() follows APPLY_ORDER with the orphan fill after the body fills.
    assert APPLY_ORDER == (
        "silhouette", "silhouette_color", "bodybox", "outline",
        "gaussian", "pixelate", "noise", "facebox", "eyebar",
    )


# ---------------------------------------------------------------------------
# Irreversibility
# ---------------------------------------------------------------------------

OPAQUE_STACKS = [
    ("facebox",),
    ("noise",),
    ("silhouette",),
    ("silhouette_color",),
    ("bodybox",),
    ("facebox", "silhouette"),
    ("noise", "bodybox"),
    ("noise", "facebox", "silhouette", "silhouette_color", "bodybox"),
]


@pytest.mark.parametrize("styles", OPAQUE_STACKS)
def test_opaque_stacks_carry_no_information_about_covered_pixels(
    styles: tuple[str, ...],
) -> None:
    params = _params(styles, color="red")
    detections = _detections("one", "populated", "both")
    p = plan(params, detections, True, SHAPE)
    assert p.action == "draw"
    covered = _drawn(p, params)
    first = _image(1)
    other = np.array(first)
    other[covered] = _image(2)[covered]
    other.setflags(write=False)
    outs = [
        ShadowMode(params, rng=np.random.default_rng(7)).render(_frame(img), detections, True)
        for img in (first, other)
    ]
    assert not np.array_equal(first, other)
    assert np.array_equal(outs[0], outs[1])


# ---------------------------------------------------------------------------
# Stacks
# ---------------------------------------------------------------------------


def test_all_nine_styles_cover_the_union() -> None:
    params = _params(STYLE_NAMES, color="red")
    detections = _detections("one", "populated", "both")
    p = plan(params, detections, True, SHAPE)
    image = _image(3)
    out = ShadowMode(params, rng=np.random.default_rng(0)).render(_frame(image), detections, True)
    _check_output(out, image)
    union = _crop_region(p)
    for rect in (*p.face_rects, *p.person_rects, *p.orphans):
        union |= _region(rect)
    red = np.all(out == np.array(RED, np.uint8), axis=2)
    black = ~out.any(axis=2)
    # The last drawn styles are opaque on every region: facebox and eyebar on
    # faces, the fills and outline elsewhere.
    assert (red | black)[union].all()
    assert np.array_equal(out[~_drawn(p, params)], image[~_drawn(p, params)])


def test_style_order_in_the_options_does_not_change_the_output() -> None:
    detections = _detections("one", "populated", "both")
    image = _image(4)
    outs = []
    for styles in (STYLE_NAMES, tuple(reversed(STYLE_NAMES)), tuple(sorted(STYLE_NAMES))):
        params = _params(styles, color="green")
        mode = ShadowMode(params, rng=np.random.default_rng(5))
        outs.append(mode.render(_frame(image), detections, True))
    assert all(np.array_equal(outs[0], o) for o in outs[1:])


# ---------------------------------------------------------------------------
# Held faces
# ---------------------------------------------------------------------------

BIG = (200, 200, 3)


def _face_at(x: int, y: int) -> Face:
    return Face(Box(x, y, 10, 10))


def _filled_rect(out: np.ndarray, image: np.ndarray) -> Rect:
    """The bounding rect of the pixels that changed, checked to be a solid red rect."""
    changed = np.any(out != image, axis=2)
    ys, xs = np.nonzero(changed)
    rect = Rect(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    block = out[rect.y0 : rect.y1, rect.x0 : rect.x1]
    assert (block == np.array(RED, np.uint8)).all()
    return rect


def _render_faces(mode: ShadowMode, image: np.ndarray, *faces: Face) -> np.ndarray:
    return mode.render(_frame(image), Detections(faces=faces), True)


def test_a_held_face_grows_until_the_cap_and_then_is_no_region() -> None:
    mode = ShadowMode(_params(["facebox"], color="red"))
    image = _image(0, BIG)
    face = _face_at(95, 95)
    for n in range(CAP):
        out = _render_faces(mode, image, _held_face(face, n))
        expected = pad_face(face.box, 0.3, 200, 200, grow=held_growth(n))
        assert _filled_rect(out, image) == expected
    assert expected.width > pad_face(face.box, 0.3, 200, 200).width  # type: ignore[union-attr]
    # From the cap on the held face alone no longer localizes the subject: black.
    for n in (CAP, CAP + 1, 15):
        assert not _render_faces(mode, image, _held_face(face, n)).any()


def test_the_held_count_is_read_from_each_box_for_faces_and_persons() -> None:
    # One instance, frames in no particular order: only the box decides.
    mode = ShadowMode(_params(["bodybox", "facebox"], color="red"))
    image = _image(0, BIG)
    face = _face_at(95, 75)
    person = Box(80, 60, 40, 100)
    for face_held, person_held in [(3, 0), (0, 5), (7, 7), (0, 0), (1, 2)]:
        detections = Detections(faces=(_held_face(face, face_held),),
                                persons=(_held_box(person, person_held),))
        out = mode.render(_frame(image), detections, True)
        want = _region_in(pad_person(person, 200, 200, grow=held_growth(person_held)), BIG)
        want |= _region_in(pad_face(face.box, 0.3, 200, 200, grow=held_growth(face_held)), BIG)
        assert np.array_equal(np.any(out != image, axis=2), want), (face_held, person_held)
        assert _all_color(out, want, RED)


def test_a_capped_held_face_is_still_drawn_when_another_region_exists() -> None:
    mode = ShadowMode(_params(["facebox"], color="red"))
    image = _image(0, BIG)
    held = _held_face(_face_at(95, 95), CAP + 2)
    # A fresh second face makes the frame draw; the capped face keeps its cover.
    fresh = _face_at(20, 20)
    out = _render_faces(mode, image, held, fresh)
    capped = pad_face(held.box, 0.3, 200, 200, grow=HELD_GROWTH_CAP)
    assert _all_color(out, _region_in(capped, BIG), RED)
    assert _all_color(out, _region_in(pad_face(fresh.box, 0.3, 200, 200), BIG), RED)


def _region_in(rect: Rect | None, shape: tuple[int, ...]) -> np.ndarray:
    assert rect is not None
    inside = np.zeros(shape[:2], bool)
    inside[rect.y0 : rect.y1, rect.x0 : rect.x1] = True
    return inside


def test_camera2_frame_280_held_face_leaves_its_person_box_an_orphan() -> None:
    # Reviewer's numbers: a face held for 2 frames and the person box of the
    # frame; the real face has walked to (439, 128, 13, 15), out of the held cover.
    held = Face(Box(429, 118, 11, 14, held=2))
    person = Box(409, 111, 60, 154)
    detections = Detections(faces=(held,), persons=(person,), motion=True)
    p = plan(_params(["facebox"], orphan="box"), detections, True, (480, 640, 3))
    assert p.action == "draw"
    assert p.face_rects == (pad_face(held.box, 0.3, 480, 640, grow=held_growth(2)),)
    assert p.orphans == (pad_person(person, 480, 640),)
    x, y, w, h = 439, 128, 13, 15
    r = p.orphans[0]
    assert r.x0 <= x and r.y0 <= y and x + w <= r.x1 and y + h <= r.y1
    # The same face detected fresh still clears the box.
    fresh = Detections(faces=(_held_face(held, 0),), persons=(person,), motion=True)
    assert plan(_params(["facebox"], orphan="box"), fresh, True, (480, 640, 3)).orphans == ()


def test_a_held_face_with_a_person_box_gives_the_orphan_fill() -> None:
    mode = ShadowMode(_params(["facebox"], color="red", orphan="box"))
    image = _image()
    detections = Detections(faces=(FACE,), persons=(PERSON_FACE,))
    first = mode.render(_frame(image), detections, True)
    orphan = _region(pad_person(PERSON_FACE, H, W))
    # Fresh face: the person box is cleared and shown outside the face cover.
    face0 = _region(pad_face(FACE.box, 0.3, H, W))
    assert np.array_equal(first[orphan & ~face0], image[orphan & ~face0])
    # Held face, fresh person box: the box is an orphan, filled black under the
    # grown face cover.
    held = Detections(faces=(_held_face(FACE, 1),), persons=(PERSON_FACE,))
    out = mode.render(_frame(image), held, True)
    face1 = _region(pad_face(FACE.box, 0.3, H, W, grow=held_growth(1)))
    box = _region(pad_person(PERSON_FACE, H, W))
    assert _all_color(out, face1, RED)
    assert _all_color(out, box & ~face1, BLACK)


def test_held_person_boxes_grow_and_stop_counting_at_the_cap() -> None:
    mode = ShadowMode(_params(["bodybox"], color="red"))
    image = _image(0, BIG)
    person = Box(80, 60, 30, 60)
    for n in range(CAP):
        held = Detections(persons=(_held_box(person, n),))
        out = mode.render(_frame(image), held, True)
        assert _filled_rect(out, image) == pad_person(person, 200, 200, grow=held_growth(n))
    for n in (CAP, CAP + 1, 15):
        held = Detections(persons=(_held_box(person, n),))
        assert not mode.render(_frame(image), held, True).any()
    # The same box detected fresh again is a region with no growth.
    out = mode.render(_frame(image), Detections(persons=(person,)), True)
    assert _filled_rect(out, image) == pad_person(person, 200, 200)


def test_held_orphans_grow() -> None:
    mode = ShadowMode(_params(["facebox"], orphan="box"))
    image = _image(0, BIG)
    person = Box(80, 60, 30, 60)
    for n in range(4):
        detections = Detections(faces=(), persons=(_held_box(person, n),))
        out = mode.render(_frame(image), detections, True)
        rect = pad_person(person, 200, 200, grow=held_growth(n))
        inside = _region_in(rect, BIG)
        assert not out[inside].any()
        assert np.array_equal(out[~inside], image[~inside])


@pytest.mark.parametrize(("pixels", "action"), [(1, "black"), (19, "black"), (20, "draw")])
def test_a_mask_counts_as_a_region_from_a_minimum_share(pixels: int, action: str) -> None:
    # 120 x 160 = 19200 pixels; 0.001 of it is 19.2, so 20 pixels are needed.
    assert MIN_MASK_SHARE == Fraction(1, 1000)
    detections = Detections(mask=_speck_mask(pixels), motion=True)
    p = plan(_params(["silhouette"]), detections, True, SHAPE)
    assert p.action == action


@pytest.mark.parametrize(("pixels", "action"), [(9, "black"), (10, "draw")])
def test_a_mask_on_the_exact_share_boundary_counts_as_a_region(pixels: int, action: str) -> None:
    # 100 x 100 = 10000 pixels, so 0.001 of it is exactly 10 pixels.
    mask = np.zeros((100, 100), bool)
    mask.reshape(-1)[:pixels] = True
    detections = Detections(mask=mask, motion=True)
    assert plan(_params(["silhouette"]), detections, True, (100, 100, 3)).action == action


def test_a_mask_speck_is_drawn_when_the_frame_is_drawn_for_another_region() -> None:
    params = _params(["silhouette", "facebox"], color="red", orphan="box")
    detections = Detections(mask=_speck_mask(), persons=(PERSON_EMPTY,))
    p = plan(params, detections, True, SHAPE)
    assert p.action == "draw" and p.mask is not None
    out = ShadowMode(params).render(_frame(_image()), detections, True)
    assert _all_color(out, _crop_region(p), BLACK)


def test_eyebar_without_landmarks_fills_the_whole_face_rect() -> None:
    face = Face(Box(62, 24, 16, 18), landmarks=None)
    params = _params(["eyebar"], color="red")
    detections = Detections(faces=(face,))
    p = plan(params, detections, True, SHAPE)
    image = _image()
    out = ShadowMode(params).render(_frame(image), detections, True)
    rect = _region(p.face_rects[0])
    assert _all_color(out, rect, RED)
    assert np.array_equal(out[~rect], image[~rect])


@pytest.mark.parametrize("styles", [("facebox",), ("bodybox",), ("facebox", "bodybox")])
def test_a_fresh_box_equal_to_the_previous_frames_box_is_not_grown_and_stays_a_region(
    styles: tuple[str, ...],
) -> None:
    # Accepted coverage change: the earlier rule took an identical box on the next
    # frame for a held one, grew it and cut it off after 10 frames.
    mode = ShadowMode(_params(styles, color="red"))
    image = _image(0, BIG)
    face = _face_at(95, 75)
    person = Box(80, 60, 40, 100)
    want = np.zeros(BIG[:2], bool)
    if "facebox" in styles:
        want |= _region_in(pad_face(face.box, 0.3, 200, 200), BIG)
    if "bodybox" in styles:
        want |= _region_in(pad_person(person, 200, 200), BIG)
    detections = Detections(
        faces=(face,) if "facebox" in styles else (), persons=(person,)
    )
    for i in range(30):
        out = mode.render(_frame(image, i), detections, True)
        # With facebox alone the fresh face also clears its person box every
        # frame, so no orphan fill appears.
        assert np.array_equal(np.any(out != image, axis=2), want), f"frame {i}"


def test_two_faces_are_grown_by_their_own_counts() -> None:
    mode = ShadowMode(_params(["facebox"], color="red"))
    image = _image(0, BIG)
    a, b = _face_at(20, 20), _face_at(150, 150)
    for counts in [(0, 0), (1, 0), (0, 3), (2, 7)]:
        out = _render_faces(mode, image, _held_face(a, counts[0]), _held_face(b, counts[1]))
        drawn = np.zeros(BIG[:2], bool)
        for face, n in zip((a, b), counts, strict=True):
            drawn |= _region_in(pad_face(face.box, 0.3, 200, 200, grow=held_growth(n)), BIG)
        assert np.array_equal(np.any(out != image, axis=2), drawn), counts


# ---------------------------------------------------------------------------
# Noise through the production path
# ---------------------------------------------------------------------------


def _ctx(
    styles: tuple[str, ...], detector: str = "person", info: SourceInfo = INFO, **options: object
) -> ModeContext:
    resolved = resolve_options(OPTIONS, {"styles": styles, **options})
    config = PetrifyConfig(
        mode="shadow", detector=detector, mode_options=mode_options_from(resolved)  # type: ignore[arg-type]
    )
    return ModeContext(info, config, resolved)


def _tracked_engine(
    mode: ShadowMode, info: SourceInfo, faces: dict[int, Face], persons: dict[int, Box],
    hold: int = 15,
) -> Engine:
    """An engine with fake face and person detectors and the real tracker for both."""
    refiners = [
        TrackerRefiner(DetectionKind.PERSONS, iou_threshold=0.3, hold_frames=hold, max_tracks=8),
        TrackerRefiner(DetectionKind.FACES, iou_threshold=0.3, hold_frames=hold, max_tracks=8,
                       landmarks=True),
    ]
    engine = Engine([_FakeFaces(faces), _FakePersons(persons)], mode, [], [_Sink()],
                    EngineConfig(lookahead=0, cold_start_frames=0), refiners=refiners)
    engine.open(info)
    return engine


BIG_INFO = dataclasses.replace(INFO, width=200, height=200)
TRACKED_FACE = Face(Box(95, 75, 10, 10), landmarks=((97.0, 78.0), (103.0, 78.0)))
TRACKED_PERSON = Box(80, 60, 40, 100)
def _big_frame(image: np.ndarray, index: int) -> Frame:
    return Frame(image=image, index=index, timestamp=index / 10.0, source=BIG_INFO)


DETECTED = range(4)
"""Frames on which the fake detectors find the face and the person; missed from then on."""


@pytest.mark.parametrize("switch", [None, 2, 6, 11])
def test_a_live_option_change_covers_exactly_as_the_old_mode_would_have(
    switch: int | None,
) -> None:
    # strength does not change facebox or bodybox output, so a mode built for it
    # must render every frame exactly like the old instance: the held count
    # travels with the box, not with the mode.
    styles = ("facebox", "bodybox")
    faces = {i: TRACKED_FACE for i in DETECTED}
    persons = {i: TRACKED_PERSON for i in DETECTED}
    image = _image(0, BIG)

    def run(switch: int | None) -> list[np.ndarray]:
        mode = SHADOW.create(_ctx(styles, info=BIG_INFO, color="red"))
        engine = _tracked_engine(mode, BIG_INFO, faces, persons)  # type: ignore[arg-type]
        outs: list[Output] = []
        for i in range(14):
            if i == switch:
                new = SHADOW.create(_ctx(styles, info=BIG_INFO, color="red", strength="low"), mode)
                assert new is not mode
                engine.reconfigure(mode=new)
            outs += engine.step(_big_frame(image, i))
        outs += engine.flush()
        engine.close()
        return [o.image for o in outs]

    continuous, switched = run(None), run(switch)
    assert len(continuous) == len(switched) == 14
    for i, (a, b) in enumerate(zip(continuous, switched, strict=True)):
        assert np.array_equal(a, b), f"frame {i}"


@pytest.mark.parametrize("styles", [("facebox",), ("facebox", "bodybox")])
def test_tracker_held_face_and_person_grow_then_turn_black_past_the_cut_off(
    styles: tuple[str, ...],
) -> None:
    # Through the real engine and the real TrackerRefiner: detected on frames 0
    # to 3, then held with counts 1, 2, ... from frame 4 on.
    faces = {i: TRACKED_FACE for i in DETECTED}
    persons = {i: TRACKED_PERSON for i in DETECTED}
    image = _image(0, BIG)
    engine = _tracked_engine(
        ShadowMode(_params(styles, color="red", orphan="box")), BIG_INFO, faces, persons
    )
    outs: list[Output] = []
    for i in range(14):
        outs += engine.step(_big_frame(image, i))
    outs += engine.flush()
    assert engine.close().failed_closed == 0
    assert [o.trace.index for o in outs] == list(range(14))
    last = DETECTED.stop - 1
    for o in outs:
        i, out = o.trace.index, o.image
        held = max(i - last, 0)
        if held >= HELD_REGION_FRAMES:
            # Both boxes are past the cut-off and nothing else localizes the subject.
            assert not out.any(), f"frame {i}"
            continue
        face_rect = _region_in(
            pad_face(TRACKED_FACE.box, 0.3, 200, 200, grow=held_growth(held)), BIG
        )
        # A held face does not clear its person box, so with facebox alone the
        # held person box is an orphan, filled black; bodybox fills it red.
        person_rect = _region_in(pad_person(TRACKED_PERSON, 200, 200, grow=held_growth(held)), BIG)
        if held == 0 and styles == ("facebox",):
            person_rect = np.zeros(BIG[:2], bool)
        assert np.array_equal(np.any(out != image, axis=2), face_rect | person_rect), f"frame {i}"
        assert _all_color(out, face_rect, RED), f"frame {i}"
        if person_rect.any():
            body_color = RED if "bodybox" in styles else BLACK
            assert _all_color(out, person_rect & ~face_rect, body_color), f"frame {i}"
    # The cut-off frame is the derived one: 8 held frames after the last detection.
    assert HELD_REGION_FRAMES == 8
    assert outs[last + 7].image.any() and not outs[last + 8].image.any()


@pytest.mark.parametrize(
    ("styles", "detector", "warns"),
    [
        (("silhouette",), "motion", True),
        (("silhouette_color", "facebox"), "motion", True),
        (("outline", "facebox"), "motion", True),
        (("silhouette",), "person", False),
        (("silhouette",), "both", False),
        (("facebox",), "motion", False),
        (("silhouette", "bodybox"), "both", False),
    ],
)
def test_create_warns_about_mask_styles_without_person_boxes(
    styles: tuple[str, ...], detector: str, warns: bool, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="petrify_shadow.factory"):
        mode = SHADOW.create(_ctx(styles, detector=detector))
        # Returning the same mode is not a new build, so it logs nothing more.
        assert SHADOW.create(_ctx(styles, detector=detector), mode) is mode
    found = [r for r in caplog.records if r.getMessage() == WARN_MASK_WITHOUT_PERSONS]
    assert len(found) == (1 if warns else 0)
    assert all(r.levelno == logging.WARNING for r in found)
    if warns:
        assert "--detector person or both" in WARN_MASK_WITHOUT_PERSONS
        assert "orphan rule" in WARN_MASK_WITHOUT_PERSONS
        # A new mode from changed options warns again, once.
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="petrify_shadow.factory"):
            SHADOW.create(_ctx(styles, detector=detector, padding=0.4), mode)
        assert [r.getMessage() for r in caplog.records] == [WARN_MASK_WITHOUT_PERSONS]


def _equal_share(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a == b))


def test_noise_is_not_reproducible_through_create() -> None:
    first, second = SHADOW.create(_ctx(("noise",))), SHADOW.create(_ctx(("noise",)))
    assert first is not second
    face = Face(Box(50, 30, 50, 50))
    detections = Detections(faces=(face,))
    rect = plan(first.params, detections, True, SHAPE).face_rects[0]  # type: ignore[attr-defined]
    assert rect.width >= 64 and rect.height >= 64
    image = _image()
    region = np.s_[rect.y0 : rect.y1, rect.x0 : rect.x1]
    a = first.render(_frame(image), detections, True)[region]
    b = second.render(_frame(image), detections, True)[region]
    assert _equal_share(a, b) < 0.05
    # One mode, consecutive frames: the noise differs between them too.
    c = first.render(_frame(image, 1), detections, True)[region]
    assert _equal_share(a, c) < 0.05


# ---------------------------------------------------------------------------
# Through the engine
# ---------------------------------------------------------------------------


class _FakeFaces:
    provides = frozenset({DetectionKind.FACES, DetectionKind.LANDMARKS})
    stateful = False

    def __init__(self, hits: dict[int, Face]) -> None:
        self.hits = hits

    def detect(self, frame: Frame) -> Detections:
        face = self.hits.get(frame.index)
        return Detections(faces=(face,) if face is not None else ())

    def reset(self) -> None:
        pass


class _FakePersons:
    provides = frozenset({DetectionKind.PERSONS})
    stateful = False

    def __init__(self, hits: dict[int, Box]) -> None:
        self.hits = hits

    def detect(self, frame: Frame) -> Detections:
        box = self.hits.get(frame.index)
        return Detections(persons=(box,) if box is not None else ())

    def reset(self) -> None:
        pass


class _Sink:
    critical = True

    def open(self, info: SourceInfo) -> None:
        pass

    def write(self, image: np.ndarray, meta: object) -> None:
        pass

    def close(self) -> None:
        pass

    def abort(self) -> None:
        pass


TOTAL = 30
PRESENT = range(8, 22)
FACE_MISSES = {12, 13, 18}
PERSON_MISSES = {12, 13, 16}


def _moving_face(i: int) -> Face:
    x = 30 + 4 * (i - PRESENT.start)
    return Face(Box(x + 8, 20, 14, 16), landmarks=((x + 12.0, 26.0), (x + 18.0, 26.0)))


def _moving_person(i: int) -> Box:
    return Box(30 + 4 * (i - PRESENT.start), 10, 30, 100)


@pytest.mark.parametrize("styles", [("facebox",), ("facebox", "bodybox"), ("noise", "eyebar")])
def test_engine_run_never_shows_an_uncovered_subject(styles: tuple[str, ...]) -> None:
    faces = {i: _moving_face(i) for i in PRESENT if i not in FACE_MISSES}
    persons = {i: _moving_person(i) for i in PRESENT if i not in PERSON_MISSES}
    params = _params(styles, color="red")
    mode = ShadowMode(params)
    engine = Engine(
        [_FakeFaces(faces), _FakePersons(persons)], mode, [], [_Sink()],
        EngineConfig(lookahead=3, cold_start_frames=0),
    )
    images = [_image(i) for i in range(TOTAL)]
    engine.open(INFO)
    outputs: list[Output] = []
    for i, image in enumerate(images):
        outputs += engine.step(_frame(image, i))
    outputs += engine.flush()
    stats = engine.close()
    assert stats.failed_closed == 0
    assert [o.trace.index for o in outputs] == list(range(TOTAL))
    filled_empty = 0
    for o in outputs:
        i, image, out = o.trace.index, images[o.trace.index], o.image
        assert not o.trace.failed_closed
        if not o.trace.final:
            assert out is image
            continue
        assert not np.may_share_memory(out, image)
        face, person = faces.get(i), persons.get(i)
        if face is None and person is None:
            # Gap-filled frames with no detection: black, never the input.
            filled_empty += 1
            assert not out.any(), f"frame {i}"
            continue
        if not out.any():
            continue  # Black is always an acceptable cover.
        detections = Detections(
            faces=(face,) if face else (), persons=(person,) if person else ()
        )
        p = plan(params, detections, True, SHAPE)
        assert p.action == "draw", f"frame {i}"
        if face is not None:
            fx, fy = face.box.x, face.box.y
            raw = out[fy : fy + face.box.h, fx : fx + face.box.w]
            assert not np.array_equal(raw, image[fy : fy + face.box.h, fx : fx + face.box.w])
        if "facebox" in styles and face is not None:
            # The held rect contains the unheld one.
            r = pad_face(face.box, 0.3, H, W)
            assert _all_color(out, _region(r), RED), f"frame {i}"
        for rect in p.orphans:
            assert not out[rect.y0 : rect.y1, rect.x0 : rect.x1].any(), f"frame {i}"
        if "bodybox" in styles:
            for rect in p.person_rects:
                assert _all_color(out, _region(rect) & ~_any_face(p), RED), f"frame {i}"
    # Lookahead frames before and after the subject, and the two-frame gap.
    assert filled_empty == 3 + 3 + 2


def _any_face(p: Plan) -> np.ndarray:
    region = np.zeros((H, W), bool)
    for rect in p.face_rects:
        region |= _region(rect)
    return region


# ---------------------------------------------------------------------------
# Input layout and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", ["fortran", "strided", "reversed"])
@pytest.mark.parametrize("group", ["all", "face", "mask"])
def test_non_contiguous_inputs_render_like_contiguous_ones(layout: str, group: str) -> None:
    base = np.array(_image(6))
    if layout == "fortran":
        image = np.asfortranarray(base)
    elif layout == "strided":
        big = np.zeros((2 * H, 2 * W, 3), np.uint8)
        big[::2, ::2] = base
        image = big[::2, ::2]
    else:
        flipped = np.ascontiguousarray(base[:, ::-1])
        image = flipped[:, ::-1]
    image.setflags(write=False)
    assert not image.flags.c_contiguous
    assert np.array_equal(image, base)
    params = _params(GROUPS[group], color="blue")
    detections = _detections("one", "populated", "both")
    want = ShadowMode(params, rng=np.random.default_rng(1)).render(
        _frame(base), detections, True
    )
    got = ShadowMode(params, rng=np.random.default_rng(1)).render(
        _frame(image), detections, True
    )
    _check_output(got, image)
    assert np.array_equal(got, want)
    black = ShadowMode(params).render(_frame(image), Detections(motion=True), True)
    _check_output(black, image)
    assert not black.any()


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((H, W), np.uint8),
        np.zeros((H, W, 4), np.uint8),
        np.zeros((H, W, 3), np.float32),
        np.zeros((H, W, 3), np.uint16),
        [[[0, 0, 0]]],
    ],
)
@pytest.mark.parametrize("present", [False, True])
def test_render_rejects_an_image_that_is_not_hxwx3_uint8(image: object, present: bool) -> None:
    mode = ShadowMode(_params(["facebox"]))
    with pytest.raises(ValueError, match="HxWx3 uint8|numpy array"):
        mode.render(_frame(image), Detections(), present)  # type: ignore[arg-type]


def test_draw_plan_fields_are_consistent_with_prepare_mask() -> None:
    params = _params(["silhouette"])
    p = plan(params, _detections("none", "populated", "none"), True, SHAPE)
    want = prepare_mask(MASKS["populated"], dilate_px(H, params.dilate))
    assert p.mask is not None and want is not None
    assert p.mask.rect == want.rect and np.array_equal(p.mask.crop, want.crop)


def test_draw_requires_a_writable_contiguous_out() -> None:
    params = _params(["facebox"])
    p = plan(params, _detections("one", "none", "none"), True, SHAPE)
    out = np.asfortranarray(np.array(_image()))
    with pytest.raises(ValueError, match="C-contiguous"):
        draw(out, p, params, np.random.default_rng(0))


# ---------------------------------------------------------------------------
# Timing (printed, not asserted)
# ---------------------------------------------------------------------------


def _timing_case(
    h: int, w: int, styles: tuple[str, ...], n_faces: int
) -> tuple[Frame, Detections]:
    image = np.random.default_rng(0).integers(1, 256, (h, w, 3), dtype=np.uint8)
    fw = max(8, w // 16)
    faces = []
    for k in range(n_faces):
        x = (k % 5) * (w // 5) + w // 20
        y = (k // 5) * (h // 3) + h // 10
        faces.append(Face(Box(x, y, fw, fw), landmarks=((x + fw * 0.3, y + fw * 0.4),
                                                        (x + fw * 0.7, y + fw * 0.4))))
    mask = None
    persons = None
    if set(styles) - {"gaussian", "pixelate", "noise", "facebox", "eyebar"}:
        m = np.zeros((h, w), np.uint8)
        cv2.ellipse(m, (w // 2, h // 2), (w // 8, h // 3), 0, 0, 360, 1, -1)
        mask = m.astype(bool)
        persons = (Box(w // 2 - w // 8, h // 6, w // 4, 2 * h // 3),)
    frame = Frame(image=image, index=0, timestamp=0.0, source=INFO)
    return frame, Detections(faces=tuple(faces), mask=mask, persons=persons)


def test_render_timing_is_printed() -> None:
    cases = [
        ("facebox, 1 face", ("facebox",), 1),
        ("facebox, 10 faces", ("facebox",), 10),
        ("silhouette + facebox", ("silhouette", "facebox"), 1),
        ("all nine", STYLE_NAMES, 1),
    ]
    lines = []
    for (h, w), (name, styles, n) in itertools.product(((480, 640), (1080, 1920)), cases):
        frame, detections = _timing_case(h, w, styles, n)
        mode = ShadowMode(_params(styles))
        mode.render(frame, detections, True)  # Warm up caches.
        runs = 30
        start = time.perf_counter()
        for _ in range(runs):
            mode.render(frame, detections, True)
        ms = (time.perf_counter() - start) / runs * 1000
        lines.append(f"{w}x{h} {name}: {ms:.2f} ms")
    print("\nshadow render timing:\n  " + "\n  ".join(lines))


# ---------------------------------------------------------------------------
# orphan = box | head
# ---------------------------------------------------------------------------

SWEEP_DIGEST = "5d0bcadc7435d7e20b29b0ab0c32211fc55c6a7bdff842fd292983d44a57c6b0"
"""SHA-256 over every render of the two decision-table sweeps above.

Recorded on the code before the ``orphan`` option existed. It pins those outputs
byte for byte, so a change that alters what the default draws fails here even
when the plan still looks right. It changes only if the sweep fixtures change.
"""


def _sweep_digest(**options: object) -> tuple[int, str]:
    """Render both decision-table sweeps with ``options`` and hash every output in order."""
    cases = [(p, f, m, pe, g, 0, 0) for p, f, m, pe, g in itertools.product(
        [False, True], FACES, MASKS, PERSONS, GROUPS)]
    cases += [(True, f, m, pe, g, fh, ph) for fh, ph, f, m, pe, g in itertools.product(
        [0, 1, CAP - 1, CAP, CAP + 3], [0, 1, CAP - 1, CAP, CAP + 3], ["empty", "one"],
        MASKS, ["none", "empty", "with_face", "without_face", "both"], GROUPS)]
    digest = hashlib.sha256()
    for n, (present, faces, mask, persons, group, fh, ph) in enumerate(cases):
        detections = _held_for(_detections(faces, mask, persons), fh, ph)
        # Seeded, so the noise style is reproducible here.
        mode = ShadowMode(_params(GROUPS[group], **options), rng=np.random.default_rng(0))
        out = mode.render(_frame(_image(n % 7)), detections, present)
        digest.update(np.ascontiguousarray(out).tobytes())
    return len(cases), digest.hexdigest()


# Only orphan=box: the default is now off, which no longer draws what the
# recorded sweeps drew.
@pytest.mark.parametrize("options", [{"orphan": "box"}], ids=["box"])
def test_orphan_box_renders_the_decision_table_sweeps_byte_identical_to_before(
    options: dict[str, object],
) -> None:
    assert _sweep_digest(**options) == (9536, SWEEP_DIGEST)


HS = (480, 640, 3)
"""A frame large enough for person boxes the head rule accepts."""
HH, HW = HS[:2]
TALL = Box(100, 60, 100, 300)
"""Eligible: 300 px tall, height / width 3. Head zone rows 60 to 134."""
TALL2 = Box(400, 60, 100, 300)
SHORT = Box(250, 60, 100, 149)
"""One row under the minimum height."""
WIDE = Box(420, 60, 150, 209)
"""Tall enough, height / width just under 7/5 (209 * 5 < 150 * 7)."""
LOW_FACE = Face(Box(130, 165, 30, 30), landmarks=((138.0, 175.0), (152.0, 175.0)))
"""Inside TALL's head rect, but its center (145, 180) is below the top third (row 160)."""
TALL_FACE = Face(Box(130, 80, 30, 30), landmarks=((138.0, 90.0), (152.0, 90.0)))
"""A face whose center lies in TALL's top third, so it clears TALL."""


def _hs_mask(*rects: tuple[int, int, int, int]) -> np.ndarray:
    """A mask of the HS frame, True on each (x0, y0, x1, y1)."""
    mask = np.zeros((HH, HW), bool)
    for x0, y0, x1, y1 in rects:
        mask[y0:y1, x0:x1] = True
    mask.setflags(write=False)
    return mask


BODY_ONLY = (100, 140, 200, 360)
"""TALL's body without its head zone: 22000 of 30000 box pixels, 0 in the head zone."""


def _hs_region(rect: Rect) -> np.ndarray:
    inside = np.zeros((HH, HW), bool)
    inside[rect.y0 : rect.y1, rect.x0 : rect.x1] = True
    return inside


def _head(box: Box) -> Rect:
    rect = head_rect(box, HH, HW)
    assert rect is not None
    return rect


def _whole(box: Box) -> Rect:
    rect = pad_person(box, HH, HW, grow=held_growth(box.held))
    assert rect is not None
    return rect


def _hs_plan(styles: Sequence[str], detections: Detections, orphan: str, **options: object) -> Plan:
    return plan(_params(styles, orphan=orphan, **options), detections, True, HS)


def test_head_rect_of_the_fixtures() -> None:
    # Pad 5 px per side and 15 px on top; the bottom edge at 60 + 150.
    assert _head(TALL) == Rect(95, 45, 205, 210)
    for box in (SHORT, WIDE, _held_box(TALL, 1)):
        assert head_rect(box, HH, HW) is None


def test_face_primary_head_fills_only_the_head_rect_of_an_eligible_orphan() -> None:
    detections = Detections(faces=(), persons=(TALL,))
    assert _hs_plan(("facebox",), detections, "box").orphans == (_whole(TALL),)
    p = _hs_plan(("facebox",), detections, "head")
    assert p.action == "draw"
    assert p.orphans == (_head(TALL),)


@pytest.mark.parametrize(
    "box", [SHORT, WIDE, _held_box(TALL, 1), _held_box(TALL, CAP - 1)],
    ids=["short", "wide", "held-1", "held-below-cap"],
)
def test_face_primary_head_keeps_the_whole_box_fill_when_the_box_is_not_eligible(
    box: Box,
) -> None:
    detections = Detections(faces=(), persons=(box,))
    for orphan in ("box", "head"):
        p = _hs_plan(("facebox",), detections, orphan)
        assert p.action == "draw"
        assert p.orphans == (_whole(box),), orphan


def test_face_primary_head_on_mixed_boxes_keeps_input_order() -> None:
    detections = Detections(faces=(TALL_FACE,), persons=(TALL, SHORT, TALL2, WIDE))
    p = _hs_plan(("facebox",), detections, "head")
    # TALL is cleared by its face; the others are orphans, each with its own fill.
    assert p.orphans == (_whole(SHORT), _head(TALL2), _whole(WIDE))


def test_a_held_face_does_not_clear_and_its_box_gets_the_head_fill() -> None:
    # The held face no longer proves where the head is, so TALL is an orphan; the
    # fresh box itself is eligible.
    detections = Detections(faces=(_held_face(TALL_FACE, 1),), persons=(TALL,))
    p = _hs_plan(("facebox",), detections, "head")
    assert p.orphans == (_head(TALL),)
    assert len(p.face_rects) == 1


def test_face_primary_head_draws_the_head_black_and_the_face_style_on_top() -> None:
    image = _image(3, HS)
    detections = Detections(faces=(LOW_FACE,), persons=(TALL,))
    params = _params(("facebox",), orphan="head", color="red")
    out = ShadowMode(params).render(Frame(image, 0, 0.0, INFO), detections, True)
    p = plan(params, detections, True, HS)
    (face_rect,) = p.face_rects
    assert p.orphans == (_head(TALL),)
    face = _hs_region(face_rect)
    head = _hs_region(_head(TALL)) & ~face
    assert face_rect.y1 <= _head(TALL).y1  # The face cover lies inside the head rect.
    assert np.all(out[face] == RED)
    assert not out[head].any()
    rest = ~(face | head)
    assert np.array_equal(out[rest], image[rest])
    # The lower body of the box, rows 210 on, is shown.
    assert np.array_equal(out[210:360, 100:200], image[210:360, 100:200])


def test_mask_primary_head_fills_the_head_rect_when_only_the_head_zone_share_fails() -> None:
    detections = Detections(mask=_hs_mask(BODY_ONLY), persons=(TALL,))
    assert _hs_plan(("silhouette",), detections, "box").orphans == (_whole(TALL),)
    p = _hs_plan(("silhouette",), detections, "head")
    assert p.action == "draw"
    assert p.orphans == (_head(TALL),)


@pytest.mark.parametrize(
    ("mask", "box"),
    [
        (None, TALL),  # no mask: the whole-box share fails
        (_hs_mask(), TALL),  # an empty mask
        (_hs_mask((100, 60, 200, 135)), TALL),  # the head zone only: 25 percent of the box
        (_hs_mask((100, 300, 200, 360), (400, 60, 500, 360)), TALL),  # 20 percent, no head
        (_hs_mask(BODY_ONLY), _held_box(TALL, 1)),  # head-zone failure, held box
        (_hs_mask((250, 120, 350, 209)), SHORT),  # head-zone failure, short box
    ],
    ids=["none", "empty", "head-only", "box-share", "held", "short"],
)
def test_mask_primary_head_keeps_the_whole_box_fill(mask: np.ndarray | None, box: Box) -> None:
    detections = Detections(mask=mask, persons=(box,))
    for orphan in ("box", "head"):
        assert _hs_plan(("silhouette",), detections, orphan).orphans == (_whole(box),), orphan


def test_mask_primary_head_draws_the_head_black_and_the_silhouette_over_the_body() -> None:
    image = _image(4, HS)
    mask = _hs_mask(BODY_ONLY)
    detections = Detections(mask=mask, persons=(TALL,))
    params = _params(("silhouette",), orphan="head", dilate=0.0)
    out = ShadowMode(params).render(Frame(image, 0, 0.0, INFO), detections, True)
    covered = _hs_region(_head(TALL)) | mask
    assert not out[covered].any()
    assert np.array_equal(out[~covered], image[~covered])


def test_bodybox_primary_has_no_orphans_with_either_value() -> None:
    detections = Detections(faces=(), mask=_hs_mask(), persons=(TALL, SHORT))
    for styles in (("bodybox",), ("bodybox", "silhouette", "facebox")):
        box = _hs_plan(styles, detections, "box")
        assert _hs_plan(styles, detections, "head") == box
        assert box.orphans == ()


def test_plan_rejects_an_unknown_orphan_value() -> None:
    params = dataclasses.replace(_params(("facebox",)), orphan="Head")  # type: ignore[type-var]
    with pytest.raises(ValueError, match="orphan"):
        plan(params, Detections(), True, HS)


def _random_box(rng: np.random.Generator) -> Box:
    return Box(int(rng.integers(-60, HW)), int(rng.integers(-60, HH)),
               int(rng.integers(0, 260)), int(rng.integers(0, 420)),
               held=int(rng.choice([0, 0, 0, 1, CAP])))


def test_head_changes_only_the_orphan_fills_and_never_whether_the_frame_has_regions() -> None:
    rng = np.random.default_rng(6)
    heads = 0
    for _ in range(400):
        persons = tuple(_random_box(rng) for _ in range(int(rng.integers(0, 4))))
        faces = tuple(
            Face(_random_box(rng), landmarks=None) for _ in range(int(rng.integers(0, 3)))
        )
        mask = rng.random((HH, HW)) < rng.random() * 0.6 if rng.random() < 0.7 else None
        group = list(GROUPS)[int(rng.integers(0, len(GROUPS)))]
        detections = Detections(faces=faces, mask=mask, persons=persons)
        present = bool(rng.random() < 0.8)
        box = plan(_params(GROUPS[group], orphan="box"), detections, present, HS)
        head = plan(_params(GROUPS[group], orphan="head"), detections, present, HS)
        assert head.action == box.action
        assert (head.faces, head.face_rects, head.person_rects) == (
            box.faces, box.face_rects, box.person_rects)
        assert (head.mask is None) == (box.mask is None)
        assert len(head.orphans) == len(box.orphans)
        for a, b in zip(head.orphans, box.orphans, strict=True):
            if a != b:
                heads += 1
                # The head rect shares the padded box's top, left and right edges.
                assert (a.x0, a.y0, a.x1) == (b.x0, b.y0, b.x1) and a.y1 < b.y1
    assert heads > 20


def _head_engine(mode: ShadowMode, info: SourceInfo) -> Engine:
    # A fresh person box on every frame and no face: always an eligible orphan.
    persons = _FakePersons({i: TALL for i in range(20)})
    engine = Engine([_FakeFaces({}), persons], mode, [], [_Sink()],
                    EngineConfig(lookahead=0, cold_start_frames=0))
    engine.open(info)
    return engine


def test_a_live_switch_of_orphan_takes_effect_on_the_next_frame() -> None:
    info = dataclasses.replace(INFO, width=HW, height=HH)
    image = _image(5, HS)
    detections = Detections(faces=(), persons=(TALL,))
    expected = {
        value: ShadowMode(_params(("facebox",), orphan=value)).render(
            Frame(image, 0, 0.0, info), detections, True)
        for value in ("box", "head")
    }
    assert not np.array_equal(expected["box"], expected["head"])
    mode = SHADOW.create(_ctx(("facebox",), info=info, orphan="box"))
    engine = _head_engine(mode, info)  # type: ignore[arg-type]
    outs: list[Output] = []
    switches = {3: "head", 6: "box"}
    for i in range(9):
        if i in switches:
            new = SHADOW.create(_ctx(("facebox",), info=info, orphan=switches[i]), mode)
            assert new is not mode
            engine.reconfigure(mode=new)
            mode = new
        outs += engine.step(Frame(image, i, i / 10.0, info))
    outs += engine.flush()
    engine.close()
    values = ["box"] * 3 + ["head"] * 3 + ["box"] * 3
    assert len(outs) == 9
    for i, (out, value) in enumerate(zip(outs, values, strict=True)):
        assert np.array_equal(out.image, expected[value]), f"frame {i}"


# ---------------------------------------------------------------------------
# orphan = off, and the webcam frame that showed the clearing defect
# ---------------------------------------------------------------------------

WEBCAM = (1080, 1920, 3)
WEBCAM_PERSON = Box(330, 175, 1500, 905)
"""The owner's webcam frame: the upper body, cut off by the bottom of the frame.

Its padded box spans x 255 to 1905 and y 129 to 1080, about the black rect seen
on the first webcam test (x 255 to 1859, y 130 to 1080).
"""
WEBCAM_FACE = Face(Box(930, 400, 300, 340), landmarks=((1020.0, 530.0), (1140.0, 530.0)))
"""A face 300 px wide centered at (1080, 570): below the top third of WEBCAM_PERSON, in its top half."""


def _webcam_render(orphan: str, face: Face = WEBCAM_FACE) -> tuple[Plan, np.ndarray, np.ndarray]:
    params = _params(("eyebar",), orphan=orphan)
    detections = Detections(faces=(face,), persons=(WEBCAM_PERSON,), motion=True)
    image = _image(9, WEBCAM)
    out = ShadowMode(params).render(_frame(image), detections, True)
    return plan(params, detections, True, WEBCAM), image, out


def test_webcam_frame_fixture_matches_what_the_owner_saw() -> None:
    p = WEBCAM_PERSON
    assert p.y + p.h == WEBCAM[0]
    assert pad_person(p, *WEBCAM[:2]) == Rect(255, 129, 1905, 1080)
    # The face center lies below the top third of the box, so the rule tuned on
    # CCTV footage alone would leave the box an orphan.
    cy2 = 2 * WEBCAM_FACE.box.y + WEBCAM_FACE.box.h
    assert cy2 * 3 > 2 * (p.y * 3 + p.h)


@pytest.mark.parametrize("orphan", ["box", "head"])
def test_webcam_frame_a_detected_face_clears_its_cut_off_box(orphan: str) -> None:
    p, image, out = _webcam_render(orphan)
    assert p.action == "draw"
    assert p.orphans == ()
    # Only the eye bar is drawn: every changed pixel lies inside the face rect,
    # and most of the padded person box is shown unchanged.
    face = _region_in(p.face_rects[0], WEBCAM)
    changed = np.any(out != image, axis=2)
    assert changed.any()
    assert not (changed & ~face).any()
    _, _, off = _webcam_render("off")
    assert np.array_equal(out, off)


def test_webcam_frame_with_orphan_off_draws_only_the_eye_bar() -> None:
    p, image, out = _webcam_render("off")
    assert p.action == "draw" and p.orphans == ()
    face = _region_in(p.face_rects[0], WEBCAM)
    assert not (np.any(out != image, axis=2) & ~face).any()


def test_webcam_frame_a_held_face_still_leaves_the_box_an_orphan() -> None:
    held = _held_face(WEBCAM_FACE, 1)
    p, _, out = _webcam_render("box", held)
    assert p.orphans == (pad_person(WEBCAM_PERSON, *WEBCAM[:2]),)
    box = _region_in(p.orphans[0], WEBCAM)
    face = _region_in(p.face_rects[0], WEBCAM)
    assert not out[box & ~face].any()
    # With off the held face is still covered, and the box is not filled.
    p_off, image, out_off = _webcam_render("off", held)
    assert p_off.orphans == ()
    assert not (np.any(out_off != image, axis=2) & ~face).any()


def test_orphan_off_reads_every_person_box() -> None:
    # A malformed person box raises with off too, so the engine fails the frame
    # closed exactly as with box and head.
    bad = Detections(faces=(FACE,), persons=(Box(1.5, 2, 3, 4),))  # type: ignore[arg-type]
    for orphan in ("off", "box", "head"):
        with pytest.raises(ValueError):
            plan(_params(("facebox",), orphan=orphan), bad, True, SHAPE)


def test_orphan_off_with_a_person_box_and_no_face_is_black() -> None:
    # The same as with no person detector: the subject is not localized.
    for persons in (None, (PERSON_EMPTY,), (PERSON_FACE, PERSON_EMPTY)):
        detections = Detections(faces=(), persons=persons, motion=True)
        assert plan(_params(("facebox",), orphan="off"), detections, True, SHAPE).action == "black"
        out = ShadowMode(_params(("facebox",))).render(_frame(_image()), detections, True)
        assert not out.any()


def test_create_builds_a_new_mode_when_orphan_changes() -> None:
    first = SHADOW.create(_ctx(("facebox",)))
    assert first.params.orphan == "off"  # type: ignore[attr-defined]
    assert SHADOW.create(_ctx(("facebox",), orphan="off"), first) is first
    previous = first
    for value in ("box", "head", "off", "box"):
        new = SHADOW.create(_ctx(("facebox",), orphan=value), previous)
        assert new is not previous and new.params.orphan == value  # type: ignore[attr-defined]
        assert SHADOW.create(_ctx(("facebox",), orphan=value), new) is new
        previous = new
