"""Fault-injection harness for the engine's fail-closed tests.

Every component here is deterministic and driven by a :class:`FaultPlan`, which maps
a frame index to the one fault injected on that frame. A component records each
fault it actually injects in :attr:`FaultPlan.fired`, so a test can tell a planned
fault from one that took effect (a refiner fault cannot fire while no refiner is
active, for example).

Pixel values are split into disjoint ranges so any image a sink receives can be
identified exactly:

- ``0``: a black (failed closed) frame.
- ``INPUT_BASE + index`` (1 to 9999): every pixel of input frame ``index``.
- ``TAG_BASE`` to ``TAG_LIMIT - 1``: :class:`TagMode` output, encoding the frame
  index, the mode's salt, and the subject flag the engine passed to the mode.
- ``STAGE_SHIFT`` and above: any of the above after :class:`FaultyStage` added
  ``STAGE_SHIFT`` to it.

Time is a :class:`StepClock` that only the components advance, so slow steps are
simulated exactly and nothing sleeps.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from petrify.core import DetectionKind, Detections, Frame, FrameMeta, SourceInfo

HEIGHT, WIDTH = 3, 4
SHAPE = (HEIGHT, WIDTH, 3)
DTYPE = np.uint16
INFO = SourceInfo(fps=10.0, width=WIDTH, height=HEIGHT, frame_count=None, seekable=False, live=False)

INPUT_BASE = 1
INPUT_LIMIT = 10_000  # Exclusive upper bound of input pixel values.
TAG_BASE = 10_000
TAG_LIMIT = 20_000  # Exclusive upper bound of TagMode values.
STAGE_SHIFT = 20_000  # FaultyStage adds this; staged values stay below 40_000.
MODE_SALTS = 4  # Distinct TagMode instances a test may tell apart.

STEP_SECONDS = 0.001  # What each component call costs on the StepClock.
BUDGET_MS = 50.0  # Frame budget the tests use; normal frames cost well under it.
OVERRUN_SECONDS = 1.0  # What an over-budget fault adds; far above any budget used.


class Fault(str, Enum):
    """The fault kinds a :class:`FaultPlan` can inject."""

    DETECTOR_RAISE = "detector_raise"
    DETECTOR_MALFORMED = "detector_malformed"
    REFINER_RAISE = "refiner_raise"
    REFINER_MALFORMED = "refiner_malformed"
    MODE_RAISE = "mode_raise"
    MODE_BAD_SHAPE = "mode_bad_shape"
    MODE_BAD_DTYPE = "mode_bad_dtype"
    MODE_NOT_ARRAY = "mode_not_array"
    MODE_ALIAS = "mode_alias"
    STAGE_RAISE = "stage_raise"
    STAGE_BAD_IMAGE = "stage_bad_image"
    STAGE_ALIAS = "stage_alias"
    DETECT_OVER_BUDGET = "detect_over_budget"
    RENDER_OVER_BUDGET = "render_over_budget"


DETECTOR_FAULTS = frozenset(
    {Fault.DETECTOR_RAISE, Fault.DETECTOR_MALFORMED, Fault.DETECT_OVER_BUDGET}
)
REFINER_FAULTS = frozenset({Fault.REFINER_RAISE, Fault.REFINER_MALFORMED})
MODE_FAULTS = frozenset(
    {
        Fault.MODE_RAISE,
        Fault.MODE_BAD_SHAPE,
        Fault.MODE_BAD_DTYPE,
        Fault.MODE_NOT_ARRAY,
        Fault.MODE_ALIAS,
        Fault.RENDER_OVER_BUDGET,
    }
)
STAGE_FAULTS = frozenset({Fault.STAGE_RAISE, Fault.STAGE_BAD_IMAGE, Fault.STAGE_ALIAS})

DETECTION_FAULTS = DETECTOR_FAULTS | REFINER_FAULTS
"""Faults known before the frame enters the lookahead buffer: the engine counts the
frame as a subject (raw True), so its neighbors are frozen."""

ALIAS_FAULTS = frozenset({Fault.MODE_ALIAS, Fault.STAGE_ALIAS})
"""Returning the input frame's own array. The engine rejects it only on a frame whose
final decision is subject present; on a pass-through frame (final False) returning
the input is the documented pass-through, so the frame is not failed closed."""

OVER_BUDGET_FAULTS = frozenset({Fault.DETECT_OVER_BUDGET, Fault.RENDER_OVER_BUDGET})


class InjectedFault(RuntimeError):
    """Raised by a harness component on a planned ``*_raise`` fault."""


@dataclass
class FaultPlan:
    """Frame index to the fault injected on that frame, plus what actually fired."""

    faults: Mapping[int, Fault] = field(default_factory=dict)
    fired: dict[int, Fault] = field(default_factory=dict)

    def take(self, index: int, kinds: frozenset[Fault]) -> Fault | None:
        """Return the fault planned for ``index`` if it is one of ``kinds``, and record it."""
        kind = self.faults.get(index)
        if kind is None or kind not in kinds:
            return None
        if index in self.fired:
            raise AssertionError(f"fault {kind.value} fired twice on frame {index}")
        self.fired[index] = kind
        return kind


class StepClock:
    """Manual clock for the engine: only components advance it."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def input_image(index: int) -> np.ndarray:
    """The synthetic input image of frame ``index``: every pixel is ``INPUT_BASE + index``."""
    value = INPUT_BASE + index
    if not INPUT_BASE <= value < INPUT_LIMIT:
        raise ValueError(f"frame index {index} is out of the encodable range")
    return np.full(SHAPE, value, dtype=DTYPE)


def make_frame(index: int) -> Frame:
    return Frame(image=input_image(index), index=index, timestamp=index / INFO.fps, source=INFO)


class FrameSource:
    """``count`` synthetic frames; keeps each yielded image so a sink can compare."""

    def __init__(self, count: int) -> None:
        self.info = INFO
        self.frames = [make_frame(i) for i in range(count)]

    def __iter__(self) -> Iterator[Frame]:
        return iter(self.frames)

    def close(self) -> None:
        pass


def tag_value(index: int, subject: bool, salt: int) -> int:
    """The pixel value :class:`TagMode` with ``salt`` writes for ``(index, subject)``."""
    if not 0 <= salt < MODE_SALTS:
        raise ValueError(f"salt must be in [0, {MODE_SALTS}), got {salt}")
    value = TAG_BASE + (index * MODE_SALTS + salt) * 2 + int(subject)
    if not TAG_BASE <= value < TAG_LIMIT:
        raise ValueError(f"frame index {index} is out of the encodable range")
    return value


def tag_image(index: int, subject: bool, salt: int) -> np.ndarray:
    return np.full(SHAPE, tag_value(index, subject, salt), dtype=DTYPE)


class FaultyDetector:
    """Reports motion on the frames in ``subjects``.

    Only a ``primary`` detector consults the plan (detector faults and the detection
    overrun); tests put one primary first in every detector list, so it always runs
    and its faults always fire.
    """

    provides = frozenset({DetectionKind.MOTION})

    def __init__(
        self,
        plan: FaultPlan,
        subjects: frozenset[int],
        clock: StepClock,
        *,
        primary: bool,
        stateful: bool,
    ) -> None:
        self.plan = plan
        self.subjects = subjects
        self.clock = clock
        self.primary = primary
        self.stateful = stateful
        self.seen: list[int] = []
        self.resets = 0

    def detect(self, frame: Frame) -> Detections:
        self.seen.append(frame.index)
        self.clock.advance(STEP_SECONDS)
        if self.primary:
            kind = self.plan.take(frame.index, DETECTOR_FAULTS)
            if kind is Fault.DETECTOR_RAISE:
                raise InjectedFault(f"detector fault on frame {frame.index}")
            if kind is Fault.DETECTOR_MALFORMED:
                return Detections(motion=1)  # type: ignore[arg-type]  # An int, not a bool.
            if kind is Fault.DETECT_OVER_BUDGET:
                self.clock.advance(OVERRUN_SECONDS)
        return Detections(motion=frame.index in self.subjects)

    def reset(self) -> None:
        self.resets += 1


class FaultyRefiner:
    """Consumes motion and provides faces; found nothing unless a fault fires."""

    consumes = frozenset({DetectionKind.MOTION})
    provides = frozenset({DetectionKind.FACES})

    def __init__(self, plan: FaultPlan, clock: StepClock) -> None:
        self.plan = plan
        self.clock = clock
        self.seen: list[int] = []
        self.resets = 0

    def refine(self, frame: Frame, detections: Detections) -> Detections:
        self.seen.append(frame.index)
        self.clock.advance(STEP_SECONDS)
        kind = self.plan.take(frame.index, REFINER_FAULTS)
        if kind is Fault.REFINER_RAISE:
            raise InjectedFault(f"refiner fault on frame {frame.index}")
        if kind is Fault.REFINER_MALFORMED:
            return Detections(faces=[])  # type: ignore[arg-type]  # A list, not a tuple.
        return Detections(faces=())

    def reset(self) -> None:
        self.resets += 1


class TagMode:
    """Writes ``tag_image(index, subject_present, salt)`` into a new array.

    Under ``"first_hit"`` it requires motion only; under ``"all"`` it also requires
    faces, which activates a :class:`FaultyRefiner` when one is configured.
    """

    mandatory: frozenset[DetectionKind] = frozenset()

    def __init__(self, plan: FaultPlan, clock: StepClock, *, salt: int, policy: str) -> None:
        if policy not in ("first_hit", "all"):
            raise ValueError(f"unknown policy {policy!r}")
        tag_value(0, False, salt)  # Validates the salt.
        self.plan = plan
        self.clock = clock
        self.salt = salt
        self.detect_policy = policy
        self.name = f"tag{salt}"
        self.requires = (
            frozenset({DetectionKind.MOTION})
            if policy == "first_hit"
            else frozenset({DetectionKind.MOTION, DetectionKind.FACES})
        )
        self.calls: list[tuple[int, bool]] = []

    def render(self, frame: Frame, detections: Detections, subject_present: bool) -> np.ndarray:
        self.calls.append((frame.index, subject_present))
        self.clock.advance(STEP_SECONDS)
        image = tag_image(frame.index, subject_present, self.salt)
        kind = self.plan.take(frame.index, MODE_FAULTS)
        if kind is Fault.MODE_RAISE:
            raise InjectedFault(f"mode fault on frame {frame.index}")
        if kind is Fault.MODE_BAD_SHAPE:
            return image[:, :-1].copy()
        if kind is Fault.MODE_BAD_DTYPE:
            return image.astype(np.uint32)
        if kind is Fault.MODE_NOT_ARRAY:
            return image.tolist()  # type: ignore[return-value]
        if kind is Fault.MODE_ALIAS:
            return frame.image
        if kind is Fault.RENDER_OVER_BUDGET:
            self.clock.advance(OVERRUN_SECONDS)
        return image


class FaultyStage:
    """Adds ``STAGE_SHIFT`` to every pixel, into a new array."""

    def __init__(self, plan: FaultPlan, clock: StepClock) -> None:
        self.plan = plan
        self.clock = clock
        self.applied: list[int] = []

    def apply(self, frame: Frame, image: np.ndarray) -> np.ndarray:
        self.applied.append(frame.index)
        self.clock.advance(STEP_SECONDS)
        kind = self.plan.take(frame.index, STAGE_FAULTS)
        if kind is Fault.STAGE_RAISE:
            raise InjectedFault(f"stage fault on frame {frame.index}")
        if kind is Fault.STAGE_BAD_IMAGE:
            return (image[:-1] + STAGE_SHIFT).astype(image.dtype)
        if kind is Fault.STAGE_ALIAS:
            return frame.image
        return image + np.asarray(STAGE_SHIFT, dtype=image.dtype)


@dataclass(frozen=True)
class Written:
    """One sink write: a copy of the image, and how it related to the input frame."""

    index: int
    image: np.ndarray
    shares_input: bool
    """``np.shares_memory`` with the input frame's image, at write time."""
    equals_input: bool
    original_shape: tuple[int, ...]
    original_dtype: np.dtype


class RecordingSink:
    """Keeps a copy of every written frame with its index; records protocol misuse."""

    critical = True

    def __init__(self, inputs: Mapping[int, np.ndarray]) -> None:
        self.inputs = inputs
        self.writes: list[Written] = []
        self.events: list[str] = []
        self.violations: list[str] = []
        self.state = "new"

    def open(self, info: SourceInfo) -> None:
        if self.state != "new":
            self.violations.append(f"open while {self.state}")
        self.state = "open"
        self.events.append("open")

    def write(self, image: np.ndarray, meta: FrameMeta) -> None:
        if self.state != "open":
            self.violations.append(f"write of frame {meta.index} while {self.state}")
        source = self.inputs.get(meta.index)
        is_array = isinstance(image, np.ndarray)
        shares = bool(is_array and source is not None and np.shares_memory(image, source))
        equals = bool(is_array and source is not None and np.array_equal(image, source))
        self.writes.append(
            Written(
                index=meta.index,
                image=np.array(image, copy=True),
                shares_input=shares,
                equals_input=equals,
                original_shape=tuple(image.shape) if is_array else (),
                original_dtype=image.dtype if is_array else np.dtype(object),
            )
        )

    def close(self) -> None:
        if self.state != "open":
            self.violations.append(f"close while {self.state}")
        self.state = "closed"
        self.events.append("close")

    def abort(self) -> None:
        self.state = "aborted"
        self.events.append("abort")
