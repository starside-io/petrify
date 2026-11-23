"""Randomized fail-closed property test at the sink boundary.

Each case runs the real :class:`~petrify.core.engine.Engine` over synthetic frames
with the components of ``tests/faults.py``: a fault plan, random subjects, a random
lookahead and cold start, a detection policy, an optional refiner and stage, and up
to two reconfigure calls. Then it checks what the sink received, frame by frame.

Properties checked on every case:

1. Every frame written to the sink is all zeros or exactly the :class:`TagMode`
   output for that frame index, salt of the mode that rendered it, and the subject
   flag the engine decided (``FrameTrace.final``), plus ``STAGE_SHIFT`` when a stage
   is configured. It has the input's shape and dtype, never equals the input frame,
   and never shares memory with it. The one documented exception is an alias fault
   (the mode or stage returning the input array) on a frame whose final decision is
   False: that is the engine's pass-through, and property 3 shows such a frame is
   outside every promised window.
2. Every frame on which a fault took effect is black and counted in
   ``failed_closed``, and no other frame is black. An alias fault takes effect only
   on a frame whose final decision is True. ``over_budget`` counts the budget faults.
3. Coverage. A frame is "raw" when the active detectors saw a subject on it, or its
   detection failed (detector or refiner raised or returned a malformed result) or
   went over budget: the engine pushes all of these as subject present. Every frame
   in the promised set below is rendered with the subject flag True, or is black.
   Never rendered as "no subject".

   The promise, derived from ``LookaheadBuffer`` and ``Engine.reconfigure``.
   Segments are the runs of frames between two reconfigure calls that rebuild the
   buffer (every call except one that changes nothing, or only the budget). Segment
   ``k`` has lookahead ``L_k``. A reconfigure call at boundary ``b`` (``b`` frames
   stepped before it) is made with old lookahead ``L_o`` and new lookahead ``L_n``.

   a. Start of stream: frames ``0`` to ``cold_start_frames - 1``.
   b. Around a raw frame ``h`` in segment ``k``: every frame ``j`` with
      ``h <= j <= h + L_k`` (forward, across any number of switches, clipped at the
      end of the stream), and every frame ``j`` with ``h - L_k <= j < h`` such that,
      for every switch ``b`` with ``j < b <= h``, ``j >= b - L_o`` holds (backward the
      engine can reach across a switch only through the old buffered tail, because
      older frames were already written; clipped at frame 0).
   c. At a switch ``b``: the old buffered tail, the frames from
      ``max(start of the old segment, b - L_o)`` to ``b - 1``; and the frames from
      ``b`` to ``b + F - 1``, where ``F`` is the largest of ``L_o``, ``L_n``, the rest
      of a cold window already in progress (frames before its end), and the new
      ``cold_start_frames`` when the call installed detectors (``detectors`` given),
      activated a refiner, or changed ``lookahead`` or ``cold_start_frames``.
      Clipped at the end of the stream.

4. Accounting: ``frozen + passed_through + failed_closed == frames`` and equals the
   number of input frames; every frame is written exactly once and in order; the
   observer and the returned outputs see the same indices; the sink is opened once,
   closed once, never aborted; nothing is written after the run ends (a step after
   flush, and again after close, raises and writes nothing).
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pytest

from petrify.core import Engine, EngineConfig, FrameTrace, Output
from tests.faults import (
    ALIAS_FAULTS,
    BUDGET_MS,
    DETECTION_FAULTS,
    DETECTOR_FAULTS,
    DTYPE,
    INFO,
    MODE_FAULTS,
    MODE_SALTS,
    OVER_BUDGET_FAULTS,
    REFINER_FAULTS,
    SHAPE,
    STAGE_FAULTS,
    STAGE_SHIFT,
    Fault,
    FaultPlan,
    FaultyDetector,
    FaultyRefiner,
    FaultyStage,
    FrameSource,
    RecordingSink,
    StepClock,
    TagMode,
    input_image,
    tag_image,
)

SEEDS = 300

# Detector lists the cases pick from: (primary is stateful, secondary or None).
# The primary always comes first, so its faults always fire.
DETECTOR_SETS: tuple[tuple[bool, bool | None], ...] = (
    (False, None),
    (True, False),
    (False, True),
)


@dataclass(frozen=True)
class Reconfig:
    """One reconfigure call, made after ``at`` frames were stepped."""

    at: int
    lookahead: int | None = None
    cold_start: int | None = None
    detector_set: int | None = None
    mode: tuple[int, str] | None = None
    """``(salt, policy)`` of a new :class:`TagMode`."""
    budget_ms: float | None = None
    """A budget-only change (used when nothing else changes)."""


@dataclass(frozen=True)
class Scenario:
    seed: int | None
    frames: int
    lookahead: int
    cold_start: int
    subjects: frozenset[int]
    faults: dict[int, Fault]
    salt: int
    policy: str
    detector_set: int
    refiner: bool
    stage: bool
    reconfigs: tuple[Reconfig, ...] = ()


@dataclass(frozen=True)
class Segment:
    """Frames ``start`` to ``end - 1``, stepped under one set of components."""

    start: int
    end: int
    lookahead: int
    salt: int
    refiner_active: bool
    rebuilt: bool
    """True when this segment starts at a reconfigure call that rebuilt the buffer."""
    tail_from: int
    """For a rebuilt segment: first frame of the old buffered tail."""
    forced_until: int
    """For a rebuilt segment (or the first): frames before this are forced final."""


@dataclass
class Result:
    scenario: Scenario
    stats: object
    traces: list[FrameTrace]
    outputs: list[Output]
    sink: RecordingSink
    plan: FaultPlan
    segments: list[Segment]
    late_write_raised: bool


def _detectors(plan: FaultPlan, scenario: Scenario, which: int, clock: StepClock) -> list:
    primary_stateful, secondary = DETECTOR_SETS[which]
    out = [
        FaultyDetector(
            plan, scenario.subjects, clock, primary=True, stateful=primary_stateful
        )
    ]
    if secondary is not None:
        out.append(
            FaultyDetector(plan, scenario.subjects, clock, primary=False, stateful=secondary)
        )
    return out


def run_scenario(scenario: Scenario) -> Result:
    """Drive the engine through ``scenario`` and record everything the checks need."""
    plan = FaultPlan(faults=dict(scenario.faults))
    clock = StepClock()
    source = FrameSource(scenario.frames)
    inputs = {frame.index: frame.image for frame in source.frames}
    sink = RecordingSink(inputs)
    traces: list[FrameTrace] = []
    refiners = [FaultyRefiner(plan, clock)] if scenario.refiner else []
    stages = [FaultyStage(plan, clock)] if scenario.stage else []
    config = EngineConfig(
        lookahead=scenario.lookahead,
        cold_start_frames=scenario.cold_start,
        frame_budget_ms=BUDGET_MS,
    )
    engine = Engine(
        _detectors(plan, scenario, scenario.detector_set, clock),
        TagMode(plan, clock, salt=scenario.salt, policy=scenario.policy),
        stages,
        [sink],
        config,
        traces.append,
        clock,
        refiners=refiners,
    )

    policy = scenario.policy
    salt = scenario.salt
    refiner_active = scenario.refiner and policy == "all"
    segments: list[Segment] = []
    seg_start = 0
    seg_rebuilt = False
    seg_tail = 0
    seg_forced = min(scenario.cold_start, scenario.frames)
    cold_end = scenario.cold_start
    outputs: list[Output] = []
    pending = sorted(scenario.reconfigs, key=lambda r: r.at)

    engine.open(INFO)

    def close_segment(end: int) -> None:
        segments.append(
            Segment(
                start=seg_start,
                end=end,
                lookahead=engine.config.lookahead,
                salt=salt,
                refiner_active=refiner_active,
                rebuilt=seg_rebuilt,
                tail_from=seg_tail,
                forced_until=seg_forced,
            )
        )

    def apply(rc: Reconfig) -> None:
        nonlocal policy, salt, refiner_active, seg_start, seg_rebuilt, seg_tail
        nonlocal seg_forced, cold_end
        old = engine.config
        new = old
        if rc.lookahead is not None or rc.cold_start is not None:
            new = replace(
                new,
                lookahead=old.lookahead if rc.lookahead is None else rc.lookahead,
                cold_start_frames=old.cold_start_frames if rc.cold_start is None else rc.cold_start,
            )
        if rc.budget_ms is not None:
            new = replace(new, frame_budget_ms=rc.budget_ms)
        detectors = (
            None if rc.detector_set is None else _detectors(plan, scenario, rc.detector_set, clock)
        )
        mode = (
            None
            if rc.mode is None
            else TagMode(plan, clock, salt=rc.mode[0], policy=rc.mode[1])
        )
        rebuild = (
            detectors is not None
            or mode is not None
            or replace(new, frame_budget_ms=old.frame_budget_ms) != old
        )
        new_policy = policy if rc.mode is None else rc.mode[1]
        new_refiner_active = scenario.refiner and new_policy == "all"
        if rebuild:
            close_segment(rc.at)
            cold = (
                detectors is not None
                or (new_refiner_active and not refiner_active)
                or new.lookahead != old.lookahead
                or new.cold_start_frames != old.cold_start_frames
            )
            forced = max(cold_end - rc.at, 0)
            if cold:
                forced = max(forced, new.cold_start_frames)
            forced = max(forced, old.lookahead, new.lookahead)
            seg_tail = max(seg_start, rc.at - old.lookahead)
            seg_start = rc.at
            seg_rebuilt = True
            seg_forced = min(rc.at + forced, scenario.frames)
            cold_end = rc.at + forced
        outputs.extend(engine.reconfigure(detectors=detectors, mode=mode, config=new))
        policy = new_policy
        refiner_active = new_refiner_active
        if rc.mode is not None:
            salt = rc.mode[0]

    for frame in source:
        while pending and pending[0].at == frame.index:
            apply(pending.pop(0))
        outputs.extend(engine.step(frame))
    while pending:
        apply(pending.pop(0))
    outputs.extend(engine.flush())
    close_segment(scenario.frames)
    late_write_raised = _step_is_refused(engine, sink, source)  # Flushed, not closed.
    stats = engine.close()
    late_write_raised = _step_is_refused(engine, sink, source) and late_write_raised
    return Result(scenario, stats, traces, outputs, sink, plan, segments, late_write_raised)


def _step_is_refused(engine: Engine, sink: RecordingSink, source: FrameSource) -> bool:
    """True when a further step raises ``RuntimeError`` and writes nothing."""
    writes_before = len(sink.writes)
    try:
        engine.step(source.frames[-1])
    except RuntimeError:
        return len(sink.writes) == writes_before
    return False


def _segment_of(segments: Sequence[Segment], index: int) -> Segment:
    for segment in segments:
        if segment.start <= index < segment.end:
            return segment
    raise AssertionError(f"frame {index} is in no segment")


def expected_fired(result: Result) -> dict[int, Fault]:
    """The planned faults whose component ran on that frame."""
    sc = result.scenario
    out: dict[int, Fault] = {}
    for index, kind in sc.faults.items():
        if index >= sc.frames:
            continue
        segment = _segment_of(result.segments, index)
        if kind in REFINER_FAULTS and not segment.refiner_active:
            continue
        if kind in STAGE_FAULTS and not sc.stage:
            continue
        out[index] = kind
    return out


def raw_frames(result: Result) -> set[int]:
    sc = result.scenario
    raw = {i for i in sc.subjects if i < sc.frames}
    raw |= {i for i, kind in result.plan.fired.items() if kind in DETECTION_FAULTS}
    return raw


def promised_frames(result: Result) -> set[int]:
    """Frames the engine promises to render with the subject flag True (or black)."""
    sc = result.scenario
    n = sc.frames
    segments = result.segments
    # (boundary, old lookahead) of every call that rebuilt the buffer, in order.
    switches = [
        (segment.start, previous.lookahead)
        for previous, segment in zip(segments, segments[1:])
        if segment.rebuilt
    ]
    promised: set[int] = set(range(min(sc.cold_start, n)))
    for segment in segments:
        if segment.rebuilt:
            promised |= set(range(segment.tail_from, segment.start))
            promised |= set(range(segment.start, min(segment.forced_until, n)))
    for h in raw_frames(result):
        lookahead = _segment_of(result.segments, h).lookahead
        promised |= set(range(h, min(h + lookahead, n - 1) + 1))
        for j in range(max(h - lookahead, 0), h):
            if all(j >= b - old for b, old in switches if j < b <= h):
                promised.add(j)
    return promised


def check(result: Result) -> list[str]:
    """Return every property violation, one line each (empty when all hold)."""
    sc = result.scenario
    n = sc.frames
    problems: list[str] = []
    traces = {t.index: t for t in result.traces}
    writes = result.sink.writes
    fired = result.plan.fired

    # Property 4: accounting and order.
    stats = result.stats
    if stats.frames != n:  # type: ignore[attr-defined]
        problems.append(f"stats.frames {stats.frames} != {n}")  # type: ignore[attr-defined]
    total = stats.frozen + stats.passed_through + stats.failed_closed  # type: ignore[attr-defined]
    if total != n:
        problems.append(f"frozen + passed_through + failed_closed = {total} != {n}")
    if [w.index for w in writes] != list(range(n)):
        problems.append(f"sink indices {[w.index for w in writes]} != 0..{n - 1}")
    if [t.index for t in result.traces] != list(range(n)):
        problems.append(f"trace indices {[t.index for t in result.traces]} != 0..{n - 1}")
    if [o.meta.index for o in result.outputs] != list(range(n)):
        problems.append("returned outputs are not each frame once, in order")
    if result.sink.violations:
        problems.append(f"sink protocol violations: {result.sink.violations}")
    if result.sink.events != ["open", "close"]:
        problems.append(f"sink events {result.sink.events} != ['open', 'close']")
    if not result.late_write_raised:
        problems.append("a step after flush or close did not raise, or wrote to the sink")
    if problems:
        return problems  # The per-frame checks below assume one write per frame.

    # The harness itself: each planned fault fired exactly when its component ran.
    want_fired = expected_fired(result)
    if fired != want_fired:
        problems.append(f"fired faults {_names(fired)} != expected {_names(want_fired)}")

    effective = {
        i for i, kind in fired.items() if kind not in ALIAS_FAULTS or traces[i].final
    }
    promised = promised_frames(result)
    raw = raw_frames(result)
    frozen = passed = failed = 0
    for written in writes:
        i = written.index
        trace = traces[i]
        segment = _segment_of(result.segments, i)
        kind = fired.get(i)
        label = f"frame {i}" + (f" ({kind.value})" if kind else "")
        if written.original_shape != SHAPE or written.original_dtype != DTYPE:
            problems.append(
                f"{label}: written shape {written.original_shape} dtype {written.original_dtype}"
            )
            continue
        black = not written.image.any()
        pass_through = kind in ALIAS_FAULTS and not trace.final
        # Property 1.
        if pass_through:
            want = input_image(i)
            if kind is Fault.MODE_ALIAS and sc.stage:
                want = want + np.asarray(STAGE_SHIFT, dtype=DTYPE)
            if not np.array_equal(written.image, want):
                problems.append(f"{label}: pass-through frame is not the expected image")
        else:
            if written.shares_input:
                problems.append(f"{label}: written image shares memory with the input")
            if written.equals_input:
                problems.append(f"{label}: written image equals the input frame")
            want = tag_image(i, trace.final, segment.salt)
            if sc.stage:
                want = want + np.asarray(STAGE_SHIFT, dtype=DTYPE)
            if not black and not np.array_equal(written.image, want):
                problems.append(
                    f"{label}: written image is neither black nor the tag for "
                    f"final={trace.final} salt={segment.salt}"
                )
        # Property 2.
        if i in effective and not black:
            problems.append(f"{label}: faulted frame is not black")
        if i in effective and not trace.failed_closed:
            problems.append(f"{label}: faulted frame not traced as failed closed")
        if i not in effective and (black or trace.failed_closed):
            problems.append(f"{label}: black or failed closed without an effective fault")
        if trace.raw != (i in raw):
            problems.append(f"{label}: trace.raw {trace.raw} != expected {i in raw}")
        # Property 3, read from the pixels: a promised frame carries the True tag or is black.
        if i in promised and not black:
            flagged = tag_image(i, True, segment.salt)
            if sc.stage:
                flagged = flagged + np.asarray(STAGE_SHIFT, dtype=DTYPE)
            if not np.array_equal(written.image, flagged):
                problems.append(f"{label}: promised frame was not rendered as subject present")
        if black:
            failed += 1
        elif trace.final:
            frozen += 1
        else:
            passed += 1

    counted = (stats.frozen, stats.passed_through, stats.failed_closed)  # type: ignore[attr-defined]
    if counted != (frozen, passed, failed):
        problems.append(
            f"stats (frozen, passed_through, failed_closed) {counted} != sink view "
            f"{(frozen, passed, failed)}"
        )
    over = sum(1 for i in effective if fired[i] in OVER_BUDGET_FAULTS)
    if stats.over_budget != over:  # type: ignore[attr-defined]
        problems.append(f"stats.over_budget {stats.over_budget} != {over}")  # type: ignore[attr-defined]
    return problems


def _names(faults: dict[int, Fault]) -> dict[int, str]:
    return {i: kind.value for i, kind in sorted(faults.items())}


def draw(seed: int) -> Scenario:
    """Draw one scenario from ``random.Random(seed)``."""
    rng = random.Random(seed)
    frames = rng.randint(1, 40)
    lookahead = rng.randint(0, 5)
    cold_start = rng.choice([0, 0, 1, 2, 4])
    density = rng.random() * 0.25
    subjects = frozenset(i for i in range(frames) if rng.random() < density)
    refiner = rng.random() < 0.5
    stage = rng.random() < 0.5
    policy = rng.choice(["first_hit", "all"])
    kinds = sorted(
        DETECTOR_FAULTS
        | MODE_FAULTS
        | (REFINER_FAULTS if refiner else frozenset())
        | (STAGE_FAULTS if stage else frozenset()),
        key=lambda k: k.value,
    )
    fault_rate = rng.random() * 0.3
    faults = {i: rng.choice(kinds) for i in range(frames) if rng.random() < fault_rate}
    reconfigs: list[Reconfig] = []
    for _ in range(rng.choice([0, 0, 1, 1, 2])):
        at = rng.randint(0, frames)
        changes = [c for c in ("lookahead", "detectors", "mode") if rng.random() < 0.5]
        rc = Reconfig(at=at)
        if "lookahead" in changes:
            rc = replace(
                rc,
                lookahead=rng.randint(0, 5),
                cold_start=rng.choice([None, 0, 1, 3]),
            )
        if "detectors" in changes:
            rc = replace(rc, detector_set=rng.randrange(len(DETECTOR_SETS)))
        if "mode" in changes:
            rc = replace(rc, mode=(rng.randrange(MODE_SALTS), rng.choice(["first_hit", "all"])))
        if not changes:
            rc = replace(rc, budget_ms=rng.choice([BUDGET_MS, 2 * BUDGET_MS]))
        reconfigs.append(rc)
    return Scenario(
        seed=seed,
        frames=frames,
        lookahead=lookahead,
        cold_start=cold_start,
        subjects=subjects,
        faults=faults,
        salt=rng.randrange(MODE_SALTS),
        policy=policy,
        detector_set=rng.randrange(len(DETECTOR_SETS)),
        refiner=refiner,
        stage=stage,
        reconfigs=tuple(reconfigs),
    )


def _assert_holds(scenario: Scenario) -> Result:
    result = run_scenario(scenario)
    problems = check(result)
    assert not problems, f"seed {scenario.seed}: {scenario}\n" + "\n".join(problems)
    return result


# ---------------------------------------------------------------------------
# Randomized
# ---------------------------------------------------------------------------


def test_every_sink_frame_is_black_or_processed_and_coverage_holds_for_300_seeds() -> None:
    """Properties 1 to 4 (module docstring) on 300 seeded random scenarios."""
    started = time.perf_counter()
    for seed in range(SEEDS):
        _assert_holds(draw(seed))
    # Guard the suite budget; this is a measurement, never an input to the engine.
    assert time.perf_counter() - started < 30.0


def test_the_random_scenarios_exercise_every_fault_kind_and_reconfigure_shape() -> None:
    """The seeds are not vacuous: each fault fires, alias faults hit both decisions."""
    fired: set[Fault] = set()
    alias_final = {Fault.MODE_ALIAS: set(), Fault.STAGE_ALIAS: set()}
    reconfigure_counts: set[int] = set()
    rebuilt = lookahead_changes = 0
    for seed in range(SEEDS):
        result = run_scenario(draw(seed))
        traces = {t.index: t for t in result.traces}
        for index, kind in result.plan.fired.items():
            fired.add(kind)
            if kind in ALIAS_FAULTS:
                alias_final[kind].add(traces[index].final)
        reconfigure_counts.add(len(result.scenario.reconfigs))
        rebuilt += sum(1 for s in result.segments if s.rebuilt)
        lookahead_changes += sum(
            1 for r in result.scenario.reconfigs if r.lookahead is not None
        )
    assert fired == set(Fault)
    assert alias_final == {Fault.MODE_ALIAS: {False, True}, Fault.STAGE_ALIAS: {False, True}}
    assert reconfigure_counts == {0, 1, 2}
    assert rebuilt > 50 and lookahead_changes > 50


# ---------------------------------------------------------------------------
# Deterministic, one per fault kind
# ---------------------------------------------------------------------------


def _single(kind: Fault, *, at: int, subjects: Sequence[int]) -> Scenario:
    return Scenario(
        seed=None,
        frames=12,
        lookahead=2,
        cold_start=0,
        subjects=frozenset(subjects),
        faults={at: kind},
        salt=1,
        policy="all",
        detector_set=1,
        refiner=True,
        stage=True,
    )


@pytest.mark.parametrize("kind", list(Fault), ids=lambda k: k.value)
def test_a_faulted_subject_frame_is_black_and_counted(kind: Fault) -> None:
    """The fault hits frame 5, which has a subject: black, counted, neighbors frozen."""
    result = _assert_holds(_single(kind, at=5, subjects=[5]))
    image = result.sink.writes[5].image
    assert not image.any(), kind.value
    assert result.traces[5].failed_closed, kind.value
    assert result.stats.failed_closed == 1, kind.value  # type: ignore[attr-defined]
    for j in (3, 4, 6, 7):
        assert result.traces[j].final, (kind.value, j)


@pytest.mark.parametrize("kind", sorted(set(Fault) - ALIAS_FAULTS, key=lambda k: k.value))
def test_a_faulted_frame_without_a_subject_is_black_and_counted(kind: Fault) -> None:
    """The fault hits frame 9, far from the subject at 2: still black and counted."""
    result = _assert_holds(_single(kind, at=9, subjects=[2]))
    assert not result.sink.writes[9].image.any(), kind.value
    assert result.traces[9].failed_closed, kind.value
    assert result.stats.failed_closed == 1, kind.value  # type: ignore[attr-defined]
    frozen_neighbors = kind in DETECTION_FAULTS
    for j in (7, 8, 10, 11):
        assert result.traces[j].final is frozen_neighbors, (kind.value, j)


@pytest.mark.parametrize("kind", sorted(ALIAS_FAULTS, key=lambda k: k.value))
def test_an_alias_on_a_pass_through_frame_passes_the_input_through(kind: Fault) -> None:
    """Documented pass-through: final False lets the input reach the sink unchanged."""
    result = _assert_holds(_single(kind, at=9, subjects=[2]))
    trace = result.traces[9]
    assert not trace.final and not trace.failed_closed
    assert result.sink.writes[9].shares_input == (kind is Fault.STAGE_ALIAS)
    assert result.stats.failed_closed == 0  # type: ignore[attr-defined]


def test_a_hit_just_after_a_lookahead_change_freezes_the_old_tail() -> None:
    """Lookahead 3 then 1 at boundary 8; a subject at 8 covers 5 to 9 and stops there."""
    scenario = replace(
        _single(Fault.MODE_RAISE, at=0, subjects=[8]),
        faults={},
        lookahead=3,
        reconfigs=(Reconfig(at=8, lookahead=1),),
    )
    result = _assert_holds(scenario)
    finals = [t.final for t in result.traces]
    assert finals[5:11] == [True] * 6  # Old tail 5..7, then max(3, 1) frames 8..10.
    assert finals[:5] == [False] * 5 and finals[11:] == [False]
