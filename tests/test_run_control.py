"""Tests for petrify.server.control.RunControl (no web dependencies, no devices)."""

from __future__ import annotations

import dataclasses
import json
import logging
import subprocess
import sys
import threading
from typing import Any

import pytest

from petrify.core import (
    BackgroundDiffConfig,
    EngineConfig,
    MotionConfig,
    PetrifyConfig,
    RoiConfig,
)
from petrify.server.control import (
    RUNTIME_FIELDS,
    RunControl,
    RunSnapshot,
    RunStats,
    SourceFacts,
    config_to_dict,
)

SIZE = (320, 256)


def _control(config: PetrifyConfig | None = None) -> RunControl:
    return RunControl(config or PetrifyConfig(), SIZE)


# ---------------------------------------------------------------------------
# Whitelist
# ---------------------------------------------------------------------------

ALLOWED: list[tuple[str, dict[str, Any], Any]] = [
    ("detector", {"detector": "both"}, lambda c: c.detector == "both"),
    ("motion.threshold", {"motion": {"threshold": 300}}, lambda c: c.motion.threshold == 300),
    ("motion.var_threshold", {"motion": {"var_threshold": 24.5}},
     lambda c: c.motion.var_threshold == 24.5),
    ("motion.detect_shadows", {"motion": {"detect_shadows": True}},
     lambda c: c.motion.detect_shadows is True),
    ("person.min_confidence", {"person": {"min_confidence": 0.8}},
     lambda c: c.person.min_confidence == 0.8),
    ("person.min_box_area", {"person": {"min_box_area": 900}},
     lambda c: c.person.min_box_area == 900),
    ("person.scale", {"person": {"scale": 1.2}}, lambda c: c.person.scale == 1.2),
    ("background_diff.enabled", {"background_diff": {"enabled": False}},
     lambda c: c.background_diff is None),
    ("background_diff.pixel_threshold", {"background_diff": {"pixel_threshold": 40}},
     lambda c: c.background_diff.pixel_threshold == 40),
    ("background_diff.area_threshold", {"background_diff": {"area_threshold": 50}},
     lambda c: c.background_diff.area_threshold == 50),
    ("engine.lookahead", {"engine": {"lookahead": 600}}, lambda c: c.engine.lookahead == 600),
    ("engine.frame_budget_ms", {"engine": {"frame_budget_ms": 40}},
     lambda c: c.engine.frame_budget_ms == 40.0),
    ("roi", {"roi": [10, 20, 100, 30]},
     lambda c: c.roi.mode == "manual" and c.roi.box == (10, 20, 100, 30)),
]


def test_every_runtime_field_has_an_allowed_case() -> None:
    assert sorted(name for name, _, _ in ALLOWED) == sorted(RUNTIME_FIELDS)


@pytest.mark.parametrize(("name", "patch", "check"), ALLOWED, ids=[a[0] for a in ALLOWED])
def test_each_runtime_field_is_accepted(name: str, patch: dict[str, Any], check: Any) -> None:
    control = _control()
    accepted = control.submit(patch)
    assert check(accepted), name
    assert control.take_pending() is accepted


def test_roi_disabled_and_budget_null_are_accepted() -> None:
    control = _control(PetrifyConfig(engine=EngineConfig(frame_budget_ms=50.0)))
    accepted = control.submit({"roi": "disabled", "engine": {"frame_budget_ms": None}})
    assert accepted.roi.mode == "disabled" and accepted.roi.box is None
    assert accepted.engine.frame_budget_ms is None


def test_differ_turned_on_again_gets_the_remembered_thresholds() -> None:
    start = PetrifyConfig(background_diff=BackgroundDiffConfig(pixel_threshold=31,
                                                               area_threshold=77))
    control = _control(start)
    off = control.submit({"background_diff": {"enabled": False}})
    assert off.background_diff is None
    on = control.submit({"background_diff": {"enabled": True, "area_threshold": 99}})
    assert on.background_diff == BackgroundDiffConfig(pixel_threshold=31, area_threshold=99)


def test_differ_off_at_start_turns_on_with_defaults_at_the_motion_downsample() -> None:
    start = dataclasses.replace(PetrifyConfig(), background_diff=None)
    on = _control(start).submit({"background_diff": {"enabled": True}})
    assert on.background_diff == BackgroundDiffConfig(downsample=start.motion.downsample)


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"background": {"strategy": "running"}}, "cannot be changed"),
        ({"encode": {"crf": 18}}, "cannot be changed"),
        ({"source": "camera:0"}, "cannot be changed"),
        ({"outputs": {"file": "x.mp4"}}, "cannot be changed"),
        ({"mode": "erase"}, "cannot be changed"),
        ({"motion": {"history": 50}}, "motion.history cannot be changed"),
        ({"motion": {"downsample": 0.25}}, "motion.downsample cannot be changed"),
        ({"person": {"win_stride": [4, 4]}}, "person.win_stride cannot be changed"),
        ({"background_diff": {"downsample": 1.0}}, "cannot be changed"),
        ({"engine": {"cold_start_frames": 0}}, "engine.cold_start_frames cannot be changed"),
        ({"engine": {"fail_closed": "pass"}}, "cannot be changed"),
        ({"sample_check": False}, "does not exist"),
        ({"nonsense": 1}, "does not exist"),
        ({"motion": {"nonsense": 1}}, "does not exist"),
        ({"roi": {"mode": "auto"}}, "roi must be"),
    ],
)
def test_forbidden_and_unknown_fields_are_rejected(patch: dict[str, Any], message: str) -> None:
    control = _control()
    with pytest.raises(ValueError, match=message):
        control.submit(patch)
    assert control.take_pending() is None


def test_unknown_key_is_not_echoed() -> None:
    with pytest.raises(ValueError) as info:
        _control().submit({"<script>alert(1)</script>": 1})
    assert "script" not in str(info.value)


@pytest.mark.parametrize(
    "patch",
    [
        None,
        [],
        "detector",
        {},
        {"detector": "face"},
        {"detector": 1},
        {"motion": 5},
        {"motion": {"threshold": True}},
        {"motion": {"threshold": 1.5}},
        {"motion": {"threshold": "300"}},
        {"motion": {"threshold": -1}},
        {"motion": {"threshold": 10_000_001}},
        {"motion": {"var_threshold": 0}},
        {"motion": {"var_threshold": 1000.5}},
        {"motion": {"var_threshold": float("nan")}},
        {"motion": {"detect_shadows": 1}},
        {"person": {"min_confidence": -0.1}},
        {"person": {"min_confidence": 10.1}},
        {"person": {"min_box_area": -5}},
        {"person": {"scale": 1.0}},
        {"person": {"scale": 2.5}},
        {"background_diff": {"pixel_threshold": 256}},
        {"background_diff": {"area_threshold": -1}},
        {"background_diff": {"enabled": "yes"}},
        {"background_diff": {"enabled": False, "pixel_threshold": 30}},
        {"engine": {"lookahead": -1}},
        {"engine": {"lookahead": 601}},
        {"engine": {"lookahead": 3.0}},
        {"engine": {"frame_budget_ms": 0}},
        {"engine": {"frame_budget_ms": -3}},
        {"engine": {"frame_budget_ms": float("inf")}},
        {"engine": {"frame_budget_ms": "fast"}},
        {"roi": "auto"},
        {"roi": [0, 0, 10]},
        {"roi": [0, 0, 0, 10]},
        {"roi": [-1, 0, 10, 10]},
        {"roi": [300, 0, 21, 10]},
        {"roi": [0, 250, 10, 7]},
        {"roi": [0, 0, 10.0, 10]},
        {"roi": [0, 0, True, 10]},
    ],
)
def test_type_and_range_errors(patch: Any) -> None:
    control = _control()
    with pytest.raises(ValueError):
        control.submit(patch)
    assert control.take_pending() is None


def test_roi_box_may_touch_the_frame_edges() -> None:
    # Boxes stay under the area cap (W1); each touches two edges of the frame.
    accepted = _control().submit({"roi": [0, 0, SIZE[0], 38]})
    assert accepted.roi.box == (0, 0, SIZE[0], 38)
    corner = (SIZE[0] - 100, SIZE[1] - 50, 100, 50)
    assert _control().submit({"roi": list(corner)}).roi.box == corner


def test_no_partial_application_on_error() -> None:
    control = _control()
    before = control.applied
    with pytest.raises(ValueError, match="engine.lookahead"):
        control.submit({"detector": "both", "motion": {"threshold": 10},
                        "engine": {"lookahead": 9999}})
    assert control.take_pending() is None
    # The next accepted patch is based on the unchanged config.
    accepted = control.submit({"engine": {"lookahead": 4}})
    assert accepted.detector == before.detector
    assert accepted.motion == before.motion


def test_submit_rejection_sets_last_error() -> None:
    control = _control()
    with pytest.raises(ValueError) as info:
        control.submit({"engine": {"lookahead": -2}})
    assert control.snapshot().last_error == str(info.value)


# ---------------------------------------------------------------------------
# Coalescing and the applied config
# ---------------------------------------------------------------------------


def test_bursts_coalesce_latest_wins_and_build_on_each_other() -> None:
    control = _control()
    control.submit({"engine": {"lookahead": 5}})
    control.submit({"motion": {"threshold": 123}})
    last = control.submit({"detector": "both"})
    pending = control.take_pending()
    assert pending is last
    assert (pending.engine.lookahead, pending.motion.threshold, pending.detector) == (
        5, 123, "both")
    assert control.take_pending() is None


def test_applied_config_is_what_the_snapshot_reports() -> None:
    control = _control()
    accepted = control.submit({"engine": {"lookahead": 7}})
    # Submitted but not applied: the snapshot keeps the old config.
    assert control.snapshot().config["engine"]["lookahead"] == 10
    assert control.applied.engine.lookahead == 10
    control.take_pending()
    control.report_applied(accepted)
    control.publish(RunSnapshot(config=config_to_dict(control.applied), running=True))
    assert control.snapshot().config["engine"]["lookahead"] == 7


def test_rejected_by_the_loop_reverts_the_base_and_records_the_error() -> None:
    control = _control()
    control.submit({"engine": {"lookahead": 7}})
    control.take_pending()
    control.report_rejected("bad combination")
    assert control.snapshot().last_error == "bad combination"
    # The next patch is based on the applied config, not on the rejected one.
    accepted = control.submit({"motion": {"threshold": 1}})
    assert accepted.engine.lookahead == 10


def test_stop_request() -> None:
    control = _control()
    assert control.stop_requested is False
    control.request_stop()
    assert control.stop_requested is True


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def test_initial_snapshot_is_starting_not_ended() -> None:
    snap = _control().snapshot()
    assert (snap.running, snap.ended) == (False, False)
    assert snap.to_dict()["config"] == config_to_dict(PetrifyConfig())


def test_snapshot_is_immutable_and_json_safe() -> None:
    snap = RunSnapshot(
        config=config_to_dict(PetrifyConfig()),
        stats=RunStats(frames=3),
        source=SourceFacts(uri="clip.mp4", width=320, height=256, fps=25.0),
        roi=(1, 2, 3, 4),
        running=True,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.running = False  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.stats.frames = 9  # type: ignore[misc]
    with pytest.raises(TypeError):
        snap.config["detector"] = "both"  # type: ignore[index]
    with pytest.raises(TypeError):
        snap.config["engine"]["lookahead"] = 1  # type: ignore[index]
    data = snap.to_dict()
    assert json.loads(json.dumps(data, allow_nan=False)) == data
    assert data["roi"] == [1, 2, 3, 4]
    assert data["config"]["person"]["win_stride"] == [8, 8]
    data["config"]["detector"] = "changed"
    assert snap.config["detector"] == "motion"  # to_dict returns a copy.


def test_config_to_dict_is_plain_json() -> None:
    data = config_to_dict(PetrifyConfig(engine=EngineConfig(frame_budget_ms=12.5)))
    assert json.loads(json.dumps(data, allow_nan=False)) == data
    assert data["engine"]["frame_budget_ms"] == 12.5


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_submit_and_take_lose_nothing_and_apply_the_latest() -> None:
    control = _control()
    writers = 4
    per_writer = 200
    start = threading.Barrier(writers + 1)
    done = threading.Event()
    taken: list[PetrifyConfig] = []

    def writer(w: int) -> None:
        start.wait()
        for i in range(per_writer):
            control.submit({"motion": {"threshold": w * 1_000 + i}})

    def taker() -> None:
        start.wait()
        while not done.is_set():
            pending = control.take_pending()
            if pending is not None:
                taken.append(pending)
                control.report_applied(pending)
        pending = control.take_pending()
        if pending is not None:
            taken.append(pending)
            control.report_applied(pending)

    threads = [threading.Thread(target=writer, args=(w,)) for w in range(writers)]
    reader = threading.Thread(target=taker)
    for t in threads:
        t.start()
    reader.start()
    for t in threads:
        t.join()
    done.set()
    reader.join()
    assert taken, "the taker saw no config"
    # Every config handed out was a valid accepted one, each at most once.
    assert len({id(c) for c in taken}) == len(taken)
    values = {w * 1_000 + i for w in range(writers) for i in range(per_writer)}
    assert all(c.motion.threshold in values for c in taken)
    # After the dust settles nothing is pending and the last one taken is applied.
    assert control.take_pending() is None
    assert control.applied is taken[-1]


def test_take_pending_waits_for_a_submit_in_progress() -> None:
    """A submit holds the lock while it validates: take never sees half a change."""
    control = _control()
    entered = threading.Event()
    release = threading.Event()
    import petrify.server.control as module

    real = module.apply_patch

    def slow(*args: Any, **kwargs: Any) -> PetrifyConfig:
        entered.set()
        assert release.wait(5)
        return real(*args, **kwargs)

    result: list[PetrifyConfig | None] = []
    module.apply_patch = slow  # type: ignore[assignment]
    try:
        submitter = threading.Thread(target=control.submit, args=({"detector": "both"},))
        submitter.start()
        assert entered.wait(5)
        taker = threading.Thread(target=lambda: result.append(control.take_pending()))
        taker.start()
        taker.join(0.2)
        assert taker.is_alive()  # Blocked on the lock held by submit.
        release.set()
        submitter.join(5)
        taker.join(5)
    finally:
        module.apply_patch = real  # type: ignore[assignment]
        release.set()
    assert result and result[0] is not None and result[0].detector == "both"


def test_control_imports_without_web_dependencies() -> None:
    code = (
        "import sys\n"
        "sys.modules['fastapi'] = None\n"
        "sys.modules['uvicorn'] = None\n"
        "import petrify.server.control, petrify.server\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == "ok"


# ---------------------------------------------------------------------------
# W1: a manual ROI is capped, because its region is shown live
# ---------------------------------------------------------------------------

# 320 x 256 = 81920 px; the default cap of 0.15 allows 12288 px.


def test_w1_roi_over_the_area_cap_is_rejected_with_the_limit() -> None:
    control = _control()
    assert control.submit({"roi": [0, 0, 256, 48]}).roi.box == (0, 0, 256, 48)  # Exactly 15 %.
    for box in ([0, 0, 256, 49], [0, 0, SIZE[0], SIZE[1]]):
        with pytest.raises(ValueError) as info:
            control.submit({"roi": box})
        message = str(info.value)
        assert "at most 15 percent" in message
        assert "shown live" in message
    assert control.take_pending().roi.box == (0, 0, 256, 48)  # type: ignore[union-attr]


def test_w1_roi_cap_is_the_runs_configured_cap() -> None:
    control = _control(PetrifyConfig(roi=RoiConfig(max_area_cap=0.05)))  # 4096 px.
    assert control.submit({"roi": [0, 0, 100, 40]}).roi.box == (0, 0, 100, 40)
    with pytest.raises(ValueError, match="at most 5 percent"):
        control.submit({"roi": [0, 0, 100, 41]})


# ---------------------------------------------------------------------------
# W2: runtime ranges keep every detector meaningful
# ---------------------------------------------------------------------------

# Every detector downsamples by 0.5 by default: 160 x 128 = 20480 px, 25 % = 5120.
BOUNDS: list[tuple[str, str, Any, Any, str]] = [
    ("motion", "threshold", 5120, 5121, "from 0 to 5120"),
    ("background_diff", "area_threshold", 5120, 5121, "from 0 to 5120"),
    ("person", "min_box_area", 5120, 5121, "from 0 to 5120"),
    ("background_diff", "pixel_threshold", 128, 129, "from 0 to 128"),
    ("motion", "var_threshold", 100, 100.5, "at most 100"),
    ("person", "min_confidence", 2, 2.01, "at most 2"),
    ("person", "scale", 1.5, 1.51, "at most 1.5"),
]


@pytest.mark.parametrize(("section", "key", "highest", "over", "message"), BOUNDS,
                         ids=[f"{b[0]}.{b[1]}" for b in BOUNDS])
def test_w2_runtime_bound(section: str, key: str, highest: Any, over: Any, message: str) -> None:
    control = _control()
    accepted = control.submit({section: {key: highest}})
    assert getattr(getattr(accepted, section), key) == highest
    with pytest.raises(ValueError) as info:
        control.submit({section: {key: over}})
    assert f"{section}.{key}" in str(info.value)
    assert message in str(info.value)


@pytest.mark.parametrize(
    "patch",
    [
        {"motion": {"threshold": 10_000_000}},
        {"background_diff": {"pixel_threshold": 255}},
        {"background_diff": {"area_threshold": 10_000_000}},
        {"motion": {"var_threshold": 1000}},
        {"person": {"min_confidence": 10}},
        {"person": {"min_box_area": 10_000_000}},
    ],
)
def test_w2_values_from_the_review_are_rejected(patch: dict[str, Any]) -> None:
    control = _control()
    with pytest.raises(ValueError):
        control.submit(patch)
    assert control.take_pending() is None


def test_w2_area_bounds_follow_the_frame_and_the_detector_downsample() -> None:
    control = RunControl(PetrifyConfig(motion=MotionConfig(downsample=0.25)), (320, 256))
    # 80 x 64 = 5120 px, 25 % = 1280.
    assert control.submit({"motion": {"threshold": 1280}}).motion.threshold == 1280
    with pytest.raises(ValueError) as info:
        control.submit({"motion": {"threshold": 1281}})
    assert "from 0 to 1280" in str(info.value) and "25 percent" in str(info.value)
    assert "80x64" in str(info.value)


def test_w2_bounds_apply_to_the_runtime_api_only() -> None:
    # A run may start with any value the command line accepts.
    start = PetrifyConfig(motion=MotionConfig(threshold=10_000_000))
    control = _control(start)
    assert control.submit({"engine": {"lookahead": 4}}).motion.threshold == 10_000_000


# ---------------------------------------------------------------------------
# W2: weakened protections
# ---------------------------------------------------------------------------

DIFFER_OFF = ("The background differ is off: a person who stops moving is no longer "
              "detected and can be shown.")
PERSON_ONLY = ("The detector is person only and the background differ is off: nothing "
               "detects a subject the person detector misses.")
LOOKAHEAD_ZERO = ("engine.lookahead is 0: no frame is held back, so the frames just before "
                  "a detection can show a person.")
RAW_PREVIEW = ("The raw preview is on: the unprocessed feed, people included, is shown on "
               "this page.")


def _threshold(field: str, value: str, start: str) -> str:
    return (f"{field} is {value}, more than twice its startup value of {start}: detection "
            "is less sensitive and can miss a person.")


def _lookahead_below(value: int, start: int) -> str:
    return (f"engine.lookahead is {value}, below its startup value of {start}: fewer frames "
            "before a detection are held back and hidden.")


def _roi_share(percent: str) -> str:
    return (f"The manual ROI covers {percent} percent of the frame, and that region is shown "
            "live (not anonymized).")


def _apply(control: RunControl, patch: dict[str, Any]) -> tuple[str, ...]:
    new = control.submit(patch)
    assert control.take_pending() is new
    control.report_applied(new)
    return control.snapshot().weakened


def test_w2_default_run_has_no_weakened_protection() -> None:
    control = _control()
    assert control.snapshot().weakened == ()
    assert control.snapshot().to_dict()["weakened"] == []


def test_w2_weakened_differ_off_and_person_only() -> None:
    control = _control()
    assert _apply(control, {"detector": "person"}) == ()  # The differ still runs.
    assert _apply(control, {"background_diff": {"enabled": False}}) == (DIFFER_OFF, PERSON_ONLY)
    assert control.snapshot().to_dict()["weakened"] == [DIFFER_OFF, PERSON_ONLY]
    assert _apply(control, {"detector": "both"}) == (DIFFER_OFF,)
    assert _apply(control, {"background_diff": {"enabled": True}}) == ()


@pytest.mark.parametrize(
    ("patch", "field", "value", "start", "not_listed"),
    [
        ({"motion": {"threshold": 1001}}, "motion.threshold", "1001", "500",
         {"motion": {"threshold": 1000}}),
        ({"motion": {"var_threshold": 33}}, "motion.var_threshold", "33", "16",
         {"motion": {"var_threshold": 32}}),
        ({"person": {"min_confidence": 0.7}}, "person.min_confidence", "0.7", "0.3",
         {"person": {"min_confidence": 0.6}}),
        ({"person": {"min_box_area": 3001}}, "person.min_box_area", "3001", "1500",
         {"person": {"min_box_area": 3000}}),
        ({"background_diff": {"pixel_threshold": 51}}, "background_diff.pixel_threshold",
         "51", "25", {"background_diff": {"pixel_threshold": 50}}),
        ({"background_diff": {"area_threshold": 401}}, "background_diff.area_threshold",
         "401", "200", {"background_diff": {"area_threshold": 400}}),
    ],
)
def test_w2_weakened_threshold_above_twice_its_startup_value(
    patch: dict[str, Any], field: str, value: str, start: str, not_listed: dict[str, Any]
) -> None:
    control = _control()
    assert _apply(control, not_listed) == ()
    assert _apply(control, patch) == (_threshold(field, value, start),)


def test_w2_weakened_lookahead() -> None:
    control = _control()
    assert _apply(control, {"engine": {"lookahead": 20}}) == ()
    assert _apply(control, {"engine": {"lookahead": 9}}) == (_lookahead_below(9, 10),)
    assert _apply(control, {"engine": {"lookahead": 0}}) == (_lookahead_below(0, 10),
                                                             LOOKAHEAD_ZERO)


def test_w2_weakened_manual_roi_share() -> None:
    control = _control()
    assert _apply(control, {"roi": [0, 0, 64, 64]}) == ()  # 5.0 %: not more than 5.
    assert _apply(control, {"roi": [0, 0, 100, 50]}) == (_roi_share("6.1"),)
    assert _apply(control, {"roi": "disabled"}) == ()


def test_w2_weakened_raw_preview() -> None:
    control = _control()
    control.mark_raw_preview()
    assert control.snapshot().weakened == (RAW_PREVIEW,)


def test_w2_absolute_conditions_are_listed_at_startup() -> None:
    start = dataclasses.replace(
        PetrifyConfig(), detector="person", background_diff=None,
        engine=EngineConfig(lookahead=0), roi=RoiConfig(mode="manual", box=(0, 0, 320, 256)),
    )
    snap = _control(start).snapshot()
    assert snap.weakened == (DIFFER_OFF, PERSON_ONLY, LOOKAHEAD_ZERO, _roi_share("100.0"))


def test_w2_thresholds_compare_with_the_startup_config_not_the_defaults() -> None:
    control = _control(PetrifyConfig(motion=MotionConfig(threshold=2000)))
    assert control.snapshot().weakened == ()
    assert _apply(control, {"motion": {"threshold": 4000}}) == ()
    assert _apply(control, {"motion": {"threshold": 4001}}) == (
        _threshold("motion.threshold", "4001", "2000"),)


def test_w2_differ_thresholds_compare_with_the_startup_differ_after_a_toggle() -> None:
    control = _control()
    _apply(control, {"background_diff": {"enabled": False}})
    assert _apply(control, {"background_diff": {"enabled": True, "pixel_threshold": 60}}) == (
        _threshold("background_diff.pixel_threshold", "60", "25"),)


def test_w2_each_newly_weakened_protection_is_logged_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    control = _control()
    with caplog.at_level(logging.WARNING, logger="petrify.server.control"):
        _control().snapshot()  # Nothing is logged without a change.
        _apply(control, {"background_diff": {"enabled": False}})
        _apply(control, {"engine": {"lookahead": 5}})
        _apply(control, {"engine": {"lookahead": 4}})  # Same protection, still weakened.
        _apply(control, {"detector": "both"})
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert messages == [
        f"Weakened protection: {DIFFER_OFF}",
        f"Weakened protection: {_lookahead_below(5, 10)}",
    ]
