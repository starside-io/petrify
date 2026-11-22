"""Tests for petrify.server.control.RunControl (no web dependencies, no devices)."""

from __future__ import annotations

import dataclasses
import json
import logging
import subprocess
import sys
import threading
from typing import Any
from unittest import mock

import pytest

from petrify.core import (
    BackgroundDiffConfig,
    EngineConfig,
    MotionConfig,
    PetrifyConfig,
    RoiConfig,
)
from petrify.core.config import TrackerConfig, mode_options_from
from petrify.core.detections import DetectionKind
from petrify.core.plugin import OptionSpec
from petrify.core.registry import ModeEntry, list_modes, load_mode
from petrify.modes.erase import ERASE
from petrify.pipeline import OutputConfig, run_stream
from petrify.server import control as control_module
from petrify.server.control import (
    RUNTIME_FIELDS,
    BackgroundState,
    RunControl,
    RunSnapshot,
    RunStats,
    SourceFacts,
    config_to_dict,
    metadata_text,
    untrusted_text,
)
from tests.test_run_stream import QUIET, HookSource, ListSink, _live_config
from tests.test_run_stream import H as STREAM_H
from tests.test_run_stream import W as STREAM_W
from tests.test_run_stream_plugins import (  # noqa: F401 (fixtures used by name)
    PLUGIN_MODULE,
    _plugin_config,
    fake_persons,
    plugin,
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


# ---------------------------------------------------------------------------
# T13: the run's mode, its options, and runtime option changes
# ---------------------------------------------------------------------------

FAKE_SPECS = (
    OptionSpec("flag", "bool", False, "A runtime flag.", runtime=True),
    OptionSpec("flag_fixed", "bool", True, "A fixed flag."),
    OptionSpec("count", "int", 3, "A runtime count.", minimum=1, maximum=10, runtime=True),
    OptionSpec("count_fixed", "int", 2, "A fixed count.", minimum=0, maximum=5),
    OptionSpec("ratio", "float", 0.5, "A runtime ratio.", minimum=0.0, maximum=2.0,
               runtime=True),
    OptionSpec("ratio_fixed", "float", 1.0, "A fixed ratio."),
    OptionSpec("shape", "choice", "box", "A runtime shape.", choices=("box", "ellipse"),
               runtime=True),
    OptionSpec("shape_fixed", "choice", "box", "A fixed shape.", choices=("box", "ellipse")),
    OptionSpec("parts", "choices", ("face",), "Runtime parts.",
               choices=("face", "body", "hands"), runtime=True),
    OptionSpec("parts_fixed", "choices", (), "Fixed parts.", choices=("face", "body")),
)
FAKE_DEFAULTS = {spec.name: spec.default for spec in FAKE_SPECS}


class FakeFactory:
    """The parts of a ModeFactory that RunControl reads; records every weakened call."""

    needs_background = False
    roi_overlay = False

    def __init__(self, weakened: Any = (), *, name: str = "fake",
                 summary: Any = "A fake mode for tests.",
                 options: tuple[OptionSpec, ...] = FAKE_SPECS) -> None:
        self.name = name
        self.summary = summary
        self.options = options
        self._weakened = weakened
        self.calls: list[Any] = []

    def weakened(self, options: Any) -> Any:
        self.calls.append(options)
        if callable(self._weakened):
            return self._weakened(options)
        return self._weakened


def _fake_config(**options: Any) -> PetrifyConfig:
    return PetrifyConfig(mode="fake", mode_options=mode_options_from({**FAKE_DEFAULTS, **options}))


def _entry(name: str, *, dist: str | None = "petrify-fake", version: str | None = "1.0",
           builtin: bool = False, problem: str | None = None) -> ModeEntry:
    return ModeEntry(name, f"fake_mod:{name.upper()}", dist, version, builtin, problem)


FAKE_MODES = (
    ModeEntry("erase", "petrify.modes.erase:ERASE", None, None, True, None),
    _entry("fake"),
    _entry("other", dist="petrify-other", version="2.3"),
    _entry("Bad-Name", problem="invalid mode name 'Bad-Name'"),
)


def _fake_control(factory: FakeFactory | None = None, **options: Any) -> RunControl:
    return RunControl(_fake_config(**options), SIZE, modes=FAKE_MODES,
                      factory=factory or FakeFactory())


def _apply_mode(control: RunControl, patch: dict[str, Any]) -> PetrifyConfig:
    accepted = control.submit(patch)
    assert control.take_pending() is accepted
    control.report_applied(accepted)
    return accepted


@pytest.mark.parametrize(
    ("name", "given", "stored"),
    [
        ("flag", True, True),
        ("count", 7, 7),
        ("count", 7.0, 7),
        ("count", 10, 10),
        ("ratio", 1, 1.0),
        ("ratio", 2, 2.0),
        ("ratio", 0.25, 0.25),
        ("shape", "ellipse", "ellipse"),
        ("parts", ["body", "face"], ("body", "face")),
        ("parts", [], ()),
    ],
)
def test_mode_option_runtime_change_is_accepted_in_its_declared_type(
    name: str, given: Any, stored: Any
) -> None:
    control = _fake_control()
    accepted = control.submit({"mode_options": {name: given}})
    options = accepted.mode_options_dict()
    assert options[name] == stored and type(options[name]) is type(stored)
    # Every other option is kept as it was, in the declared order and type.
    assert {k: v for k, v in options.items() if k != name} == {
        k: v for k, v in FAKE_DEFAULTS.items() if k != name
    }
    assert [k for k, _ in accepted.mode_options] == sorted(FAKE_DEFAULTS)
    assert accepted.mode == "fake"
    assert control.take_pending() is accepted


def test_mode_option_changes_build_on_each_other_and_mix_with_other_settings() -> None:
    control = _fake_control()
    control.submit({"mode_options": {"count": 4}})
    accepted = control.submit({"mode_options": {"ratio": 1, "flag": True},
                               "engine": {"lookahead": 5}})
    options = accepted.mode_options_dict()
    assert (options["count"], options["ratio"], options["flag"]) == (4, 1.0, True)
    assert type(options["ratio"]) is float
    assert accepted.engine.lookahead == 5
    assert control.take_pending() is accepted


def test_empty_mode_options_object_changes_nothing() -> None:
    control = _fake_control()
    accepted = control.submit({"mode_options": {}})
    assert accepted.mode_options == _fake_config().mode_options


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"mode_options": {"nope": 1}}, "mode option that does not exist"),
        ({"mode_options": {"__proto__": 1}}, "mode option that does not exist"),
        ({"mode_options": {"constructor": 1}}, "mode option that does not exist"),
        ({"mode_options": {"flag_fixed": False}}, "mode_options.flag_fixed cannot be changed"),
        ({"mode_options": {"count_fixed": 1}}, "mode_options.count_fixed cannot be changed"),
        ({"mode_options": {"ratio_fixed": 1.5}}, "mode_options.ratio_fixed cannot be changed"),
        ({"mode_options": {"shape_fixed": "ellipse"}},
         "mode_options.shape_fixed cannot be changed"),
        ({"mode_options": {"parts_fixed": ["body"]}},
         "mode_options.parts_fixed cannot be changed"),
        ({"mode_options": {"flag": 1}}, "mode_options.flag must be true or false"),
        ({"mode_options": {"flag": "true"}}, "mode_options.flag must be true or false"),
        ({"mode_options": {"flag": None}}, "mode_options.flag must be true or false"),
        ({"mode_options": {"count": 1.5}}, "mode_options.count must be a whole number from 1"),
        ({"mode_options": {"count": True}}, "mode_options.count must be a whole number"),
        ({"mode_options": {"count": "3"}}, "mode_options.count must be a whole number"),
        ({"mode_options": {"count": 0}}, "mode_options.count must be a whole number from 1 to 10"),
        ({"mode_options": {"count": 11}}, "mode_options.count must be a whole number from 1"),
        ({"mode_options": {"ratio": True}}, "mode_options.ratio must be a number from 0 to 2"),
        ({"mode_options": {"ratio": "1"}}, "mode_options.ratio must be a number"),
        ({"mode_options": {"ratio": float("nan")}}, "mode_options.ratio must be a number"),
        ({"mode_options": {"ratio": float("inf")}}, "mode_options.ratio must be a number"),
        ({"mode_options": {"ratio": 2.5}}, "mode_options.ratio must be a number from 0 to 2"),
        ({"mode_options": {"ratio": -0.1}}, "mode_options.ratio must be a number"),
        ({"mode_options": {"shape": "circle"}}, "mode_options.shape must be one of: box, ellipse"),
        ({"mode_options": {"shape": 1}}, "mode_options.shape must be one of"),
        ({"mode_options": {"parts": "face"}}, "mode_options.parts must be a list of distinct"),
        ({"mode_options": {"parts": ["face", "face"]}}, "mode_options.parts must be a list"),
        ({"mode_options": {"parts": ["nope"]}}, "entries from: face, body, hands"),
        ({"mode_options": []}, "mode_options must be an object"),
        ({"mode_options": "count=3"}, "mode_options must be an object"),
        ({"mode_options": None}, "mode_options must be an object"),
        ({"mode_options": 1}, "mode_options must be an object"),
        ({"mode": "other"}, "The mode cannot be changed"),
        ({"mode": "fake", "mode_options": {"count": 4}}, "The mode cannot be changed"),
        # A valid change next to an invalid one: nothing is applied.
        ({"mode_options": {"count": 4, "ratio": 9}}, "mode_options.ratio must be"),
        ({"engine": {"lookahead": 5}, "mode_options": {"count": 99}}, "mode_options.count"),
        ({"mode_options": {"count": 4}, "engine": {"lookahead": 9999}}, "engine.lookahead"),
    ],
)
def test_invalid_mode_option_patch_is_rejected_and_changes_nothing(
    patch: dict[str, Any], message: str
) -> None:
    control = _fake_control()
    base = _apply_mode(control, {"mode_options": {"count": 5}})
    with pytest.raises(ValueError, match=message):
        control.submit(patch)
    assert control.take_pending() is None
    assert control.applied is base
    assert control.snapshot().last_error is not None
    # The next change builds on the unchanged config.
    after = control.submit({"detector": "motion"})
    assert after.mode_options == base.mode_options
    assert after.engine == base.engine
    assert dataclasses.replace(after, detector=base.detector) == base


def test_rejected_mode_option_patch_keeps_an_earlier_pending_change() -> None:
    control = _fake_control()
    first = control.submit({"mode_options": {"count": 4}})
    with pytest.raises(ValueError):
        control.submit({"mode_options": {"count": 0}})
    assert control.take_pending() is first


@pytest.mark.parametrize("value", ["<b>x</b>", 123456789, [7, 8], {"k": "v"}])
def test_mode_option_errors_do_not_echo_the_value_or_an_unknown_name(value: Any) -> None:
    control = _fake_control()
    for patch in ({"mode_options": {"count": value}}, {"mode_options": {"<i>name</i>": value}}):
        with pytest.raises(ValueError) as info:
            control.submit(patch)
        text = str(info.value)
        assert "<" not in text and "123456789" not in text and "'k'" not in text


@pytest.mark.parametrize("patch", [{"mode_options": {"count": 4}}, {"mode_options": {}},
                                   {"mode_options": "x"}])
def test_without_a_factory_every_mode_options_patch_is_rejected(patch: dict[str, Any]) -> None:
    control = RunControl(_fake_config(), SIZE)
    with pytest.raises(ValueError, match="this run does not report its mode"):
        control.submit(patch)
    assert control.take_pending() is None


def test_erase_has_no_options_to_change() -> None:
    control = RunControl(PetrifyConfig(), SIZE, modes=list_modes([]), factory=ERASE)
    with pytest.raises(ValueError, match="mode option that does not exist"):
        control.submit({"mode_options": {"strength": 1}})
    assert control.take_pending() is None


def test_factory_must_be_the_runs_mode() -> None:
    with pytest.raises(ValueError, match="the run's mode is 'fake'"):
        RunControl(_fake_config(), SIZE, factory=FakeFactory(name="other"))


# -- snapshot ---------------------------------------------------------------


def test_defaults_keep_the_old_snapshot() -> None:
    data = _control().snapshot().to_dict()
    assert set(data) == {"config", "stats", "timings", "source", "background", "roi",
                         "running", "ended", "last_error", "weakened", "person_detector"}
    assert _control().snapshot().mode is None and _control().snapshot().modes is None


def test_erase_snapshot_keeps_every_existing_field_and_value() -> None:
    plain = _control().snapshot().to_dict()
    with_mode = RunControl(PetrifyConfig(), SIZE, modes=list_modes([]),
                           factory=ERASE).snapshot().to_dict()
    assert {k: v for k, v in with_mode.items() if k not in ("mode", "modes")} == plain
    assert with_mode["mode"] == {
        "name": "erase",
        "origin": "built in",
        "builtin": True,
        "summary": ERASE.summary,
        "options": [],
    }
    assert with_mode["modes"] == {
        "listed": [{"name": "erase", "origin": "built in", "builtin": True,
                    "selectable": True, "problem": None, "active": True}],
        "omitted": 0,
    }


def test_snapshot_lists_the_modes_and_the_active_modes_options() -> None:
    control = _fake_control()
    data = control.snapshot().to_dict()
    assert json.loads(json.dumps(data, allow_nan=False)) == data
    mode = data["mode"]
    assert (mode["name"], mode["origin"], mode["builtin"], mode["summary"]) == (
        "fake", "petrify-fake 1.0", False, "A fake mode for tests.")
    by_name = {opt["name"]: opt for opt in mode["options"]}
    assert [opt["name"] for opt in mode["options"]] == [spec.name for spec in FAKE_SPECS]
    assert by_name["count"] == {
        "name": "count", "kind": "int", "value": 3, "default": 3, "choices": [],
        "minimum": 1, "maximum": 10, "help": "A runtime count.", "runtime": True,
    }
    assert by_name["parts"] == {
        "name": "parts", "kind": "choices", "value": ["face"], "default": ["face"],
        "choices": ["face", "body", "hands"], "minimum": None, "maximum": None,
        "help": "Runtime parts.", "runtime": True,
    }
    assert by_name["ratio_fixed"]["runtime"] is False
    assert by_name["ratio_fixed"]["value"] == 1.0
    assert by_name["shape"]["choices"] == ["box", "ellipse"]
    assert data["modes"] == {
        "listed": [
            {"name": "erase", "origin": "built in", "builtin": True, "selectable": True,
             "problem": None, "active": False},
            {"name": "fake", "origin": "petrify-fake 1.0", "builtin": False,
             "selectable": True, "problem": None, "active": True},
            {"name": "other", "origin": "petrify-other 2.3", "builtin": False,
             "selectable": True, "problem": None, "active": False},
            {"name": "Bad-Name", "origin": "petrify-fake 1.0", "builtin": False,
             "selectable": False, "problem": "invalid mode name 'Bad-Name'", "active": False},
        ],
        "omitted": 0,
    }


def test_snapshot_option_values_follow_the_applied_config() -> None:
    control = _fake_control()
    accepted = control.submit({"mode_options": {"count": 8, "parts": ["hands"]}})

    def values() -> dict[str, Any]:
        return {o["name"]: o["value"] for o in control.snapshot().to_dict()["mode"]["options"]}

    assert values()["count"] == 3  # Accepted, not yet applied by the loop.
    control.take_pending()
    control.report_applied(accepted)
    assert values()["count"] == 8 and values()["parts"] == ["hands"]
    assert values()["count_fixed"] == 2


def test_a_mode_without_an_entry_and_a_cap_on_listed_modes() -> None:
    many = tuple(_entry(f"m{i}") for i in range(70))
    control = RunControl(_fake_config(), SIZE, modes=many, factory=FakeFactory())
    data = control.snapshot().to_dict()
    assert data["mode"]["origin"] == "unknown origin" and data["mode"]["builtin"] is False
    assert len(data["modes"]["listed"]) == 50 and data["modes"]["omitted"] == 20
    assert not any(item["active"] for item in data["modes"]["listed"])


def test_snapshot_imports_no_listed_plugin() -> None:
    _fake_control().snapshot().to_dict()
    assert "fake_mod" not in sys.modules


def _is_plain(text: str, limit: int) -> bool:
    return len(text) <= limit and all(ch.isprintable() for ch in text) and "  " not in text


def test_untrusted_strings_are_plain_text_and_capped() -> None:
    nasty = "<script>alert(1)</script>\n\r\t\x00\x1b[31m\u202eevil\u2066 " * 200
    spec = OptionSpec("count", "int", 3, nasty, minimum=0, maximum=10, runtime=True)
    factory = FakeFactory(name="fake", summary=nasty, options=(spec,),
                          weakened=(nasty, nasty + "x"))
    modes = (
        ModeEntry("fake", "m:F", nasty, nasty, False, None),
        ModeEntry(nasty, "m:F", nasty, None, False, nasty),
    )
    config = PetrifyConfig(mode="fake", mode_options=(("count", 3),))
    data = RunControl(config, SIZE, modes=modes, factory=factory).snapshot().to_dict()
    mode = data["mode"]
    assert _is_plain(mode["summary"], control_module.SUMMARY_TEXT_LIMIT)
    assert _is_plain(mode["origin"], control_module.ORIGIN_TEXT_LIMIT)
    assert _is_plain(mode["options"][0]["help"], control_module.HELP_TEXT_LIMIT)
    assert mode["summary"].endswith("...")
    for item in data["modes"]["listed"]:
        assert _is_plain(item["name"], control_module.NAME_TEXT_LIMIT)
        assert _is_plain(item["origin"], control_module.ORIGIN_TEXT_LIMIT)
        if item["problem"] is not None:
            assert _is_plain(item["problem"], control_module.PROBLEM_TEXT_LIMIT)
    assert data["weakened"]
    for sentence in data["weakened"]:
        assert _is_plain(sentence, control_module.WEAKENED_TEXT_LIMIT)
    assert len(data["weakened"]) == 1  # Both sentences are the same once capped.


def test_untrusted_text_never_raises() -> None:
    class Hostile:
        def __str__(self) -> str:
            raise RuntimeError("no")

    assert untrusted_text(Hostile(), 40) == "<Hostile object>"
    assert untrusted_text("a\nb\u200bc", 40) == "a b c"
    assert untrusted_text("x" * 50, 10) == "xxxxxxx..."


def test_a_summary_that_cannot_be_read_shows_no_summary() -> None:
    class Raising(FakeFactory):
        def __getattribute__(self, name: str) -> Any:
            if name == "summary":
                raise RuntimeError("broken")
            return super().__getattribute__(name)

    control = RunControl(_fake_config(), SIZE, factory=Raising())
    assert control.snapshot().to_dict()["mode"]["summary"] == ""


# -- weakened ---------------------------------------------------------------


def _high_count(options: Any) -> tuple[str, ...]:
    if options["count"] > 5:
        return (f"count is {options['count']}: fewer frames are hidden.",)
    return ()


def test_mode_weakened_sentences_are_listed_and_recomputed_after_each_change(
    caplog: pytest.LogCaptureFixture,
) -> None:
    factory = FakeFactory(_high_count)
    control = _fake_control(factory)
    assert control.snapshot().to_dict()["weakened"] == []
    with caplog.at_level(logging.WARNING, logger="petrify.server.control"):
        _apply_mode(control, {"mode_options": {"count": 8}})
        assert control.snapshot().weakened == ("count is 8: fewer frames are hidden.",)
        _apply_mode(control, {"mode_options": {"count": 9}})
        _apply_mode(control, {"mode_options": {"count": 2}})
    assert control.snapshot().weakened == ()
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert messages == [
        "Weakened protection: count is 8: fewer frames are hidden.",
        "Weakened protection: count is 9: fewer frames are hidden.",
    ]
    # The factory saw every option, read-only.
    last = factory.calls[-1]
    assert dict(last) == {**FAKE_DEFAULTS, "count": 2}
    with pytest.raises(TypeError):
        last["count"] = 1


def test_mode_weakened_sentences_follow_the_built_in_ones() -> None:
    factory = FakeFactory(("The mode shows faces.",))
    control = RunControl(
        dataclasses.replace(_fake_config(), background_diff=None), SIZE, factory=factory
    )
    assert control.snapshot().weakened == (DIFFER_OFF, "The mode shows faces.")


def test_mode_weakened_is_capped_in_number_and_says_how_many_are_not_listed() -> None:
    factory = FakeFactory(tuple(f"Sentence {i}." for i in range(30)))
    weakened = _fake_control(factory).snapshot().weakened
    cap = control_module.MAX_MODE_WEAKENED
    assert len(weakened) == cap + 1
    assert weakened[:cap] == tuple(f"Sentence {i}." for i in range(cap))
    assert weakened[cap] == control_module.MODE_WEAKENED_MORE.format(name="fake", count=30 - cap)
    assert weakened[cap] == (
        "The mode fake reported 20 more weakened protections that are not listed here."
    )


def test_mode_weakened_at_the_cap_adds_no_count_sentence() -> None:
    cap = control_module.MAX_MODE_WEAKENED
    factory = FakeFactory(tuple(f"Sentence {i}." for i in range(cap)))
    assert len(_fake_control(factory).snapshot().weakened) == cap


@pytest.mark.parametrize(
    "sentences",
    [("",), ("   ",), ("\u200b\u200d",), ("\u3164\u2800",), ("\u0301\u0302",),
     ("Real sentence.", "\u200b"), ("\ufe0f", "Real sentence.", "\n\t")],
    ids=["empty", "spaces", "zero-width", "blank-letters", "marks", "real-then-blank",
         "blank-real-blank"],
)
def test_a_weakened_sentence_that_cleans_to_empty_shows_the_fixed_sentence_once(
    sentences: tuple[str, ...],
) -> None:
    weakened = _fake_control(FakeFactory(sentences)).snapshot().weakened
    unknown = control_module.MODE_WEAKENED_UNKNOWN.format(name="fake")
    real = tuple(s for s in sentences if s == "Real sentence.")
    assert weakened == (*real, unknown)


def _raises(options: Any) -> Any:
    raise RuntimeError("plugin bug")


def _exits(options: Any) -> Any:
    raise SystemExit(3)


@pytest.mark.parametrize(
    "weakened",
    [_raises, _exits, ["a list"], "a string", None, ("ok", 3), (b"bytes",)],
    ids=["raises", "system-exit", "list", "string", "none", "non-string-item", "bytes"],
)
def test_mode_weakened_failure_shows_a_fixed_sentence(weakened: Any) -> None:
    control = _fake_control(FakeFactory(weakened))
    expected = control_module.MODE_WEAKENED_UNKNOWN.format(name="fake")
    assert control.snapshot().weakened == (expected,)
    # The snapshot still works and is JSON-safe; a change is still accepted and applied.
    _apply_mode(control, {"mode_options": {"count": 4}})
    data = control.snapshot().to_dict()
    assert data["weakened"] == [expected]
    assert json.loads(json.dumps(data)) == data


# -- end to end through run_stream ------------------------------------------


@pytest.mark.usefixtures("plugin", "fake_persons")
def test_runtime_option_change_reaches_the_mode_while_the_run_continues() -> None:
    plugin_module = sys.modules[PLUGIN_MODULE]
    config = _plugin_config()
    factory = load_mode("paint")
    control = RunControl(config, (STREAM_W, STREAM_H), modes=list_modes(), factory=factory)
    seen: dict[str, Any] = {}

    def change() -> None:
        seen["accepted"] = control.submit({"mode_options": {"strength": 7}})

    def check() -> None:
        seen["mode"] = control.snapshot().to_dict()["mode"]

    sink = ListSink()
    report = run_stream(HookSource(QUIET, hooks={30: change, 60: check}), config,
                        OutputConfig(), sinks=[sink], control=control)
    assert len(sink.images) == QUIET.total  # The run went on to the end.
    assert report.stats.reconfigurations == 1
    strengths = [ctx.options["strength"] for ctx in plugin_module.CONTEXTS]
    assert strengths[0] == 3 and strengths[-1] == 7
    assert dict(plugin_module.CONTEXTS[-1].config.mode_options)["strength"] == 7
    assert control.applied is seen["accepted"]
    options = {opt["name"]: opt for opt in seen["mode"]["options"]}
    assert options["strength"]["value"] == 7 and options["strength"]["runtime"] is True
    assert options["fail"]["runtime"] is False
    assert seen["mode"]["origin"] == "petrify-t11-plugins 0.1"
    assert control.snapshot().last_error is None


# ---------------------------------------------------------------------------
# T13b: plugin code outside the lock, ordering of weakened results
# ---------------------------------------------------------------------------


class _Gate:
    """A weakened callable that blocks for configs with ``count == block_at`` until released."""

    def __init__(self, block_at: int) -> None:
        self.block_at = block_at
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, options: Any) -> tuple[str, ...]:
        if options["count"] == self.block_at:
            self.entered.set()
            assert self.release.wait(10)
        return (f"count {options['count']} is weakened.",)


def _in_thread(target: Any, *args: Any) -> threading.Thread:
    thread = threading.Thread(target=target, args=args, daemon=True)
    thread.start()
    return thread


def test_a_slow_weakened_call_does_not_hold_the_lock() -> None:
    gate = _Gate(block_at=8)
    control = _fake_control(FakeFactory(gate))
    slow = _in_thread(control.report_applied, _fake_config(count=8))
    try:
        assert gate.entered.wait(5)
        # While the plugin runs, every server-side call returns at once.
        done = threading.Event()

        def server_calls() -> None:
            control.snapshot().to_dict()
            control.submit({"mode_options": {"ratio": 1.5}})
            assert control.applied.mode_options_dict()["count"] == 8
            done.set()

        _in_thread(server_calls)
        assert done.wait(2), "a server call waited for the plugin's weakened call"
        # Until the result lands, the previous mode sentences stay listed.
        assert control.snapshot().weakened == ("count 3 is weakened.",)
    finally:
        gate.release.set()
        slow.join(5)
    assert control.snapshot().weakened == ("count 8 is weakened.",)


def test_a_newer_applied_config_wins_over_a_slower_older_result(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gate = _Gate(block_at=8)
    control = _fake_control(FakeFactory(gate))
    with caplog.at_level(logging.WARNING, logger="petrify.server.control"):
        older = _in_thread(control.report_applied, _fake_config(count=8))
        try:
            assert gate.entered.wait(5)
            control.report_applied(_fake_config(count=9))  # Completes while the older one waits.
            assert control.snapshot().weakened == ("count 9 is weakened.",)
        finally:
            gate.release.set()
            older.join(5)
    # The older, slower result was discarded: the list belongs to the applied config.
    assert control.applied.mode_options_dict()["count"] == 9
    assert control.snapshot().weakened == ("count 9 is weakened.",)
    options = {o["name"]: o["value"] for o in control.snapshot().to_dict()["mode"]["options"]}
    assert options["count"] == 9
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert messages == ["Weakened protection: count 9 is weakened."]


def test_built_in_weakened_protections_are_logged_when_the_mode_part_is_superseded(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gate = _Gate(block_at=8)
    control = _fake_control(FakeFactory(gate))
    off = dataclasses.replace(_fake_config(count=8), background_diff=None)
    with caplog.at_level(logging.WARNING, logger="petrify.server.control"):
        older = _in_thread(control.report_applied, off)
        try:
            assert gate.entered.wait(5)
            control.report_applied(dataclasses.replace(off, mode_options=_fake_config(
                count=9).mode_options))
        finally:
            gate.release.set()
            older.join(5)
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert messages == [f"Weakened protection: {DIFFER_OFF}",
                        "Weakened protection: count 9 is weakened."]


def test_option_spec_subclasses_are_used_as_plain_specs() -> None:
    armed = threading.Event()

    class Slow(OptionSpec):
        def validate(self, value: object) -> Any:
            if armed.is_set():
                raise RuntimeError("plugin validate must not run under the lock")
            return super().validate(value)

    spec = Slow("count", "int", 3, "count", minimum=1, maximum=10, runtime=True)
    armed.set()
    control = RunControl(PetrifyConfig(mode="fake", mode_options=(("count", 3),)), SIZE,
                         factory=FakeFactory(options=(spec,)))
    assert control.submit({"mode_options": {"count": 4}}).mode_options == (("count", 4),)
    with pytest.raises(ValueError, match="from 1 to 10"):
        control.submit({"mode_options": {"count": 11}})


# ---------------------------------------------------------------------------
# T13b: rejection messages, and the change accepted after a rejected one
# ---------------------------------------------------------------------------

HOSTILE_ERROR = "x" * 20_000 + "\nsecond line \u202eevil\u0301"


def _is_clean(text: str | None, limit: int) -> bool:
    return (text is not None and len(text) <= limit and all(ch.isprintable() for ch in text)
            and "\u202e" not in text and "\u0301" not in text)


def test_rejection_messages_are_cleaned_and_capped() -> None:
    control = _fake_control()
    control.submit({"mode_options": {"count": 4}})
    control.take_pending()
    control.report_rejected(HOSTILE_ERROR)
    error = control.snapshot().last_error
    assert _is_clean(error, control_module.PROBLEM_TEXT_LIMIT)
    assert error is not None and error.endswith("...") and "\n" not in error


def test_the_pipeline_logs_and_reports_a_cleaned_rejection(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import petrify.pipeline as pipeline_module

    control = _fake_control()
    control.submit({"mode_options": {"count": 4}})

    class OpenEngine:
        is_open = True

    old = mock.Mock(config=control.applied)
    with (
        mock.patch.object(pipeline_module, "_rebuild", side_effect=ValueError(HOSTILE_ERROR)),
        caplog.at_level(logging.WARNING, logger="petrify.pipeline"),
    ):
        assert pipeline_module._apply_pending(OpenEngine(), old, control, False) is old  # type: ignore[arg-type]
    records = [r for r in caplog.records if r.name == "petrify.pipeline"]
    assert len(records) == 1
    logged = records[0].getMessage()
    prefix = "The new settings were not applied, the previous ones stay: "
    assert logged.startswith(prefix)
    assert _is_clean(logged[len(prefix):], control_module.PROBLEM_TEXT_LIMIT)
    assert logged[len(prefix):] == control.snapshot().last_error


def test_a_change_accepted_after_a_rejected_one_stands_on_its_own() -> None:
    control = _fake_control()
    control.submit({"mode_options": {"count": 4}})  # A
    assert control.take_pending() is not None  # The loop starts applying A.
    b = control.submit({"mode_options": {"ratio": 1.5}})  # B, built on A: returns 200.
    assert b.mode_options_dict()["count"] == 4
    control.report_rejected("A is not acceptable")
    assert control.snapshot().last_error == "A is not acceptable"
    rebuilt = control.take_pending()
    assert rebuilt is not None
    options = rebuilt.mode_options_dict()
    assert (options["count"], options["ratio"]) == (3, 1.5)  # B without A's value.
    control.report_applied(rebuilt)
    assert control.applied.mode_options_dict()["ratio"] == 1.5


def test_the_next_change_builds_on_the_rebuilt_one_not_on_the_applied_config() -> None:
    control = _fake_control()
    control.submit({"mode_options": {"count": 4}})  # A
    control.take_pending()
    control.submit({"mode_options": {"ratio": 1.5}})  # B
    control.report_rejected("A is not acceptable")
    c = control.submit({"mode_options": {"flag": True}})  # C
    options = c.mode_options_dict()
    assert (options["count"], options["ratio"], options["flag"]) == (3, 1.5, True)
    assert control.take_pending() is c


def test_a_later_change_that_needs_the_rejected_one_is_dropped_with_a_clear_error() -> None:
    control = RunControl(dataclasses.replace(_fake_config(), background_diff=None), SIZE,
                         modes=FAKE_MODES, factory=FakeFactory())
    control.submit({"background_diff": {"enabled": True}})  # A
    control.take_pending()
    control.submit({"background_diff": {"pixel_threshold": 30}})  # B, valid only after A.
    control.report_rejected("A is not acceptable")
    assert control.take_pending() is None
    error = control.snapshot().last_error
    assert error is not None
    assert error.startswith("A is not acceptable A later change was dropped too")
    assert "The background differ is off" in error
    # The next change builds on the applied config.
    assert control.submit({"detector": "both"}).background_diff is None


# ---------------------------------------------------------------------------
# T13b: spoofing text from package metadata and from the plugin
# ---------------------------------------------------------------------------

ZALGO = "x" + "\u0336\u0337\u0338\u0489" * 100
SPOOFS = {
    "zalgo": ZALGO,
    "hangul-filler": "era\u3164se",
    "halfwidth-filler": "era\uffa0se",
    "jungseong-filler": "era\u1160se",
    "braille-blank": "era\u2800se",
    "cyrillic-e": "\u0435rase",
    "rtl-letters": "\u05d0\u05d1\u05d2 erase",
    "variation-selector": "erase\ufe0f\U000e0100",
    "fake-status": "fake (built in): the mode of this run;",
}


def _ascii_printable(text: str) -> bool:
    return all(" " <= ch <= "~" for ch in text)


@pytest.mark.parametrize("spoof", list(SPOOFS.values()), ids=list(SPOOFS))
def test_metadata_strings_are_reduced_to_printable_ascii(spoof: str) -> None:
    modes = (
        ModeEntry("fake", "m:F", "petrify-fake", "1.0", False, None),
        ModeEntry(spoof, "m:F", spoof, spoof, False, f"invalid mode name {spoof!r}"),
    )
    data = RunControl(_fake_config(), SIZE, modes=modes,
                      factory=FakeFactory()).snapshot().to_dict()
    listed = data["modes"]["listed"][1]
    for key in ("name", "origin", "problem"):
        assert _ascii_printable(listed[key]), (key, listed[key])
        assert listed[key]
    if not _ascii_printable(spoof):
        assert "?" in listed["name"] and listed["name"] != "erase"


def test_metadata_text_replaces_each_foreign_character_with_a_placeholder() -> None:
    assert metadata_text("\u0435rase", 64) == "?rase"
    assert metadata_text("era\u3164se", 64) == "era?se"
    assert metadata_text("erase\ufe0f", 64) == "erase?"
    assert metadata_text("a\nb\tc", 64) == "a b c"
    assert metadata_text(ZALGO, 64) == "x" + "?" * 60 + "..."
    assert metadata_text("fake (built in): the mode of this run;", 64) == (
        "fake (built in): the mode of this run;")


def test_plugin_strings_keep_other_scripts_but_lose_marks_and_blank_characters() -> None:
    assert untrusted_text("\u041f\u0440\u0438\u0432\u0435\u0442 \u05e9\u05dc\u05d5\u05dd", 64) == (
        "\u041f\u0440\u0438\u0432\u0435\u0442 \u05e9\u05dc\u05d5\u05dd")
    assert untrusted_text(ZALGO, 64) == "x"
    assert untrusted_text("a\u3164b\uffa0c\u1160d\u2800e", 64) == "a b c d e"
    assert untrusted_text("ok\ufe0f\U000e0100", 64) == "ok"
    spec = OptionSpec("count", "int", 3, "Help " + ZALGO + " \u3164", minimum=0, maximum=5,
                      runtime=True)
    factory = FakeFactory(summary="Summary " + ZALGO, options=(spec,))
    data = RunControl(PetrifyConfig(mode="fake", mode_options=(("count", 3),)), SIZE,
                      factory=factory).snapshot().to_dict()
    assert data["mode"]["summary"] == "Summary x"
    assert data["mode"]["options"][0]["help"] == "Help x"


# ---------------------------------------------------------------------------
# T13b: the background state of a mode that uses none
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("plugin", "fake_persons")
def test_a_mode_without_a_background_reports_that_none_is_in_use() -> None:
    config = _plugin_config()
    control = RunControl(config, (STREAM_W, STREAM_H), modes=list_modes(),
                         factory=load_mode("paint"))
    seen: dict[str, Any] = {}

    def check() -> None:
        seen["background"] = control.snapshot().to_dict()["background"]

    run_stream(HookSource(QUIET, hooks={20: check}), config, OutputConfig(),
               sinks=[ListSink()], control=control)
    assert seen["background"] == {"strategy": "running", "ready": False, "rejected": 0,
                                  "in_use": False}


def test_an_erase_run_keeps_the_background_state_shape() -> None:
    config = _live_config("running")
    control = RunControl(config, (STREAM_W, STREAM_H), modes=list_modes([]), factory=ERASE)
    seen: dict[str, Any] = {}

    def check() -> None:
        seen["background"] = control.snapshot().to_dict()["background"]

    run_stream(HookSource(QUIET, hooks={20: check}), config, OutputConfig(),
               sinks=[ListSink()], control=control)
    assert set(seen["background"]) == {"strategy", "ready", "rejected"}
    assert BackgroundState().to_dict() == {"strategy": "", "ready": False, "rejected": 0}


# ---------------------------------------------------------------------------
# T8b: the HOG settings while the DNN finds the person boxes
# ---------------------------------------------------------------------------

DNN_PERSON_ERROR = (
    "The HOG person settings cannot be changed: this run uses the DNN person detector, "
    "which they do not affect"
)


def _dnn_config() -> PetrifyConfig:
    from petrify.core.config import PersonDnnConfig

    return PetrifyConfig(detector="both", person_dnn=PersonDnnConfig())


@pytest.mark.parametrize("field", ["min_confidence", "min_box_area", "scale"])
def test_t8b_hog_settings_are_rejected_while_the_dnn_runs(field: str) -> None:
    control = _control(_dnn_config())
    value = {"min_confidence": 0.5, "min_box_area": 900, "scale": 1.1}[field]
    with pytest.raises(ValueError) as excinfo:
        control.submit({"person": {field: value}})
    assert str(excinfo.value) == DNN_PERSON_ERROR
    assert control.snapshot().last_error == DNN_PERSON_ERROR
    assert control.take_pending() is None
    # Nothing else in the same patch is applied either.
    with pytest.raises(ValueError, match="HOG person settings cannot be changed"):
        control.submit({"detector": "person", "person": {field: value}})
    assert control.take_pending() is None


def test_t8b_other_settings_still_change_while_the_dnn_runs() -> None:
    control = _control(_dnn_config())
    assert control.submit({"detector": "person"}).detector == "person"
    assert control.submit({"motion": {"threshold": 300}}).motion.threshold == 300


def test_t8b_hog_settings_still_change_with_hog() -> None:
    control = _control(PetrifyConfig(detector="both"))
    assert control.submit({"person": {"min_confidence": 0.5}}).person.min_confidence == 0.5


def test_t8b_no_hog_weakened_sentence_while_the_dnn_runs() -> None:
    dnn = _dnn_config()
    tripled = dataclasses.replace(
        dnn, person=dataclasses.replace(dnn.person, min_confidence=0.9, min_box_area=5000)
    )
    found = control_module.weakened_protections(tripled, dnn, SIZE)
    assert [key for key, _ in found if key.startswith("threshold:person.")] == []
    hog = PetrifyConfig(detector="both")
    hog_tripled = dataclasses.replace(tripled, person_dnn=None)
    found = control_module.weakened_protections(hog_tripled, hog, SIZE)
    assert [key for key, _ in found if key.startswith("threshold:person.")] == [
        "threshold:person.min_confidence", "threshold:person.min_box_area",
    ]


def test_t8b_the_snapshot_names_the_active_person_detector() -> None:
    assert _control().snapshot().to_dict()["person_detector"] == "hog"
    assert _control(_dnn_config()).snapshot().to_dict()["person_detector"] == "dnn"


class _RequiresFactory(FakeFactory):
    """A FakeFactory whose mode requires motion and persons."""

    def requires(self, options: Any) -> Any:
        return frozenset({DetectionKind.MOTION, DetectionKind.PERSONS})


def test_a_kind_the_tracker_cannot_hold_is_listed_as_weakened() -> None:
    untracked = control_module.TRACKER_UNTRACKED_WEAKENED.format(kind="persons")
    config = dataclasses.replace(_fake_config(), detector="motion", tracker=TrackerConfig())
    control = RunControl(config, SIZE, factory=_RequiresFactory())
    assert untracked in control.snapshot().weakened
    control.report_applied(dataclasses.replace(config, detector="both"))
    assert untracked not in control.snapshot().weakened
    control.report_applied(config)
    assert untracked in control.snapshot().weakened
    control.report_applied(dataclasses.replace(config, tracker=None))
    assert untracked not in control.snapshot().weakened


def test_a_tracked_kind_is_not_listed_and_a_failing_requires_lists_nothing() -> None:
    config = dataclasses.replace(_fake_config(), detector="both", tracker=TrackerConfig())
    weakened = RunControl(config, SIZE, factory=_RequiresFactory()).snapshot().weakened
    assert not any("tracker" in sentence for sentence in weakened)
    motion = dataclasses.replace(config, detector="motion")
    weakened = RunControl(motion, SIZE, factory=FakeFactory()).snapshot().weakened
    assert not any("tracker" in sentence for sentence in weakened)  # No requires method.
