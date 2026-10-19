"""Outcome-independent40-state declaration; NOT160 new physical executions."""

from dataclasses import asdict, replace

import pytest

from rosclaw_soccer.rsi import postcontact_robustness_plan as mod
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_course_preflight import (
    FirstTouchCourse,
    preflight_receiving_courses,
)
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

MEAN = "sha256:" + "a" * 64


def test_balanced_whole40_four_arms_with_fixed_randomization():
    plan = mod.declare_robustness40(mean_model_hash=MEAN)
    assert plan == mod.declare_robustness40(mean_model_hash=MEAN)
    assert plan["strata_counts"] == [10] * 4
    assert len({tuple(r["course"]) for r in plan["records"]}) == 40
    assert len({r["sampling_seed"] for r in plan["records"]}) == 40
    assert len({r["source_cluster"] for r in plan["records"]}) == 10
    assert all(
        len(r["arms"]) == 4 and r["same_sampling_seed_all_four_arms"] for r in plan["records"]
    )
    assert plan["new_native_step_budget"] == 160 * 3000
    assert plan["maximum_forecast_step_budget"] == 80 * 560
    assert plan["combined_step_budget"] == 524800
    assert plan["baseline_source_executions_reused"] == 0
    assert not plan["teacher_qualified"] and not plan["training_authorized"]
    assert plan["report_hash"] == hash_json({k: v for k, v in plan.items() if k != "report_hash"})


def test_all_inputs_preflight_before_any_worker(monkeypatch):
    original = mod.balanced_public_first_touch_jobs
    jobs = original(mod.COURSE_GENERATOR_SEEDS, repeats=1)
    bad = (replace(jobs[0], course=(2.400001, 0.1, 0.4)), *jobs[1:])
    monkeypatch.setattr(mod, "balanced_public_first_touch_jobs", lambda *a, **k: bad)
    with pytest.raises(ValueError, match="entire public course"):
        mod.declare_robustness40(mean_model_hash=MEAN)


@pytest.mark.parametrize("mean", [None, "parent", "sha256:" + "z" * 64, "sha256:" + "a" * 63])
def test_exact_hash_required(mean):
    with pytest.raises(ValueError):
        mod.declare_robustness40(mean_model_hash=mean)


def test_distinct_domains_do_not_coerce_or_change_legacy_receiving_hash():
    old = ReceivingCourse("blue.defender", 123, 0.5, -0.12)
    assert preflight_receiving_courses((old,)) == hash_json(
        dict(
            schema="soccer.receiving_course_preflight.v1",
            courses=[asdict(old)],
            physics_qualified=False,
        )
    )
    new = FirstTouchCourse("cpu.first_touch.0", 456, (2.65, 0.1, 0.3))
    result = preflight_receiving_courses((new,))
    assert result == preflight_receiving_courses((new,))
    assert result != preflight_receiving_courses((old,))
    with pytest.raises(ValueError):
        preflight_receiving_courses((old, new))
    # A real RSI0.3m/s launch is NOT a legacy receiving0.5m/s course.
    with pytest.raises(ValueError):
        preflight_receiving_courses((replace(old, speed_mps=0.3),))


@pytest.mark.parametrize(
    "bad", ["duplicate_seed", "duplicate_state", "agent", "seed", "domain", "tuple", "nan"]
)
def test_explicit_firsttouch_bank_rejects_invalid_or_duplicated_inputs(bad):
    good = FirstTouchCourse("cpu.first_touch.0", 456, (2.65, 0.1, 0.3))
    other = replace(good, seed=457, course=(2.66, 0.1, 0.3))
    if bad == "duplicate_seed":
        other = replace(other, seed=456)
    elif bad == "duplicate_state":
        other = replace(other, course=good.course)
    elif bad == "agent":
        other = replace(other, agent_id="bad name")
    elif bad == "seed":
        other = replace(other, seed=True)
    elif bad == "domain":
        other = replace(other, course=(2.8, 0.1, 0.3))
    elif bad == "tuple":
        other = replace(other, course=[2.66, 0.1, 0.3])
    else:
        other = replace(other, course=(2.66, 0.1, float("nan")))
    with pytest.raises(ValueError):
        preflight_receiving_courses((good, other))
