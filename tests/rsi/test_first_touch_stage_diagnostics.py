"""Synthetic primitive-label diagnostics, never additional physical successes."""

import copy

import pytest

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.first_touch_stage_diagnostics import assess_first_touch_stages
from tests.rsi.test_first_touch_outcome import trace


def outcome(frame, *, weak=False, unsafe=False):
    t = trace(frame)
    if weak:
        t["ball_position_after_step_m"] *= 0.5
    if unsafe:
        t["force_n"][280, 0, 5] = 2.0
    return label_first_touch(t).to_dict()


def test_m0_tie_does_not_hide_acquisition_regression_or_create_fake_learning():
    base = dict(g28=outcome(None), g76=outcome(151, weak=True), g12=outcome(231))
    changed = dict(g28=outcome(None), g76=outcome(None), g12=outcome(231))
    before = copy.deepcopy((base, changed))
    r = assess_first_touch_stages(base, changed, declared_pair_ids=("g28", "g76", "g12"))
    assert r["original_m0_pairs"]["net_success_gain"] == 0
    assert r["original_m0_pairs"]["regressed_pairs"] == 0
    assert r["contact_regressions"] == r["quality_observation_losses"] == 1
    assert r["baseline"]["acquired"] == 2
    assert r["evaluated"]["acquired"] == 1
    assert r["baseline"]["m0_pass_given_safe_observed_contact"] == 0.5
    assert r["evaluated"]["m0_pass_given_safe_observed_contact"] == 1.0
    # The conditional rate improved only because a failed contact disappeared.
    assert r["baseline"]["m0_task_pass_rate"] == r["evaluated"]["m0_task_pass_rate"] == 1 / 3
    assert (base, changed) == before
    assert not r["teacher_qualified"] and not r["receiving_skill_qualified"]
    assert not r["successor_readiness_qualified"]


@pytest.mark.parametrize("frame", [240, 260, 299])
def test_late_contact_is_acquisition_not_a_fabricated_zero_quality_failure(frame):
    b, e = dict(case=outcome(239)), dict(case=outcome(frame))
    r = assess_first_touch_stages(b, e, declared_pair_ids=["case"])
    assert r["evaluated"]["acquired"] == r["evaluated"]["censored"] == 1
    assert r["evaluated"]["observed"] == 0
    assert r["evaluated"]["m0_pass_given_safe_observed_contact"] is None
    assert r["evaluated"]["measured_quality_coverage_given_contact"] == 0
    assert r["contact_regressions"] == 0 and r["quality_observation_losses"] == 1
    assert r["original_m0_pairs"]["regressed_pairs"] == 1
    assert r["original_m0_pairs"]["rows"][0]["evaluated"]["forward_60_m"] is None


@pytest.mark.parametrize("frame", [0, 239, 240, 299, None])
def test_safety_overrides_status_but_does_not_erase_measured_or_censored_events(frame):
    o = outcome(frame, unsafe=True)
    r = assess_first_touch_stages(dict(case=o), dict(case=o), declared_pair_ids=["case"])
    assert r["evaluated"]["safe_acquired"] == r["evaluated"]["task_pass"] == 0
    assert r["evaluated"]["acquired"] == int(frame is not None)
    assert r["evaluated"]["observed"] == int(frame in (0, 239))
    assert r["evaluated"]["censored"] == int(frame in (240, 299))


def test_no_contact_rates_with_empty_denominators_remain_unknown():
    o = dict(case=outcome(None))
    r = assess_first_touch_stages(o, o, declared_pair_ids=["case"])
    assert r["baseline"]["contact_rate"] == 0
    assert r["baseline"]["measured_quality_coverage_given_contact"] is None
    assert r["baseline"]["m0_pass_given_safe_observed_contact"] is None
    assert r["new_physical_steps"] == r["new_optimizer_updates"] == 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("first_foot_contact_frame", True),
        ("first_foot_contact_frame", 300),
        ("first_foot_contact_frame", 239.0),
        ("quality_observed", 1),
        ("quality_observed", False),
        ("measurement_status", "RIGHT_CENSORED"),
        ("forward_60_m", None),
        ("lateral_60_m", float("nan")),
        ("minimum_pelvis_z_m", 0.64),
        ("maximum_lateral_excursion_m", -0.1),
        ("task_pass", False),
        ("safe", 1),
        ("event_status", "NO_FOOT_CONTACT"),
        ("failure_reason", "NO_FOOT_CONTACT"),
        ("outcome_contract_hash", "other"),
        ("terminated", False),
        ("truncated", True),
        ("promotion_authorized", True),
        ("hardware_authorized", True),
    ],
)
def test_inconsistent_primitive_labels_do_not_get_repaired_or_retagged(key, value):
    b = dict(case=outcome(239))
    e = copy.deepcopy(b)
    e["case"][key] = value
    with pytest.raises(ValueError):
        assess_first_touch_stages(b, e, declared_pair_ids=["case"])


@pytest.mark.parametrize("frame", [240, None])
def test_missing_quality_must_never_be_replaced_with_zero(frame):
    o = dict(case=outcome(frame))
    o["case"]["forward_60_m"] = 0.0
    with pytest.raises(ValueError, match="fabricate"):
        assess_first_touch_stages(o, o, declared_pair_ids=["case"])


@pytest.mark.parametrize("mutation", ["partial", "foreign", "duplicates", "empty", "string"])
def test_explicit_whole_pair_declaration_required(mutation):
    b, e, ids = dict(case=outcome(239)), dict(case=outcome(239)), ["case"]
    if mutation == "partial":
        e.clear()
    elif mutation == "foreign":
        e["other"] = e.pop("case")
    elif mutation == "duplicates":
        ids *= 2
    elif mutation == "empty":
        ids.clear()
    else:
        ids = "case"
    with pytest.raises(ValueError):
        assess_first_touch_stages(b, e, declared_pair_ids=ids)
