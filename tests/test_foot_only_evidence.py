import pytest

from rosclaw_soccer.sim.foot_only_evidence import FootOnlyContactTracker
from rosclaw_soccer.sim.strike_quality import StrikeQualityConfig


def tracker():
    return FootOnlyContactTracker(
        geometry_contract_hash="sha256:" + "a" * 64,
        config=StrikeQualityConfig(duration_sec=0.01),
    )


def sample(t, i, foot=0.0, other=0.0, safe=True):
    t.observe(
        elapsed_sec=i * 0.002,
        body_safe=safe,
        foot_ball_normal_force_n=foot,
        nonfoot_robot_ball_normal_force_n=other,
    )


@pytest.mark.parametrize("nonfoot_tick", [1, 2, 4, 5])
def test_prior_simultaneous_late_and_final_nonfoot_rejected(nonfoot_tick):
    t = tracker()
    for i in range(1, 6):
        sample(t, i, foot=2.0 if i == 2 else 0.0, other=1.0 if i == nonfoot_tick else 0.0)
    assert not t.result()["foot_only_complete"]
    assert t.result()["first_nonfoot_robot_sec"] == nonfoot_tick * 0.002


def test_complete_clean_foot_at_threshold_is_required():
    t = tracker()
    for i in range(1, 5):
        sample(t, i, foot=1.0 if i == 2 else 0.0)
    assert not t.result()["foot_only_complete"]
    sample(t, 5)
    r = t.result()
    assert r["foot_only_complete"] and r["first_foot_sec"] == 0.004
    assert r["promotion_eligible"] is False


@pytest.mark.parametrize(
    "failure", ["no_touch", "fall", "gap", "duplicate", "nan", "negative", "boolean_force", "extra"]
)
def test_incomplete_or_invalid_evidence_never_admitted(failure):
    t = tracker()
    if failure in ("no_touch", "fall"):
        for i in range(1, 6):
            sample(
                t,
                i,
                foot=0.0 if failure == "no_touch" else 2.0,
                safe=not (failure == "fall" and i == 5),
            )
    elif failure == "extra":
        for i in range(1, 6):
            sample(t, i, foot=2.0)
        with pytest.raises(ValueError):
            sample(t, 6)
    else:
        sample(t, 1, foot=2.0)
        values = dict(i=2, foot=0.0, other=0.0)
        if failure == "gap":
            values["i"] = 3
        if failure == "duplicate":
            values["i"] = 1
        if failure == "nan":
            values["foot"] = float("nan")
        if failure == "negative":
            values["other"] = -1.0
        if failure == "boolean_force":
            values["foot"] = True
        with pytest.raises(ValueError):
            sample(t, **values)
        with pytest.raises(ValueError):
            sample(t, 2)
    assert not t.result()["foot_only_complete"]


def test_geometry_hash_cannot_be_omitted_or_malformed():
    with pytest.raises(ValueError):
        FootOnlyContactTracker(geometry_contract_hash="unknown", config=StrikeQualityConfig())
