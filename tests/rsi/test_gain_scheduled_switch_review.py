"""Synthetic held-gain provenance/mutation checks."""

from dataclasses import replace

import numpy as np
import pytest
from test_switched_velocity_sonic_review import fixture

from rosclaw_soccer.physics.g1_gain_scheduled_episode import G1GainSchedule
from rosclaw_soccer.rsi.gain_scheduled_switch_review import review_gain_switch
from rosclaw_soccer.rsi.switched_velocity_sonic_review import review_switched


def setup(mode):
    raw, n, p, c, w = fixture()
    s = G1GainSchedule(
        p.stiffness,
        p.damping,
        tuple(float(x) for x in c.kp),
        tuple(float(x) for x in c.kd),
        30,
        mode,
    )
    values = [s.gains(f) for f in range(300)]
    raw.update(
        held_pd_kp=np.array([x[0] for x in values]), held_pd_kd=np.array([x[1] for x in values])
    )
    return raw, n, p, c, w, s


def test_constant_metadata_preserved_and_ramp_is_explicit():
    raw, n, p, c, w, s = setup("constant")
    common = {k: v for k, v in raw.items() if not k.startswith("held_pd_")}
    assert review_gain_switch(raw, n, p, c, w, handoff=30, schedule=s) == review_switched(
        common, n, p, c, w, handoff=30
    )
    raw, n, p, c, w, s = setup("linear10")
    r = review_gain_switch(raw, n, p, c, w, handoff=30, schedule=s)
    assert r["gain_switch_performed"] and r["all_declared_pd_and_policy_inputs_reconstructed"]
    assert not r["common_fixed_pd_is_velocity_deployment_not_sonic_deployment"]
    assert not r["score"]["qualified"]


@pytest.mark.parametrize("kind", ["gain", "dtype", "missing", "source", "boundary", "pd"])
def test_wrong_declared_or_actual_gains_fail_closed(kind):
    raw, n, p, c, w, s = setup("linear10")
    h = 30
    if kind == "gain":
        raw["held_pd_kp"][31, 0] += 0.01
    elif kind == "dtype":
        raw["held_pd_kd"] = raw["held_pd_kd"].astype(np.float32)
    elif kind == "missing":
        del raw["held_pd_kp"]
    elif kind == "source":
        s = replace(s, final_kp=(1.0,) * 29)
    elif kind == "boundary":
        h = 31
    else:
        raw["pd_requested_torque_nm"][40, 0, 0] = 1.0
    with pytest.raises(ValueError):
        review_gain_switch(raw, n, p, c, w, handoff=h, schedule=s)
