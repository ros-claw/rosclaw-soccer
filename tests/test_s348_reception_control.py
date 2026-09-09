from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.growth.reception_control import (
    ReceptionControlConfig,
    measure_reception_control,
)


def samples():
    n = 51
    t = np.arange(n) * 0.02
    ball = np.zeros((n, 3))
    ball[:, 2] = 0.115
    v = np.zeros((n, 3))
    v[:11, 0] = 1.0
    v[11:, 0] = 0.2
    feet = np.zeros((n, 2, 3))
    feet[:, :, 1] = [-0.1, 0.1]
    force = np.zeros(n)
    force[11] = 5
    return dict(
        time=t,
        ball_position=ball,
        ball_velocity=v,
        receiver_feet=feet,
        foot_contact_force=force,
        forbidden_contact=np.zeros(n, dtype=bool),
        body_safe=np.ones(n, dtype=bool),
        contact_index=11,
    )


def test_real_touch_followed_by_sustained_slow_close_ball_is_controlled():
    a = samples()
    r = measure_reception_control(**a)
    assert r["controlled_reception"] and r["confirmation_index"] == 36
    assert not r["promotion_eligible"]


@pytest.mark.parametrize(
    "cause",
    ["graze", "escape", "nonfoot", "opponent", "unsafe", "no-touch", "short", "slow-incoming"],
)
def test_touch_alone_or_interrupted_or_incomplete_is_not_control(cause):
    a = samples()
    if cause == "graze":
        a["ball_velocity"][11:, 0] = 1.0
    if cause == "escape":
        a["ball_position"][30:, 0] = 2.0
    if cause in ("nonfoot", "opponent"):
        a["forbidden_contact"][20] = True
    if cause == "unsafe":
        a["body_safe"][20] = False
    if cause == "no-touch":
        a["foot_contact_force"][:] = 0
    if cause == "slow-incoming":
        a["ball_velocity"][10] = 0
    if cause == "short":
        a = {k: v[:20] if isinstance(v, np.ndarray) else v for k, v in a.items()}
    r = measure_reception_control(**a)
    assert not r["controlled_reception"]


@pytest.mark.parametrize("cause", ["nan", "gap", "duplicate", "bad-force", "bad-safe", "bad-index"])
def test_invalid_physical_evidence_rejected(cause):
    a = samples()
    if cause == "nan":
        a["ball_velocity"][1, 0] = np.nan
    if cause == "gap":
        a["time"][20:] += 1
    if cause == "duplicate":
        a["time"][20] = a["time"][19]
    if cause == "bad-force":
        a["foot_contact_force"][0] = -1
    if cause == "bad-safe":
        a["body_safe"] = a["body_safe"].astype(int)
    if cause == "bad-index":
        a["contact_index"] = True
    with pytest.raises(ValueError):
        measure_reception_control(**a)


def test_config_rejects_relaxed_or_nonfinite_thresholds():
    for values in [
        dict(maximum_speed_mps=5.0),
        dict(maximum_foot_distance_m=float("inf")),
        dict(observation_sec=True),
    ]:
        with pytest.raises(ValueError):
            replace(ReceptionControlConfig(), **values)


def test_numeric_overflow_and_wrong_config_cannot_create_report():
    a = samples()
    a["ball_velocity"][:] = 1e308
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="numeric envelope"):
        measure_reception_control(**a)
    with pytest.raises(ValueError, match="typed"):
        measure_reception_control(**samples(), config=False)
