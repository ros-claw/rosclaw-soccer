from types import SimpleNamespace

import numpy as np
import pytest

from scripts.rsi_review_passer_taskspace import summarize


def fixture():
    model = SimpleNamespace(nu=58, nq=79, nv=76)
    a = SimpleNamespace(ai=np.arange(29), bv=70)
    b = SimpleNamespace(ai=np.arange(29, 58), bq=72, rq=36, bv=70, rv=35)
    raw = dict(
        torque=np.zeros((81, 10, 58)),
        parent_torque=np.zeros((81, 10, 58)),
        applied_torque_change=np.zeros((81, 10, 58)),
        ball_forces=np.zeros((81, 10, 2, 3)),
        qpos=np.zeros((81, 10, 79)),
        qvel=np.zeros((81, 10, 76)),
        body_failed=np.zeros((81, 2), dtype=bool),
    )
    return model, a, b, raw


def test_actual_passer_only_and_receiver_first_contact():
    m, a, b, z = fixture()
    z["torque"][80, :, 0] = 2
    z["applied_torque_change"][80, :, 0] = 2
    z["ball_forces"][79, 0, 0, 0] = 12
    z["ball_forces"][79, 3, 1, 0] = 12
    r = summarize(z, m, a, b, True, 10, 10)
    assert r["changed_native_commands"] == 10
    assert r["receiver_foot_first"]
    assert r["receiver_post_0p2s"] is None
    z["applied_torque_change"][80, 0, 30] = 2
    with pytest.raises(ValueError, match="accounting"):
        summarize(z, m, a, b, True, 10, 10)


def test_rejects_torque_before_fixed_start_and_unbounded_delta():
    m, a, b, z = fixture()
    z["torque"][79, 0, 0] = z["applied_torque_change"][79, 0, 0] = 1
    with pytest.raises(ValueError, match="accounting"):
        summarize(z, m, a, b, True, 10, 10)
    z["torque"][79, 0, 0] = z["applied_torque_change"][79, 0, 0] = 0
    z["torque"][80, 0, 0] = z["applied_torque_change"][80, 0, 0] = 15
    with pytest.raises(ValueError, match="bounded"):
        summarize(z, m, a, b, True, 10, 10)


def test_tempo_measurement_does_not_claim_recorded_torque_composition():
    m, a, b, z = fixture()
    del z["parent_torque"]
    del z["applied_torque_change"]
    z["torque"][80, :, 0] = 2
    r = summarize(z, m, a, b, True, 0, 0, tempo=True)
    assert r["changed_native_commands"] is None
    assert r["max_abs_delta_nm"] is None
