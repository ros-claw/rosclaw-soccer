import numpy as np
import pytest

from rosclaw_soccer.rsi.pass_transfer_outcome import pass_transfer_outcome


def trajectory():
    p, v, f = np.zeros((4000, 3)), np.zeros((4000, 3)), np.zeros((4000, 2, 3))
    p[:, 0] = 2
    p[100:200, 0] = np.linspace(2, 0.05, 100)
    v[:, 0] = 2
    f[100, 0, 0] = 11
    f[200, 1, 0] = 2
    return p, v, f


def score(p, v, f, **kwargs):
    return pass_transfer_outcome(
        p,
        v,
        f,
        np.zeros(3),
        **(
            dict(
                whole_body_healthy=True,
                passer_recovery_observed=True,
                receiver_first_ready_control=30,
                receiver_excluded_contact=False,
            )
            | kwargs
        ),
    )


def test_actual_delivery_and_causal_receive_are_both_required():
    p, v, f = trajectory()
    assert score(p, v, f)["combined_precision_and_functional_receive"]
    assert not score(p, v, f, receiver_first_ready_control=None)[
        "combined_precision_and_functional_receive"
    ]
    assert not score(p, v, f, receiver_excluded_contact=True)[
        "combined_precision_and_functional_receive"
    ]
    assert not score(p, v, f, passer_recovery_observed=False)[
        "combined_precision_and_functional_receive"
    ]


def test_B_dribble_cannot_repair_A_delivery_error():
    p, v, f = trajectory()
    p[:200, 0] = 2
    p[200:, 0] = 0
    r = score(p, v, f)
    assert r["B_legal_functional_receive_before_deadline"]
    assert r["incoming"]["incoming_target_error_3d_before_B_contact_m"] == 2
    assert not r["precise_moderate_delivery"]


def test_recovery_tail_cannot_improve_formal_arrival_or_receive():
    p, v, f = trajectory()
    p[:, 0] = 2
    p[3000:, 0] = 0
    f[:, 1, :] = 0
    f[3100, 1, 0] = 2
    r = score(p, v, f, receiver_first_ready_control=330)
    assert r["formal_native_steps"] == 3000
    assert not r["precise_moderate_delivery"]
    assert not r["B_legal_functional_receive_before_deadline"]


def test_no_contact_speed_nonfoot_and_failure_remain_negative():
    p, v, f = trajectory()
    v[:, 0] = 3.01
    assert not score(p, v, f)["precise_moderate_delivery"]
    v[:, 0] = 2
    f[3999, 0, 2] = 2
    assert not score(p, v, f)["precise_moderate_delivery"]
    f[:, 0, :] = 0
    r = score(p, v, f)
    assert r["incoming_arrival_speed_mps"] is None
    assert not r["precise_moderate_delivery"]
    assert not score(p, v, f, whole_body_healthy=False)["combined_precision_and_functional_receive"]


def test_invalid_future_readiness_is_rejected():
    p, v, f = trajectory()
    with pytest.raises(ValueError):
        score(p, v, f, receiver_first_ready_control=401)
