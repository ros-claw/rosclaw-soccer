from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.online_receiving_boundary import OnlineReceivingBoundary
from rosclaw_soccer.rsi.receiving_witness import body_contact_witness, readiness_witness


def test_readiness_witness_tracks_actual_gate_without_mutating_world():
    q, v = np.zeros(43), np.zeros(41)
    q[:7] = [0, 0, 0.8, 1, 0, 0, 0]
    q[36:43] = [0.25, 0.1, 0.11, 1, 0, 0, 0]
    w = SimpleNamespace(
        control_steps=0,
        failed=False,
        data=None,
        last_ball_contact_bodies={"b/": {}},
        layouts={"b/": SimpleNamespace(canonical=lambda d: (q.copy(), v.copy()))},
        _require_current_boundary=lambda **kwargs: None,
    )
    boundary = OnlineReceivingBoundary(w, "b/")
    w.last_ball_contact_bodies["b/"] = {"b/right_knee_link": 100.0}
    for control in range(1, 16):
        w.control_steps = control
        actual = boundary.update()
        evidence = readiness_witness(boundary, actual)
        assert evidence["consecutive"] == control
        assert evidence["good"] and evidence["ready"] == (control == 15)
    assert evidence["first_ready_control"] == 15
    with pytest.raises(ValueError, match="disagree"):
        readiness_witness(boundary, False)
    w.control_steps += 1
    with pytest.raises(ValueError, match="updated"):
        readiness_witness(boundary, True)
    w.last_ball_contact_bodies["b/"] = {"b/left_wrist_roll_link": 2.0}
    actual = boundary.update()
    evidence = readiness_witness(boundary, actual)
    assert evidence["excluded_contact"] and not evidence["good"]
    assert evidence["consecutive"] == 0
    np.testing.assert_array_equal(q[:7], [0, 0, 0.8, 1, 0, 0, 0])


def test_body_witness_copies_both_players_and_requests_actual_boundary():
    seen = []
    contacts = {"a/": {"a/left_ankle_roll_link": 9.0}, "b/": {}}
    w = SimpleNamespace(
        last_ball_contact_bodies=contacts,
        _require_current_boundary=lambda **kwargs: seen.append(kwargs),
    )
    evidence = body_contact_witness(w)
    contacts["a/"]["a/left_ankle_roll_link"] = 3.0
    assert evidence["a/"]["a/left_ankle_roll_link"] == 9.0
    assert evidence["b/"] == {} and seen == [{"allow_failed": True}]
