import numpy as np
import pytest

from rosclaw_soccer.rsi.precontact_fullbody_gate import precontact_fullbody_gate


def test_only_actual_past_pass_healthy_approaching_precontact_triggers():
    q = np.zeros(43)
    v = np.zeros(41)
    q[2], q[36], q[38], v[35] = 0.8, 0.7, 0.11, -2.0
    args = dict(passer_foot_seen=True, receiver_contact_seen=False, world_healthy=True)
    assert precontact_fullbody_gate(q, v, **args)
    assert not precontact_fullbody_gate(q, v, **(args | {"passer_foot_seen": False}))
    assert not precontact_fullbody_gate(q, v, **(args | {"receiver_contact_seen": True}))
    assert not precontact_fullbody_gate(q, v, **(args | {"world_healthy": False}))
    v[35] = 2.0
    assert not precontact_fullbody_gate(q, v, **args)
    q[36] = np.nan
    with pytest.raises(ValueError):
        precontact_fullbody_gate(q, v, **args)
