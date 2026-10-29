import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.receiving_witness_audit import audit_receiving_witness


def evidence():
    q, v = np.zeros((15, 10, 79)), np.zeros((15, 10, 76))
    q[:, :, 2] = 0.8
    q[:, :, 3] = 1
    q[:, :, 40:43] = [0.25, 0.1, 0.11]
    forces = np.zeros((15, 10, 2, 3))
    forces[0, 2, 1, 2] = 2
    physics = dict(qpos=q, qvel=v, ball_forces=forces, body_failed=np.zeros((15, 2), bool))
    contacts = [{"a/": {}, "b/": {}} for _ in range(15)]
    contacts[0]["b/"] = {"b/right_knee_link": 2.0}
    rows = [
        dict(
            control=i + 1,
            contact_seen=True,
            excluded_contact=False,
            good=True,
            consecutive=i + 1,
            ready=i == 14,
            first_ready_control=15 if i == 14 else None,
            distance_xy_m=float(np.linalg.norm([0.25, 0.1])),
            relative_speed_xy_mps=0.0,
            ball_height_m=0.11,
            forward_m=0.25,
            root_height_m=0.8,
            upright=1.0,
        )
        for i in range(15)
    ]
    return physics, dict(contacts=contacts, readiness=rows)


def test_independently_reconstructs_gate_and_rejects_tampering():
    physics, witness = evidence()
    options = dict(receiver_q=0, receiver_v=0, ball_q=40, ball_v=40)
    report = audit_receiving_witness(physics, witness, **options)
    assert report["best_consecutive_controls"] == 15
    assert report["first_ready_control"] == 15
    changed = copy.deepcopy(witness)
    changed["contacts"][0]["b/"]["b/right_knee_link"] = 3
    with pytest.raises(ValueError, match="force groups"):
        audit_receiving_witness(physics, changed, **options)
    changed = copy.deepcopy(witness)
    changed["readiness"][4]["distance_xy_m"] = 0.1
    with pytest.raises(ValueError, match="physical state"):
        audit_receiving_witness(physics, changed, **options)
    changed = copy.deepcopy(witness)
    changed["readiness"][4]["consecutive"] = 99
    with pytest.raises(ValueError, match="latches"):
        audit_receiving_witness(physics, changed, **options)
