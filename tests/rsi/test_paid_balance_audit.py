from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.paid_balance_audit import audit_balance_recovery


def fixture():
    """Algebra-only stationary fixture, not a MuJoCo physics success."""
    n = 100
    q, v = np.zeros(43), np.zeros(41)
    q[2], q[3], q[38], q[39] = 0.78, 1, 0.11, 1
    entry = dict(
        qpos=q.copy(),
        qvel=v.copy(),
        ctrl=np.zeros(29),
        qacc_warmstart=v.copy(),
        mocap_pos=np.array([[6.0, 0, 0.11]]),
        mocap_quat=np.array([[1.0, 0, 0, 0]]),
        qfrc_applied=v.copy(),
        xfrc_applied=np.zeros((33, 6)),
        act=np.zeros(0),
        time=np.array(0.0),
    )
    raw = dict(
        native_qpos_before=np.tile(q, (n, 1)),
        native_qvel_before=np.tile(v, (n, 1)),
        target=np.zeros((n, 29)),
        neural_control=np.zeros(n, bool),
        time_before=np.arange(n) * 0.02,
        qpos=np.tile(q, (n, 10, 1)),
        qvel=np.tile(v, (n, 10, 1)),
        torque=np.zeros((n, 10, 29)),
        forces=np.zeros((n, 10, 3)),
        body_failed=np.zeros(n, bool),
        RECOVERY_requested_delta_HG=np.zeros((n, 29)),
        RECOVERY_actual_delta_HG=np.zeros((n, 29)),
        RECOVERY_actual_action=np.zeros((n, 29), np.float32),
        RECOVERY_anchor_HG=np.zeros((n, 29)),
        RECOVERY_body_feedback=np.zeros((n, 8)),
        **{"initial_" + k: value.copy() for k, value in entry.items()},
    )
    auditor = SimpleNamespace(
        canonical=lambda q, v: (q, v),
        offset=np.zeros(29),
        scale=np.ones(29),
        order=np.arange(29),
        kp=np.ones(29) * 100,
        kd=np.ones(29) * 4,
        effort=np.ones(29) * 100,
    )
    args = dict(
        entry=entry,
        anchor=np.zeros(29),
        anchor_action=np.zeros(29, np.float32),
        latent=np.zeros((9, 4)),
    )
    return auditor, raw, args


def test_reconstructs_complete_fixed_anchor_causal_feedback_and_bounded_pd():
    auditor, raw, args = fixture()
    result = audit_balance_recovery(auditor, raw, **args)
    assert result["healthy"] and result["recovery_stability_observed"]
    assert result["cost"] == 0
    assert result["next_skill_ready_proven"] is False


@pytest.mark.parametrize(
    "field,index,value",
    [
        ("RECOVERY_requested_delta_HG", (0, 0), 0.03),
        ("RECOVERY_body_feedback", (0, 0), 0.03),
        ("RECOVERY_actual_action", (0, 0), 0.03),
        ("RECOVERY_anchor_HG", (0, 0), 0.03),
        ("torque", (0, 0, 0), 2.0),
        ("time_before", (5,), 0.3),
        ("initial_qvel", (0,), 0.2),
        ("neural_control", (0,), True),
    ],
)
def test_changed_motor_state_or_proposal_cannot_receive_a_pass(field, index, value):
    auditor, raw, args = fixture()
    raw[field][index] = value
    with pytest.raises(ValueError):
        audit_balance_recovery(auditor, raw, **args)
