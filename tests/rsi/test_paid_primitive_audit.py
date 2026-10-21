from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.native_contact_window import native_contact_window
from rosclaw_soccer.rsi.paid_primitive_audit import PaidPrimitiveAuditor


def synthetic_raw():
    """Primitive unit fixture, not neural/physics execution or football success."""
    q = np.zeros(43)
    q[2:4] = [0.8, 1]
    q[36:] = [0.8, 0.15, 0.11, 1, 0, 0, 0]
    z = dict(
        target=np.zeros((61, 29), np.float32),
        NN_action=np.zeros((1, 29), np.float32),
        NN_nominal_action=np.zeros((1, 29), np.float32),
        neural_control=np.arange(61) == 0,
        NN_obs=np.zeros((1, 160), np.float32),
        qpos=np.broadcast_to(q, (61, 10, 43)).copy(),
        qvel=np.zeros((61, 10, 41)),
        forces=np.zeros((61, 10, 3)),
        torque=np.zeros((61, 10, 29)),
        body_failed=np.zeros(61, bool),
        native_qpos_before=np.tile(q, (61, 1)),
        native_qvel_before=np.zeros((61, 41)),
        initial_qpos=q.copy(),
        initial_qvel=np.zeros(41),
        initial_time=np.array(0.0),
        time_before=np.arange(61) * 0.02,
        initial_mocap_pos=np.array([[6.0, 0, 0.11]]),
    )
    for key in ("h_before", "h_after", "c_before", "c_after"):
        z["NN_" + key] = np.zeros((1, 2, 1, 128), np.float32)
    post = native_contact_window(
        force=z["forces"].reshape(-1, 3)[:, :2],
        canonical_qpos=z["qpos"].reshape(-1, 43),
        canonical_qvel=z["qvel"].reshape(-1, 41),
        goal=z["initial_mocap_pos"][0],
        intended_foot=0,
    )
    row = dict(
        source="left.npz",
        neural_rows=1,
        native_steps=610,
        body_failed=False,
        forbidden=False,
        terminal_reward=-8.0,
        nonfoot_impulse_Ns=0.0,
        postcontact=post,
    )
    params = {
        k: np.full(29, value, np.float32)
        for k, value in (
            ("default_joint_pos", 0),
            ("action_scale", 1),
            ("joint_stiffness", 1),
            ("joint_damping", 1),
        )
    }
    layout = SimpleNamespace(rq=0, rv=0, bq=36, bv=35, qi=np.arange(7, 36), vi=np.arange(6, 35))
    auditor = PaidPrimitiveAuditor(
        layout=layout, parameters=params, order=np.arange(29), effort=np.ones(29), goal_mocap_id=0
    )
    return auditor, z, row


def test_missing_contact_is_audited_as_failure_not_synthetic_success():
    a, z, row = synthetic_raw()
    result = a.audit(z, row, horizon=1, previous_action=np.zeros(29, np.float32))
    assert result == dict(native_steps=610, neural_rows=1, quality=False, body_failed=False)


def test_new_reference_prefix_never_becomes_old_completed_exam():
    a, z, row = synthetic_raw()
    prefix = {
        k: (v.copy() if k.startswith(("NN_", "initial_")) else v[:1].copy()) for k, v in z.items()
    }
    row.update(
        native_steps=10,
        reference_prefix_only=True,
        official_exam_result=False,
        held_tail_controls_omitted=60,
    )
    row["postcontact"] = native_contact_window(
        force=prefix["forces"].reshape(-1, 3)[:, :2],
        canonical_qpos=prefix["qpos"].reshape(-1, 43),
        canonical_qvel=prefix["qvel"].reshape(-1, 41),
        goal=prefix["initial_mocap_pos"][0],
        intended_foot=0,
    )
    with pytest.raises(ValueError, match="tail"):
        a.audit(prefix, row, horizon=1, previous_action=np.zeros(29, np.float32))
    result = a.audit(
        prefix, row, horizon=1, previous_action=np.zeros(29, np.float32), reference_prefix=True
    )
    assert result["native_steps"] == 10 and result["quality"] is False
    row["official_exam_result"] = True
    with pytest.raises(ValueError, match="masquerade"):
        a.audit(
            prefix, row, horizon=1, previous_action=np.zeros(29, np.float32), reference_prefix=True
        )


def test_declared_hold_checks_no_neural_padding_and_actual_torque():
    a, z, _ = synthetic_raw()
    keys = (
        "target",
        "native_qpos_before",
        "native_qvel_before",
        "qpos",
        "qvel",
        "forces",
        "torque",
        "body_failed",
        "time_before",
    )
    hold = {k: z[k][:3].copy() for k in keys}
    hold.update(neural_control=np.zeros(3, bool), post_reference_hold=np.ones(3, bool))
    entry = {k: z["initial_" + k] for k in ("qpos", "qvel", "time")}
    assert a.audit_hold(hold, entry, z["target"][0]) == dict(
        native_steps=30, body_failed=False, settled=True
    )
    hold["torque"][0, 0, 0] = 0.1
    with pytest.raises(ValueError, match="torque"):
        a.audit_hold(hold, entry, z["target"][0])
    hold["torque"][0, 0, 0] = 0
    hold["NN_fake"] = np.array(0.0)
    with pytest.raises(ValueError, match="padding"):
        a.audit_hold(hold, entry, z["target"][0])


@pytest.mark.parametrize(
    "fault", ["time", "body_reset", "torque", "action", "h", "previous", "reward", "censor"]
)
def test_inconsistent_raw_or_summary_is_rejected(fault):
    a, z, row = synthetic_raw()
    if fault == "time":
        z["time_before"][1] = 0
    elif fault == "body_reset":
        z["native_qpos_before"][1, 0] = 0.1
    elif fault == "torque":
        z["torque"][0, 0, 0] = 0.1
    elif fault == "action":
        z["NN_action"][0, 0] = 0.1
    elif fault == "h":
        z["NN_h_before"][0, 0, 0, 0] = 0.1
    elif fault == "previous":
        z["NN_obs"][0, 125] = 0.1
    elif fault == "reward":
        row["terminal_reward"] = 10
    else:
        z["neural_control"][-1] = True
    with pytest.raises(ValueError):
        a.audit(z, row, horizon=1, previous_action=np.zeros(29, np.float32))
