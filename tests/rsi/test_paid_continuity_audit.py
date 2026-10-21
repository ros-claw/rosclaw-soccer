import copy

import numpy as np
import pytest
from test_paid_primitive_audit import synthetic_raw

from rosclaw_soccer.rsi.paid_continuity_audit import audit_practice_boundary, audit_terminal


def fixture():
    a, z, _ = synthetic_raw()
    a.layout.ai = np.arange(29)
    terminal = dict(
        qpos=z["native_qpos_before"][-1].copy(),
        qvel=z["native_qvel_before"][-1].copy(),
        ctrl=np.zeros(29),
        time=np.array(1.22),
        qacc_warmstart=np.zeros(41),
        mocap_pos=z["initial_mocap_pos"].copy(),
        mocap_quat=np.array([[1.0, 0, 0, 0]]),
        qfrc_applied=np.zeros(41),
        xfrc_applied=np.zeros((2, 6)),
        act=np.zeros(0),
    )
    drop = dict(performed=np.array(False), **{"before_" + k: v.copy() for k, v in terminal.items()})
    boundary = {
        **{"before_" + k: v.copy() for k, v in terminal.items()},
        **{"after_" + k: v.copy() for k, v in terminal.items()},
        "previous_actual_action": np.zeros(29, np.float32),
    }
    return a, z, terminal, drop, boundary


def test_healthy_no_return_boundary_preserves_complete_state_and_action():
    a, z, t, d, b = fixture()
    audit_terminal(a, z, t)
    result = audit_practice_boundary(
        a, previous=z, terminal=t, drop=d, hold=None, boundary=b, source="left.npz"
    )
    assert result["status"] == "READY" and result["native_steps"] == 0
    assert set(result["state"]) == set(t)


@pytest.mark.parametrize(
    "fault",
    [
        "root",
        "velocity",
        "warmstart",
        "force",
        "goal",
        "missing",
        "previous",
        "eligibility",
        "omitted",
        "hidden_hold",
    ],
)
def test_reset_erased_history_missing_state_or_undeclared_physics_rejected(fault):
    a, z, t, d, b = fixture()
    if fault == "root":
        b["after_qpos"][0] += 0.01
    elif fault == "velocity":
        b["after_qvel"][0] += 0.01
    elif fault == "warmstart":
        b["after_qacc_warmstart"][0] += 0.01
    elif fault == "force":
        b["after_qfrc_applied"][0] += 0.01
    elif fault == "goal":
        b["after_mocap_pos"][0, 0] += 0.01
    elif fault == "missing":
        b.pop("after_qacc_warmstart")
        b.pop("before_qacc_warmstart")
    elif fault == "previous":
        b["previous_actual_action"][0] = 1
    elif fault == "eligibility":
        d["performed"] = np.array(True)
    elif fault == "omitted":
        b = None
    hold = {} if fault == "hidden_hold" else None
    with pytest.raises(ValueError):
        audit_practice_boundary(
            a, previous=z, terminal=t, drop=d, hold=hold, boundary=b, source="left.npz"
        )


def test_incorrect_native_terminal_is_not_a_valid_handoff():
    a, z, t, _, _ = fixture()
    changed = copy.deepcopy(t)
    changed["ctrl"][0] = 1
    with pytest.raises(ValueError):
        audit_terminal(a, z, changed)
