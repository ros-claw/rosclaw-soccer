from copy import deepcopy

import numpy as np
import pytest
from test_paid_primitive_audit import synthetic_raw

from rosclaw_soccer.rsi.native_contact_window import native_contact_window
from rosclaw_soccer.rsi.paid_recovery_trial_audit import audit_recovery_trial


def fixture():
    """Algebra-only failed-world reader fixture, not physics or NN execution."""
    auditor, z, row = synthetic_raw()
    auditor.layout.ai = np.arange(29)
    z["qpos"][-1, -1, 2] = 0.5
    z["body_failed"][-1] = True
    row.update(body_failed=True, terminal_reward=-12.0)
    row["postcontact"] = native_contact_window(
        force=z["forces"].reshape(-1, 3)[:, :2],
        canonical_qpos=z["qpos"].reshape(-1, 43),
        canonical_qvel=z["qvel"].reshape(-1, 41),
        goal=z["initial_mocap_pos"][0],
        intended_foot=0,
    )
    terminal = dict(
        qpos=z["qpos"][-1, -1].copy(),
        qvel=z["qvel"][-1, -1].copy(),
        ctrl=np.zeros(29),
        qacc_warmstart=np.zeros(41),
        mocap_pos=z["initial_mocap_pos"].copy(),
        mocap_quat=np.array([[1.0, 0, 0, 0]]),
        qfrc_applied=np.zeros(41),
        xfrc_applied=np.zeros((33, 6)),
        act=np.zeros(0),
        time=np.array(1.22),
    )
    data = dict(skill0=z, terminal0=terminal)
    world = dict(
        skills=[dict(index=0, result=row)],
        boundaries=[],
        recovery=None,
        status="FIRST_BODY_FAILURE",
        stable=False,
        quality_by_skill=[False],
        body_initializations=1,
        body_resets=0,
        native_steps=610,
        neural_rows=1,
        final_sim_time=1.22,
    )
    args = dict(
        data=data,
        source_blocks=deepcopy(data),
        first_boundary_status="NOT_REACHED",
        motion=dict(q=np.zeros((1, 29))),
        actor=None,
        latent=np.zeros((9, 4)),
    )
    return auditor, world, args


def test_failed_first_body_remains_failed_without_next_skill_or_neural_work():
    a, w, args = fixture()
    result = audit_recovery_trial(a, w, **args)
    assert not result["stable"] and result["own_neural"] == 0
    assert result["status"] == "FIRST_BODY_FAILURE"


@pytest.mark.parametrize("fault", ["success", "extra", "reset"])
def test_fabricated_success_unrecorded_phase_or_body_reset_is_rejected(fault):
    a, w, args = fixture()
    if fault == "success":
        w["stable"] = True
    if fault == "extra":
        args["data"]["hidden-phase"] = {}
    if fault == "reset":
        w["body_resets"] = 1
    with pytest.raises(ValueError):
        audit_recovery_trial(a, w, **args)
