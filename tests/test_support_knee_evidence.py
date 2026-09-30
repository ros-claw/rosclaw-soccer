"""The saved support-leg action must be reproducible from causal measurements."""

from __future__ import annotations

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi import support_knee_evidence
from rosclaw_soccer.rsi.support_knee_nullspace import support_knee_nullspace_delta
from rosclaw_soccer.rsi.taskspace_swing_evidence import LEG_NAMES


class _Replay(dict[str, np.ndarray]):
    @property
    def files(self) -> list[str]:
        return list(self)


def _fixture() -> tuple[_Replay, dict[str, object]]:
    order = list(G1_DDS_JOINT_NAMES)
    baseline = np.zeros((1, 1, 29), dtype=np.float32)
    limits = np.tile(np.array((-1.0, 1.0), dtype=np.float32), (1, 29, 1))
    feet = np.zeros((1, 1, 2, 3), dtype=np.float32)
    feet[0, 0, 0, 2] = 0.03  # airborne left foot, grounded right support
    foot_jac = np.zeros((1, 1, 2, 3, 6), dtype=np.float32)
    foot_jac[0, 0, 1, :, :3] = np.eye(3)
    knee_jac = np.zeros((1, 1, 2, 6), dtype=np.float32)
    knee_jac[0, 0, 1, 3] = 1.0
    ids = [order.index(name) for name in LEG_NAMES[1]]
    proposal = support_knee_nullspace_delta(
        foot_jac[0, 0, 1],
        knee_jac[0, 0, 1],
        baseline[0, 0, ids],
        limits[0, ids],
        retract_m=0.04,
        support_foot_grounded=True,
        swing_foot_airborne=True,
    )
    applied = np.zeros_like(baseline)
    applied[0, 0, ids] = np.asarray(proposal.joint_delta_rad, dtype=np.float32)
    replay = _Replay(
        pre_step_foot_link_position_w=feet,
        pre_step_foot_linear_jacobian_w=foot_jac,
        pre_step_knee_x_jacobian_w=knee_jac,
        taskspace_selected_side=np.array([[0]], dtype=np.int64),
        applied_support_knee_joint_delta_rad=applied,
        applied_taskspace_joint_delta_rad=applied.copy(),
        baseline_taskspace_joint_target_rad=baseline,
        executed_taskspace_joint_target_rad=baseline + applied,
        taskspace_joint_limits_rad=limits,
        observed_ball_body_contact_force_peak_n=np.zeros((1, 1, 6), dtype=np.float32),
    )
    report: dict[str, object] = {
        "support_knee_retract_m": 0.04,
        "taskspace_joint_order": order,
        "selected_taskspace_mask": [True],
    }
    return replay, report


def test_causal_support_action_replays(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        support_knee_evidence,
        "audit_taskspace_swing_trace",
        lambda *_args, **_kwargs: {"taskspace_action_audited": True},
    )
    replay, report = _fixture()
    audit = support_knee_evidence.audit_support_knee_action_trace(replay, report, frames=1, count=1)
    assert audit["support_knee_action_audited"]
    assert audit["support_knee_applied_lane_frames"] == 1
    assert audit["support_knee_peak_predicted_retract_m"] > 0


def test_support_lane_must_bind_to_existing_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        support_knee_evidence,
        "audit_taskspace_swing_trace",
        lambda *_args, **_kwargs: {"taskspace_action_audited": True},
    )
    replay, report = _fixture()
    report["support_knee_lane"] = 0
    assert (
        support_knee_evidence.audit_support_knee_action_trace(replay, report, frames=1, count=1)[
            "support_knee_applied_lane_frames"
        ]
        == 1
    )
    report["support_knee_lane"] = 1
    with pytest.raises(ValueError, match="uncommitted"):
        support_knee_evidence.audit_support_knee_action_trace(replay, report, frames=1, count=1)


@pytest.mark.parametrize("tamper", ["target", "measured_knee", "contact_gate"])
def test_tampered_support_action_is_rejected(monkeypatch: pytest.MonkeyPatch, tamper: str) -> None:
    monkeypatch.setattr(
        support_knee_evidence,
        "audit_taskspace_swing_trace",
        lambda *_args, **_kwargs: {"taskspace_action_audited": True},
    )
    replay, report = _fixture()
    if tamper == "target":
        replay["applied_support_knee_joint_delta_rad"][0, 0, 0] = 0.01
    elif tamper == "measured_knee":
        replay["pre_step_knee_x_jacobian_w"][0, 0, 1] = 0
    else:
        replay["pre_step_foot_link_position_w"][0, 0, 1, 2] = 0.2
    with pytest.raises(ValueError, match="support-knee joint action drift"):
        support_knee_evidence.audit_support_knee_action_trace(replay, report, frames=1, count=1)
