"""Synthetic diagnostic contract checks, not actual G1 performance evidence."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.original_ik_planner_diagnostics import ik_contact_readout, ik_motion_readout
from rosclaw_soccer.rsi.original_ik_planner_scope import assess_ik_scope
from rosclaw_soccer.rsi.reference_feedback_divergence import CHANNELS
from tests.rsi.test_original_ik_planner_scope import receipts
from tests.rsi.test_pre_planner_pulse_cohort import dense


@pytest.mark.parametrize(
    "mutation", [None, "missing", "nonfoot", "newton", "retag", "label", "steps"]
)
def test_original_ik_contact_cannot_claim_learning_or_drop_failures(mutation):
    scope, rows = receipts()
    d, c = dict(scope=scope), dict(results=rows, assessment=assess_ik_scope(scope, rows))
    replay, metrics = dense(d, c)
    if mutation == "missing":
        metrics["rows"].pop()
    elif mutation == "nonfoot":
        metrics["rows"][1]["metrics"]["first_nonfoot_robot_contact_substep"] = 20
    elif mutation == "newton":
        metrics["rows"][0]["metrics"]["maximum_linear_newton_residual_n"] = 1e-6
    elif mutation == "retag":
        replay["rows"][0]["case_id"] = "planner_pulse_off_g28"
    elif mutation == "label":
        metrics["rows"][1]["original_outcome"] = {}
    elif mutation == "steps":
        replay["integrated_substeps"] = 36000
    if mutation in (None, "nonfoot"):
        r = ik_contact_readout(d, c, replay, metrics)
        assert len(r["rows"]) == 6
        assert not r["teacher_qualified"] and not r["neural_training_authorized"]
        assert r["new_closed_loop_policy_trials"] == r["new_optimizer_updates"] == 0
        assert r["pulse_endpoints_with_nonfoot_over1n"] == 1
        assert r["research_decision"] == "UNSAFE_OR_NONFOOT_IK_COMPOSITION_NOT_ADMITTED"
    else:
        with pytest.raises(ValueError):
            ik_contact_readout(d, c, replay, metrics)


def motion_traces():
    t = {key: np.zeros((300, 1, *tail)) for _, key, tail, _ in CHANNELS[2:]}
    t.update(
        canonical_qpos=np.zeros((300, 1, 43)),
        canonical_qvel=np.zeros((300, 1, 41)),
        navigation_command=np.zeros((300, 1, 3)),
        taskspace_joint_increment_rad=np.zeros((300, 1, 29)),
    )
    for i in range(33 - len(t)):
        t[f"extra_original_array_{i}"] = np.zeros((300, 1, 1))
    return t, copy.deepcopy(t)


@pytest.mark.parametrize("mutation", [None, "early", "reference", "nan", "short"])
def test_original_ik_motion_uses_real_channels_without_fabricating_gru_inputs(mutation):
    off, pulse = motion_traces()
    pulse["navigation_command"][100:160, 0, 1] = 0.08
    pulse["torque_nm"][103:, 0] = 0.01
    if mutation == "early":
        pulse["joint_position_rad"][99, 0, 0] = 0.001
    elif mutation == "reference":
        pulse["reference_student_requested_reference_increment_rad"] = np.zeros((300, 1, 12))
    elif mutation == "nan":
        pulse["torque_nm"][150, 0, 0, 0] = np.nan
    elif mutation == "short":
        pulse["torque_nm"] = pulse["torque_nm"][:-1]
    if mutation is None:
        r = ik_motion_readout(off, pulse)
        assert r["before_action_prefix"]["compared_frames"] == 100
        assert not r["reference_gru_arrays_fabricated"]
        assert r["measured_channels"]["navigation"]["first_frame_above"]["1e-08"] == 100
        assert r["measured_channels"]["pd_torque"]["first_frame_above"]["1e-08"] == 103
        assert r["first_difference_is_not_proven_causal_order"]
    else:
        with pytest.raises(ValueError):
            ik_motion_readout(off, pulse)
