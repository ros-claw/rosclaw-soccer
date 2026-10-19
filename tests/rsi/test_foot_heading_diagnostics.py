"""Synthetic dense contracts and old integrator guards, not a new rollout."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.foot_heading_diagnostics import (
    heading_contact_readout,
    heading_motion_readout,
)
from rosclaw_soccer.rsi.foot_heading_evidence import FIELDS, PREFIX
from rosclaw_soccer.rsi.foot_heading_scope import assess_heading_scope
from scripts.rsi_replay_first_touch_contacts import fixed_replay_scope
from tests.rsi.test_foot_heading_parity import baseline
from tests.rsi.test_foot_heading_scope import receipts
from tests.rsi.test_pre_planner_pulse_cohort import dense


@pytest.mark.parametrize(
    "mutation", [None, "partial", "nonfoot", "newton", "retag", "label", "steps"]
)
def test_all_heading_endpoints_keep_failure_and_nonfoot_veto(mutation):
    s, rows = receipts()
    d, c = dict(scope=s), dict(results=rows, assessment=assess_heading_scope(s, rows))
    replay, metrics = dense(d, c)
    if mutation == "partial":
        metrics["rows"].pop()
    elif mutation == "nonfoot":
        metrics["rows"][3]["metrics"]["first_nonfoot_robot_contact_substep"] = 100
    elif mutation == "newton":
        metrics["rows"][0]["metrics"]["maximum_linear_newton_residual_n"] = 1e-6
    elif mutation == "retag":
        replay["rows"][0]["case_id"] = "ik_pulse_off_g28"
    elif mutation == "label":
        metrics["rows"][1]["original_outcome"] = {}
    elif mutation == "steps":
        replay["integrated_substeps"] = 36000
    if mutation in (None, "nonfoot"):
        r = heading_contact_readout(d, c, replay, metrics)
        assert len(r["rows"]) == 6
        assert r["heading_endpoints_with_nonfoot_over1n"] == (2 if mutation == "nonfoot" else 1)
        assert not r["teacher_qualified"] and not r["neural_training_authorized"]
        assert r["research_decision"] == "UNSAFE_OR_NONFOOT_HEADING_COMPOSITION_NOT_ADMITTED"
    else:
        with pytest.raises(ValueError):
            heading_contact_readout(d, c, replay, metrics)


@pytest.mark.parametrize("mutation", [None, "early", "reference", "nan", "partial"])
def test_motion_preserves_original36_prefix_and_reports_heading_not_a_learning_gain(mutation):
    off = baseline()
    off.update({PREFIX + k: np.zeros((300, 1, w)) for k, w in FIELDS.items()})
    on = copy.deepcopy(off)
    on["foot_heading_additional_applied_rad"][30:, 0, 0] = 0.05
    on["torque_nm"][31:, 0] = 0.01
    if mutation == "early":
        on["joint_target_rad"][29, 0, 0] = 0.001
    elif mutation == "reference":
        on["reference_student_input"] = np.zeros((300, 1, 135))
    elif mutation == "nan":
        on["torque_nm"][299, 0, 0, 0] = np.nan
    elif mutation == "partial":
        del on["foot_heading_state"]
    if mutation is None:
        r = heading_motion_readout(off, on)
        assert r["before_action_prefix"]["compared_frames"] == 30
        assert (
            r["measured_channels"]["heading_additional_applied"]["first_frame_above"]["1e-08"] == 30
        )
        assert r["measured_channels"]["pd_torque"]["first_frame_above"]["1e-08"] == 31
        assert r["first_difference_is_not_proven_causal_order"]
        assert r["new_physical_steps"] == r["optimizer_updates"] == 0
    else:
        with pytest.raises(ValueError):
            heading_motion_readout(off, on)


@pytest.mark.parametrize("mutation", [None, "order", "duplicate", "partial", "budget", "oldscope"])
def test_fixed_heading_diagnostic_cannot_select_easy_cases(mutation):
    s, _ = receipts()
    cfg = dict(
        replay_profile="foot-heading-contact-six",
        behavior_model_hash=s["source_plan"]["behavior_model_hash"],
        new_replay_budget=6,
        new_substep_budget=18000,
        cases=[
            dict(case_id=j["job"].replace("-", "_"), receipt_hash=f"r{i}")
            for i, j in enumerate(s["jobs"])
        ],
    )
    if mutation == "order":
        cfg["cases"].reverse()
    elif mutation == "duplicate":
        cfg["cases"][1]["receipt_hash"] = "r0"
    elif mutation == "partial":
        cfg["cases"].pop()
    elif mutation == "budget":
        cfg["new_substep_budget"] = 18000.0
    elif mutation == "oldscope":
        cfg["cases"][0]["case_id"] = "ik_pulse_off_g28"
    if mutation is None:
        assert fixed_replay_scope(cfg) == (6, 18000)
    else:
        with pytest.raises(ValueError):
            fixed_replay_scope(cfg)
