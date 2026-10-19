"""Synthetic paired evidence tests, not G1 physical or neural performance."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi import precontact_heading_pairs as mod
from rosclaw_soccer.rsi.foot_heading_evidence import FIELDS
from rosclaw_soccer.rsi.foot_heading_parity import SHAPES


def trace(enabled):
    t = {}
    for name, shape in SHAPES.items():
        dtype = (
            np.bool_
            if name == "recurrent_sampling_performed"
            else (np.float32 if name.startswith("foundation_neural_") else np.float64)
        )
        t[name] = np.zeros((300, 1, *shape), dtype=dtype)
    for name, width in FIELDS.items():
        t["foot_heading_" + name] = np.zeros((300, 1, width), dtype=np.float64)
    t["foot_heading_state"][:, 0, 3] = int(enabled)
    t["canonical_qpos"][:, 0, 2] = 0.793
    t["canonical_qpos"][:, 0, 3] = 1.0
    t["canonical_qpos"][:, 0, 39] = 1.0
    t["root_pose_xyzw_m"][:, 0, 2] = 0.793
    t["pelvis_z_per_substep_m"][:] = 0.793
    t["force_n"][50, 0, 0] = 10.0
    t["ball_position_after_step_m"][110, 0, :2] = (1.5, 0.8 if enabled else 0.1)
    t["postcontact_release_requested_offset_nm"] = np.zeros((300, 1, 29))
    t["postcontact_release_applied_offset_nm"] = np.zeros((300, 1, 10, 29))
    t["postcontact_release_forecast_steps"] = np.zeros((300, 1, 1), dtype=np.int64)
    return t


def contexts(index=0):
    oi, ni = mod.PAIRS[index]
    off, on = trace(False), trace(True)
    context = []
    for ci, t, si in ((oi, off, 0), (ni, on, 1)):
        rows = [
            dict(
                context_index=ci,
                action_index=ai,
                action_name=name,
                course=[2.5, 0.1, 0.3],
                original_policy_hash="policy",
                outcome=mod.label_first_touch(t).to_dict(),
            )
            for ai, name in enumerate([*mod.ACTION_NAMES, "baseline"])
        ]
        context.append(
            dict(
                context_index=ci,
                source_index=si,
                rows=rows,
                partition="CONSUMED_LOCAL_DEVELOPMENT"
                if index < 6
                else "CONSUMED_ORIGINAL_RETENTION",
                report_hash=f"context-{ci}",
            )
        )
    return context[0], context[1], ([off] * 14, [on] * 14)


def test_complete_pair_keeps_all_outcomes_and_current_only_input():
    off, on, tensors = contexts()
    r = mod.paired_context(off, on, tensors, index=0)
    assert r["observation_width"] == 112 and len(r["current_features"]) == 112
    assert len(r["outcomes_by_heading_then_action"]) == 2
    assert r["observed_safe_passing_actions_by_heading"] == [list(range(13)), []]
    assert r["finite_horizon_task_pass_known_for_all26"]
    assert not r["training_authorized"] and not r["motor_policy_activation_authorized"]


def test_post_treatment_target_and_future_differ_but_features_do_not():
    off, on = trace(False), trace(True)
    before, _ = mod.compare_pair(off, on)
    on["joint_target_rad"][30] = 0.5  # Not a permitted pre-heading feature.
    on["canonical_qpos"][250, 0, 0] = 9.0
    after, _ = mod.compare_pair(off, on)
    assert before.tobytes() == after.tobytes()
    assert not after.flags.writeable


@pytest.mark.parametrize(
    "what", ["early", "current_q", "nav", "force", "heading", "extra", "release", "forecast"]
)
def test_mismatched_observation_or_prefix_rejected(what):
    off, on = trace(False), trace(True)
    if what == "early":
        on["foundation_neural_encoder_features"][29, 0, 0] = 1.0
    elif what == "current_q":
        on["canonical_qpos"][30, 0, 0] = 1.0
    elif what == "nav":
        on["foundation_navigation_joint_target_rad"][30, 0, 0] = 0.1
    elif what == "force":
        on["force_n"][29, 0, 0] = 2.0
    elif what == "heading":
        on["foot_heading_state"][40, 0, 3] = 0.0
    elif what == "extra":
        on["future_outcome"] = np.zeros(1)
    elif what == "release":
        on["postcontact_release_requested_offset_nm"][30, 0, 0] = 0.1
    else:
        on["postcontact_release_forecast_steps"][250] = 1
    with pytest.raises(ValueError):
        mod.compare_pair(off, on)


@pytest.mark.parametrize("what", ["drop", "policy", "course", "order", "partition", "label"])
def test_dropped_or_unmatched_outcome_matrix_rejected(what):
    off, on, tensors = contexts()
    on = copy.deepcopy(on)
    if what == "drop":
        on["rows"].pop()
    elif what == "policy":
        on["rows"][0]["original_policy_hash"] = "other"
    elif what == "course":
        on["rows"][0]["course"][0] += 0.1
    elif what == "order":
        on["rows"][0]["action_index"] = 1
    elif what == "partition":
        on["partition"] = "CONSUMED_ORIGINAL_RETENTION"
    else:
        on["rows"][0]["outcome"]["task_pass"] = True
    with pytest.raises(ValueError):
        mod.paired_context(off, on, tensors, index=0)


def test_retention_partition_stays_distinct():
    off, on, tensors = contexts(index=6)
    r = mod.paired_context(off, on, tensors, index=6)
    assert r["partition"] == "CONSUMED_ORIGINAL_RETENTION"
