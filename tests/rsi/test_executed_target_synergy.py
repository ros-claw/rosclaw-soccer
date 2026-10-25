import copy

import numpy as np
import pytest
from test_shared_nn_forecast import setup

from rosclaw_soccer.rsi.executed_target_synergy import (
    ExecutedTargetSynergy,
    apply_executed_target_synergy,
)
from rosclaw_soccer.rsi.shared_nn_forecast import policy_state_hash
from scripts.rsi_review_joint_response_pilot import verify_target_edits


def plan(start=0, value=0.01):
    return ExecutedTargetSynergy(start, ((1.0,) + (0.0,) * 28,), ((value,), (-value,)))


def test_smooth_actual_target_plan_has_finite_bounded_entry_midpoint_exit():
    offsets = plan(value=0.06).offsets()
    assert offsets.shape == (49, 29)
    assert abs(offsets).max() <= 0.06
    assert not offsets[[0, 24, 48]].any()
    assert abs(np.diff(offsets, axis=0)).max() < 0.008


def test_zero_preserves_exact_pending_policy_and_array_identity(tmp_path):
    p, _ = setup(tmp_path)
    before = policy_state_hash(p.policy)
    raw, target = p.policy.last_action_il, p.pending
    result, receipt = apply_executed_target_synergy(p, plan(value=0.0))
    assert before == policy_state_hash(p.policy)
    assert p.pending is target and p.policy.last_action_il is raw
    np.testing.assert_array_equal(result, target)
    assert receipt["zero_branch_exact"] and not receipt["applied"]


def test_composite_target_and_future_nn_history_agree_without_changing_past(tmp_path):
    p, _ = setup(tmp_path)
    q, v, time = p.world.data.qpos.copy(), p.world.data.qvel.copy(), p.world.data.time
    history = copy.deepcopy(p.policy.history)
    old = p.pending.copy()
    target, receipt = apply_executed_target_synergy(p, plan())
    assert receipt["applied"] and not receipt["reference_modified"]
    assert target[0] > old[0] and np.array_equal(target[1:], old[1:])
    np.testing.assert_array_equal(target, p.policy.last_action_il.astype(np.float64))
    for a, b in zip(history, p.policy.history, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(q, p.world.data.qpos)
    np.testing.assert_array_equal(v, p.world.data.qvel)
    assert time == p.world.data.time
    last = p.policy.last_action_il.copy()
    p.world.advance({p.prefix: target})
    p.commit()
    p.propose()
    np.testing.assert_array_equal(p.policy.history[-1], last)


def test_nonzero_plan_forecast_matches_actual_multiframe_nn_and_physics(tmp_path):
    p, f = setup(tmp_path, budget=160)
    p.world.advance({p.prefix: p.pending})
    p.commit()
    before = policy_state_hash(p.policy)
    original = f.forecast(controls=8, before_proposal=True)
    predicted = f.forecast(controls=8, before_proposal=True, target_plan=plan())
    assert policy_state_hash(p.policy) == before
    assert not np.array_equal(original["qpos"], predicted["qpos"])
    records, targets = [], []
    for _ in range(8):
        p.propose()
        target, _ = apply_executed_target_synergy(p, plan())
        targets.append(target)
        records.append(p.world.advance({p.prefix: target}))
        p.commit()
    for key in ("qpos", "qvel", "torque", "ball_forces"):
        np.testing.assert_array_equal(predicted[key], np.concatenate([r[key] for r in records]))
    np.testing.assert_array_equal(predicted["joint_targets"], targets)
    assert predicted["executed_target_plan"]
    assert len(predicted["executed_target_edits"]) == 8
    assert predicted["executed_target_edits"][0]["control_step"] == 2
    assert predicted["executed_target_edits"][0]["owned_control_step"] == 0
    assert p.policy.enters == 1


def test_second_target_edit_at_same_boundary_is_rejected(tmp_path):
    p, _ = setup(tmp_path)
    apply_executed_target_synergy(p, plan())
    before = policy_state_hash(p.policy)
    with pytest.raises(ValueError, match="pending NN boundary"):
        apply_executed_target_synergy(p, plan())
    assert policy_state_hash(p.policy) == before


def test_target_plan_cannot_modify_pending_forecast_or_mix_reference(tmp_path):
    from rosclaw_soccer.rsi.motion_reference_timing import MotionTimingPlan

    p, f = setup(tmp_path)
    with pytest.raises(ValueError, match="separate explicit"):
        f.forecast(controls=4, target_plan=plan())
    p.world.advance({p.prefix: p.pending})
    p.commit()
    with pytest.raises(ValueError, match="separate explicit"):
        f.forecast(
            controls=4,
            before_proposal=True,
            target_plan=plan(),
            reference_plan=MotionTimingPlan(2, 24, 3.0),
        )
    assert f.counters()["attempted_native_steps"] == 0


def test_empty_action_joint_intersection_falls_back_atomically(tmp_path):
    p, _ = setup(tmp_path)
    p.policy.action_clip_lo_il[0] = 3
    p.policy.action_clip_hi_il[0] = 4
    before = policy_state_hash(p.policy)
    target, receipt = apply_executed_target_synergy(p, plan())
    assert receipt["fallback"] and not receipt["applied"]
    assert policy_state_hash(p.policy) == before
    np.testing.assert_array_equal(target, p.pending)


def test_real_mode_and_invalid_encoder_rejected_without_mutation(tmp_path):
    p, _ = setup(tmp_path)
    p.policy.runtime_mode = "real"
    before = policy_state_hash(p.policy)
    with pytest.raises(ValueError, match="SIM"):
        apply_executed_target_synergy(p, plan())
    assert policy_state_hash(p.policy) == before
    p.policy.runtime_mode = "sim"
    p.pending[0] += 0.001
    with pytest.raises(ValueError, match="exactly decode"):
        apply_executed_target_synergy(p, plan())


@pytest.mark.parametrize("value", [float("nan"), 0.061])
def test_unbounded_or_nonfinite_plan_is_not_an_actuator_command(value):
    with pytest.raises(ValueError, match="finite normalized"):
        plan(value=value).offsets()


def test_independent_target_codec_readback_and_tamper_rejection(tmp_path):
    p, _ = setup(tmp_path)
    target, edit = apply_executed_target_synergy(p, plan())
    codec = (
        p.policy.default_q_mj,
        p.policy.action_scale_mj,
        np.arange(29),
        p.policy.action_clip_lo_il,
        p.policy.action_clip_hi_il,
    )
    kwargs = dict(
        start=p.world.control_steps,
        plan=plan(),
        codec=codec,
        limits=p.world.model.jnt_range[p.layout.joints],
    )
    verify_target_edits([edit], target[None], **kwargs)
    wrong = copy.deepcopy(edit)
    wrong["executed_raw_action_il"][0] += 0.001
    with pytest.raises(ValueError, match="codec"):
        verify_target_edits([wrong], target[None], **kwargs)
    wrong = copy.deepcopy(edit)
    wrong["control_step"] = 0
    with pytest.raises(ValueError, match="clock"):
        verify_target_edits([wrong], target[None], **kwargs)
    with pytest.raises(ValueError, match="every controlled"):
        verify_target_edits([], target[None], **kwargs)
