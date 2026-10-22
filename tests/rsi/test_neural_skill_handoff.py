from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_shared_native_motor_world import world
from test_shared_native_neural_player import players

from rosclaw_soccer.rsi.online_receiving_boundary import OnlineReceivingBoundary
from rosclaw_soccer.rsi.paid_skill_handoff import snapshot
from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters
from rosclaw_soccer.rsi.shared_shadow_motor import SharedShadowMotor


def test_profile_switch_preserves_every_physical_field_and_original_effort_caps():
    w = world()
    w.advance({p: np.zeros(29) for p in w.layouts})
    before = snapshot(w.data)
    profile = NativePDParameters(np.full(29, 30.0), np.full(29, 2.0), np.full(29, 25.0))
    record = w.switch_pd_profile("b/", profile)
    assert record["physics_unchanged"] and record["control_step"] == 1
    for key, value in before.items():
        np.testing.assert_array_equal(
            value, getattr(w.data, key) if key != "time" else np.array(w.data.time)
        )
    profile.stiffness[:] = 100
    assert w.parameters["b/"].stiffness[0] == 30
    with pytest.raises(ValueError, match="same original effort"):
        w.switch_pd_profile("b/", NativePDParameters(np.ones(29), np.ones(29), np.full(29, 26.0)))
    w.data.qpos[0] += 0.01
    with pytest.raises(ValueError, match="unchanged"):
        w.switch_pd_profile("b/", profile)


def test_shadow_context_carries_executed_other_motor_actions_not_its_discarded_proposal():
    w, ps = players()
    template = ps["b/"]
    shadow = SharedShadowMotor(
        w,
        "b/",
        template.actor,
        clip=template.clip,
        torso_index=template.torso_index,
        build_observation=template.build_observation,
        goal=template.goal,
        offset=template.offset,
        scale=template.scale,
        order=template.order,
    )
    evidence = shadow.observe()
    with pytest.raises(ValueError):
        shadow.observe()
    with pytest.raises(ValueError):
        shadow.commit_actual()
    actual = np.full(29, 0.08)
    w.advance({"a/": np.zeros(29), "b/": actual})
    shadow.commit_actual()
    np.testing.assert_allclose(shadow.previous_action, 0.16, atol=1e-7)
    assert not np.array_equal(shadow.previous_action, evidence["discarded_action"])
    before = snapshot(w.data)
    warm = shadow.handoff(goal=template.goal, carry_memory=True)
    cold = shadow.handoff(goal=template.goal, carry_memory=False)
    assert torch.equal(warm.state[0], shadow.state[0])
    assert warm.state[0].data_ptr() != shadow.state[0].data_ptr()
    assert not cold.state[0].count_nonzero()
    np.testing.assert_array_equal(warm.last_action, shadow.previous_action)
    np.testing.assert_array_equal(cold.last_action, shadow.previous_action)
    np.testing.assert_array_equal(warm.ref_pos, cold.ref_pos)
    for key, value in before.items():
        np.testing.assert_array_equal(
            value, getattr(w.data, key) if key != "time" else np.array(w.data.time)
        )
    target, _ = warm.propose()
    w.advance({"a/": np.zeros(29), "b/": target})
    warm.commit()
    assert w.control_steps == 2 and warm.decisions == 1


def test_receiving_readiness_uses_only_observed_past_and_never_future_whole_trial_health():
    q, v = np.zeros(43), np.zeros(41)
    q[:7] = [0, 0, 0.8, 1, 0, 0, 0]
    q[36:43] = [0.25, 0.1, 0.11, 1, 0, 0, 0]
    w = SimpleNamespace(
        control_steps=0,
        failed=False,
        data=None,
        last_ball_contact_bodies={"b/": {}},
        layouts={"b/": SimpleNamespace(canonical=lambda d: (q.copy(), v.copy()))},
    )
    ready = OnlineReceivingBoundary(w, "b/")
    with pytest.raises(ValueError):
        ready.update()
    w.last_ball_contact_bodies["b/"] = {"b/right_knee_link": 100.0}
    for control in range(1, 16):
        w.control_steps = control
        assert ready.update() == (control == 15)
    assert ready.first_ready_control == 15
    # A later excluded touch must not be peeked at before it physically occurs.
    w.control_steps = 16
    w.last_ball_contact_bodies["b/"] = {"b/left_wrist_roll_link": 2.0}
    assert not ready.update() and ready.excluded_contact
    assert ready.first_ready_control == 15
