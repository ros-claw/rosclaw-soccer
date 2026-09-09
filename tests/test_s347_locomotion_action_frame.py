from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.locomotion_action_frame import remap_previous_locomotion_action
from rosclaw_soccer.providers.g1.mujoco_primitives import mirror_g1_joint_positions
from rosclaw_soccer.skills.team.independent_team_world import (
    IndependentTeamWorldConfig,
    _run_locomotion,
)


def parameters():
    return dict(
        default_angles=np.linspace(-0.15, 0.15, 29),
        action_scale=0.25,
        joint_to_motor=np.roll(np.arange(29), 5),
    )


def physical(action, params, reflected):
    target = np.empty(29)
    target[params["joint_to_motor"]] = (
        action.astype(float) * params["action_scale"] + params["default_angles"]
    )
    return mirror_g1_joint_positions(target) if reflected else target


@pytest.mark.parametrize("old,new", [(False, False), (True, True), (False, True), (True, False)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_frame_change_preserves_physical_target_and_source(old, new, dtype):
    action = np.linspace(-0.5, 0.5, 29).astype(dtype)
    saved = action.copy()
    p = parameters()
    result = remap_previous_locomotion_action(
        action, **p, previous_reflected=old, next_reflected=new
    )
    np.testing.assert_allclose(physical(action, p, old), physical(result, p, new), atol=1e-7)
    np.testing.assert_array_equal(action, saved)
    assert not np.shares_memory(action, result)
    restored = remap_previous_locomotion_action(
        result, **p, previous_reflected=new, next_reflected=old
    )
    np.testing.assert_allclose(restored, action, atol=2e-7)
    if old == new:
        assert result.tobytes() == action.tobytes()


@pytest.mark.parametrize(
    "change",
    [
        dict(action_scale=0.0),
        dict(action_scale=True),
        dict(action_scale=float("nan")),
        dict(joint_to_motor=np.zeros(29, dtype=int)),
        dict(joint_to_motor=np.arange(29, dtype=float)),
        dict(default_angles=np.full(29, np.inf)),
        dict(previous_reflected=1),
        dict(next_reflected=None),
    ],
)
def test_invalid_frame_contract_rejected_before_mutation(change):
    action = np.zeros(29)
    args = dict(**parameters(), previous_reflected=False, next_reflected=True)
    args.update(change)
    with pytest.raises(ValueError):
        remap_previous_locomotion_action(action, **args)
    np.testing.assert_array_equal(action, 0)


def controller():
    p = parameters()
    a = np.linspace(-0.5, 0.5, 29).astype(np.float32)
    state = SimpleNamespace(
        q=np.zeros(29),
        dq=np.zeros(29),
        gravity_ori=np.array([0.0, 0.0, -1.0]),
        ang_vel=np.zeros(3),
        vel_cmd=np.array([0.0, -0.2, 0.0]),
    )
    output = SimpleNamespace(actions=np.zeros(29), kps=np.ones(29), kds=np.ones(29))
    policy = SimpleNamespace(
        action=a.copy(),
        default_angles=p["default_angles"],
        action_scale=p["action_scale"],
        joint2motor_idx=p["joint_to_motor"],
    )
    seen = []

    def run():
        seen.append(policy.action.copy())
        output.actions = physical(policy.action, p, False)

    policy.run = run
    c = SimpleNamespace(
        policy=policy,
        state=state,
        output=output,
        locomotion_reflection_frame=False,
        locomotion_frame_switch_count=0,
    )
    return c, seen, a, p


def test_adapter_converts_once_per_switch_and_keeps_measurements():
    c, seen, a, p = controller()
    q = c.state.q.copy()
    for mode in (True, True, False):
        _run_locomotion(c, mirror=mode, correct_mirrored_yaw=True, synchronize_action_frame=True)
        np.testing.assert_allclose(c.output.actions, physical(a, p, False), atol=1e-7)
        np.testing.assert_array_equal(c.state.q, q)
    assert c.locomotion_frame_switch_count == 2
    np.testing.assert_array_equal(seen[0], seen[1])
    np.testing.assert_allclose(seen[2], a, atol=2e-7)


def test_failed_inference_restores_previous_action_and_frame_marker():
    c, _, a, _ = controller()

    def fail():
        c.policy.action[:] = 5
        raise RuntimeError("injected policy failure")

    c.policy.run = fail
    with pytest.raises(RuntimeError):
        _run_locomotion(c, mirror=True, synchronize_action_frame=True)
    np.testing.assert_array_equal(c.policy.action, a)
    np.testing.assert_array_equal(c.state.q, np.zeros(29))
    assert c.locomotion_reflection_frame is False and c.locomotion_frame_switch_count == 0


def test_first_inference_has_no_fabricated_previous_frame():
    c, seen, a, _ = controller()
    c.locomotion_reflection_frame = None
    _run_locomotion(c, mirror=True, synchronize_action_frame=True)
    np.testing.assert_array_equal(seen[0], a)
    assert c.locomotion_reflection_frame is True and c.locomotion_frame_switch_count == 0


def test_disabled_adapter_preserves_legacy_and_default_config_hash():
    c, seen, a, _ = controller()
    _run_locomotion(c, mirror=True)
    np.testing.assert_array_equal(seen[0], a)
    old = IndependentTeamWorldConfig()
    assert (
        old.config_hash == "sha256:fd442bce83f737c64e2ee59ecc09dc204376100f9476a21361b41b538d8c41d1"
    )
    assert replace(old, locomotion_action_frame_sync=True).config_hash != old.config_hash
    with pytest.raises(ValueError):
        replace(old, locomotion_action_frame_sync=1)
