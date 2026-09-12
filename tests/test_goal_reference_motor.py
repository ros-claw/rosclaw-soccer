import sys
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.ball_motor import g1_ball_motor_observation
from rosclaw_soccer.providers.g1.goal_reference_motor import G1FrozenGoalReferenceMotor
from rosclaw_soccer.sim.contracts import hash_bytes


@pytest.fixture
def checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from rosclaw_soccer.training.ball_residual import build_ball_residual_actor_critic
    from rosclaw_soccer.training.goal_reference_actor import build_goal_reference_actor_critic

    torch.set_num_threads(1)
    torch.manual_seed(773)
    actor = build_goal_reference_actor_critic(build_ball_residual_actor_critic().state_dict())
    with torch.no_grad():
        for name, parameter in actor.named_parameters():
            if name.startswith(("goal_", "reference_")):
                parameter.normal_(0, 0.03)
    weights = tmp_path / "actor.npz"
    np.savez_compressed(weights, **{k: v.detach().numpy() for k, v in actor.state_dict().items()})
    kwargs = dict(
        expected_actor_hash=hash_bytes(weights.read_bytes()),
        foundation_hash="sha256:" + "a" * 64,
        reference_library_hash="sha256:" + "b" * 64,
        course_transform_hash="sha256:" + "c" * 64,
        default_angles=np.zeros(29),
    )
    return weights, kwargs, actor


def state():
    q, v = np.zeros(43), np.zeros(41)
    q[2], q[3], q[36], q[37], q[38], q[39] = 0.8, 1, 0.3, -0.2, 0.115, 1
    return q, v


def propose(motor, frame=0, **overrides):
    q, v = state()
    return motor.propose(
        **(dict(frame=frame, course_qpos=q, course_qvel=v, teacher_target=np.zeros(29)) | overrides)
    )


@pytest.mark.parametrize("center", [0.0, -0.36, 0.36, -0.6, 0.6])
def test_full_episode_matches_torch_actor_and_both_history_filters(checkpoint, center):
    import torch

    from rosclaw_soccer.training.ball_residual import advance_ball_residual
    from rosclaw_soccer.training.goal_reference_actor import (
        ReferenceHeadingEnvelope,
        advance_reference_heading,
    )

    weights, kwargs, actor = checkpoint
    motor = G1FrozenGoalReferenceMotor(weights, **kwargs, reference_center_rad=center)
    goal = np.array([3.0, 1.0])
    motor.begin_episode(target_position_xy=goal)
    goal[:] = 99  # Caller-owned target mutation must not alter this episode.
    goal = np.array([3.0, 1.0])
    previous, heading = torch.zeros(1, 29), torch.full((1,), center)
    rng = np.random.default_rng(773)
    for frame in range(200):
        q, v = state()
        q[7:36] = rng.uniform(-0.1, 0.1, 29)
        v[6:35] = rng.uniform(-0.3, 0.3, 29)
        base = rng.uniform(-0.05, 0.05, 29)
        observation = g1_ball_motor_observation(
            course_qpos=q,
            course_qvel=v,
            default_angles=np.zeros(29),
            teacher_target=base,
            previous_residual=previous[0].numpy(),
            frame=frame,
        )
        ray = goal - q[36:38]
        ray /= max(float(np.linalg.norm(ray)), 0.01)
        obs = torch.from_numpy(
            np.concatenate((observation, ray, heading.numpy())).astype(np.float32)
        )[None]
        with torch.no_grad():
            raw, _ = actor(obs)
            residual = advance_ball_residual(raw[:, :29], previous)
            following = advance_reference_heading(
                raw[:, 29],
                heading,
                ReferenceHeadingEnvelope(center_rad=center, maximum_offset_rad=0.1),
            )
        result = motor.propose(frame=frame, course_qpos=q, course_qvel=v, teacher_target=base)
        np.testing.assert_allclose(result.target_rad, base + residual[0].numpy(), atol=1e-7, rtol=0)
        assert result.reference_heading_used_rad == pytest.approx(float(heading[0]), abs=1e-7)
        assert result.next_reference_heading_rad == pytest.approx(float(following[0]), abs=1e-7)
        assert result.activation_ceiling == "SIM_ONLY"
        assert center - 0.1 - 1e-7 <= result.next_reference_heading_rad <= center + 0.1 + 1e-7
        previous, heading = residual, following
    with pytest.raises(ValueError):
        propose(motor, 200)


@pytest.mark.parametrize("center", [True, None, "-0.36", float("nan"), float("inf"), -0.61, 0.61])
def test_invalid_reference_center_rejected(checkpoint, center):
    weights, kwargs, _ = checkpoint
    with pytest.raises(ValueError, match="reference center"):
        G1FrozenGoalReferenceMotor(weights, **kwargs, reference_center_rad=center)


def test_reference_center_bound_to_contract_and_restored_on_reset(checkpoint):
    weights, kwargs, _ = checkpoint
    default = G1FrozenGoalReferenceMotor(weights, **kwargs)
    zero = G1FrozenGoalReferenceMotor(weights, **kwargs, reference_center_rad=0.0)
    shifted = G1FrozenGoalReferenceMotor(weights, **kwargs, reference_center_rad=-0.36)
    assert default.contract_hash == zero.contract_hash != shifted.contract_hash
    for motor in (default, shifted):
        motor.begin_episode(target_position_xy=np.array([3.0, 1.0]))
    first = propose(shifted)
    assert first.episode_hash != propose(default).episode_hash
    assert first.reference_heading_used_rad == pytest.approx(-0.36)
    propose(shifted, 1)
    shifted.begin_episode(target_position_xy=np.array([3.0, 1.0]))
    assert propose(shifted) == first


def test_runtime_does_not_require_torch_and_returns_immutable_proposal(checkpoint, monkeypatch):
    weights, kwargs, _ = checkpoint
    monkeypatch.setitem(sys.modules, "torch", None)
    motor = G1FrozenGoalReferenceMotor(weights, **kwargs)
    motor.begin_episode(target_position_xy=np.array([3.0, 1.0]))
    result = propose(motor)
    with pytest.raises(FrozenInstanceError):
        result.frame = 2
    assert all(not p.flags.writeable for p in motor._parameters.values())


def test_private_history_and_target_bound_episode_hash(checkpoint):
    weights, kwargs, _ = checkpoint
    a, b = [G1FrozenGoalReferenceMotor(weights, **kwargs) for _ in range(2)]
    with pytest.raises(ValueError):
        propose(a)
    for motor in (a, b):
        motor.begin_episode(target_position_xy=np.array([3.0, 1.0]))
    first = propose(a)
    propose(a, 1)
    assert propose(b) == first
    assert not np.shares_memory(a._previous, b._previous)
    a.begin_episode(target_position_xy=np.array([6.0, 2.0]))
    other = propose(a)
    assert other.contract_hash == first.contract_hash
    assert other.episode_hash != first.episode_hash


@pytest.mark.parametrize("bad_frame", [True, -1, 1, 200])
def test_bad_frame_latches_until_explicit_reset(checkpoint, bad_frame):
    weights, kwargs, _ = checkpoint
    motor = G1FrozenGoalReferenceMotor(weights, **kwargs)
    motor.begin_episode(target_position_xy=np.zeros(2))
    with pytest.raises(ValueError):
        propose(motor, bad_frame)
    with pytest.raises(ValueError):
        propose(motor, 0)
    motor.begin_episode(target_position_xy=np.zeros(2))
    propose(motor, 0)


def test_duplicate_frame_and_nonfinite_observation_latch(checkpoint):
    weights, kwargs, _ = checkpoint
    motor = G1FrozenGoalReferenceMotor(weights, **kwargs)
    motor.begin_episode(target_position_xy=np.zeros(2))
    propose(motor)
    with pytest.raises(ValueError):
        propose(motor)
    with pytest.raises(ValueError):
        propose(motor, 1)
    motor.begin_episode(target_position_xy=np.zeros(2))
    q, _ = state()
    q[7] = np.nan
    with pytest.raises(ValueError):
        propose(motor, course_qpos=q)
    with pytest.raises(ValueError):
        propose(motor)


@pytest.mark.parametrize(
    "target",
    [np.array([np.nan, 0.0]), np.array([1001.0, 0.0]), np.zeros(3), np.array([True, False])],
)
def test_invalid_target_reset_disables_previous_episode(checkpoint, target):
    weights, kwargs, _ = checkpoint
    motor = G1FrozenGoalReferenceMotor(weights, **kwargs)
    motor.begin_episode(target_position_xy=np.zeros(2))
    with pytest.raises(ValueError):
        motor.begin_episode(target_position_xy=target)
    with pytest.raises(ValueError):
        propose(motor)


@pytest.mark.parametrize(
    "key",
    ["expected_actor_hash", "foundation_hash", "reference_library_hash", "course_transform_hash"],
)
def test_missing_content_binding_rejected(checkpoint, key):
    weights, kwargs, _ = checkpoint
    with pytest.raises(ValueError):
        G1FrozenGoalReferenceMotor(weights, **(kwargs | {key: "unbound"}))


def test_wrong_checkpoint_and_numeric_overflow_fail_closed(checkpoint, tmp_path):
    weights, kwargs, _ = checkpoint
    with pytest.raises(ValueError):
        G1FrozenGoalReferenceMotor(
            weights, **(kwargs | {"expected_actor_hash": "sha256:" + "0" * 64})
        )
    with np.load(weights) as z:
        data = {k: z[k] for k in z.files}
    data["actor.0.weight"] = np.full_like(data["actor.0.weight"], np.finfo(np.float32).max)
    bad = tmp_path / "overflow.npz"
    np.savez_compressed(bad, **data)
    motor = G1FrozenGoalReferenceMotor(
        bad, **(kwargs | {"expected_actor_hash": hash_bytes(bad.read_bytes())})
    )
    motor.begin_episode(target_position_xy=np.zeros(2))
    with pytest.raises((ValueError, FloatingPointError)):
        propose(motor, teacher_target=np.full(29, 10.0))
    with pytest.raises(ValueError):
        propose(motor)
