import sys
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.ball_motor import g1_ball_motor_observation
from rosclaw_soccer.providers.g1.event_protected_motor import G1FrozenEventProtectedMotor
from rosclaw_soccer.providers.g1.goal_reference_motor import G1FrozenGoalReferenceMotor
from rosclaw_soccer.sim.contracts import hash_bytes


@pytest.fixture
def checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from rosclaw_soccer.training.ball_residual import build_ball_residual_actor_critic
    from rosclaw_soccer.training.event_protected_actor import build_event_protected_actor_critic
    from rosclaw_soccer.training.goal_reference_actor import build_goal_reference_actor_critic

    torch.set_num_threads(1)
    torch.manual_seed(91101)
    seed = build_goal_reference_actor_critic(build_ball_residual_actor_critic().state_dict())
    with torch.no_grad():
        seed.actor[-1].weight.normal_(0, 0.03)
        seed.reference_actor[-1].weight.normal_(0, 0.03)
    model = build_event_protected_actor_critic(seed, observation_size=136, action_size=30)
    with torch.no_grad():
        model.plastic.actor[-1].bias.add_(0.2)
        model.plastic.reference_actor[-1].bias.add_(0.1)
    weights = tmp_path / "protected.npz"
    original = tmp_path / "anchor.npz"
    for path, actor in ((weights, model), (original, seed)):
        np.savez_compressed(path, **{k: v.detach().numpy() for k, v in actor.state_dict().items()})
    kwargs = dict(
        expected_actor_hash=hash_bytes(weights.read_bytes()),
        foundation_hash="sha256:" + "a" * 64,
        reference_library_hash="sha256:" + "b" * 64,
        course_transform_hash="sha256:" + "c" * 64,
        default_angles=np.zeros(29),
        reference_center_rad=-0.337,
    )
    return weights, original, kwargs, model


def state():
    q, v = np.zeros(43), np.zeros(41)
    q[2], q[3], q[36], q[37], q[38], q[39] = 0.8, 1, 0.3, -0.2, 0.115, 1
    return q, v


def propose(motor, frame=0, flag=False):
    q, v = state()
    return motor.propose(
        frame=frame,
        course_qpos=q,
        course_qvel=v,
        teacher_target=np.zeros(29),
        successor_enabled=flag,
    )


def test_full_200_frame_torch_bridge_shares_actual_history_across_event(checkpoint):
    import torch

    from rosclaw_soccer.training.ball_residual import advance_ball_residual
    from rosclaw_soccer.training.goal_reference_actor import (
        ReferenceHeadingEnvelope,
        advance_reference_heading,
    )

    weights, _, kwargs, actor = checkpoint
    motor = G1FrozenEventProtectedMotor(weights, **kwargs)
    goal = np.array([3.0, 1.0])
    motor.begin_episode(target_position_xy=goal)
    previous, heading = torch.zeros(1, 29), torch.full((1,), -0.337)
    rng = np.random.default_rng(91101)
    for frame in range(200):
        q, v = state()
        q[7:36] = rng.uniform(-0.1, 0.1, 29)
        base = rng.uniform(-0.05, 0.05, 29)
        body = g1_ball_motor_observation(
            course_qpos=q,
            course_qvel=v,
            default_angles=np.zeros(29),
            teacher_target=base,
            previous_residual=previous[0].numpy(),
            frame=frame,
        )
        ray = goal - q[36:38]
        ray /= np.linalg.norm(ray)
        flag = frame >= 40  # Synthetic interface test, NOT contact evidence.
        obs = torch.from_numpy(
            np.concatenate((body, ray, heading.numpy(), [flag])).astype(np.float32)
        )[None]
        with torch.no_grad():
            raw, _ = actor(obs)
            residual = advance_ball_residual(raw[:, :29], previous)
            following = advance_reference_heading(
                raw[:, 29],
                heading,
                ReferenceHeadingEnvelope(center_rad=-0.337, maximum_offset_rad=0.1),
            )
        result = motor.propose(
            frame=frame, course_qpos=q, course_qvel=v, teacher_target=base, successor_enabled=flag
        )
        np.testing.assert_allclose(result.target_rad, base + residual[0].numpy(), atol=1e-7, rtol=0)
        assert result.next_reference_heading_rad == pytest.approx(float(following[0]), abs=1e-7)
        assert result.successor_enabled is flag and result.activation_ceiling == "SIM_ONLY"
        previous, heading = residual, following


def test_exact_numeric_anchor_prefix_and_independent_history(checkpoint):
    weights, original, kwargs, _ = checkpoint
    protected = G1FrozenEventProtectedMotor(weights, **kwargs)
    anchor = G1FrozenGoalReferenceMotor(
        original, **(kwargs | {"expected_actor_hash": hash_bytes(original.read_bytes())})
    )
    for motor in (protected, anchor):
        motor.begin_episode(target_position_xy=np.array([3.0, 1.0]))
    q, v = state()
    for i in range(30):
        a = propose(protected, i)
        b = anchor.propose(frame=i, course_qpos=q, course_qvel=v, teacher_target=np.zeros(29))
        assert (
            a.target_rad == b.target_rad
            and a.next_reference_heading_rad == b.next_reference_heading_rad
        )
    assert not np.shares_memory(protected._previous, anchor._previous)


@pytest.mark.parametrize("flag", [None, 0, 1, 0.5, "true", True])
def test_invalid_or_premature_event_latches_until_reset(checkpoint, flag):
    weights, _, kwargs, _ = checkpoint
    motor = G1FrozenEventProtectedMotor(weights, **kwargs)
    motor.begin_episode(target_position_xy=np.zeros(2))
    with pytest.raises(ValueError):
        propose(motor, flag=flag)
    with pytest.raises(ValueError):
        propose(motor)
    motor.begin_episode(target_position_xy=np.zeros(2))
    propose(motor)


def test_event_reversal_rejected_and_reset_clears_latch(checkpoint):
    weights, _, kwargs, _ = checkpoint
    motor = G1FrozenEventProtectedMotor(weights, **kwargs)
    motor.begin_episode(target_position_xy=np.zeros(2))
    first = propose(motor)
    propose(motor, 1, True)
    with pytest.raises(ValueError):
        propose(motor, 2, False)
    with pytest.raises(ValueError):
        propose(motor, 2, True)
    motor.begin_episode(target_position_xy=np.zeros(2))
    assert propose(motor) == first


def test_no_torch_runtime_and_immutable_parameters(checkpoint, monkeypatch):
    weights, _, kwargs, _ = checkpoint
    monkeypatch.setitem(sys.modules, "torch", None)
    motor = G1FrozenEventProtectedMotor(weights, **kwargs)
    assert all(not p.flags.writeable for p in motor._successor_parameters.values())
    motor.begin_episode(target_position_xy=np.zeros(2))
    result = propose(motor)
    with pytest.raises(FrozenInstanceError):
        result.successor_enabled = True


def test_changed_exploration_or_extra_checkpoint_keys_rejected(checkpoint, tmp_path):
    weights, _, kwargs, _ = checkpoint
    with np.load(weights, allow_pickle=False) as z:
        values = {k: z[k].copy() for k in z.files}
    for change in ("exploration", "extra"):
        candidate = {k: v.copy() for k, v in values.items()}
        if change == "exploration":
            candidate["plastic.logstd"][0] += 0.1
        else:
            candidate["unexpected"] = np.zeros(1, dtype=np.float32)
        path = tmp_path / f"{change}.npz"
        np.savez_compressed(path, **candidate)
        with pytest.raises(ValueError):
            G1FrozenEventProtectedMotor(
                path, **(kwargs | {"expected_actor_hash": hash_bytes(path.read_bytes())})
            )
