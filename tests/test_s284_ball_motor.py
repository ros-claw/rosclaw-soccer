import numpy as np
import pytest

from rosclaw_soccer.providers.g1.ball_motor import (
    G1FrozenBallMotor,
    g1_ball_motor_observation,
)
from rosclaw_soccer.sim.contracts import hash_bytes
from rosclaw_soccer.training.ball_residual import BallResidualEnvelope


def state():
    q = np.zeros(43)
    q[2], q[3], q[36], q[38], q[39] = 0.75, 1, 0.6, 0.115, 1
    return q, np.zeros(41)


def artifact(tmp_path):
    values = {"logstd": np.full(29, -1, dtype=np.float32)}
    for head, output in (("actor", 29), ("critic", 1)):
        for layer, n_in, n_out in ((0, 133, 128), (2, 128, 128), (4, 128, output)):
            values[f"{head}.{layer}.weight"] = np.zeros((n_out, n_in), dtype=np.float32)
            values[f"{head}.{layer}.bias"] = np.zeros(n_out, dtype=np.float32)
    values["actor.4.bias"][:] = 0.5
    path = tmp_path / "actor.npz"
    np.savez_compressed(path, **values)
    return path


def load(path, **kwargs):
    options = dict(
        expected_actor_hash=hash_bytes(path.read_bytes()),
        foundation_hash="sha256:" + "a" * 64,
        reference_library_hash="sha256:" + "b" * 64,
        default_angles=np.zeros(29),
    )
    options.update(kwargs)
    return G1FrozenBallMotor(path, **options)


def propose(motor, frame=0):
    q, v = state()
    return motor.propose(frame=frame, course_qpos=q, course_qvel=v, teacher_target=np.zeros(29))


def test_motor_history_is_per_player_and_bounded(tmp_path):
    path = artifact(tmp_path)
    a, b = load(path), load(path)
    with pytest.raises(ValueError):
        propose(a)
    a.begin_episode()
    b.begin_episode()
    first = propose(a)
    second = propose(a, 1)
    assert first == propose(b)
    assert second.residual_rad[0] > first.residual_rad[0]
    assert first.activation_ceiling == "SIM_ONLY"
    previous = np.array(second.residual_rad)
    for i in range(2, 200):
        proposal = propose(a, i)
        residual = np.array(proposal.residual_rad)
        assert np.max(np.abs(residual)) <= 0.25
        assert np.max(np.abs(residual - previous)) <= 0.025001
        previous = residual
    with pytest.raises(ValueError):
        propose(a, 200)
    a.begin_episode()
    assert propose(a) == first


@pytest.mark.parametrize("bad_frame", [True, 2, -1, 0.0])
def test_invalid_clock_latches_until_explicit_episode(tmp_path, bad_frame):
    a = load(artifact(tmp_path))
    a.begin_episode()
    with pytest.raises(ValueError):
        propose(a, bad_frame)
    with pytest.raises(ValueError):
        propose(a)
    a.begin_episode()
    propose(a)
    with pytest.raises(ValueError):
        propose(a)


def test_observation_is_training_compatible_and_read_only():
    q, v = state()
    q[7], v[6], v[0], v[3], v[35] = 0.3, 2, 0.7, 0.5, 0.2
    saved_q, saved_v = q.copy(), v.copy()
    obs = g1_ball_motor_observation(
        course_qpos=q,
        course_qvel=v,
        default_angles=np.full(29, 0.1),
        teacher_target=np.full(29, 0.3),
        previous_residual=np.full(29, 0.02),
        frame=0,
    )
    assert obs.shape == (133,)
    assert obs.dtype == np.float32
    np.testing.assert_allclose(
        obs[[0, 29, 60, 61, 64, 67, 70, 73, 74, 75, 104]],
        [0.2, 0.2, -1, 0.7, 0.1, 0.6, 0.2, 0, 1, 0.02, 0.2],
    )
    np.testing.assert_array_equal(q, saved_q)
    np.testing.assert_array_equal(v, saved_v)


@pytest.mark.parametrize("bad", [np.full(29, np.nan), np.zeros(28), np.ones(29, dtype=bool)])
def test_bad_teacher_latches(tmp_path, bad):
    a = load(artifact(tmp_path))
    a.begin_episode()
    q, v = state()
    with pytest.raises(ValueError):
        a.propose(frame=0, course_qpos=q, course_qvel=v, teacher_target=bad)
    with pytest.raises(ValueError):
        propose(a)


def test_checkpoint_binding_and_envelope_are_not_silently_changed(tmp_path):
    path = artifact(tmp_path)
    with pytest.raises(ValueError):
        load(path, expected_actor_hash="sha256:" + "0" * 64)
    with pytest.raises(ValueError):
        load(path, envelope=BallResidualEnvelope(maximum_offset_rad=0.3))
    with pytest.raises(ValueError):
        load(path, foundation_hash="unbound")
    a = load(path)
    b = load(path, reference_library_hash="sha256:" + "c" * 64)
    assert a.contract_hash != b.contract_hash


def test_npz_object_or_missing_tensor_rejected(tmp_path):
    path = artifact(tmp_path)
    with np.load(path, allow_pickle=False) as z:
        weights = {key: z[key] for key in z.files}
    weights["actor.0.bias"] = np.array([object()] * 128, dtype=object)
    np.savez_compressed(path, **weights)
    with pytest.raises(ValueError):
        load(path)
    del weights["actor.0.bias"]
    np.savez_compressed(path, **weights)
    with pytest.raises(ValueError):
        load(path)

