import numpy as np
import pytest

from rosclaw_soccer.rsi.neural_motor_student import (
    FEATURE_LAYOUT,
    FEATURE_WIDTH,
    create_motor_student,
    motor_student_features,
    recorded_student_sequence,
)


def state():
    q = np.zeros(43)
    q[2:4] = [0.793, 1]
    q[36:39] = [0.35, -0.1, 0.13]
    return q, np.zeros(41), np.zeros(29, dtype=np.float32)


def test_current_observation_layout_translation_and_quaternion_sign_invariance():
    q, v, p = state()
    saved = q.copy()
    f = motor_student_features(q, v, p, frame=20)
    assert f.shape == (FEATURE_WIDTH,) and f.dtype == np.float32
    assert [index for _, lo, hi in FEATURE_LAYOUT for index in range(lo, hi)] == list(range(108))
    np.testing.assert_array_equal(q, saved)
    q[:2] += [3, 7]
    q[36:38] += [3, 7]
    q[3:7] *= -1
    np.testing.assert_allclose(f, motor_student_features(q, v, p, frame=20), atol=1e-7)
    np.testing.assert_array_equal(f[79:], p)


def test_actor_features_do_not_contain_current_or_future_action_label():
    q, v, p = state()
    raw = dict(
        canonical_qpos=np.tile(q, (300, 1)),
        canonical_qvel=np.tile(v, (300, 1)),
        neural_action_isaac=np.zeros((300, 29), dtype=np.float32),
    )
    x, y = recorded_student_sequence(raw)
    raw["neural_action_isaac"][50] = 1
    changed, targets = recorded_student_sequence(raw)
    np.testing.assert_array_equal(x[:51], changed[:51])
    np.testing.assert_array_equal(changed[51, 79:], np.ones(29))
    assert targets[50, 0] == 1 and y[50, 0] == 0


@pytest.mark.parametrize("kind", ["nan", "shape", "frame", "quat"])
def test_invalid_proprioception_is_fatal(kind):
    q, v, p = state()
    frame = 0
    if kind == "nan":
        v[0] = np.nan
    elif kind == "shape":
        q = q[:-1]
    elif kind == "frame":
        frame = 300
    else:
        q[3] = 2
    with pytest.raises(ValueError):
        motor_student_features(q, v, p, frame=frame)


def test_real_recurrent29_motor_head_backprop_and_step_sequence_equivalence():
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(47)
    model = create_motor_student(hidden_size=32)
    x = torch.randn(2, 7, 108)
    action, value, hidden = model(x)
    assert action.shape == (2, 7, 29) and value.shape == (2, 7) and hidden.shape == (1, 2, 32)
    assert action.abs().max() <= 8
    state = None
    parts = []
    for i in range(7):
        current, _, state = model(x[:, i : i + 1], state)
        parts.append(current)
    torch.testing.assert_close(torch.cat(parts, dim=1), action, atol=1e-6, rtol=1e-6)
    loss = (action - 0.1).square().mean() + value.square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(float(p.grad.abs().sum()) for p in model.actor.parameters()) > 0


def test_architecture_cannot_be_unbounded():
    with pytest.raises(ValueError):
        create_motor_student(action_limit=1000)


@pytest.mark.parametrize("bad", ["hidden", "normalizer", "weights"])
def test_nonfinite_network_state_does_not_return_a_motor_action(bad):
    import torch

    model = create_motor_student(hidden_size=32)
    hidden = None
    with torch.no_grad():
        if bad == "hidden":
            hidden = torch.full((1, 1, 32), float("nan"))
        elif bad == "normalizer":
            model.observation_std[0] = 0
        else:
            model.actor.weight[0, 0] = float("nan")
    with pytest.raises(ValueError):
        model(torch.zeros(1, 1, 108), hidden)
