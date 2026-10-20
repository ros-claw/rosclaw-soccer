import numpy as np
import pytest


def model():
    import torch

    from rosclaw_soccer.rsi.kinematic_motion_residual import create_kinematic_motion_residual
    from rosclaw_soccer.rsi.motion_actor_critic import create_motion_actor_critic

    class Geometry:
        def from_student_features(self, x):
            # Test fake encodes current values only, no physics qualification.
            return x[12:38].copy()

    torch.set_num_threads(1)
    torch.manual_seed(4)
    parent = create_motion_actor_critic().eval().requires_grad_(False)
    return parent, create_kinematic_motion_residual(parent, Geometry())


def test_zero_geometry_column_preserves_frozen_parent_and_owned_memory():
    import torch

    parent, child = model()
    child.eval()
    h1 = h2 = None
    generator = torch.Generator().manual_seed(42)
    for _ in range(12):
        x = torch.randn(1, 1, 108)
        a = parent.sample(x, h1, generator=generator, stochastic=False)
        b = child.sample(x, h2, generator=generator, stochastic=False)
        for key in a:
            assert torch.equal(a[key], b[key]), key
        h1, h2 = a["hidden"], b["hidden"]
    with pytest.raises(ValueError):
        child.network_call_counts()


def test_new_column_learns_without_changing_any_parent_parameter():
    import torch

    parent, child = model()
    before = {k: v.clone() for k, v in parent.state_dict().items()}
    child.train()
    assert not parent.training
    x = torch.randn(2, 9, 108)
    geo = child.geometry(x)
    actual = child.parameters_for_state(x, geometry=geo)
    cached = child.parameters_for_state(x)
    for a, b in zip(actual, cached, strict=True):
        assert torch.equal(a, b)
    optimizer = torch.optim.Adam([p for p in child.parameters() if p.requires_grad], lr=1e-3)
    mean, _, value, _ = actual
    loss = (mean - 0.1).square().mean() + (value - 0.3).square().mean()
    loss.backward()
    assert child.adapter[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in parent.parameters())
    optimizer.step()
    for k, v in parent.state_dict().items():
        assert torch.equal(v, before[k])
    mean, _, _, _ = child.parameters_for_state(x)
    original = parent.parameters_for_state(x)[0]
    assert (mean - original).abs().max() <= 0.125
    assert not torch.equal(mean, original)


def test_bad_geometry_and_frozen_normalizers_fail_closed():
    import torch

    _, child = model()
    x = torch.zeros(1, 1, 108)
    with pytest.raises(ValueError):
        child.parameters_for_state(x, geometry=torch.full((1, 1, 26), float("nan")))
    child.geometry_std[0] = 0
    with pytest.raises(ValueError):
        child.parameters_for_state(x)


def test_current108_reconstruction_matches_original_measured_geometry():
    from rosclaw_soccer.rsi.neural_motor_student import motor_student_features
    from tests.rsi.test_measured_foot_kinematics import setup

    _, donor, extractor = setup()
    q, v = donor.qpos.copy(), donor.qvel.copy()
    q[:2] = [3, -2]
    v[:6] = [0.2, -0.3, 0.1, 0.2, 0.1, -0.1]
    v[35:38] = [0.3, -0.1, 0.1]
    feature = motor_student_features(q, v, np.zeros(29, dtype=np.float32), frame=20)
    np.testing.assert_allclose(
        extractor.features(q, v), extractor.from_student_features(feature), atol=1e-6
    )
