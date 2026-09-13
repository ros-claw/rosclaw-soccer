import pytest

from rosclaw_soccer.training.ball_carry_credit import begin_ball_carry
from rosclaw_soccer.training.carry_guidance_teacher import (
    CarryGuidanceTeacherConfig,
    carry_guidance_teacher,
)

torch = pytest.importorskip("torch")


def inputs():
    b = torch.tensor([[0.4, 0.0, 0.115], [0.4, 0.0, 0.115]])
    r = torch.tensor([[0.0, 0.0, 0.75], [0.0, 0.0, 0.75]])
    d = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    h = begin_ball_carry(ball_position=b, root_position=r, direction=d)
    return h, dict(
        ball_position=b,
        root_position=r,
        root_quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        direction=d.double(),
        selected_foot=torch.tensor([0, 1]),
    )


def test_mirrored_foot_alignment_and_bounds_without_mutation():
    h, kw = inputs()
    before = {k: v.clone() for k, v in kw.items()}
    out = carry_guidance_teacher(h, **kw)
    torch.testing.assert_close(
        out,
        torch.tensor([[0.4, -0.2, 0.0], [0.4, 0.2, 0.0]], dtype=torch.float64),
        atol=1e-7,
        rtol=0,
    )
    assert torch.all(torch.linalg.vector_norm(out[:, :2], dim=1) <= 0.7)
    for key, value in kw.items():
        assert torch.equal(value, before[key])


@pytest.mark.parametrize("fault", ["nan", "direction", "foot", "dtype", "shape", "grad", "config"])
def test_invalid_inputs_rejected(fault):
    h, kw = inputs()
    if fault == "nan":
        kw["direction"][0, 0] = float("nan")
    elif fault == "direction":
        kw["direction"] *= -1
    elif fault == "foot":
        kw["selected_foot"][0] = 2
    elif fault == "dtype":
        kw["selected_foot"] = kw["selected_foot"].float()
    elif fault == "shape":
        kw["direction"] = kw["direction"][:1]
    elif fault == "grad":
        kw["direction"].requires_grad_(True)
    else:
        kw["config"] = None
    with pytest.raises(ValueError):
        carry_guidance_teacher(h, **kw)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 0.5, True])
def test_invalid_config(value):
    with pytest.raises(ValueError):
        CarryGuidanceTeacherConfig(standoff_m=value)
