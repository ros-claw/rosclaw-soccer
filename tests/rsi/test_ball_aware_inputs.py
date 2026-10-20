import numpy as np
import pytest

from rosclaw_soccer.rsi.ball_aware_inputs import ball_goal9


def fixture():
    return dict(
        pelvis_rotation=np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        pelvis_position=np.array([10.0, 20.0, 1.0]),
        pelvis_linear_velocity=np.array([1.0, 0.0, 0.0]),
        ball_position=np.array([11.0, 22.0, 0.11]),
        ball_linear_velocity=np.array([0.0, 2.0, 0.0]),
        target_position=np.array([13.0, 24.0, 2.0]),
    )


def test_measured_ball_and_relative_world_velocity_use_current_pelvis_basis():
    values = fixture()
    result = ball_goal9(**values)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [2.0, -1.0, -0.89, 2.0, 1.0, 0.0, 4.0, -3.0, 1.0], atol=1e-7)
    for name in ("pelvis_position", "ball_position", "target_position"):
        values[name] += np.array([40.0, -25.0, 0.0])
    np.testing.assert_allclose(ball_goal9(**values), result, atol=1e-7)
    values["ball_position"][1] += 1
    assert ball_goal9(**values)[0] == 3


@pytest.mark.parametrize("kind", ["nan", "improper", "shape", "dtype"])
def test_invalid_perception_rejected_without_repair(kind):
    values = fixture()
    if kind == "nan":
        values["ball_linear_velocity"][0] = np.nan
    elif kind == "improper":
        values["pelvis_rotation"][2, 2] = -1
    elif kind == "shape":
        values["target_position"] = np.zeros(2)
    else:
        values["ball_position"] = np.zeros(3, dtype=int)
    with pytest.raises(ValueError):
        ball_goal9(**values)
