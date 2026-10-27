import numpy as np
import pytest

from rosclaw_soccer.rsi.team_coverage_actor import TeamCoverageActor


def model():
    return dict(
        schema="soccer.team_coverage_actor.v1",
        center=[0.0] * 10,
        scale=[1.0] * 10,
        weight1=np.zeros((32, 10)).tolist(),
        bias1=[0.0] * 32,
        weight2=np.zeros((17, 32)).tolist(),
        bias2=[0.0, 1.0] + [0.0] * 15,
        support=np.zeros((6, 10)).tolist(),
        allowed=[[False, True] + [False] * 14 for _ in range(6)],
        offsets=np.zeros((16, 12)).tolist(),
        radius=0.004,
        promotion_authorized=False,
        hardware_authorized=False,
    )


def test_actor_requires_support_and_known_safe_action():
    value = model()
    actor = TeamCoverageActor.from_dict(value)
    assert actor.select((0.0,) * 10) == (1, False)
    assert actor.select((1.0,) * 10) == (16, True)
    value["allowed"][0][1] = False
    assert TeamCoverageActor.from_dict(value).select((0.0,) * 10) == (16, True)


def test_actor_rejects_nonfinite_and_nonboolean_safety():
    value = model()
    value["allowed"][0][1] = 1
    with pytest.raises(ValueError):
        TeamCoverageActor.from_dict(value)
    with pytest.raises(ValueError):
        TeamCoverageActor.from_dict(model()).select((float("nan"),) * 10)
