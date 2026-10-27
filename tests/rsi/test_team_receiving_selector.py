import numpy as np
import pytest

from rosclaw_soccer.rsi.team_receiving_selector import TeamReceivingSelector


def model():
    return dict(
        schema="soccer.team_receiving_offline_actor.v1",
        hardware_authorized=False,
        promotion_authorized=False,
        center=[0.0] * 10,
        scale=[1.0] * 10,
        weight1=np.zeros((32, 10)).tolist(),
        bias1=[0.0] * 32,
        weight2=np.zeros((16, 32)).tolist(),
        bias2=list(np.arange(16, dtype=float)),
        offsets=np.zeros((16, 12)).tolist(),
    )


def test_selector_and_out_of_support_fallback():
    selector = TeamReceivingSelector.from_dict(model())
    assert selector.select((0.0,) * 10) == (15, False)
    assert selector.select((6.0,) + (0.0,) * 9) == (0, True)
    with pytest.raises(ValueError):
        selector.select((float("nan"),) + (0.0,) * 9)


def test_bad_model_refused():
    value = model()
    value["scale"][0] = 0
    with pytest.raises(ValueError):
        TeamReceivingSelector.from_dict(value)
