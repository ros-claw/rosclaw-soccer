"""Compact measured-state residual stays bounded and retains anchor conditions."""

from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.receiving_lateral_piecewise_expert import ReceivingLateralPiecewiseExpert
from rosclaw_soccer.rsi.receiving_middle_basis_expert import ReceivingMiddleBasisExpert
from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def _actor(**kwargs: object) -> ReceivingMiddleBasisExpert:
    schedule = ReceivingOracleSchedule(
        "red.finisher", "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    return ReceivingMiddleBasisExpert(
        "red.finisher",
        schedule.contract_hash,
        ReceiveContactMailbox("red.finisher"),
        (0.0,) * 8,
        (0.0,) * 12,
        (0.0,) * 12,
        middle_weights=(1.0,) + (0.0,) * 11,
        **kwargs,
    )


@pytest.mark.parametrize(
    "weights",
    [
        (float("nan"),) + (0.0,) * 11,
        (1.01,) + (0.0,) * 11,
        (0.0,) * 11,
    ],
)
def test_middle_basis_rejects_invalid_weights(weights: tuple[float, ...]) -> None:
    schedule = ReceivingOracleSchedule(
        "red.finisher", "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    with pytest.raises(ValueError, match="bounded measured middle-course"):
        ReceivingMiddleBasisExpert(
            "red.finisher",
            schedule.contract_hash,
            ReceiveContactMailbox("red.finisher"),
            (0.0,) * 8,
            (0.0,) * 12,
            (0.0,) * 12,
            middle_weights=weights,
        )


@pytest.mark.parametrize(
    ("lateral", "side", "frame", "expected"),
    [
        (0.1395, 0, 20, 0.25),
        (0.1395, 0, 17, 0.125),
        (0.1395, 0, 42, 0.0),
        (0.13067944497433248, 0, 20, 0.0),
        (0.14861864755818321, 0, 20, 0.0),
        (0.1395, 1, 20, 0.0),
    ],
)
def test_middle_basis_compact_support_and_side_guard(
    monkeypatch: pytest.MonkeyPatch,
    lateral: float,
    side: int,
    frame: int,
    expected: float,
) -> None:
    monkeypatch.setattr(
        ReceivingLateralPiecewiseExpert,
        "propose",
        lambda self, observation: (0.0,) * 29,
    )
    actor = _actor()
    actor.measured_lateral_m = lateral
    actor._selected_side = side
    output = actor.propose(SimpleNamespace(frame=frame))
    assert actor.activation_ceiling == "SIM_ONLY"
    assert output[0] == pytest.approx(expected)
    assert np.array_equal(np.asarray(output[1:]), np.zeros(28))
