"""The zone selector uses only first measured ball/body state and then freezes."""

from types import SimpleNamespace

import pytest

from rosclaw_soccer.rsi.receiving_middle_basis_expert import ReceivingMiddleBasisExpert
from rosclaw_soccer.rsi.receiving_three_zone_expert import ReceivingThreeZoneExpert
from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def _actor(**kwargs: object) -> ReceivingThreeZoneExpert:
    schedule = ReceivingOracleSchedule(
        "red.finisher", "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    return ReceivingThreeZoneExpert(
        "red.finisher",
        schedule.contract_hash,
        ReceiveContactMailbox("red.finisher"),
        (0.0,) * 8,
        (0.0,) * 12,
        (0.0,) * 12,
        low_weights=(0.1,) + (0.0,) * 11,
        center_weights=(0.2,) + (0.0,) * 11,
        high_weights=(0.3,) + (0.0,) * 11,
        **kwargs,
    )


def _observation(lateral: float) -> SimpleNamespace:
    qpos = [0.0] * 43
    qpos[37] = lateral
    return SimpleNamespace(qpos=tuple(qpos))


@pytest.mark.parametrize(
    ("lateral", "zone", "value"),
    [(0.135, "low", 0.1), (0.140, "center", 0.2), (0.145, "high", 0.3)],
)
def test_selection_is_measured_and_immutable(
    monkeypatch: pytest.MonkeyPatch, lateral: float, zone: str, value: float
) -> None:
    monkeypatch.setattr(
        ReceivingMiddleBasisExpert,
        "propose",
        lambda self, observation: (self.middle_weights[0],) + (0.0,) * 28,
    )
    actor = _actor()
    first = actor.propose(_observation(lateral))
    later = actor.propose(_observation(0.2))
    assert actor.selected_zone == zone
    assert actor.activation_ceiling == "SIM_ONLY"
    assert first[0] == later[0] == value


def test_invalid_zone_configuration_fails_closed() -> None:
    with pytest.raises(ValueError, match="three-zone"):
        _actor(low_boundary_m=0.145, high_boundary_m=0.142)
    with pytest.raises(ValueError, match="three-zone"):
        _actor(low_boundary_m=0.137, high_boundary_m=float("nan"))


def test_nonfinite_measured_ball_state_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ReceivingMiddleBasisExpert,
        "propose",
        lambda self, observation: (0.0,) * 29,
    )
    with pytest.raises(ValueError, match="finite measured initial"):
        _actor().propose(_observation(float("nan")))
