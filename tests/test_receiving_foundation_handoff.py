from dataclasses import replace

import pytest

from rosclaw_soccer.training.receiving_foundation_handoff import ReceivingFoundationHandoff
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def fixture():
    schedule = ReceivingOracleSchedule("blue.playmaker", "A0_leg12", 30, 20, ((0.0,) * 12,))
    handoff = ReceivingFoundationHandoff(
        "blue.playmaker", 150, schedule.contract_hash, "sha256:" + "1" * 64
    )
    args = dict(
        motor_agent_id=handoff.agent_id,
        motor_contract_hash=handoff.motor_contract_hash,
        motor_entry_frame=150,
        idle_residual_fallback=True,
    )
    return schedule, handoff, args


def test_explicit_binding_and_irreversible_frame_boundary():
    schedule, handoff, args = fixture()
    handoff.validate_binding(schedule, **args)
    assert not handoff.predecessor_retired(149)
    assert handoff.predecessor_retired(150) and handoff.predecessor_retired(999)
    assert handoff.activation_ceiling == "SIM_ONLY"
    assert handoff.contract_hash != replace(handoff, entry_frame=151).contract_hash


@pytest.mark.parametrize(
    "key,value",
    [
        ("motor_agent_id", "red.playmaker"),
        ("motor_contract_hash", "sha256:" + "2" * 64),
        ("motor_entry_frame", 149),
        ("motor_entry_frame", True),
        ("idle_residual_fallback", False),
    ],
)
def test_foreign_or_mistimed_successor_rejected(key, value):
    schedule, handoff, args = fixture()
    args[key] = value
    with pytest.raises(ValueError):
        handoff.validate_binding(schedule, **args)


@pytest.mark.parametrize(
    "schedule",
    [
        ReceivingOracleSchedule("blue.playmaker", "A3_sonic_residual", 30, 20, ((0.0,) * 29,)),
        ReceivingOracleSchedule("blue.playmaker", "A0_leg12", 150, 20, ((0.0,) * 12,)),
        ReceivingOracleSchedule("red.playmaker", "A0_leg12", 30, 20, ((0.0,) * 12,)),
    ],
)
def test_wrong_predecessor_rejected(schedule):
    _, handoff, args = fixture()
    with pytest.raises(ValueError):
        handoff.validate_binding(schedule, **args)


@pytest.mark.parametrize("frame", [True, -1, 1000, float("nan"), 1.0])
def test_invalid_clock_rejected(frame):
    _, handoff, _ = fixture()
    with pytest.raises(ValueError):
        handoff.predecessor_retired(frame)


def test_no_unbound_contract_or_real_ceiling():
    _, handoff, _ = fixture()
    with pytest.raises(ValueError):
        replace(handoff, motor_contract_hash="unbound")
    with pytest.raises(TypeError):
        ReceivingFoundationHandoff(
            handoff.agent_id,
            150,
            handoff.schedule_hash,
            handoff.motor_contract_hash,
            activation_ceiling="REAL",
        )


@pytest.mark.parametrize("fault", ["missing_motor", "wrong_type", "bridge", "hash"])
def test_world_rejects_unbound_handoff_before_loading_assets(monkeypatch, fault):
    from pathlib import Path
    from types import SimpleNamespace

    from test_receiving_sonic import Navigation

    from rosclaw_soccer.providers.g1.receiving_sonic import ReceivingSonicOption
    from rosclaw_soccer.skills.team.independent_team_world import (
        IndependentTeamWorldConfig,
        simulate_independent_team_world,
    )

    monkeypatch.setattr("rosclaw_soccer.providers.g1.receiving_sonic.G1SonicNavigation", Navigation)
    schedule, handoff, _ = fixture()
    option = ReceivingSonicOption(None, handoff.agent_id, start_frame=150)
    handoff = replace(handoff, motor_contract_hash=option.contract_hash)
    motors = {handoff.agent_id: option}
    if fault == "missing_motor":
        motors = {}
    elif fault == "wrong_type":
        motors = {handoff.agent_id: object()}
    elif fault == "hash":
        handoff = replace(handoff, motor_contract_hash="sha256:" + "f" * 64)
    with pytest.raises(ValueError, match="handoff"):
        simulate_independent_team_world(
            asset_root=Path("must-not-be-opened"),
            roster=SimpleNamespace(agents=[SimpleNamespace(agent_id=handoff.agent_id)]),
            cells=(),
            players=(),
            scenario=None,
            goal=None,
            config=IndependentTeamWorldConfig(motor_idle_residual_fallback=True),
            motor_options=motors,
            option_bridge_config=object() if fault == "bridge" else None,
            receiving_oracle=schedule,
            receiving_foundation_handoff=handoff,
        )
