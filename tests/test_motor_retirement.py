from dataclasses import replace

import pytest

from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldConfig
from rosclaw_soccer.skills.team.motor_retirement import (
    TeamMotorRetirement,
    validate_motor_retirement,
)


def test_only_same_frame_idle_owner_can_relinquish_control():
    request = TeamMotorRetirement("blue.finisher", 100, 2.0, "sha256:" + "a" * 64)
    kw = dict(
        agent_id=request.agent_id,
        frame=100,
        time_sec=2.0,
        contract_hash=request.contract_hash,
        proposed_target=False,
    )
    validate_motor_retirement(request, **kw)
    for change in (
        {"proposed_target": True},
        {"frame": 101},
        {"frame": 100.0},
        {"time_sec": 2.02},
        {"agent_id": "red.finisher"},
        {"contract_hash": "sha256:" + "b" * 64},
    ):
        with pytest.raises(ValueError):
            validate_motor_retirement(request, **{**kw, **change})
    object.__setattr__(request, "activation_ceiling", "REAL")
    with pytest.raises(ValueError):
        validate_motor_retirement(request, **kw)


def test_retirement_is_explicit_and_default_world_identity_is_unchanged():
    config = IndependentTeamWorldConfig()
    assert (
        config.config_hash
        == "sha256:fd442bce83f737c64e2ee59ecc09dc204376100f9476a21361b41b538d8c41d1"
    )
    with pytest.raises(ValueError):
        replace(config, retire_completed_motors=True)
    with pytest.raises(ValueError):
        replace(config, retire_completed_motors=1, disjoint_motor_backends=True)
    assert (
        replace(config, retire_completed_motors=True, disjoint_motor_backends=True).config_hash
        != config.config_hash
    )
