from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw_soccer.growth.locomotion_contact_teacher import G1RollingOptionBridgeConfig
from rosclaw_soccer.skills.team.independent_team_world import (
    IndependentTeamWorldConfig,
    simulate_independent_team_world,
)


@pytest.mark.parametrize("optin,per_player", [(False, False), (False, True), (True, False)])
def test_mixed_backends_need_explicit_disjoint_per_player_contract(optin, per_player):
    with pytest.raises(ValueError, match="per-player motor"):
        simulate_independent_team_world(
            asset_root=Path("must-not-be-opened"),
            roster=SimpleNamespace(agents=[SimpleNamespace(agent_id="red.a")]),
            cells=(),
            players=(),
            scenario=None,
            goal=None,
            config=IndependentTeamWorldConfig(disjoint_motor_backends=optin),
            option_bridge_config=G1RollingOptionBridgeConfig(
                per_player_options_enabled=per_player, task_context_bound=True
            ),
            motor_options={"red.a": SimpleNamespace(contract_hash="sha256:" + "a" * 64)},
        )


def test_boolean_optin_and_disabled_hash_compatibility():
    base = IndependentTeamWorldConfig()
    with pytest.raises(ValueError):
        replace(base, disjoint_motor_backends=1)
    assert replace(base, disjoint_motor_backends=True).config_hash != base.config_hash
    assert asdict(base)["disjoint_motor_backends"] is False
    assert (
        base.config_hash
        == "sha256:fd442bce83f737c64e2ee59ecc09dc204376100f9476a21361b41b538d8c41d1"
    )
