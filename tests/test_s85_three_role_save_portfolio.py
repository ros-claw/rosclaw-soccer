from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rosclaw_soccer.skills.team.shared_world import G1GoalkeeperConfig
from rosclaw_soccer.training.three_role_save_portfolio import (
    ThreeRoleSaveLane,
    ThreeRoleSavePortfolioConfig,
    three_role_save_lane_kwargs,
)


def test_save_portfolio_is_sim_only_and_requires_diverse_lanes() -> None:
    config = ThreeRoleSavePortfolioConfig()
    assert config.activation_ceiling == "SIM_ONLY"
    assert config.hardware_authorized is False
    assert len(config.lanes) == 4
    assert config.minimum_contact_span_m == pytest.approx(0.75)
    with pytest.raises(ValueError, match="at least three"):
        replace(config, lanes=config.lanes[:2])
    with pytest.raises(ValueError, match="unique"):
        replace(config, lanes=(config.lanes[0], config.lanes[0], config.lanes[1]))
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, hardware_authorized=True)


def test_save_lane_rejects_unqualified_goalkeeper_pocket() -> None:
    with pytest.raises(ValueError, match="goalkeeper pocket"):
        ThreeRoleSaveLane("bad-lane", "BAD", -0.8, 0.0)
    with pytest.raises(ValueError, match="offsets"):
        ThreeRoleSaveLane("bad-lane", "BAD", -1.3, -1.2)
    with pytest.raises(ValueError, match="initial lateral"):
        G1GoalkeeperConfig(initial_lateral_position_m=1.51)


def test_save_lane_translation_preserves_local_strike_pocket(tmp_path: Path) -> None:
    artifacts = tuple(tmp_path / name for name in ("striker", "goalkeeper", "gmt", "skill"))
    for path in artifacts:
        path.write_text("bound", encoding="utf-8")
    lane = ThreeRoleSaveLane("center-channel", "CENTER", -0.45, -0.35)
    kwargs = three_role_save_lane_kwargs(
        lane=lane,
        striker_actor_path=artifacts[0],
        goalkeeper_actor_path=artifacts[1],
        gmt_model_path=artifacts[2],
        gmt_skill_path=artifacts[3],
    )
    assert kwargs["shooter_origin"] == pytest.approx((0.0, -0.45, 0.0))
    assert kwargs["pass_reception_target_m"] == pytest.approx((1.275, -0.47, 0.115))
    assert kwargs["passer_origin"] == pytest.approx((5.10, -0.614060065039216, 0.0))
    assert kwargs["shooter_target"] == pytest.approx((7.50, 0.44, 0.115))
    assert kwargs["goal_spec"].target_y_m == pytest.approx(0.44)
    assert kwargs["goalkeeper_config"].initial_lateral_position_m == pytest.approx(-0.35)
    # The goal and receiving pocket moved together with the front G1; the
    # frozen policy still sees the exact same local target.
    assert kwargs["shooter_policy_target"] == pytest.approx((7.50, 0.70, 0.50))
