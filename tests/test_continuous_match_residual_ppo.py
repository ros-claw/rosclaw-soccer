from pathlib import Path

import pytest

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.training.continuous_match_residual_ppo import train


def test_outlet_stance_scope_excludes_finishers_and_handoff_is_bound():
    from rosclaw_soccer.training.continuous_match_residual_ppo import collection_world

    old = collection_world(None, False)
    new = collection_world(0.24, True)
    assert old.owned_contact_policy is None
    assert new.owned_contact_roles == ("defender", "goalkeeper", "playmaker")
    assert new.owned_contact_policy.depth_m == 0.24
    assert new.strict_receive_handoff
    assert new.config_hash != old.config_hash
    for depth in (-0.2, 0.0, 0.5, float("nan")):
        with pytest.raises(ValueError):
            collection_world(depth, True)


def test_finisher_option_learning_is_a_distinct_explicit_control_contract():
    from rosclaw_soccer.training.continuous_match_residual_ppo import collection_world

    old = collection_world(None, False)
    new = collection_world(None, False, True)
    assert new.option_only_residual_roles == ("finisher",)
    assert new.config_hash != old.config_hash
    assert old.option_only_residual_roles is None


@pytest.mark.parametrize("iterations,workers", [(0, 1), (21, 1), (True, 1), (1, 0), (1, 5)])
def test_budget_rejected_without_artifact_creation(tmp_path, iterations, workers):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="bounded"):
        train(
            assets=Path("unused"),
            output=output,
            checkpoint=Path("unused"),
            scope=("red.defender",),
            iterations=iterations,
            workers=workers,
        )
    assert not output.exists()


@pytest.mark.parametrize("scope", [(), ("red.unknown",), ("red.defender", "red.defender")])
def test_scope_is_checked_before_collection(tmp_path, monkeypatch, scope):
    from types import SimpleNamespace

    monkeypatch.setattr(
        NearBallResidualPolicy,
        "load",
        lambda _: SimpleNamespace(agent_ids=("blue.defender", "red.defender")),
    )
    output = tmp_path / "run"
    with pytest.raises(ValueError):
        train(assets=Path("unused"), output=output, checkpoint=Path("unused"), scope=scope)
    assert not output.exists()


@pytest.mark.parametrize("explore", [True, False])
def test_collection_constructs_valid_same_clock_match_and_routes_scope(
    tmp_path, monkeypatch, explore
):
    from dataclasses import dataclass
    from types import SimpleNamespace

    from rosclaw_soccer.training import continuous_match_residual_ppo as module

    @dataclass
    class Fixture:
        cells: tuple = ()
        goal: object = None

    fixture = Fixture(goal=SimpleNamespace(ball_radius_m=0.115))
    monkeypatch.setattr(module, "build_four_vs_four_fixture", lambda *a, **k: fixture)
    monkeypatch.setattr(NearBallResidualPolicy, "load", lambda _: "checkpoint")
    captured = {}

    def run(**kwargs):
        captured.update(kwargs)
        return {"exact_replay": True}

    monkeypatch.setattr(module, "run_continuous_competitive_match_growth", run)
    scope = ("red.defender",)
    module.collect(
        module.MatchCollection(
            tmp_path, tmp_path / "out", tmp_path / "weights", 123, scope, explore
        )
    )
    assert captured["world_config"].simulation_duration_sec == 25
    assert captured["scenario"].ball_initial_position_m == (3, 0, 0.115)
    assert captured["near_ball_explore"] is explore
    assert captured["near_ball_exploration_agent_ids"] == (scope if explore else None)
    assert captured["near_ball_seed"] == 123
