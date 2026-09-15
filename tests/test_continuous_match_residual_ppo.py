from pathlib import Path

import pytest

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.training.continuous_match_residual_ppo import train


def test_balanced_motor_kickoffs_are_half_turn_pairs_not_hidden_resets():
    from rosclaw_soccer.training.continuous_match_residual_ppo import training_kickoffs

    positions = training_kickoffs(varied=False, balanced_motor=True)
    assert positions == ((3.2, -0.3), (2.8, 0.3))
    assert positions[1] == (6.0 - positions[0][0], -positions[0][1])
    with pytest.raises(ValueError, match="nonoverlapping"):
        training_kickoffs(varied=True, balanced_motor=True)


def test_collection_options_bind_task_context_and_motor_entry_geometry():
    from rosclaw_soccer.training.continuous_match_residual_ppo import (
        collection_options,
        collection_world,
    )

    world = collection_world(None, False, True)
    original = collection_options(world)
    bound = collection_options(world, prospective=True, bound_context=True, lateral_limit_m=0.3)
    assert bound.bilateral_enabled and bound.prospective_enabled and bound.task_context_bound
    assert bound.maximum_strike_lateral_error_m == 0.3
    assert bound.config_hash != original.config_hash


def test_balanced_motor_training_cannot_silently_use_contact_only_admission(tmp_path):
    with pytest.raises(ValueError, match="prospective finisher learning"):
        train(
            assets=tmp_path,
            output=tmp_path / "out",
            checkpoint=tmp_path / "unused",
            scope=("red.finisher",),
            balanced_motor_kickoffs=True,
            finisher_option_learning=True,
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("commitment", [None, "sha256:" + "0" * 64])
def test_new_motor_curriculum_requires_correct_option_commitment(tmp_path, monkeypatch, commitment):
    import json
    from types import SimpleNamespace

    from rosclaw_soccer.sim.contracts import hash_json
    from rosclaw_soccer.training.continuous_match_learning_audit import audit

    ids = tuple(
        f"{team}.{role}"
        for team in ("blue", "red")
        for role in ("defender", "finisher", "goalkeeper", "playmaker")
    )
    monkeypatch.setattr(
        NearBallResidualPolicy,
        "load",
        lambda _: SimpleNamespace(
            agent_ids=ids,
            policy_hash="sha256:" + "1" * 64,
        ),
    )
    manifest = {
        "schema": "rosclaw_soccer.continuous_match_residual_ppo.v1",
        "activation_ceiling": "SIM_ONLY",
        "promotion_eligible": False,
        "iterations": [{"generation": 5}],
        "trainable_agent_ids": ["red.finisher"],
        "exploration_agent_ids": ["red.finisher"],
        "initial_policy_hash": "sha256:" + "1" * 64,
        "optimizer_epochs": 8,
        "gamma": 0.997,
        "trace_decay": 0.997,
        "reward_shaping": "motor_task_contact_v1",
        "training_ball_y_m": [0.0],
        "prospective_motor": True,
    }
    if commitment is not None:
        manifest["collection_option_config_hash"] = commitment
    manifest["manifest_hash"] = hash_json(manifest)
    (tmp_path / "training.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="option"):
        audit(tmp_path)


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
