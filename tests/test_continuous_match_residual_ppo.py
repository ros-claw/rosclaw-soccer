from pathlib import Path

import pytest

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.training.continuous_match_residual_ppo import train


def test_collection_motor_concurrency_and_rearm_are_explicit_and_hash_bound():
    from rosclaw_soccer.training.continuous_match_residual_ppo import (
        collection_options,
        collection_world,
    )

    world = collection_world(None, False)
    old = collection_options(world, bound_context=True)
    new = collection_options(
        world, bound_context=True, per_player_options=True, continuous_motor_rearm=True
    )
    assert new.per_player_options_enabled and new.continuous_rearm_enabled
    assert old.config_hash != new.config_hash
    assert collection_options(world, bound_context=True, per_player_options=False) == old
    with pytest.raises(ValueError):
        collection_options(world, per_player_options=True)
    for bad in (1, "yes"):
        with pytest.raises(ValueError):
            collection_options(world, bound_context=True, per_player_options=bad)


def test_buildup_curriculum_covers_both_keepers_and_defenders_with_disjoint_exams():
    from rosclaw_soccer.training.continuous_match_residual_ppo import (
        evaluation_kickoffs,
        training_kickoffs,
    )

    points = training_kickoffs(varied=False, balanced_motor=False, buildup=True)
    assert len(points) == 4
    for x, y in points:
        assert any(abs(xx - (6 - x)) < 1e-9 and abs(yy + y) < 1e-9 for xx, yy in points)
    exams = evaluation_kickoffs(buildup=True)
    assert len(exams) == 9 and len({name for name, _, _ in exams}) == 9
    assert not set(points).intersection((x, y) for _, x, y in exams)
    assert evaluation_kickoffs() == (
        ("-0.06", 3.0, -0.06),
        ("+0.00", 3.0, 0.0),
        ("+0.06", 3.0, 0.06),
    )
    for kwargs in (
        dict(varied=True, balanced_motor=False, buildup=True),
        dict(varied=False, balanced_motor=True, buildup=True),
        dict(varied=False, balanced_motor=False, buildup=1),
    ):
        with pytest.raises(ValueError, match="nonoverlapping"):
            training_kickoffs(**kwargs)


def test_collection_fixture_binds_keeper_preview_and_default_preserves_cells(monkeypatch):
    from dataclasses import replace

    import test_s220_basic_ball_play as basic

    from rosclaw_soccer.training import continuous_match_residual_ppo as trainer

    fixture = basic.course.__wrapped__(monkeypatch)
    monkeypatch.setattr(trainer, "build_four_vs_four_fixture", lambda *a, **k: fixture)
    old = trainer.collection_fixture(Path("unused"))
    expected = replace(
        fixture,
        cells=tuple(
            replace(c, tactical_profile=replace(c.tactical_profile, anticipatory_contact=True))
            for c in fixture.cells
        ),
    )
    assert old.fixture_hash == expected.fixture_hash
    new = trainer.collection_fixture(Path("unused"), keeper_preview=True)
    assert new.fixture_hash != old.fixture_hash
    assert all(c.tactical_profile.keeper_distribution_preview for c in new.cells)
    with pytest.raises(ValueError, match="explicit"):
        trainer.collection_fixture(Path("unused"), keeper_preview=1)


@pytest.mark.parametrize(
    "fixture_hash,match",
    [(None, "fixture commitment"), ("sha256:" + "2" * 64, "held-out evaluation")],
)
def test_preview_protocol_cannot_omit_fixture_or_exam_commitments(
    tmp_path, monkeypatch, fixture_hash, match
):
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
        lambda _: SimpleNamespace(agent_ids=ids, policy_hash="sha256:" + "1" * 64),
    )
    manifest = dict(
        schema="rosclaw_soccer.continuous_match_residual_ppo.v1",
        activation_ceiling="SIM_ONLY",
        promotion_eligible=False,
        iterations=[{"generation": 5}],
        trainable_agent_ids=["red.goalkeeper"],
        exploration_agent_ids=["red.goalkeeper"],
        initial_policy_hash="sha256:" + "1" * 64,
        optimizer_epochs=8,
        gamma=0.997,
        trace_decay=0.997,
        reward_shaping="motor_task_contact_v1",
        training_ball_y_m=[0.0],
        keeper_distribution_preview=True,
    )
    if fixture_hash is not None:
        manifest["collection_fixture_hash"] = fixture_hash
    manifest["manifest_hash"] = hash_json(manifest)
    (tmp_path / "training.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        audit(tmp_path)


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
