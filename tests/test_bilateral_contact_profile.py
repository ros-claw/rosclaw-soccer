import json
import pickle
from dataclasses import asdict, replace
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.growth.locomotion_contact_teacher import G1LocomotionContactTeacherConfig
from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldConfig
from rosclaw_soccer.training import near_ball_residual_ppo as trainer
from rosclaw_soccer.training.contact_control_profile import ContactControlProfile


@pytest.mark.parametrize("valid_receipt", [True, False])
def test_bilateral_training_commits_profile_through_collection_and_evaluation(
    tmp_path, monkeypatch, valid_receipt
):
    ids = tuple(f"agent.{i}" for i in range(8))
    cells = tuple(
        SimpleNamespace(
            agent_id=agent, growth_scope=SimpleNamespace(body_hash="sha256:" + "a" * 64)
        )
        for agent in ids
    )
    monkeypatch.setattr(
        trainer, "build_four_vs_four_fixture", lambda _: SimpleNamespace(cells=cells)
    )
    profile = ContactControlProfile(
        separation_m=1.2, stiffness_scale=1.0, guard_margin_rad=0.04, strike_residual_enabled=True
    )
    world, teacher = profile.apply(IndependentTeamWorldConfig(), G1LocomotionContactTeacherConfig())
    report = {
        "exact_replay": True,
        "report_hash": "test",
        "world_config": asdict(world),
        "contact_teacher_config": asdict(teacher),
    }
    if not valid_receipt:
        report["world_config"]["strike_residual_enabled"] = False
    monkeypatch.setattr(trainer, "validate_probe", lambda _: report)
    seen = []

    def collect(job, *, contact_control_profile):
        seen.append(contact_control_profile)
        path = trainer.Path(job[1])
        path.mkdir()
        np.savez(path / "primary.npz", sample=np.zeros(1))
        return str(path / "probe.json")

    monkeypatch.setattr(trainer, "_collect", collect)
    updates = []

    def update(policy, *_args, **_kwargs):
        updates.append(True)
        return replace(policy, generation=policy.generation + 1), []

    monkeypatch.setattr(trainer, "update_private_actors", update)
    evaluations = []

    def evaluate(**kwargs):
        evaluations.append(kwargs["contact_control_profile"])
        raise RuntimeError("evaluation reached")

    monkeypatch.setattr(trainer, "run_probe", evaluate)
    output = tmp_path / "out"
    with pytest.raises(RuntimeError if valid_receipt else ValueError):
        trainer.train(
            assets=tmp_path,
            output=output,
            iterations=1,
            duration=5,
            contact_control_profile=profile,
        )
    saved = json.loads((output / "training.json").read_text())
    assert saved["contact_control_profile"] == asdict(profile)
    assert seen == [profile, profile]
    assert updates == ([True] if valid_receipt else [])
    assert evaluations == ([profile] if valid_receipt else [])


def test_worker_passes_opt_in_profile_without_touching_other_probe_contracts(tmp_path, monkeypatch):
    checkpoint = tmp_path / "policy.npz"
    policy = trainer.NearBallResidualPolicy.initialize(
        tuple(f"agent.{i}" for i in range(8)), "sha256:" + "a" * 64
    )
    policy.save(checkpoint)
    profile = ContactControlProfile(strike_residual_enabled=True)
    calls = []
    monkeypatch.setattr(trainer, "run_probe", lambda **kwargs: calls.append(kwargs))
    trainer._collect(
        (
            str(tmp_path),
            str(tmp_path / "out"),
            str(checkpoint),
            5.0,
            True,
            3,
            0.08,
            True,
            True,
            True,
        ),
        contact_control_profile=profile,
    )
    assert calls[0]["contact_control_profile"] is profile
    assert calls[0]["near_ball_explore"] and calls[0]["strict_receive_handoff"]


def test_exact_shoot_only_profile_and_spawn_serialization():
    baseline = replace(IndependentTeamWorldConfig(), minimum_player_separation_m=1.2)
    teacher = G1LocomotionContactTeacherConfig()
    profile = ContactControlProfile(
        separation_m=1.2, stiffness_scale=1.0, guard_margin_rad=0.04, strike_residual_enabled=True
    )
    world, updated_teacher = profile.apply(baseline, teacher)
    assert world == replace(baseline, strike_residual_enabled=True)
    assert updated_teacher == teacher
    worker = partial(trainer._collect, contact_control_profile=profile)
    restored = pickle.loads(pickle.dumps(worker))
    assert restored.func is trainer._collect
    assert restored.keywords == {"contact_control_profile": profile}
