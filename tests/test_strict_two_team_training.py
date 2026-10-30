"""The small bilateral PPO course must use the declared physical handoff."""

from pathlib import Path

import pytest

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.training import near_ball_residual_ppo as module


@pytest.mark.parametrize("strict", [False, True])
def test_two_team_rollout_binds_handoff_contract(monkeypatch, tmp_path: Path, strict: bool):
    checkpoint = tmp_path / "policy.npz"
    policy = NearBallResidualPolicy.initialize(
        tuple(f"agent.{i}" for i in range(8)), "sha256:" + "a" * 64
    )
    policy.save(checkpoint)
    calls = []
    monkeypatch.setattr(module, "run_probe", lambda **kwargs: calls.append(kwargs))
    target = tmp_path / "output"
    result = module._collect(
        (str(tmp_path), str(target), str(checkpoint), 5.0, True, 12, 0.08, True, True, strict)
    )
    assert result == str(target / "probe.json")
    assert calls[0]["strict_receive_handoff"] is strict
    assert calls[0]["near_ball_explore"] is True
    assert calls[0]["near_ball_policy"].policy_hash == policy.policy_hash


@pytest.mark.parametrize("prospective,clearance", [(False, False), (True, False), (False, True)])
def test_strict_training_requires_prospective_and_clearance(tmp_path, prospective, clearance):
    with pytest.raises(ValueError, match="bounded online training"):
        module.train(
            assets=tmp_path,
            output=tmp_path / "out",
            iterations=1,
            duration=5.0,
            prospective_curriculum=prospective,
            all_role_clearance=clearance,
            strict_receive_handoff=True,
        )
