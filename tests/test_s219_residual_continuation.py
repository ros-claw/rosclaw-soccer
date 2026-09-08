from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.training import near_ball_residual_ppo as module


def foundation(monkeypatch, body="sha256:" + "a" * 64):
    ids = tuple(f"agent.{i}" for i in range(8))
    monkeypatch.setattr(
        module,
        "build_four_vs_four_fixture",
        lambda _: SimpleNamespace(
            cells=tuple(
                SimpleNamespace(agent_id=i, growth_scope=SimpleNamespace(body_hash=body))
                for i in ids
            )
        ),
    )
    return ids


def test_continuation_collects_from_parent_not_zero(monkeypatch, tmp_path: Path):
    ids = foundation(monkeypatch)
    parent = replace(NearBallResidualPolicy.initialize(ids, "sha256:" + "a" * 64), generation=12)
    checkpoint = tmp_path / "parent.npz"
    parent.save(checkpoint)

    def collect(job):
        assert Path(job[2]).name == "generation-012.npz"
        assert NearBallResidualPolicy.load(Path(job[2])).policy_hash == parent.policy_hash
        assert job[-1] is True
        assert job[6] == -0.16
        raise RuntimeError("collection-boundary-reached")

    monkeypatch.setattr(module, "_collect", collect)
    with pytest.raises(RuntimeError, match="collection-boundary"):
        module.train(
            assets=tmp_path,
            output=tmp_path / "run",
            iterations=1,
            duration=5,
            initial_checkpoint=checkpoint,
            all_role_clearance=True,
            diverse_ball_positions=True,
        )
    assert not (tmp_path / "run" / "generation-000.npz").exists()


def test_wrong_body_parent_rejected_before_output_creation(monkeypatch, tmp_path: Path):
    ids = foundation(monkeypatch, "sha256:" + "b" * 64)
    path = tmp_path / "wrong.npz"
    NearBallResidualPolicy.initialize(ids, "sha256:" + "a" * 64).save(path)
    with pytest.raises(ValueError, match="body/roster"):
        module.train(
            assets=tmp_path,
            output=tmp_path / "run",
            iterations=1,
            duration=5,
            initial_checkpoint=path,
        )
    assert not (tmp_path / "run").exists()
