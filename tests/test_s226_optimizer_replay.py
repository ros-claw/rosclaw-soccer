import json

import numpy as np
import pytest

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training import active_team_probe, near_ball_residual_ppo
from rosclaw_soccer.training.near_ball_plasticity import replay_recorded_update


@pytest.mark.parametrize("role,rounds,count", [(False, 1, 2), (True, 1, 8), (True, 5, 40)])
def test_replay_uses_committed_order_for_legacy_and_role_batches(
    tmp_path, monkeypatch, role, rounds, count
):
    parent = NearBallResidualPolicy.initialize(
        tuple(f"agent.{i}" for i in range(8)), "sha256:" + "a" * 64
    )
    child = NearBallResidualPolicy(
        parent.agent_ids, parent.body_hash, 1, parent.policy_hash, parent.weights
    )
    parent.save(tmp_path / "generation-000.npz")
    child.save(tmp_path / "generation-001.npz")
    proofs = []
    for i in range(count):
        directory = tmp_path / f"rollout-{count - i:03d}"
        directory.mkdir()
        proof = str(hash_json({"sample": i}))
        proofs.append(proof)
        (directory / "probe.json").write_text(json.dumps({"report_hash": proof}))
        np.savez(directory / "primary.npz", marker=np.asarray(i))
    monkeypatch.setattr(active_team_probe, "validate_probe", lambda p: json.loads(p.read_text()))

    def update(parent, traces, **kwargs):
        assert [int(t["marker"]) for t in traces] == list(range(count))
        return child, []

    monkeypatch.setattr(near_ball_residual_ppo, "update_private_actors", update)
    manifest = {
        "role_curriculum": role,
        "role_batch_rounds": rounds,
        "iterations": [{"rollout_report_hashes": proofs}],
    }
    manifest["manifest_hash"] = hash_json(manifest)
    (tmp_path / "training.json").write_text(json.dumps(manifest))
    result = replay_recorded_update(tmp_path, tmp_path / "proof.json")
    assert result["exact_checkpoint_reproduced"] and result["dataset_reports"] == proofs
    duplicate = tmp_path / "duplicate"
    duplicate.mkdir()
    (duplicate / "probe.json").write_text(json.dumps({"report_hash": proofs[0]}))
    with pytest.raises(ValueError, match="duplicate"):
        replay_recorded_update(tmp_path, tmp_path / "another.json")
