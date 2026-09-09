import json
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.keeper_context_gate import KeeperContextGate, gate_config_hash


def payload():
    return dict(
        schema="keeper-context-gate.v1",
        activation_ceiling="SIM_ONLY",
        promotion_authorized=False,
        parent_policy_hash="parent",
        config_hash="config",
        center=1.4,
        scale=0.1,
        weight=2.0,
        bias=0.0,
    )


def load(tmp_path, value):
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(value))
    return KeeperContextGate(path, parent_policy_hash="parent", config_hash="config")


def test_numeric_selection_and_content_binding(tmp_path):
    gate = load(tmp_path, payload())
    assert not gate.select_reach(1.3)
    assert gate.select_reach(1.5)
    changed = load(tmp_path, dict(payload(), weight=-2.0))
    assert changed.policy_hash != gate.policy_hash
    assert changed.select_reach(1.3)
    for value in (np.nan, np.inf, 0.4, 2.1):
        with pytest.raises(ValueError):
            gate.select_reach(value)


@pytest.mark.parametrize(
    "change",
    [
        dict(activation_ceiling="REAL"),
        dict(promotion_authorized=True),
        dict(parent_policy_hash="other"),
        dict(config_hash="other"),
        dict(scale=0),
        dict(weight=float("nan")),
        dict(bias=101),
        dict(center=None),
    ],
)
def test_rejects_authority_parent_and_numeric_errors(tmp_path, change):
    with pytest.raises(ValueError):
        load(tmp_path, dict(payload(), **change))


def test_missing_weight_rejected(tmp_path):
    value = payload()
    del value["weight"]
    with pytest.raises(ValueError):
        load(tmp_path, value)


def test_common_contract_excludes_only_model_location_and_choice():
    base = dict(gain_scale=1.5, muscle_actor_path="a", muscle_reach_correction=False)
    assert gate_config_hash(base) == gate_config_hash(
        dict(base, muscle_actor_path="b", muscle_reach_correction=True, muscle_gate_path="c")
    )
    assert gate_config_hash(base) != gate_config_hash(dict(base, gain_scale=1.6))


def test_learning_uses_bound_physical_labels_and_single_focal_core_audit(tmp_path, monkeypatch):
    from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
    from rosclaw_soccer.training import keeper_context_learning as learning

    ids = [
        f"{team}.{role}"
        for team in ("red", "blue")
        for role in ("goalkeeper", "defender", "playmaker", "finisher")
    ]
    fixture = SimpleNamespace(
        fixture_hash=str(hash_json("fixture")),
        foundation_policy_hash=str(hash_json("foundation")),
        cells=[SimpleNamespace(agent_id=name) for name in ids],
    )
    monkeypatch.setattr(learning, "build_four_vs_four_fixture", lambda _: fixture)
    exams = tmp_path / "exams"
    for height in (1.3, 1.5):
        for correction in (False, True):
            directory = exams / f"{height}-{correction}"
            directory.mkdir(parents=True)
            trajectory = directory / "trajectory.npz"
            np.savez(trajectory, context_ready=[False, True], context_height_m=[0.0, height])
            record = dict(
                schema_version="s229.shared_keeper_reach_exam.v3",
                activation_ceiling="SIM_ONLY",
                candidate_promoted=False,
                source_integrity_verified=True,
                fixture_hash=fixture.fixture_hash,
                gate_policy_hash=None,
                trajectory_hash=str(hash_bytes(trajectory.read_bytes())),
                muscle_policy_hash=str(hash_json("parent")),
                config=dict(muscle_reach_correction=correction, gain_scale=1.5),
                team="red",
                lateral=0.0,
                height=height,
                stable_save=correction == (height == 1.5),
                physical_safe=True,
            )
            (directory / "result.json").write_text(json.dumps(record))
    report = learning.train(tmp_path, exams, tmp_path / "out")
    assert report["predictions"] == [False, True]
    audit = report["core_learning_boundary"]["audit"]
    assert audit["passed"]
    assert audit["changed_agent_ids"] == ["red.goalkeeper"]
    assert not report["physical_replay_passed"]
    assert not report["candidate_promoted"]
    (exams / "1.3-False" / "trajectory.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="trajectory changed"):
        learning.train(tmp_path, exams, tmp_path / "invalid")
