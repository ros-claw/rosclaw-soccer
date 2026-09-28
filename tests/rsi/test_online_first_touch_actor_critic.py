"""On-policy update requires authenticated, unique physical batch evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi import online_first_touch_actor_critic as learner
from rosclaw_soccer.rsi.first_touch_candidate import candidate_manifest

PARENT = "sha256:" + "a" * 64
COURSES = ((2.4, -0.1, -0.5), (2.6, 0.1, 0.5))


def _evidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path, Path]:
    parent = tmp_path / "parent"
    parent.mkdir()
    rows = [
        {
            "course": {"ball_x_m": x, "ball_y_local_m": y, "ball_vx_m_s": vx},
            "contact_body_indices": [0],
            "minimum_pelvis_z_m": 0.7,
        }
        for x, y, vx in COURSES
    ]
    (parent / "report.json").write_text(
        json.dumps({"report_hash": PARENT, "environments": rows}), encoding="utf-8"
    )
    monkeypatch.setattr(
        learner,
        "audit_vector_first_touch",
        lambda _: {"source_report_hash": PARENT, "clean_foot_only_episode_count": 2},
    )
    monkeypatch.setattr(
        learner,
        "audit_first_touch_candidate_execution",
        lambda folder, **_: {
            "report_hash": "sha256:" + folder.name.rjust(64, "b"),
            "reward_per_course": [0.8, -1.0],
            "candidate_clean_foot_only_count": 1,
        },
    )
    execution = tmp_path / "batch"
    execution.mkdir()
    manifest_path = tmp_path / "candidate.json"
    actions = np.random.default_rng(17).normal(0.0, 0.02, (2, 6))
    manifest_path.write_text(
        json.dumps(
            candidate_manifest(
                courses=COURSES,
                parent_report_hash=PARENT,
                actions_rad=tuple(tuple(float(value) for value in row) for row in actions),
                seed=17,
            )
        ),
        encoding="utf-8",
    )
    return parent, execution, manifest_path


def test_gen0_update_consumes_exact_sample_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parent, execution, manifest = _evidence(monkeypatch, tmp_path)
    initial = learner.initial_state(PARENT)
    state, report = learner.update_actor_critic(parent, initial, ((execution, manifest),))
    assert state["generation"] == 1
    assert report["physical_episode_count"] == 2
    assert report["fresh_opened"] is False
    with pytest.raises(ValueError, match="off-policy or already consumed"):
        learner.update_actor_critic(parent, state, ((execution, manifest),))


def test_mismatched_action_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parent, execution, manifest_path = _evidence(monkeypatch, tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["actions_rad"][0][0] += 0.001
    manifest.pop("candidate_hash")
    from rosclaw_soccer.sim.contracts import hash_json

    manifest["candidate_hash"] = hash_json(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="differ from bound actor sample"):
        learner.update_actor_critic(
            parent, learner.initial_state(PARENT), ((execution, manifest_path),)
        )


def test_sample_is_state_bound(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parent, _, _ = _evidence(monkeypatch, tmp_path)
    state = learner.initial_state(PARENT)
    sampled = learner.sample_candidate(parent, state, seed=23)
    assert sampled["actor_state_hash"] == state["state_hash"]
    assert max(abs(value) for row in sampled["actions_rad"] for value in row) <= 0.08
    tampered = {**state, "actor_weights": [[0.1] * 6] * 4}
    with pytest.raises(ValueError, match="digest invalid"):
        learner.sample_candidate(parent, tampered, seed=23)


def test_gen1_sample_can_update_only_bound_actor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parent, execution, manifest_path = _evidence(monkeypatch, tmp_path)
    initial = learner.initial_state(PARENT)
    updated, _ = learner.update_actor_critic(parent, initial, ((execution, manifest_path),))
    sampled = learner.sample_candidate(parent, updated, seed=23)
    second_execution = tmp_path / "second_batch"
    second_execution.mkdir()
    second_manifest = tmp_path / "second_candidate.json"
    second_manifest.write_text(json.dumps(sampled), encoding="utf-8")
    second_state, _ = learner.update_actor_critic(
        parent, updated, ((second_execution, second_manifest),)
    )
    assert second_state["generation"] == 2
    sampled["actor_state_hash"] = initial["state_hash"]
    sampled.pop("candidate_hash")
    from rosclaw_soccer.sim.contracts import hash_json

    sampled["candidate_hash"] = hash_json(sampled)
    second_manifest.write_text(json.dumps(sampled), encoding="utf-8")
    with pytest.raises(ValueError, match="off-policy"):
        learner.update_actor_critic(parent, updated, ((second_execution, second_manifest),))
