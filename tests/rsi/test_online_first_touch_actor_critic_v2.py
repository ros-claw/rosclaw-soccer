"""Contextual v2 updates require authenticated, nonreused physical feedback."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi import online_first_touch_actor_critic_v2 as learner
from rosclaw_soccer.rsi.first_touch_candidate import candidate_manifest
from rosclaw_soccer.rsi.first_touch_course_catalog import sample_training_courses
from rosclaw_soccer.sim.contracts import hash_json


def _setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    courses = sample_training_courses(20260928)
    parent = {
        "report_hash": "sha256:" + "a" * 64,
        "training_course_seed": 20260928,
        "course_catalog_hash": hash_json(courses),
        "environments": [{"contact_body_indices": [4], "minimum_pelvis_z_m": 0.7} for _ in courses],
    }
    monkeypatch.setattr(learner, "_parent", lambda _: (parent, courses))
    monkeypatch.setattr(
        learner,
        "audit_first_touch_candidate_execution",
        lambda folder, **_: {
            "report_hash": "sha256:" + "b" * 64,
            "reward_per_course": [1.0] * 6 + [-1.0] * 10,
            "candidate_clean_foot_only_count": 6,
        },
    )
    folder = tmp_path / "candidate"
    folder.mkdir()
    manifest_path = tmp_path / "manifest.json"
    actions = np.clip(np.random.default_rng(101).normal(0, 0.05, (16, 6)), -0.08, 0.08)
    manifest = candidate_manifest(
        courses=courses,
        parent_report_hash=parent["report_hash"],
        actions_rad=tuple(tuple(float(value) for value in row) for row in actions),
        seed=101,
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return parent, folder, manifest_path


def test_v2_update_is_bounded_and_cannot_reconsume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parent, folder, manifest_path = _setup(monkeypatch, tmp_path)
    state = learner.initial_state(tmp_path)
    updated, report = learner.update_actor_critic(tmp_path, state, ((folder, manifest_path),))
    assert updated["generation"] == 1
    assert report["physical_episode_count"] == 16
    assert report["maximum_actor_mean_shift_rad"] <= 0.01
    assert report["fresh_opened"] is False
    assert updated["parent_report_hash"] == parent["report_hash"]
    with pytest.raises(ValueError, match="already consumed"):
        learner.update_actor_critic(tmp_path, updated, ((folder, manifest_path),))
    sampled = learner.sample_candidate(tmp_path, updated, seed=102)
    assert sampled["actor_state_hash"] == updated["state_hash"]
    assert max(abs(value) for row in sampled["actions_rad"] for value in row) <= 0.08
    mean = learner.deterministic_mean_candidate(tmp_path, updated)
    assert mean["evaluation_mode"] == "FROZEN_ACTOR_MEAN"
    assert mean["actor_state_hash"] == updated["state_hash"]
    assert np.max(np.abs(mean["actions_rad"])) <= 0.04
    assert mean["actions_rad"] != sampled["actions_rad"]


def test_v2_rejects_relabelled_action_and_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, folder, manifest_path = _setup(monkeypatch, tmp_path)
    state = learner.initial_state(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["actions_rad"][0][0] += 0.001
    manifest["candidate_hash"] = hash_json(
        {key: value for key, value in manifest.items() if key != "candidate_hash"}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="differ from bound actor sample"):
        learner.update_actor_critic(tmp_path, state, ((folder, manifest_path),))
    state["actor_weights"][0][0] = 0.02
    with pytest.raises(ValueError, match="commitment"):
        learner.sample_candidate(tmp_path, state, seed=102)


def test_v2_training_domain_rejects_changed_navigation_speed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    courses = sample_training_courses(20260928)
    report = {
        "report_hash": "sha256:" + "a" * 64,
        "training_course_seed": 20260928,
        "course_catalog_hash": hash_json(courses),
        "navigation_speed_mps": 1.2,
        "environments": [{} for _ in courses],
    }
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        learner,
        "audit_vector_first_touch",
        lambda _: {"source_report_hash": report["report_hash"]},
    )
    with pytest.raises(ValueError, match="seeded sixteen-course Parent"):
        learner._parent(tmp_path)
