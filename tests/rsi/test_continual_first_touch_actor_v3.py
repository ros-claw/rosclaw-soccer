"""Cross-course actor must preserve a frozen domain and one-use train seeds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi import continual_first_touch_actor_v3 as learner
from rosclaw_soccer.rsi import online_first_touch_actor_critic_v2 as v2
from rosclaw_soccer.rsi.first_touch_course_catalog import sample_training_courses
from rosclaw_soccer.sim.contracts import hash_json


def _setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    parent_a = tmp_path / "seed_a"
    parent_b = tmp_path / "seed_b"
    parent_a.mkdir()
    parent_b.mkdir()

    def report(seed: int, symbol: str):
        courses = sample_training_courses(seed)
        return {
            "report_hash": "sha256:" + symbol * 64,
            "training_course_seed": seed,
            "course_catalog_hash": hash_json(courses),
            "asset_hash": "sha256:" + "c" * 64,
            "sonic_qualification_hash": "sha256:" + "d" * 64,
            "torch_batch_plan_only": True,
            "environments": [
                {"contact_body_indices": [4], "minimum_pelvis_z_m": 0.7} for _ in courses
            ],
        }, courses

    parents = {parent_a: report(101, "a"), parent_b: report(102, "b")}
    monkeypatch.setattr(v2, "_parent", lambda folder: parents[folder])
    monkeypatch.setattr(
        learner,
        "audit_first_touch_candidate_execution",
        lambda *_args, **_kwargs: {
            "report_hash": "sha256:" + "e" * 64,
            "reward_per_course": [1.0] * 6 + [-1.0] * 10,
            "candidate_clean_foot_only_count": 6,
        },
    )
    v2_state = v2._state(
        parents[parent_a][0], 1, np.zeros((4, 6)), np.zeros(4), ["sha256:" + "f" * 64]
    )
    return parent_a, parent_b, v2_state


def test_continual_update_accepts_new_seed_and_retains_anchors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parent_a, parent_b, _ = _setup(monkeypatch, tmp_path)
    state = learner.initial_state(parent_a)
    manifest = learner.sample_candidate(parent_b, state, seed=103)
    mean = learner.deterministic_mean_candidate(parent_b, state)
    assert mean["evaluation_mode"] == "FROZEN_TRANSFER_MEAN"
    assert mean["actor_state_hash"] == state["state_hash"]
    assert np.max(np.abs(mean["actions_rad"])) <= 0.04
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    candidate_folder = tmp_path / "candidate"
    candidate_folder.mkdir()
    updated, report = learner.update(parent_b, state, candidate_folder, manifest_path)
    assert updated["status"] == "PROVISIONAL"
    assert report["next_state_status"] == "PROVISIONAL_REQUIRES_TWO_SEED_MEAN_RETENTION"
    assert updated["consumed_training_seeds"] == [101, 102]
    assert report["physical_episode_count"] == 16
    assert report["maximum_per_generation_anchor_shift_rad"] <= 0.01
    assert report["maximum_global_anchor_shift_rad"] <= 0.04
    assert report["fresh_opened"] is False
    with pytest.raises(ValueError, match="commitment"):
        learner.sample_candidate(parent_b, updated, seed=104)
    with pytest.raises(ValueError, match="commitment"):
        learner.update(parent_b, updated, candidate_folder, manifest_path)


def test_continual_update_rejects_relabelled_action_or_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parent_a, parent_b, _ = _setup(monkeypatch, tmp_path)
    state = learner.initial_state(parent_a)
    manifest = learner.sample_candidate(parent_b, state, seed=103)
    manifest["actions_rad"][0][0] += 0.001
    manifest["candidate_hash"] = hash_json(
        {key: value for key, value in manifest.items() if key != "candidate_hash"}
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    candidate_folder = tmp_path / "candidate"
    candidate_folder.mkdir()
    with pytest.raises(ValueError, match="differ from committed"):
        learner.update(parent_b, state, candidate_folder, manifest_path)
    state["actor_weights"][0][0] = 0.01
    with pytest.raises(ValueError, match="commitment"):
        learner.sample_candidate(parent_b, state, seed=104)


def test_failed_learned_mean_cannot_be_migrated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parent_a, _, v2_state = _setup(monkeypatch, tmp_path)
    mean_manifest = tmp_path / "mean.json"
    expected = {"candidate_hash": "sha256:" + "f" * 64}
    mean_manifest.write_text(json.dumps(expected), encoding="utf-8")
    monkeypatch.setattr(v2, "deterministic_mean_candidate", lambda *_: expected)
    monkeypatch.setattr(
        learner,
        "audit_vector_first_touch",
        lambda *_: {"clean_foot_only_episode_count": 6},
    )
    monkeypatch.setattr(
        learner,
        "audit_first_touch_candidate_execution",
        lambda *_args, **_kwargs: {
            "candidate_hash": expected["candidate_hash"],
            "candidate_clean_foot_only_count": 5,
        },
    )
    with pytest.raises(ValueError, match="retention gate"):
        learner.migrate_v2(
            parent_a, v2_state, mean_folder=tmp_path, mean_manifest_path=mean_manifest
        )


def test_provisional_mean_failure_cannot_activate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parent_a, parent_b, _ = _setup(monkeypatch, tmp_path)
    state = learner.initial_state(parent_a)
    sampled = learner.sample_candidate(parent_b, state, seed=103)
    sampled_path = tmp_path / "sampled.json"
    sampled_path.write_text(json.dumps(sampled), encoding="utf-8")
    candidate_folder = tmp_path / "candidate"
    candidate_folder.mkdir()
    provisional, _ = learner.update(parent_b, state, candidate_folder, sampled_path)
    old_mean = learner.provisional_mean_candidate(parent_a, provisional)
    new_mean = learner.provisional_mean_candidate(parent_b, provisional)
    old_path, new_path = tmp_path / "old_mean.json", tmp_path / "new_mean.json"
    old_path.write_text(json.dumps(old_mean), encoding="utf-8")
    new_path.write_text(json.dumps(new_mean), encoding="utf-8")
    old_folder, new_folder = tmp_path / "old_eval", tmp_path / "new_eval"
    for folder in (old_folder, new_folder):
        folder.mkdir()
        (folder / "report.json").write_text(
            json.dumps({"environments": [{"minimum_pelvis_z_m": 0.7}]}), encoding="utf-8"
        )
    monkeypatch.setattr(
        learner,
        "audit_vector_first_touch",
        lambda *_: {"clean_foot_only_episode_count": 6},
    )
    monkeypatch.setattr(
        learner,
        "audit_first_touch_candidate_execution",
        lambda _folder, **kwargs: {
            "candidate_hash": json.loads(kwargs["candidate_path"].read_text())["candidate_hash"],
            "candidate_clean_foot_only_count": 5 if _folder == new_folder else 6,
            "report_hash": "sha256:" + ("2" if _folder == new_folder else "1") * 64,
        },
    )
    with pytest.raises(ValueError, match="retention gate"):
        learner.qualify_provisional(
            state,
            provisional,
            old_parent_folder=parent_a,
            old_mean_folder=old_folder,
            old_mean_manifest=old_path,
            new_parent_folder=parent_b,
            new_mean_folder=new_folder,
            new_mean_manifest=new_path,
        )
    monkeypatch.setattr(
        learner,
        "audit_first_touch_candidate_execution",
        lambda _folder, **kwargs: {
            "candidate_hash": json.loads(kwargs["candidate_path"].read_text())["candidate_hash"],
            "candidate_clean_foot_only_count": 6,
            "report_hash": "sha256:" + ("2" if _folder == new_folder else "1") * 64,
        },
    )
    active, receipt = learner.qualify_provisional(
        state,
        provisional,
        old_parent_folder=parent_a,
        old_mean_folder=old_folder,
        old_mean_manifest=old_path,
        new_parent_folder=parent_b,
        new_mean_folder=new_folder,
        new_mean_manifest=new_path,
    )
    assert learner.load_state(active)["state_hash"] == receipt["qualified_state_hash"]
    assert active["qualification_hash"] == receipt["qualification_hash"]
