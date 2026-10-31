import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.training import near_ball_residual_ppo as trainer
from rosclaw_soccer.training.near_ball_residual_ppo import _bilateral_jobs, _coverage_extra_jobs
from rosclaw_soccer.training.role_sample_coverage import role_sample_coverage


def test_counts_intersect_all_masks() -> None:
    trace = {
        "residual_active": np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
        "residual_exploration_mask": np.array([[1, 0], [1, 1], [1, 1]], dtype=bool),
        "residual_learning_mask": np.array([[1, 1], [0, 1], [1, 1]], dtype=bool),
    }
    assert role_sample_coverage([trace, trace], agent_ids=("a", "b"), required=("a", "b")) == {
        "a": 4,
        "b": 2,
    }


def test_rejects_invalid_masks_and_inconsistent_learning_scope() -> None:
    active = np.ones((3, 2), dtype=bool)
    base = {"residual_active": active}
    for key in ("residual_active", "residual_exploration_mask", "residual_learning_mask"):
        for invalid in (np.ones((3, 2), dtype=int), np.ones((3, 1), dtype=bool)):
            with pytest.raises(ValueError):
                role_sample_coverage(
                    [{**base, key: invalid}], agent_ids=("a", "b"), required=("a",)
                )
    with pytest.raises(ValueError, match="every rollout"):
        role_sample_coverage(
            [base, {**base, "residual_learning_mask": active}],
            agent_ids=("a", "b"),
            required=("a",),
        )
    with pytest.raises(ValueError):
        role_sample_coverage([], agent_ids=("a", "b"), required=("a",))
    with pytest.raises(ValueError):
        role_sample_coverage([base], agent_ids=("b", "a"), required=("a",))


def test_extra_jobs_are_fresh_bilateral_and_parent_bound(tmp_path: Path) -> None:
    assets, output = tmp_path / "assets", tmp_path / "output"
    ordinary_seeds = set()
    extra_seeds = set()
    paths = set()
    for iteration in range(100):
        ordinary = _bilateral_jobs(
            assets, output, iteration, iteration, 7.0, (0.0,), 8, True, True, True
        )
        ordinary_seeds.update(job[5] for job in ordinary)
        for index in range(8):
            jobs = _coverage_extra_jobs(
                assets, output, iteration, iteration, 7.0, (0.0,), index, True, True, True
            )
            assert [job[4] for job in jobs] == [False, True]
            for job in jobs:
                assert job[2] == str(output / f"generation-{iteration:03d}.npz")
                assert job[1] not in paths
                assert job[5] not in extra_seeds
                paths.add(job[1])
                extra_seeds.add(job[5])
    assert ordinary_seeds.isdisjoint(extra_seeds)
    with pytest.raises(ValueError):
        _coverage_extra_jobs(assets, output, 0, 0, 7.0, (0.0,), 8, True, True, True)


@pytest.mark.parametrize("frames_per_rollout,should_update", [(8, True), (1, False)])
def test_adaptive_collection_updates_only_after_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frames_per_rollout: int, should_update: bool
) -> None:
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

    def collect(job):
        dest = Path(job[1])
        dest.mkdir()
        active = np.zeros((frames_per_rollout, 8), dtype=bool)
        active[:, :2] = True
        np.savez(dest / "primary.npz", residual_active=active)
        return str(dest / "probe.json")

    monkeypatch.setattr(trainer, "_collect", collect)
    monkeypatch.setattr(
        trainer, "validate_probe", lambda _: {"exact_replay": True, "report_hash": "test"}
    )
    monkeypatch.setattr(trainer, "_validate_on_policy", lambda *_: None)
    updates = []

    def update(policy, traces, **kwargs):
        updates.append(len(traces))
        return replace(policy, generation=policy.generation + 1), []

    monkeypatch.setattr(trainer, "update_private_actors", update)
    monkeypatch.setattr(
        trainer, "run_probe", lambda **_: (_ for _ in ()).throw(RuntimeError("evaluation reached"))
    )
    output = tmp_path / "run"
    with pytest.raises(RuntimeError if should_update else ValueError):
        trainer.train(
            assets=tmp_path,
            output=output,
            iterations=1,
            duration=5,
            prospective_curriculum=True,
            all_role_clearance=True,
            strict_receive_handoff=True,
            trainable_agent_ids=ids[:2],
            minimum_actor_samples=32,
            maximum_coverage_pairs=1,
        )
    saved = json.loads((output / "training.json").read_text())
    coverage = saved["adaptive_coverage"]["iterations"][0]
    assert coverage["complete"] is should_update
    assert len(coverage["extra_rounds"]) == 1
    assert updates == ([4] if should_update else [])
    assert (output / "generation-001.npz").exists() is should_update
