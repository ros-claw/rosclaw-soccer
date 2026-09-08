from types import SimpleNamespace

import pytest

from rosclaw_soccer.training import near_ball_residual_ppo as module


@pytest.mark.parametrize("rounds", [1, 5])
def test_eight_role_workers_preserve_fixed_job_identity_and_budget(monkeypatch, tmp_path, rounds):
    body = "sha256:" + "a" * 64
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

    class Pool:
        def __init__(self, *, max_workers, mp_context):
            assert max_workers == 8 and mp_context.get_start_method() == "spawn"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, jobs):
            assert function is module.collect_role_course
            assert len(jobs) == len({j.destination for j in jobs}) == 8 * rounds
            assert {j.seed for j in jobs} == set(range(22100, 22100 + 8 * rounds))
            assert all(j.strict_handoff and j.explore for j in jobs)
            assert len({j.checkpoint for j in jobs}) == 1
            raise RuntimeError("collection boundary reached")

    monkeypatch.setattr(module, "ProcessPoolExecutor", Pool)
    with pytest.raises(RuntimeError, match="collection boundary"):
        module.train(
            assets=tmp_path,
            output=tmp_path / "run",
            iterations=1,
            duration=12,
            workers=8,
            prospective_curriculum=True,
            all_role_clearance=True,
            role_curriculum=True,
            strict_receive_handoff=True,
            role_batch_rounds=rounds,
        )


@pytest.mark.parametrize("workers,role", [(9, True), (3, False), (True, True)])
def test_excessive_or_implicit_worker_budget_rejected_before_io(tmp_path, workers, role):
    with pytest.raises(ValueError, match="budget"):
        module.train(
            assets=tmp_path,
            output=tmp_path / "run",
            iterations=1,
            duration=12,
            workers=workers,
            role_curriculum=role,
            prospective_curriculum=True,
            all_role_clearance=True,
        )
    assert not (tmp_path / "run").exists()
