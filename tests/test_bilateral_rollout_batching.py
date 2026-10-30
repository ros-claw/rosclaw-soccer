"""On-policy bilateral batches are diverse, bounded and collision-free."""

from pathlib import Path

import pytest

from rosclaw_soccer.training.near_ball_residual_ppo import _bilateral_jobs, train


def jobs(tmp_path: Path, iteration: int, count: int):
    return _bilateral_jobs(
        tmp_path / "assets",
        tmp_path / "out",
        iteration,
        iteration,
        7.0,
        (-0.16, -0.04, 0.04, 0.16),
        count,
        True,
        True,
        True,
    )


def test_batch_shares_frozen_parent_and_covers_both_teams_with_distinct_seeds(tmp_path):
    batch = jobs(tmp_path, 0, 4)
    assert len(batch) == 8
    assert len({j[1] for j in batch}) == len({j[5] for j in batch}) == 8
    assert len({j[2] for j in batch}) == 1
    assert {j[4] for j in batch} == {False, True}
    for side in (False, True):
        assert {j[6] for j in batch if j[4] == side} == {-0.16, -0.04, 0.04, 0.16}
    assert all(j[7:] == (True, True, True) for j in batch)
    assert not {j[5] for j in batch} & {j[5] for j in jobs(tmp_path, 1, 4)}


def test_single_episode_keeps_historical_names_and_seeds(tmp_path):
    batch = jobs(tmp_path, 1, 1)
    assert [Path(j[1]).name for j in batch] == ["train-001-red", "train-001-blue"]
    assert [j[5] for j in batch] == [21502, 21503]
    assert [j[6] for j in batch] == [-0.04, -0.04]


@pytest.mark.parametrize("value", [0, 9, True, 1.5, None])
def test_invalid_batch_size_fails_before_creating_output(tmp_path, value):
    with pytest.raises(ValueError, match="bounded online training"):
        train(
            assets=tmp_path,
            output=tmp_path / "out",
            iterations=1,
            duration=7.0,
            rollouts_per_side=value,
        )
    assert not (tmp_path / "out").exists()


def test_role_and_bilateral_batching_cannot_be_combined(tmp_path):
    with pytest.raises(ValueError, match="bounded online training"):
        train(
            assets=tmp_path,
            output=tmp_path / "out",
            iterations=1,
            duration=7.0,
            prospective_curriculum=True,
            all_role_clearance=True,
            role_curriculum=True,
            rollouts_per_side=2,
        )
