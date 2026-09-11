"""Event termination must not manufacture policy actions in recovery."""

import pytest

from rosclaw_soccer.training.event_recovery_credit import fold_event_recovery_tail
from rosclaw_soccer.training.fixed_recovery_credit import fold_fixed_recovery_tail

torch = pytest.importorskip("torch")


def fixture():
    result = {name: torch.ones(4, 2) for name in ("logp", "value", "reward", "alive", "next_alive")}
    result["next_alive"][-1] = 0
    result["obs"] = torch.ones(4, 2, 3)
    result["raw"] = torch.ones(4, 2, 2)
    return result


def test_distinct_events_and_dead_padding():
    source = fixture()
    before = {k: v.clone() for k, v in source.items()}
    result = fold_event_recovery_tail(source, prefix_frames=torch.tensor([1, 3]), gamma=0.9)
    assert result["reward"].shape == (3, 2)
    assert result["reward"][0, 0].item() == pytest.approx(1 + 0.9 + 0.9**2 + 0.9**3)
    assert result["reward"][2, 1].item() == pytest.approx(1.9)
    assert result["next_alive"][0, 0] == 0
    assert result["next_alive"][2, 1] == 0
    for name in result:
        assert torch.count_nonzero(result[name][1:, 0]) == 0
        assert torch.equal(source[name], before[name])
        assert result[name].data_ptr() != source[name].data_ptr()
    assert torch.equal(result["alive"][1:], result["next_alive"][:-1])


@pytest.mark.parametrize("count", [1, 2, 4])
def test_uniform_events_equal_fixed_tail(count):
    source = fixture()
    actual = fold_event_recovery_tail(
        source, prefix_frames=torch.tensor([count, count]), gamma=0.995
    )
    expected = fold_fixed_recovery_tail(source, prefix_frames=count, gamma=0.995)
    for key in actual:
        assert torch.equal(actual[key], expected[key])


@pytest.mark.parametrize("counts", [[0, 2], [5, 2], [1], [True, False], [1.0, 2.0]])
def test_invalid_event_counts(counts):
    with pytest.raises(ValueError):
        fold_event_recovery_tail(fixture(), prefix_frames=torch.tensor(counts), gamma=0.9)


def test_dead_world_remains_dead():
    source = fixture()
    source["alive"][1:, 0] = 0
    source["next_alive"][:, 0] = 0
    source["reward"][1:, 0] = 0
    result = fold_event_recovery_tail(source, prefix_frames=torch.tensor([1, 3]), gamma=0.9)
    assert result["reward"][0, 0] == 1
    assert torch.count_nonzero(result["alive"][1:, 0]) == 0


def test_each_world_matches_independent_fixed_fold():
    source = fixture()
    source["reward"] = torch.tensor([[2.0, -3.0], [4.0, 5.0], [-6.0, 7.0], [8.0, -9.0]])
    counts = torch.tensor([1, 3])
    actual = fold_event_recovery_tail(source, prefix_frames=counts, gamma=0.995)
    for world, count in enumerate(counts.tolist()):
        isolated = {key: value[:, world : world + 1] for key, value in source.items()}
        expected = fold_fixed_recovery_tail(isolated, prefix_frames=count, gamma=0.995)
        for key in actual:
            assert torch.equal(actual[key][:count, world : world + 1], expected[key])


def test_short_event_overflow_rejected_even_if_longest_prefix_does_not_overflow():
    source = fixture()
    source["reward"][:2, 0] = torch.finfo(torch.float32).max * 0.8
    with pytest.raises(ValueError, match="overflow"):
        fold_event_recovery_tail(source, prefix_frames=torch.tensor([1, 4]), gamma=0.995)


@pytest.mark.parametrize("problem", ["nan", "revive", "dead_reward", "keys"])
def test_invalid_full_evidence_rejected(problem):
    source = fixture()
    if problem == "nan":
        source["reward"][3, 0] = float("nan")
    elif problem == "revive":
        source["alive"][1, 0] = 0
    elif problem == "dead_reward":
        source["alive"][1:, 0] = 0
        source["next_alive"][:, 0] = 0
    else:
        source["extra"] = torch.ones(4, 2)
    with pytest.raises(ValueError):
        fold_event_recovery_tail(source, prefix_frames=torch.tensor([1, 3]), gamma=0.9)
