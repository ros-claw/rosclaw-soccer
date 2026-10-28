import pytest

from scripts.rsi_train_incoming_pilot import curriculum_episode


def test_mixed_course_banks_each_advance_independently():
    schedule = [curriculum_episode(i, True) for i in range(12)]
    assert schedule == [(i % 2 == 0, i // 2) for i in range(12)]
    # Restarting at an episode boundary must reproduce the same course indices.
    assert [curriculum_episode(i, True) for i in range(5, 12)] == schedule[5:]
    assert [curriculum_episode(i, False) for i in range(4)] == [
        (False, 0),
        (False, 1),
        (False, 2),
        (False, 3),
    ]


@pytest.mark.parametrize("episode,mixed", [(-1, True), (True, True), (0, 1)])
def test_invalid_curriculum_schedule_rejected(episode, mixed):
    with pytest.raises(ValueError):
        curriculum_episode(episode, mixed)
