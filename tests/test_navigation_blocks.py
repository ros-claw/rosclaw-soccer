import numpy as np
import pytest

from rosclaw_soccer.training.navigation_blocks import navigation_block_credit


def test_blocks_discount_real_frames_and_do_not_count_held_actions_repeatedly():
    rewards = np.arange(26, dtype=np.float32)
    applied = np.zeros(26, dtype=bool)
    applied[15] = True
    credit, influence = navigation_block_credit(
        rewards, np.array([3, 13, 23]), applied, hold_frames=10, frame_gamma=0.997
    )
    expected = [
        sum(float(rewards[j]) * 0.997 ** (j - s) for j in range(s, min(s + 10, 26)))
        for s in (3, 13, 23)
    ]
    np.testing.assert_array_equal(credit, np.asarray(expected, np.float32))
    assert influence.tolist() == [False, True, False]
    assert len(credit) == 3


@pytest.mark.parametrize(
    "frames",
    [
        np.array([3, 14, 23]),
        np.array([3, 13]),
        np.array([-1, 9, 19]),
        np.array([True]),
        np.array([2**64 - 7, 3, 13, 23], dtype=np.uint64),
    ],
)
def test_missing_repeated_or_invalid_physical_blocks_fail_closed(frames):
    with pytest.raises(ValueError):
        navigation_block_credit(
            np.ones(26), frames, np.ones(26, dtype=bool), hold_frames=10, frame_gamma=0.997
        )


def test_block_credit_requires_explicit_finite_rewards_and_actual_boolean_influence():
    for rewards, applied in (
        (np.full(10, np.nan), np.ones(10, dtype=bool)),
        (np.ones(10), np.ones(10)),
        (np.ones(10, dtype=int), np.ones(10, dtype=bool)),
    ):
        with pytest.raises(ValueError):
            navigation_block_credit(
                rewards, np.array([0]), applied, hold_frames=10, frame_gamma=0.997
            )
