import numpy as np
import pytest

from rosclaw_soccer.training.pass_commitment_windows import enumerate_pass_commitment_windows


def run(source, target, **kwargs):
    return enumerate_pass_commitment_windows(
        control_time_sec=np.arange(1, len(source) + 1) * 0.02,
        source_codes=np.asarray(source),
        receiver_codes=np.asarray(target),
        agent_count=4,
        **kwargs,
    )


def test_repeated_pair_and_target_switch_are_separate_and_last_is_censored():
    windows = run([0, 1, 1, 0, 1, 1, 1], [0, 2, 2, 0, 2, 3, 3])
    assert [(w.sender_code, w.receiver_code, w.first_frame, w.last_frame) for w in windows] == [
        (1, 2, 1, 2),
        (1, 2, 4, 4),
        (1, 3, 5, 6),
    ]
    assert windows[0].next_observation_sec == pytest.approx(0.08)
    assert windows[-1].right_censored and windows[-1].next_observation_sec is None
    assert not windows[0].right_censored


@pytest.mark.parametrize(
    "src,dst", [([0, 1], [0, 0]), ([0, 1], [0, 1]), ([0, 5], [0, 2]), ([0, -1], [0, 2])]
)
def test_partial_same_or_out_of_roster_identity_fails(src, dst):
    with pytest.raises(ValueError):
        run(src, dst)


def test_clock_gaps_and_nonfinite_values_fail_closed():
    for times in (np.array([0.02, 0.06]), np.array([0.02, np.nan])):
        with pytest.raises(ValueError):
            enumerate_pass_commitment_windows(
                control_time_sec=times,
                source_codes=np.array([1, 1]),
                receiver_codes=np.array([2, 2]),
                agent_count=4,
            )


def test_empty_commitment_is_not_an_attempt():
    assert run([0, 0], [0, 0]) == ()
