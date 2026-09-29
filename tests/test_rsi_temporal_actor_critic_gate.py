"""Online motor learning never promotes truncated or unsafe receiving rollouts."""

from rsi_r1_temporal_actor_critic_v193 import clean, reward, score
from rsi_r1_temporal_zero_v192 import TRAIN_COURSES


def _row(*, safe: bool = True, controlled: bool = True) -> dict[str, object]:
    return {
        "safe": safe,
        "fault_agents": [],
        "active_substeps": 32,
        "first_foot_frame": 33,
        "own_nonfoot_frames": [],
        "controlled_reception": controlled,
        "tail_maximum_foot_distance_m": 0.2,
        "tail_maximum_ball_speed_mps": 0.2,
    }


def test_truncated_rollout_is_failure_even_with_reported_foot_contact() -> None:
    row = _row(safe=False)
    assert not clean(row)
    assert reward(row) == -60.0


def test_retention_precedes_development_score() -> None:
    baseline = [_row(controlled=False) for _ in TRAIN_COURSES]
    baseline[-2:] = [_row(), _row()]
    more_development = [_row() for _ in TRAIN_COURSES]
    more_development[-1] = _row(controlled=False)
    assert score(baseline)[0] == 1
    assert score(more_development)[0] == 0
    assert score(baseline) > score(more_development)
