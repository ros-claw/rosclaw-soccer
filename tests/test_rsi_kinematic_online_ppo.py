"""Online phase returns preserve safety ordering and observed-frame alignment."""

from rsi_r1_kinematic_online_ppo_v204 import _returns, _terminal_quality


def _row() -> dict:
    return {
        "safe": True,
        "fault_agents": [],
        "first_foot_frame": 31,
        "own_nonfoot_frames": [],
        "tail_maximum_foot_distance_m": 0.3,
        "tail_maximum_ball_speed_mps": 0.2,
        "controlled_reception": True,
        "observed_frames": [15, 20, 21],
        "shaped_reward": [0.0] * 100,
    }


def test_online_terminal_priority() -> None:
    clean = _row()
    assert _terminal_quality(clean) > 0
    contaminated = {**clean, "own_nonfoot_frames": [37]}
    assert _terminal_quality(contaminated) < _terminal_quality(clean)
    unsafe = {**clean, "safe": False}
    assert _terminal_quality(unsafe) < _terminal_quality(contaminated)


def test_returns_align_current_observation_to_future_outcome() -> None:
    row = _row()
    row["shaped_reward"][0] = 2.0
    before, frame20, frame21 = _returns(row)
    assert before == frame20
    assert frame20 > frame21
    assert len(_returns({**row, "shaped_reward": []})) == 3
