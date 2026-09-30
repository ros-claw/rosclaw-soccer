"""Causal receiving phase detection cannot command motion or use future labels."""

import pytest

from rosclaw_soccer.rsi.receiving_contact_episode_gate import ReceivingContactEpisodeGate


def test_measured_approach_contact_and_release() -> None:
    gate = ReceivingContactEpisodeGate("red.finisher")
    for frame in range(60):
        distance = 0.8 - 0.025 * frame
        state = gate.observe(
            frame=frame,
            time_sec=0.02 * (frame + 1),
            ball_relative_position_xy_m=(distance, 0.0),
            ball_relative_velocity_xy_mps=(-1.25, 0.0),
            own_foot_contact=frame == 25,
        )
        if frame == 16:
            assert state == "APPROACH"
        if frame == 25:
            assert state == "CONTACT"
    assert len(gate.episodes) == 1
    episode = gate.episodes[0]
    assert episode.arm_frame == 16
    assert episode.contact_frame == 25
    assert episode.outcome == "MEASURED_FOOT_CONTACT_AND_RECOVERY"


def test_only_closing_ball_arms_and_frames_are_sequential() -> None:
    gate = ReceivingContactEpisodeGate("red.finisher")
    assert (
        gate.observe(
            frame=0,
            time_sec=0.02,
            ball_relative_position_xy_m=(0.35, 0.0),
            ball_relative_velocity_xy_mps=(0.5, 0.0),
            own_foot_contact=False,
        )
        == "WAITING"
    )
    with pytest.raises(ValueError, match="sequential"):
        gate.observe(
            frame=2,
            time_sec=0.06,
            ball_relative_position_xy_m=(0.35, 0.0),
            ball_relative_velocity_xy_mps=(-0.5, 0.0),
            own_foot_contact=False,
        )
    with pytest.raises(ValueError, match="bounded"):
        ReceivingContactEpisodeGate("red.finisher", arm_radius_m=float("nan"))


def test_missed_contact_times_out_without_claiming_reception() -> None:
    gate = ReceivingContactEpisodeGate("blue.playmaker")
    for frame in range(45):
        gate.observe(
            frame=frame,
            time_sec=0.02 * (frame + 1),
            ball_relative_position_xy_m=(0.30, 0.0),
            ball_relative_velocity_xy_mps=(-0.50, 0.0) if frame == 0 else (0.0, 0.0),
            own_foot_contact=False,
        )
    assert len(gate.episodes) == 1
    assert gate.episodes[0].contact_frame is None
    assert gate.episodes[0].outcome == "TIMED_OUT_WITHOUT_FOOT_CONTACT"
    assert gate.state == "WAITING"


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_sensor_frame_rejected_without_state_progress(invalid: float) -> None:
    gate = ReceivingContactEpisodeGate("red.finisher")
    with pytest.raises(ValueError, match="finite"):
        gate.observe(
            frame=0,
            time_sec=0.02,
            ball_relative_position_xy_m=(invalid, 0.0),
            ball_relative_velocity_xy_mps=(-0.5, 0.0),
            own_foot_contact=False,
        )
    assert gate.last_frame is None
    assert gate.episodes == []
