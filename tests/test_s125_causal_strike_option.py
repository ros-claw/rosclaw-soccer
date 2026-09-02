from __future__ import annotations

from dataclasses import replace

import pytest

from rosclaw_soccer.growth.causal_strike_option import (
    CausalStrikeOptionObservation,
    CausalStrikeOptionPhase,
    G1CausalStrikeOptionConfig,
    G1CausalStrikeOptionController,
)


def _observation(frame: int, **overrides: object) -> CausalStrikeOptionObservation:
    values: dict[str, object] = {
        "timestamp_sec": frame * 0.02,
        "predecessor_policy_frame": frame,
        "receiver_pelvis_height_m": 0.75,
        "receiver_roll_rad": 0.02,
        "receiver_pitch_rad": -0.03,
        "receiver_joint_velocity_rms_rad_s": 0.10,
        "receiver_ball_local_x_m": 3.9,
        "receiver_ball_local_vx_mps": 0.0,
    }
    values.update(overrides)
    return CausalStrikeOptionObservation(**values)  # type: ignore[arg-type]


def test_causal_strike_option_commits_once_after_ready_tracking() -> None:
    config = G1CausalStrikeOptionConfig()
    controller = G1CausalStrikeOptionController(config)

    prepare = controller.step(_observation(config.predecessor_track_policy_frame - 1))
    track = controller.step(_observation(config.predecessor_track_policy_frame))
    commit = controller.step(_observation(config.predecessor_commit_policy_frame))
    latched = controller.step(_observation(config.predecessor_commit_policy_frame + 1))
    controller.observe_contact()

    assert prepare.phase == CausalStrikeOptionPhase.PREPARE
    assert track.phase == CausalStrikeOptionPhase.TRACK
    assert commit.phase == CausalStrikeOptionPhase.COMMIT
    assert commit.begin_bridge
    assert commit.strike_phase_start_frame == 100
    assert not latched.begin_bridge
    assert controller.phase == CausalStrikeOptionPhase.RECOVER


def test_causal_strike_option_aborts_when_readiness_never_recovers() -> None:
    config = G1CausalStrikeOptionConfig()
    controller = G1CausalStrikeOptionController(config)

    controller.step(
        _observation(
            config.predecessor_track_policy_frame,
            receiver_pelvis_height_m=0.50,
        )
    )
    decision = controller.step(
        _observation(
            config.predecessor_abort_policy_frame,
            receiver_pelvis_height_m=0.50,
        )
    )

    assert decision.phase == CausalStrikeOptionPhase.ABORTED
    assert not decision.begin_bridge
    assert decision.reason == "readiness_deadline_missed"


def test_causal_strike_option_reports_only_causal_ball_eta() -> None:
    config = G1CausalStrikeOptionConfig()
    controller = G1CausalStrikeOptionController(config)

    static = controller.step(_observation(config.predecessor_track_policy_frame))
    incoming = controller.step(
        _observation(
            config.predecessor_track_policy_frame + 1,
            receiver_ball_local_x_m=3.0,
            receiver_ball_local_vx_mps=-1.5,
        )
    )

    assert static.ball_arrival_eta_sec is None
    assert incoming.incoming_ball
    assert incoming.ball_arrival_eta_sec == pytest.approx((3.0 - 1.25) / 1.5)
    assert incoming.ready


def test_causal_strike_option_bounds_arrival_alignment() -> None:
    config = G1CausalStrikeOptionConfig(maximum_arrival_hold_frames=2)
    controller = G1CausalStrikeOptionController(config)
    controller.step(_observation(config.predecessor_track_policy_frame))
    controller.step(_observation(config.predecessor_commit_policy_frame))
    for frame in range(
        config.predecessor_commit_policy_frame + 1,
        config.predecessor_commit_policy_frame + 1 + config.minimum_incoming_observations,
    ):
        controller.step(
            _observation(
                frame,
                receiver_ball_local_x_m=3.9,
                receiver_ball_local_vx_mps=-1.5,
            )
        )

    first = controller.align_repeat_count(
        policy_frame=config.arrival_alignment_start_policy_frame,
        nominal_repeat=1,
    )
    second = controller.align_repeat_count(
        policy_frame=config.arrival_alignment_start_policy_frame,
        nominal_repeat=1,
    )
    exhausted = controller.align_repeat_count(
        policy_frame=config.arrival_alignment_start_policy_frame,
        nominal_repeat=1,
    )

    assert first == second == (0, -1)
    assert exhausted == (1, 0)


def test_causal_strike_option_aborts_committed_motion_when_pass_never_arrives() -> None:
    config = G1CausalStrikeOptionConfig()
    controller = G1CausalStrikeOptionController(config)
    controller.step(_observation(config.predecessor_track_policy_frame))
    commit = controller.step(_observation(config.predecessor_commit_policy_frame))
    aborted = controller.step(_observation(config.missing_ball_abort_predecessor_policy_frame))

    assert commit.phase == CausalStrikeOptionPhase.COMMIT
    assert aborted.phase == CausalStrikeOptionPhase.ABORTED
    assert aborted.reason == "incoming_ball_deadline_missed"
    assert not controller.stable_incoming_observed


def test_causal_strike_option_rejects_schema_or_time_rewind() -> None:
    config = G1CausalStrikeOptionConfig()
    with pytest.raises(ValueError, match="SIM-only envelope"):
        replace(config, direct_joint_torque_output=True)
    controller = G1CausalStrikeOptionController(config)
    controller.step(_observation(config.predecessor_track_policy_frame))
    with pytest.raises(ValueError, match="monotonic"):
        controller.step(_observation(config.predecessor_track_policy_frame - 1))
