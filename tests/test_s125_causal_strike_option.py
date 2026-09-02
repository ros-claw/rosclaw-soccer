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
    assert incoming.ball_arrival_eta_sec == pytest.approx(2.0)


def test_causal_strike_option_rejects_schema_or_time_rewind() -> None:
    config = G1CausalStrikeOptionConfig()
    with pytest.raises(ValueError, match="SIM-only envelope"):
        replace(config, direct_joint_torque_output=True)
    controller = G1CausalStrikeOptionController(config)
    controller.step(_observation(config.predecessor_track_policy_frame))
    with pytest.raises(ValueError, match="monotonic"):
        controller.step(_observation(config.predecessor_track_policy_frame - 1))
