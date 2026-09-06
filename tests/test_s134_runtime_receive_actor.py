from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rosclaw_soccer.growth.causal_strike_option import (
    CausalStrikeOptionObservation,
    G1CausalStrikeOptionConfig,
    G1CausalStrikeOptionController,
)
from rosclaw_soccer.growth.runtime_receive_actor import (
    G1RuntimeReceiveActor,
    RuntimeReceiveAction,
    RuntimeReceiveMemory,
    load_runtime_receive_actor,
    runtime_receive_features,
    save_runtime_receive_actor,
)

_HASH_A = "sha256:" + "a" * 64
_HASH_B = "sha256:" + "b" * 64
_HASH_C = "sha256:" + "c" * 64
_HASH_D = "sha256:" + "d" * 64
_HASH_E = "sha256:" + "e" * 64
_HASH_F = "sha256:" + "f" * 64


def _memory(
    index: int,
    lateral: float,
    action: RuntimeReceiveAction,
    quality: float,
) -> RuntimeReceiveMemory:
    digit = format(index, "x")
    return RuntimeReceiveMemory(
        context_hash="sha256:" + digit * 64,
        trajectory_hash="sha256:" + "0" * 63 + digit,
        features=(3.5, lateral, -1.8, 0.0, 1.2, lateral, 0.0, 0.1, 190.0),
        action=action,
        quality_score=quality,
    )


def _actor() -> G1RuntimeReceiveActor:
    center = RuntimeReceiveAction(stance_offset_y_m=-0.04, foot_yaw_offset_rad=0.08)
    wide = RuntimeReceiveAction(stance_offset_y_m=-0.04, foot_yaw_offset_rad=-0.04)
    return G1RuntimeReceiveActor(
        body_hash=_HASH_A,
        kick_prior_hash=_HASH_B,
        roster_hash=_HASH_C,
        finisher_self_model_hash=_HASH_D,
        source_evidence_hashes=(_HASH_E,),
        training_snapshot_hash=_HASH_F,
        feature_center=(0.0,) * 9,
        feature_scale=(1.0,) * 9,
        successful_memories=(
            _memory(1, 0.0, center, 9.0),
            _memory(2, -0.10, wide, 9.5),
        ),
        failed_memories=(
            _memory(3, -0.10, center, 3.0),
            _memory(4, 0.0, wide, 3.0),
        ),
    )


def test_runtime_receive_features_predict_contact_pocket_lateral_state() -> None:
    features = runtime_receive_features(
        ball_local_position_m=(3.5, 0.1, 0.115),
        ball_local_velocity_mps=(-2.0, -0.2, 0.0),
        ball_arrival_eta_sec=1.25,
        pelvis_local_position_m=(0.0, -0.02, 0.78),
        joint_velocity_rad_s=(0.2,) * 29,
        policy_frame=190,
    )

    assert features == pytest.approx((3.5, 0.1, -2.0, -0.2, 1.25, -0.15, -0.02, 0.2, 190.0))


def test_runtime_receive_actor_routes_role_local_contact_geometry() -> None:
    actor = _actor()
    center = actor.decide((3.5, 0.0, -1.8, 0.0, 1.2, 0.0, 0.0, 0.1, 190.0))
    wide = actor.decide((3.5, -0.10, -1.8, 0.0, 1.2, -0.10, 0.0, 0.1, 190.0))

    assert center.accepted and center.action is not None
    assert center.action.foot_yaw_offset_rad == 0.08
    assert wide.accepted and wide.action is not None
    assert wide.action.foot_yaw_offset_rad == -0.04
    assert actor.agent_id == "red.finisher"
    assert actor.tactical_intent == "receive"
    assert actor.owned_skill == "first_touch"


def test_runtime_receive_actor_round_trip_is_content_bound(tmp_path: Path) -> None:
    path = tmp_path / "runtime-receive.json"
    actor = _actor()
    save_runtime_receive_actor(actor, path)

    assert load_runtime_receive_actor(path) == actor
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            '"maximum_support_distance": 2.75',
            '"maximum_support_distance": 2.5',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        load_runtime_receive_actor(path)


def test_runtime_receive_action_cannot_claim_torque_or_hardware_authority() -> None:
    with pytest.raises(ValueError, match="SIM-only envelope"):
        RuntimeReceiveAction(direct_joint_torque_output=True)
    with pytest.raises(ValueError, match="SIM-only envelope"):
        RuntimeReceiveAction(arrival_alignment_tolerance_sec=0.01)
    with pytest.raises(ValueError, match="SIM-only envelope"):
        RuntimeReceiveAction(stance_offset_y_m=0.13)


def test_latched_receive_law_tightens_continuous_phase_feedback() -> None:
    controller = G1CausalStrikeOptionController(
        replace(G1CausalStrikeOptionConfig(), maximum_arrival_advance_frames=18)
    )
    controller.arm_runtime_route()
    for index in range(5):
        decision = controller.step(
            CausalStrikeOptionObservation(
                timestamp_sec=index * 0.02,
                predecessor_policy_frame=166 + index,
                receiver_pelvis_height_m=0.78,
                receiver_roll_rad=0.0,
                receiver_pitch_rad=0.0,
                receiver_joint_velocity_rms_rad_s=0.1,
                receiver_ball_local_x_m=1.28,
                receiver_ball_local_vx_mps=-1.0,
            )
        )
    assert decision.begin_bridge
    controller.select_arrival_route(18, arrival_alignment_tolerance_sec=0.02)

    repeat, correction = controller.align_repeat_count(policy_frame=248, nominal_repeat=1)

    assert repeat == 2
    assert correction == 1
