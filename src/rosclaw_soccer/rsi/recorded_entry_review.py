"""Recorded-control native replay, NOT encoder/decoder/planner math replay."""

from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.sonic_reference_preparation import prepare_sonic_reference
from rosclaw_soccer.providers.g1.sonic_runup import (
    ISAACLAB_TO_MUJOCO,
    G1SonicRunupController,
    _sonic_control_parameters,
)
from rosclaw_soccer.providers.g1.sonic_wrist_quarantine import quarantine_wrist_reference
from rosclaw_soccer.rsi.ball_entry_teacher import ball_entry_decision
from rosclaw_soccer.rsi.contact_reference_offset import offset_contact_reference
from rosclaw_soccer.rsi.full_task_branch import integration_hash


def validate_recording(raw: dict[str, np.ndarray], *, frames: int, nbody: int) -> None:
    shapes = dict(
        canonical_qpos=(frames, 43),
        canonical_qvel=(frames, 41),
        joint_target_rad=(frames, 29),
        pd_requested_torque_nm=(frames, 10, 29),
        torque_nm=(frames, 10, 29),
        actual_actuator_force_nm=(frames, 10, 29),
        pelvis_z_per_substep_m=(frames, 10),
        force_n=(frames, 6),
        ball_position_after_step_m=(frames, 3),
        target_body_canonical_qpos_after_substep=(frames, 10, 43),
        target_body_canonical_qvel_after_substep=(frames, 10, 41),
        ball_contact_force_by_body_n=(frames, 10, nbody),
        neural_action_isaac=(frames, 29),
        encoder_reference_poses=(frames, 10, 36),
        entry_stage=(frames,),
        entry_command=(frames, 3),
        entry_target_root_xy_m=(frames, 2),
        entry_error_m=(frames,),
    )
    if frames not in (20, 280, 300) or set(raw) != set(shapes):
        raise ValueError("complete original18-field recording required")
    for key, array in raw.items():
        dtype = (
            np.int64
            if key == "entry_stage"
            else np.float32
            if key == "neural_action_isaac"
            else np.float64
        )
        if (
            type(array) is not np.ndarray
            or array.shape != shapes[key]
            or array.dtype != dtype
            or not np.isfinite(array).all()
        ):
            raise ValueError("finite typed complete native recording required: " + key)
    if np.any(raw["ball_contact_force_by_body_n"] < 0):
        raise ValueError("nonnegative solved contact magnitudes required")


def recorded_target(raw: dict[str, np.ndarray], index: int) -> np.ndarray:
    """Check original action permutation/scale, not neural origin of the action."""
    _, _, scale = _sonic_control_parameters(1.0, (1.0,) * 29)
    target: np.ndarray = (
        G1SonicRunupController.default_angles
        + raw["neural_action_isaac"][index][ISAACLAB_TO_MUJOCO] * scale
    )
    np.testing.assert_array_equal(target, raw["joint_target_rad"][index])
    return target


def check_physical_step(body: Any, raw: dict[str, np.ndarray], frame: int, index: int) -> None:
    physical = body.step(frame, recorded_target(raw, index))
    for key, value in physical.items():
        np.testing.assert_array_equal(value[0], raw[key][index])


def replay_prefix(body: Any, raw: dict[str, np.ndarray], affordance: Any) -> None:
    validate_recording(raw, frames=20, nbody=body.model.nbody)
    if body.next_frame != 0:
        raise ValueError("new native body required for prefix review")
    for frame in range(20):
        q, v = body.canonical()
        decision = ball_entry_decision(q, v, frame=frame, affordance=affordance)
        np.testing.assert_array_equal(decision.command, raw["entry_command"][frame])
        np.testing.assert_array_equal(
            decision.target_root_xy_m, raw["entry_target_root_xy_m"][frame]
        )
        if (
            raw["entry_stage"][frame] != 0
            or raw["entry_error_m"][frame] != decision.horizontal_error_m
        ):
            raise ValueError("original measured prefix entry differs")
        check_physical_step(body, raw, frame, frame)


def replay_suffix(
    body: Any,
    raw: dict[str, np.ndarray],
    *,
    law: Any,
    offset_root_m: tuple[float, float, float],
    affordance: Any,
    source_poses: np.ndarray,
    source_evidence_hash: str,
    report: dict[str, Any],
) -> None:
    validate_recording(raw, frames=280, nbody=body.model.nbody)
    if body.next_frame != 20:
        raise ValueError("own measured prefix20 required for suffix review")
    stage, handoff, reference = 0, None, None
    goal, error = np.zeros(2), 0.0
    for frame in range(20, 300):
        index = frame - 20
        q, v = body.canonical()
        decision, command = None, (0.0, 0.0, 0.0)
        if stage in (0, 2):
            decision = ball_entry_decision(q, v, frame=frame, affordance=affordance)
            goal, error = np.asarray(decision.target_root_xy_m), decision.horizontal_error_m
            command = law.command(q, v, frame=frame, affordance=affordance)
            if decision.timed_out:
                stage = 2
        if decision is not None and decision.ready:
            handoff, stage = frame, 1
            if (
                integration_hash(body.model, body.data)
                != report["handoff_native_state_unchanged_hash"]
            ):
                raise ValueError("source full native handoff state differs")
            held, quarantine = quarantine_wrist_reference(source_poses, q[:36])
            prepared = prepare_sonic_reference(
                held[286:767],
                q[:36],
                source_frequency_hz=120,
                entry_intervals=50,
                source_evidence_hash=source_evidence_hash,
            )
            adapted, geometry = offset_contact_reference(
                body.model, prepared.reference, offset_root_m=offset_root_m
            )
            if (
                geometry != report["contact_reference_adaptation"]
                or quarantine.contract_hash != report["wrist_reference_quarantine_hash"]
                or prepared.contract_hash != report["reference_preparation_hash"]
            ):
                raise ValueError("source reference/quarantine/contact request differs")
            reference = adapted.initialize(q[:36], frames=410)
        if reference is not None:
            assert handoff is not None
            local = frame - handoff
            np.testing.assert_array_equal(
                reference[local : local + 10], raw["encoder_reference_poses"][index]
            )
        np.testing.assert_array_equal(command, raw["entry_command"][index])
        np.testing.assert_array_equal(goal, raw["entry_target_root_xy_m"][index])
        if raw["entry_stage"][index] != stage or raw["entry_error_m"][index] != error:
            raise ValueError("actual-state entry trace differs")
        check_physical_step(body, raw, frame, index)
    if handoff != report["handoff_frame"]:
        raise ValueError("original handoff time differs")
