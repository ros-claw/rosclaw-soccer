"""SIM-only measured-ball entry -> same-body neural reference execution.

The entry policy is a privileged deterministic teacher, not a new learned
actor. After handoff the contact reference remains open to proprioception,
not task retargeting. Misses/timeouts are ordinary full physical episodes.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
from rosclaw_soccer.providers.g1.sonic_navigation import G1SonicNavigation, SonicNavigationConfig
from rosclaw_soccer.providers.g1.sonic_pose_reference import SonicPoseReference
from rosclaw_soccer.providers.g1.sonic_reference_preparation import (
    PreparedSonicReference,
    prepare_sonic_reference,
)
from rosclaw_soccer.providers.g1.sonic_wrist_quarantine import (
    WristReferenceQuarantine,
    quarantine_wrist_reference,
)
from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance, ball_entry_decision
from rosclaw_soccer.rsi.full_task_branch import integration_hash
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation


def run_ball_entry_episode(
    model_root: Path,
    world: Path,
    *,
    world_hash: str,
    source_poses: np.ndarray,
    source_evidence_hash: str,
    source_interval: tuple[int, int],
    affordance: SkillEntryAffordance,
    course: tuple[float, float, float],
    arm: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """One fixed H300, no body resets or clock writes across option handoff."""
    if arm not in ("fixed_reference", "ball_entry"):
        raise ValueError("explicit fixed reference or ball-entry teacher arm required")
    if source_interval != (286, 767):
        raise ValueError("pilot uses the same frozen historical skill reference window")
    source = source_poses.copy()
    preparation: PreparedSonicReference | None = None

    def reference(entry: np.ndarray) -> tuple[SonicPoseReference, WristReferenceQuarantine]:
        nonlocal preparation
        held, quarantine = quarantine_wrist_reference(source, entry)
        preparation = prepare_sonic_reference(
            held[source_interval[0] : source_interval[1]],
            entry,
            source_frequency_hz=120,
            entry_intervals=50,
            source_evidence_hash=source_evidence_hash,
        )
        return preparation.reference, quarantine

    # The initial measurement is fixed by the native body contract, not a source
    # pose imposed on a simulator. Its exact match is checked after construction.
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController

    defaults = G1SonicRunupController.default_angles.copy()
    initial = np.r_[0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0, defaults]
    fixed_ref, fixed_quarantine = reference(initial)
    nav = G1SonicNavigation(
        model_root,
        "tracker",
        SonicNavigationConfig(
            maximum_frames=300,
            model_variant="low_latency",
            inference_threads=1,
            pose_reference=fixed_ref if arm == "fixed_reference" else None,
            record_default_planner_events=arm == "ball_entry",
        ),
    )
    model = mujoco.MjModel.from_binary_path(str(world))
    body = G1TargetEpisode(
        model,
        world_hash=world_hash,
        kp=nav.backend.kp.copy(),
        kd=nav.backend.kd.copy(),
        course=course,
        defaults=defaults,
        record_all_ball_contacts=True,
    )
    if not np.array_equal(body.canonical()[0][:36], initial):
        raise ValueError("native body does not match measured entry contract")
    stage = 1 if arm == "fixed_reference" else 0
    handoff = None
    handoff_frame = None
    handoff_state_hash = None
    motion_quarantine = fixed_quarantine
    source_nav = nav
    traces, actions, reference_inputs = [], [], []
    stages, commands, goals, errors = [], [], [], []
    target_xy = np.zeros(2)
    error = 0.0
    for frame in range(300):
        q, v = body.canonical()
        command = (0.0, 0.0, 0.0)
        decision = None
        if stage in (0, 2):
            decision = ball_entry_decision(q, v, frame=frame, affordance=affordance)
            command, target_xy, error = (
                decision.command,
                np.asarray(decision.target_root_xy_m),
                decision.horizontal_error_m,
            )
            if decision.timed_out:
                stage = 2
        observation = TeamMotorObservation(
            agent_id="tracker",
            frame=frame,
            time_sec=float(body.data.time),
            intent="shoot",
            prospective_owner=True,
            qpos=tuple(float(x) for x in q),
            qvel=tuple(float(x) for x in v),
            target_position_m=(float(q[36]), float(q[37]), float(q[38])),
            navigation_command=command,
        )
        if decision is not None and decision.ready:
            ref, motion_quarantine = reference(q[:36])
            destination = G1SonicNavigation(
                model_root,
                "tracker",
                SonicNavigationConfig(
                    maximum_frames=300,
                    model_variant="low_latency",
                    inference_threads=1,
                    pose_reference=ref,
                ),
            )
            before = integration_hash(model, body.data)
            handoff = nav.handoff_to_navigation(destination, observation=observation)
            if integration_hash(model, body.data) != before:
                raise ValueError("option handoff modified native integration state")
            handoff_state_hash, handoff_frame = before, frame
            nav = destination
            stage = 1
        proposal = nav.propose(observation)
        if not (np.array_equal(proposal.kp, body.kp) and np.array_equal(proposal.kd, body.kd)):
            raise ValueError("PD gains changed during task entry/reference handoff")
        local = frame - (handoff_frame if handoff_frame is not None else 0)
        reference_inputs.append(nav.backend.reference[local : local + 10].copy())
        actions.append(nav.backend.action.copy())
        stages.append(stage)
        commands.append(command)
        goals.append(target_xy.copy())
        errors.append(error)
        traces.append(body.step(frame, np.asarray(proposal.target_rad, dtype=np.float64)))
    arrays = {k: np.concatenate([t[k] for t in traces]) for k in traces[0]}
    arrays.update(
        neural_action_isaac=np.asarray(actions),
        encoder_reference_poses=np.asarray(reference_inputs),
        entry_stage=np.asarray(stages, dtype=np.int64),
        entry_command=np.asarray(commands),
        entry_target_root_xy_m=np.asarray(goals),
        entry_error_m=np.asarray(errors),
    )
    assert preparation is not None
    report = dict(
        body=body.review(),
        arm=arm,
        course=list(course),
        handoff_frame=handoff_frame,
        handoff_receipt=asdict(handoff) if handoff is not None else None,
        handoff_native_state_unchanged_hash=handoff_state_hash,
        affordance_hash=affordance.contract_hash,
        reference_preparation_hash=preparation.contract_hash,
        wrist_reference_quarantine_hash=motion_quarantine.contract_hash,
        entry_timed_out=bool(stage == 2),
        motion_reference_executed=arm == "fixed_reference" or handoff_frame is not None,
        minimum_entry_error_m=float(min(errors)) if arm == "ball_entry" else None,
        minimum_native_pelvis_z_m=float(arrays["pelvis_z_per_substep_m"].min()),
        final_root_position_m=body.canonical()[0][:3].tolist(),
        navigation_planner_events=source_nav.backend.events,
        navigation_contract=source_nav.contract_hash,
        final_option_contract=nav.contract_hash,
        neural_policy_recomputed_on_own_state=True,
        entry_teacher_is_learned=False,
        contact_controller_retargets_to_ball_after_handoff=False,
        optimizer_steps=0,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    return arrays, report
