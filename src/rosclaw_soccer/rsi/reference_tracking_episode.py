"""Actual own-proprioception SONIC reference tracking in the fixed SIM H300 body.

No dataset, scene mutation, source filtering, task labels or hardware authority.
The caller pins assets/source, admits reference provenance and owns artifacts.
"""

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
from rosclaw_soccer.providers.g1.sonic_navigation import G1SonicNavigation, SonicNavigationConfig
from rosclaw_soccer.providers.g1.sonic_pose_reference import SonicPoseReference
from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation


def run_reference_tracking_episode(
    model_root: Path,
    world: Path,
    *,
    world_hash: str,
    reference: SonicPoseReference,
    course: tuple[float, float, float] = (10.0, 0.0, 0.0),
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Exactly3000 NEW native steps for this one fresh body.

    Every frame's29-joint target is recomputed from this body's measurement.
    The initial history is explicit measured cold-start padding, not fabricated
    observed past motion. This helper does not independently certify its policy.
    """
    reference.__post_init__()
    defaults = G1SonicRunupController.default_angles.copy()
    expected_entry = np.r_[0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0, defaults]
    planned = reference.initialize(expected_entry, frames=410)[:300]
    nav = G1SonicNavigation(
        model_root,
        "tracker",
        SonicNavigationConfig(
            maximum_frames=300,
            model_variant="low_latency",
            pose_reference=reference,
            inference_threads=1,
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
    )
    if not np.array_equal(body.canonical()[0][:36], expected_entry):
        raise ValueError("native reference entry differs from measured body")
    traces, actions = [], []
    for frame in range(300):
        q, v = body.canonical()
        proposal = nav.propose(
            TeamMotorObservation(
                agent_id="tracker",
                frame=frame,
                time_sec=float(body.data.time),
                intent="other",
                prospective_owner=False,
                qpos=tuple(float(x) for x in q),
                qvel=tuple(float(x) for x in v),
                target_position_m=(0.0, 0.0, 0.793),
                navigation_command=(0.0, 0.0, 0.0),
            )
        )
        if not (np.array_equal(proposal.kp, body.kp) and np.array_equal(proposal.kd, body.kd)):
            raise ValueError("native PD gains changed")
        traces.append(body.step(frame, np.asarray(proposal.target_rad, dtype=np.float64)))
        actions.append(nav.backend.action.copy())
    arrays = {key: np.concatenate([t[key] for t in traces]) for key in traces[0]}
    arrays["neural_action_isaac"] = np.asarray(actions)
    arrays["planned_reference"] = planned
    report = dict(
        body=body.review(),
        course=list(course),
        reference_hash=reference.contract_hash,
        navigation_contract=nav.contract_hash,
        foundation=nav.backend.qualification.qualification_hash,
        planner_calls=nav.backend.planner_calls,
        joint_reference_rmse_rad=float(
            np.sqrt(np.mean((arrays["canonical_qpos"][:, 7:36] - planned[:, 7:]) ** 2))
        ),
        minimum_native_pelvis_z_m=float(arrays["pelvis_z_per_substep_m"].min()),
        final_root_position_m=body.canonical()[0][:3].tolist(),
        neural_policy_recomputed_on_own_state=True,
        optimizer_steps=0,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    return arrays, report
