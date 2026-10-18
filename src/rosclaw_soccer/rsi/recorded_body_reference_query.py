"""Counterfactual bounded IK on a recorded body; no dynamics or admission.

Rebuild the ORIGINAL cached Euler kinematic clock from the previous recorded
2ms pre-step pose, not fresh FK of the current 50Hz pose. Source authentication
and expert-domain qualification belong to the caller. A label is not success.
"""

import mujoco
import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.contact_reference_labels import reference_labels
from rosclaw_soccer.rsi.taskspace_swing_probe import (
    choose_swing_side,
    continuous_swing_joint_delta,
    release_joint_delta,
)
from rosclaw_soccer.sim.contracts import hash_bytes


def query_recorded_body(trace, substeps, model, *, control):
    """Return proposed same-body labels, not labels from another trajectory.

    At tick t use qpos before substep 10*t-1 for cached x/J (t>0), and
    current tick's qpos/qvel as in the original Euler-native controller.
    Teacher phase uses only contacts completed strictly before this tick.
    No future outcome/contact/teacher action or model prediction is consumed.
    """
    if (
        type(control) is not BoundedFirstTouchControl
        or (model.nq, model.nv, model.nu) != (43, 41, 29)
        or model.nplugin
        or model.opt.timestep != 0.002
        or model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER
    ):
        raise ValueError(
            "original bounded controller and unchanged canonical 2ms Euler model required"
        )
    control.__post_init__()
    shapes = {
        "canonical_qpos": (300, 1, 43),
        "canonical_qvel": (300, 1, 41),
        "foot_geometry_position_before_step_m": (300, 1, 4, 3),
        "root_pose_xyzw_m": (300, 1, 7),
        "ball_position_before_step_m": (300, 1, 3),
        "force_n": (300, 1, 6),
    }
    for name, shape in shapes.items():
        a = trace.get(name)
        if (
            type(a) is not np.ndarray
            or a.shape != shape
            or a.dtype != np.float64
            or not np.isfinite(a).all()
        ):
            raise ValueError("complete finite original measured body trace required")
    q, v, force = (trace[n] for n in ("canonical_qpos", "canonical_qvel", "force_n"))
    pose = substeps.get("canonical_qpos_before_substep")
    if (
        type(pose) is not np.ndarray
        or pose.shape != (3000, 43)
        or pose.dtype != np.float64
        or not np.isfinite(pose).all()
        or np.any(force < 0)
        or pose[::10].tobytes() != q[:, 0].tobytes()
    ):
        raise ValueError(
            "exact original 2ms/50Hz pose clocks and completed nonnegative forces required"
        )
    navigation = reference_labels(trace)["navigation_joint_target_rad"][:, 0]
    joints = [model.joint(n).id for n in G1_DDS_JOINT_NAMES]
    qi, vi = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    feet = [
        model.body(n).id
        for n in (
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_knee_link",
            "right_knee_link",
        )
    ]
    pelvis, ball = model.body("pelvis").id, model.body("ball").id
    bj = model.body_jntadr[ball]
    bq, bv = model.jnt_qposadr[bj], model.jnt_dofadr[bj]
    limits = model.jnt_range[joints]
    data = mujoco.MjData(model)  # Scratch kinematics only, never integrated.

    def assign(position, velocity):
        data.qpos[:7], data.qpos[qi], data.qpos[bq : bq + 7] = (
            position[:7],
            position[7:36],
            position[36:],
        )
        data.qvel[:6], data.qvel[vi], data.qvel[bv : bv + 6] = (
            velocity[:6],
            velocity[6:35],
            velocity[35:],
        )

    labels, sides, contact, side = [], [], None, -1
    contact_delta = np.zeros(6)
    maximum_cache_error = 0.0
    for frame in range(300):
        cached = q[0, 0] if frame == 0 else pose[frame * 10 - 1]
        assign(cached, v[frame, 0])
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        # Restore generalized state WITHOUT refreshing the native's cached
        # kinematic quantities; reproduces its actual Euler observation clock.
        assign(q[frame, 0], v[frame, 0])
        expected = np.concatenate(
            (
                trace["foot_geometry_position_before_step_m"][frame, 0].ravel(),
                trace["ball_position_before_step_m"][frame, 0],
                trace["root_pose_xyzw_m"][frame, 0, :3],
            )
        )
        actual = np.concatenate((data.xpos[feet].ravel(), data.xpos[ball], data.xpos[pelvis]))
        error = float(np.max(np.abs(expected - actual)))
        maximum_cache_error = max(maximum_cache_error, error)
        if error > 1e-12:
            raise ValueError("original cached body/ball kinematics cannot be reconstructed")
        delta = np.zeros(6)
        target = navigation[frame].copy()
        if frame >= 30:
            if contact is None:
                side = choose_swing_side(
                    data.xpos[feet[:2]],
                    data.xpos[ball],
                    side,
                    acquisition_max_gap_m=0.95,
                    revalidate_swing_side=True,
                )
            if side >= 0:
                ids = np.arange(side * 6, side * 6 + 6)
                if contact is not None:
                    delta = release_joint_delta(contact_delta, frame - contact)
                else:
                    jac = np.zeros((3, model.nv))
                    mujoco.mj_jacBody(model, data, jac, None, feet[side])
                    delta = continuous_swing_joint_delta(
                        data.xpos[feet[side]],
                        data.xpos[ball],
                        jac[:, vi[ids]],
                        target[ids],
                        limits[ids],
                        forward_cap_m=control.foot_forward_cap_m,
                        lateral_cap_m=control.foot_lateral_cap_m,
                        vertical_offset_m=control.foot_vertical_offset_m,
                    )
                target[ids] += delta
        labels.append((target - navigation[frame])[:12])
        sides.append(side)
        # This is after the query: a frame's completed contact can affect the
        # NEXT label only, even when the recorded student path was unsafe.
        if contact is None and np.any(force[frame, 0] > 1):
            contact, contact_delta = frame, delta.copy()
    labels = np.stack(labels)
    return dict(
        proposed_reference_increment_rad=labels,
        selected_side=np.array(sides, dtype=np.int64),
        diagnostic=dict(
            schema="soccer.rsi.recorded_body_reference_query.v1",
            proposed_labels_hash=hash_bytes(labels.tobytes()),
            frames_queried=300,
            kinematic_cache_frames_verified=300,
            maximum_cache_position_error_m=maximum_cache_error,
            first_recorded_contact_frame=contact,
            future_contacts_or_outcomes_used=False,
            uses_actual_recorded_body_not_another_trajectory=True,
            new_physical_steps=0,
            new_foundation_inferences=0,
            optimizer_updates=0,
            expert_rollout_on_this_body_state_verified=False,
            teacher_qualified=False,
            dataset_admission_authorized=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
