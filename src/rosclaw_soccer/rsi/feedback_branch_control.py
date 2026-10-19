"""One next-frame SIM target from owned Foundation, IK and residual histories.

No future recorded target/state is accepted. This restricted branch cannot
cross a planner refresh, and is not model qualification or a live controller.
"""

import copy
from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.providers.g1.sonic_branch_state import fork_sonic_tracker
from rosclaw_soccer.providers.g1.sonic_runup import (
    ISAACLAB_TO_MUJOCO,
    G1SonicRunupController,
    _sonic_control_parameters,
)
from rosclaw_soccer.rsi.fixed_recurrent_reference_audit import _numeric_graph_hash
from rosclaw_soccer.rsi.foot_heading_evidence import FootHeadingCpuReconstruction
from rosclaw_soccer.rsi.recurrent_sampling_motor import CompiledRecurrentSamplingMotor
from rosclaw_soccer.sim.root_velocity_reference import root_velocity_world

BODY_SHAPES = {
    "joint_position_rad": (29,),
    "joint_velocity_rad_s": (29,),
    "root_pose_xyzw_m": (7,),
    "root_velocity_world": (6,),
    "ball_position_before_step_m": (3,),
    "ball_linear_velocity_before_step_m_s": (3,),
    "foot_geometry_position_before_step_m": (4, 3),
}


def fork_motor(decoder: Any, *, next_frame: int) -> CompiledRecurrentSamplingMotor:
    if (
        type(decoder) is not CompiledRecurrentSamplingMotor
        or type(next_frame) is not int
        or not 31 <= next_frame < 300
        or decoder._active_frame is not None
        or decoder._memory.last_frame != next_frame - 1
        or decoder._recurrent.next_index != next_frame - 30
    ):
        raise ValueError("completed current recurrent decision and exact next-frame clock required")
    before = _numeric_graph_hash(decoder)  # Also enforces a bounded allowlisted numeric graph.
    owned = copy.deepcopy(decoder)
    if _numeric_graph_hash(owned) != before or _numeric_graph_hash(decoder) != before:
        raise ValueError("whole owned residual branch state differs")
    return owned


def fork_ik(ik: Any, *, frame: int) -> FootHeadingCpuReconstruction:
    if (
        type(ik) is not FootHeadingCpuReconstruction
        or ik.next_frame != frame
        or ik.pending is None
        or ik.pending[0] != frame
        or set(vars(ik))
        != {
            "model",
            "feet",
            "ball",
            "vi",
            "limits",
            "side",
            "contact",
            "next_frame",
            "contact_delta",
            "pending",
            "probe",
            "last_record",
        }
    ):
        raise ValueError("reviewed current original-IK/heading pending boundary required")
    owned = object.__new__(FootHeadingCpuReconstruction)
    for key, value in vars(ik).items():
        setattr(owned, key, value if key == "model" else copy.deepcopy(value))
    return owned


class G1FeedbackBranchControl:
    """Owned next-target stepper, restricted to old cached/body-COM observation."""

    def __init__(
        self,
        model: Any,
        *,
        frame: int,
        tracker: Any,
        decoder: Any,
        ik: Any,
        policy: dict[str, Any],
        body_prefix: dict[str, np.ndarray],
        previous_motor_delta_rad: np.ndarray,
    ) -> None:
        if (
            type(frame) is not int
            or not 30 <= frame < 298
            or (frame + 1) % 20 == 0
            or not isinstance(model, mujoco.MjModel)
            or (model.nq, model.nv, model.nu) != (43, 41, 29)
            or tracker._next_frame != frame + 1
            or ik.model is not model
            or policy.get("policy_hash") != decoder._policy_hash
            or set(body_prefix) != set(BODY_SHAPES)
        ):
            raise ValueError(
                "same qualified causal pending controller without next-frame replan required"
            )
        for key, shape in BODY_SHAPES.items():
            a = body_prefix[key]
            if (
                not isinstance(a, np.ndarray)
                or a.shape != (frame + 1, 1, *shape)
                or a.dtype != np.float64
                or not np.isfinite(a).all()
            ):
                raise ValueError("exact current/past-only finite native body prefix required")
        if (
            not isinstance(previous_motor_delta_rad, np.ndarray)
            or previous_motor_delta_rad.shape != (12,)
            or previous_motor_delta_rad.dtype != np.float64
            or not np.isfinite(previous_motor_delta_rad).all()
            or np.max(abs(previous_motor_delta_rad)) > 0.16 + 1e-8
        ):
            raise ValueError("current bounded executed residual required")
        self.model, self.frame = model, frame
        self.tracker = fork_sonic_tracker(tracker)
        self.decoder = fork_motor(decoder, next_frame=frame + 1)
        self.ik = fork_ik(ik, frame=frame)
        # Compiled decoder checks this fixed identity; never mutate the shared policy.
        self.policy = policy
        self.body = {k: v.copy() for k, v in body_prefix.items()}
        self.previous = previous_motor_delta_rad.copy()
        joints = [int(model.joint(n).id) for n in G1_DDS_JOINT_NAMES]
        self.qi, self.vi = (
            [int(model.jnt_qposadr[j]) for j in joints],
            [int(model.jnt_dofadr[j]) for j in joints],
        )
        self.limits = model.jnt_range[joints].copy()
        self.pelvis, self.ball = int(model.body("pelvis").id), int(model.body("ball").id)
        bj = int(model.body_jntadr[self.ball])
        self.bq, self.bv = int(model.jnt_qposadr[bj]), int(model.jnt_dofadr[bj])
        self.geometry = [
            int(model.body(n).id)
            for n in (
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_knee_link",
                "right_knee_link",
            )
        ]
        self.consumed = False

    def next_target(
        self, data: Any, completed_force_n: np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if (
            self.consumed
            or not isinstance(data, mujoco.MjData)
            or data.model is not self.model
            or not np.isclose(data.time, (self.frame + 1) * 0.02, rtol=0, atol=1e-8)
            or not isinstance(completed_force_n, np.ndarray)
            or completed_force_n.shape != (6,)
            or completed_force_n.dtype != np.float64
            or not np.isfinite(completed_force_n).all()
            or np.any(completed_force_n < 0)
        ):
            raise ValueError("one complete next-frame simulation observation required")
        self.consumed = True  # A failed target reconstruction is terminal too.
        self.ik.after_step(self.frame, completed_force_n)
        q = np.concatenate((data.qpos[:7], data.qpos[self.qi], data.qpos[self.bq : self.bq + 7]))
        v = np.concatenate((data.qvel[:6], data.qvel[self.vi], data.qvel[self.bv : self.bv + 6]))
        self.tracker.observe(q[None], v[None])
        self.tracker.update(self.frame + 1, q[None], v[None])
        _, _, scale = _sonic_control_parameters(1.0, (1.0,) * 29)
        navigation = (
            np.asarray(G1SonicRunupController.default_angles)
            + self.tracker.action[0].cpu().numpy()[ISAACLAB_TO_MUJOCO] * scale
        )
        increment = self.ik.before_step(data, self.frame + 1, data.xpos, navigation)
        nominal = navigation + increment
        ball_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, data, mujoco.mjtObj.mjOBJ_BODY, self.ball, ball_velocity, 0
        )
        current = dict(
            joint_position_rad=data.qpos[self.qi],
            joint_velocity_rad_s=data.qvel[self.vi],
            root_pose_xyzw_m=np.concatenate(
                (data.xpos[self.pelvis], data.xquat[self.pelvis][[1, 2, 3, 0]])
            ),
            root_velocity_world=root_velocity_world(self.model, data, self.pelvis, "body-com"),
            ball_position_before_step_m=data.xpos[self.ball],
            ball_linear_velocity_before_step_m_s=ball_velocity[3:],
            foot_geometry_position_before_step_m=data.xpos[self.geometry],
        )
        for key, value in current.items():
            self.body[key] = np.concatenate((self.body[key], value[None, None].copy()))
        residual = self.decoder.delta_at_frame(
            self.policy,
            self.body,
            frame=self.frame + 1,
            nominal_target=nominal,
            baseline=nominal[:12],
            limits=self.limits[:12],
            previous=self.previous,
            previous_contact_forces=completed_force_n,
        )
        target = nominal.copy()
        target[:12] += residual
        if not np.isfinite(target).all():
            raise ValueError("nonfinite feedback target")
        return target, dict(
            navigation_target_rad=navigation,
            taskspace_increment_rad=increment,
            motor_delta_rad=residual.copy(),
            joint_target_rad=target.copy(),
            canonical_qpos=q,
            canonical_qvel=v,
        )
