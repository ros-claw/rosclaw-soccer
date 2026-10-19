"""Own causal IK/residual history for an offline independently integrated body.

Uses the original shared IK and compiled residual algorithms, not a different
numerical implementation. The caller must construct the complete independent
reference decoder and authenticate weights/physics/Foundation evidence. No
producer controller, future body prefix, action search or hardware is called.
"""

from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.contact_motor_primitive import CAP_RAD
from rosclaw_soccer.rsi.feedback_branch_control import BODY_SHAPES
from rosclaw_soccer.rsi.foot_heading_control import FootHeadingContactProbe
from rosclaw_soccer.rsi.foot_heading_evidence import FootHeadingCpuReconstruction
from rosclaw_soccer.rsi.original_ik_planner_pulse import OriginalIkPlannerPulse
from rosclaw_soccer.rsi.recurrent_sampling_motor import CompiledRecurrentSamplingMotor
from rosclaw_soccer.sim.root_velocity_reference import root_velocity_world


class OriginalMotorComponentAudit:
    def __init__(
        self,
        model: Any,
        *,
        policy: dict[str, Any],
        decoder: CompiledRecurrentSamplingMotor,
        heading: FootHeadingContactProbe,
        profile: BoundedFirstTouchControl,
    ) -> None:
        if (
            not isinstance(model, mujoco.MjModel)
            or (model.nq, model.nv, model.nu) != (43, 41, 29)
            or type(decoder) is not CompiledRecurrentSamplingMotor
            or type(heading) is not FootHeadingContactProbe
            or type(profile) is not BoundedFirstTouchControl
            or type(policy) is not dict
            or policy.get("policy_hash") != decoder._policy_hash
            or decoder._memory.last_frame != -1
            or decoder._recurrent.next_index != 0
            or decoder._active_frame is not None
        ):
            raise ValueError("fresh independently constructed original reference decoder required")
        heading.__post_init__()
        profile.__post_init__()
        self.profile, self.pulse = profile, OriginalIkPlannerPulse(0.08)
        self.model, self.policy, self.decoder = model, policy, decoder
        self.ik = FootHeadingCpuReconstruction(model, heading)
        joints = [int(model.joint(n).id) for n in G1_DDS_JOINT_NAMES]
        self.qi = [int(model.jnt_qposadr[j]) for j in joints]
        self.vi = [int(model.jnt_dofadr[j]) for j in joints]
        self.limits = model.jnt_range[joints].copy()
        self.pelvis, self.ball = int(model.body("pelvis").id), int(model.body("ball").id)
        self.geometry = [
            int(model.body(n).id)
            for n in (
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_knee_link",
                "right_knee_link",
            )
        ]
        self.body: dict[str, list[np.ndarray]] = {k: [] for k in BODY_SHAPES}
        self.previous, self.previous_forces = np.zeros(12), np.zeros(6)
        self.next_frame = 0
        self.pending = self.failed = False
        self.errors: dict[str, float] = {}

    def _compare(self, name: str, actual: np.ndarray, expected: dict[str, np.ndarray]) -> None:
        e = expected.get(name)
        if (
            not isinstance(e, np.ndarray)
            or e.shape != actual.shape
            or e.dtype != actual.dtype
            or not np.isfinite(e).all()
            or not np.isfinite(actual).all()
        ):
            raise ValueError("whole finite same-frame motor capture required: " + name)
        error = (
            float(np.any(actual != e))
            if actual.dtype == np.bool_
            else float(np.max(abs(actual - e)))
        )
        if error > 1e-8:
            raise ValueError("independently reconstructed current motor differs: " + name)
        self.errors[name] = max(self.errors.get(name, 0.0), error)

    def before_frame(self, data: Any, frame: int, expected: dict[str, np.ndarray]) -> np.ndarray:
        if (
            self.failed
            or self.pending
            or type(frame) is not int
            or frame != self.next_frame
            or not 0 <= frame < 300
            or not isinstance(data, mujoco.MjData)
            or data.model is not self.model
            or not np.isclose(data.time, frame * 0.02, rtol=0, atol=1e-8)
        ):
            self.failed = True
            raise ValueError("contiguous own current-body motor reconstruction required; no retry")
        self.failed = True
        m, d = self.model, data
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.ball, velocity, 0)
        current = dict(
            joint_position_rad=d.qpos[self.qi],
            joint_velocity_rad_s=d.qvel[self.vi],
            root_pose_xyzw_m=np.concatenate(
                (d.xpos[self.pelvis], d.xquat[self.pelvis][[1, 2, 3, 0]])
            ),
            root_velocity_world=root_velocity_world(m, d, self.pelvis, "body-com"),
            ball_position_before_step_m=d.xpos[self.ball],
            ball_linear_velocity_before_step_m_s=velocity[3:],
            foot_geometry_position_before_step_m=d.xpos[self.geometry],
        )
        for key, value in current.items():
            self._compare(key, value, expected)
            self.body[key].append(value.copy()[None])
        gap = d.xpos[self.ball] - d.xpos[self.pelvis]
        requested, baseline = self.profile.navigation(
            gap_x=float(gap[0]),
            gap_y=float(gap[1]),
            ball_vx=float(velocity[3]),
            before_contact=self.ik.contact is None,
        )
        residual, command = self.pulse.compose(frame, baseline)
        for key, value in (
            ("approach_requested_navigation_command", requested),
            ("navigation_command", command),
            ("ik_planner_pulse_baseline_command", baseline),
            ("ik_planner_pulse_requested_residual", residual),
            ("ik_planner_pulse_applied_command", command),
        ):
            self._compare(key, np.asarray(value, dtype=np.float64), expected)
        nav = expected.get("foundation_navigation_joint_target_rad")
        if (
            not isinstance(nav, np.ndarray)
            or nav.shape != (29,)
            or nav.dtype != np.float64
            or not np.isfinite(nav).all()
        ):
            raise ValueError(
                "current Foundation target from independently checked component required"
            )
        increment = self.ik.before_step(d, frame, d.xpos, nav)
        if self.ik.last_record is None:
            raise ValueError("whole own original IK/heading capture required")
        for key, value in self.ik.last_record.items():
            self._compare(key, value, expected)
        nominal = nav.copy()
        if self.ik.side >= 0:
            ids = slice(self.ik.side * 6, self.ik.side * 6 + 6)
            nominal[ids] += increment[ids]
        self._compare("taskspace_joint_increment_rad", nominal - nav, expected)
        self._compare("pre_motor_joint_target_rad", nominal, expected)
        # Only this independently measured past/current prefix reaches the
        # compiled residual. No full trace or future body array is supplied.
        observed = {k: np.asarray(v, dtype=np.float64) for k, v in self.body.items()}
        delta = self.decoder.delta_at_frame(
            self.policy,
            observed,
            frame=frame,
            nominal_target=nominal,
            baseline=nominal[:12],
            limits=self.limits[:12],
            previous=self.previous,
            previous_contact_forces=self.previous_forces,
        )
        self._compare("motor_delta_rad", delta, expected)
        self._compare("recurrent_residual_hidden_state", self.decoder.hidden_state, expected)
        for key, value in self.decoder.sampled_transition.items():
            self._compare(key, value, expected)
        pre_projection = (
            np.zeros(12)
            if frame < 30
            else CAP_RAD
            * np.tanh(self.decoder.sampled_transition["recurrent_sampling_latent_action"])
        )
        self._compare("motor_pre_projection_requested_rad", pre_projection, expected)
        target = nominal.copy()
        target[:12] += delta
        self._compare("joint_target_rad", target, expected)
        self.previous = delta.copy()
        self.pending = True
        return target

    def after_frame(self, frame: int, measured_forces: np.ndarray) -> None:
        pending = self.pending
        self.pending = False
        self.failed = True
        if (
            not pending
            or type(frame) is not int
            or frame != self.next_frame
            or not isinstance(measured_forces, np.ndarray)
            or measured_forces.shape != (6,)
            or measured_forces.dtype != np.float64
            or not np.isfinite(measured_forces).all()
            or np.any(measured_forces < 0)
        ):
            raise ValueError("one completed own-physics force observation required; no retry")
        self.ik.after_step(frame, measured_forces)
        self.previous_forces = measured_forces.copy()
        self.next_frame += 1
        self.failed = False

    def review(self) -> dict[str, Any]:
        if (
            self.failed
            or self.pending
            or self.next_frame != 300
            or self.decoder._recurrent.next_index != 270
        ):
            raise ValueError("whole300 current-body motor decisions required")
        return dict(
            schema="soccer.rsi.original_motor_component_audit.v1",
            motor_decisions=300,
            recurrent_updates=270,
            maximum_absolute_errors=self.errors,
            body_history_reconstructed_from_own_physics=True,
            future_body_prefix_consumed=False,
            foundation_targets_consumed_from_recording=True,
            navigation_command_reconstructed_from_own_physics=True,
            shared_ik_and_residual_algorithms=True,
            independent_decoder_construction_required_from_caller=True,
            action_search_reconstructed=False,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
