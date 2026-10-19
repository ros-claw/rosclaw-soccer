"""Separate current-body CPU rollout for the fixed positive-IK impact study.

No recorded trace, future target, source contact frame, output callback or
private course reader enters this controller. The caller may compare returned
measurements AFTER each step. Shared frozen controller algorithms are disclosed;
this is not an independent neural implementation or a teacher promotion.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.physics.g1_pd_torque_forecast import G1PDTorqueForecaster
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.providers.g1.sonic_navigation import G1SonicNavigation, SonicNavigationConfig
from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
from rosclaw_soccer.providers.g1.sonic_torch import FrozenSonicG1Torch
from rosclaw_soccer.providers.g1.sonic_vector import BatchedSonicTracker
from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.contact_motor_primitive import CAP_RAD
from rosclaw_soccer.rsi.early_impact_controller import EarlyImpactCommand, EarlyImpactController
from rosclaw_soccer.rsi.early_impact_evidence import EarlyImpactEvidence
from rosclaw_soccer.rsi.feedback_branch_control import BODY_SHAPES
from rosclaw_soccer.rsi.foot_heading_control import FootHeadingContactProbe, require_heading_request
from rosclaw_soccer.rsi.foot_heading_evidence import FootHeadingCpuReconstruction
from rosclaw_soccer.rsi.impact_feedback_teacher import ImpactFeedbackPlan, ImpactFeedbackTeacher
from rosclaw_soccer.rsi.native_first_touch_initial_state import NativeFirstTouchInitialState
from rosclaw_soccer.rsi.original_ik_planner_pulse import OriginalIkPlannerPulse
from rosclaw_soccer.rsi.recurrent_sampling_motor import CompiledRecurrentSamplingMotor
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS
from rosclaw_soccer.sim.root_velocity_reference import root_velocity_world
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation


@dataclass(frozen=True)
class CausalImpactFrame:
    frame: int
    arrays: dict[str, np.ndarray]
    impact_plan: ImpactFeedbackPlan | EarlyImpactCommand | None


class CausalImpactReplay:
    """Exactly300 frames,3000 native steps; optional bounded SIM forecast profile."""

    def __init__(
        self,
        model: Any,
        *,
        world_hash: str,
        kp: np.ndarray,
        kd: np.ndarray,
        model_root: Path,
        foundation: FrozenSonicG1Torch,
        expected_foundation_hash: str,
        policy: dict[str, Any],
        decoder: CompiledRecurrentSamplingMotor,
        expected_policy_hash: str,
        course: tuple[float, float, float],
        profile: BoundedFirstTouchControl,
        heading: FootHeadingContactProbe,
        impact_enabled: bool,
        early_impact_enabled: bool = False,
        early_evidence_root: Path | None = None,
        reuse_owned_early_updates: bool = False,
    ) -> None:
        if (
            type(impact_enabled) is not bool
            or type(early_impact_enabled) is not bool
            or (impact_enabled and early_impact_enabled)
            or (early_impact_enabled != (early_evidence_root is not None))
            or type(reuse_owned_early_updates) is not bool
            or (reuse_owned_early_updates and not early_impact_enabled)
            or type(profile) is not BoundedFirstTouchControl
            or type(heading) is not FootHeadingContactProbe
            or type(foundation) is not FrozenSonicG1Torch
            or str(foundation.device) != "cpu"
            or foundation.variant != "low_latency"
            or type(decoder) is not CompiledRecurrentSamplingMotor
            or decoder._memory.last_frame != -1
            or decoder._recurrent.next_index != 0
            or decoder._active_frame is not None
            or policy.get("policy_hash") != expected_policy_hash
            or decoder._policy_hash != expected_policy_hash
            or foundation.qualification.qualification_hash != expected_foundation_hash
        ):
            raise ValueError(
                "same frozen CPU Foundation, owned original residual and fixed study required"
            )
        profile.__post_init__()
        heading.__post_init__()
        require_heading_request(
            heading,
            pulse=OriginalIkPlannerPulse(0.08),
            reference_model=None,
            profile=profile,
            record_planner_events=True,
        )
        foundation.qualification.require_eligible()
        checked = G1PDTorqueForecaster(
            model, expected_world_hash=world_hash, kp=kp, kd=kd, max_forecast_steps=20
        )
        self.model, self.data = model, mujoco.MjData(model)
        self.qi, self.vi, self.ai = checked._qi, checked._vi, checked._ai
        self.bq, self.bv, self.ball = checked._bq, checked._bv, checked._ball
        self.bg, self.pelvis = checked._bg, checked._pelvis
        self.geometry = [
            int(model.body(n).id)
            for n in (
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_knee_link",
                "right_knee_link",
            )
        ]
        self.contacts = (
            self.geometry[:2]
            + [int(model.body(n).id) for n in ("left_ankle_pitch_link", "right_ankle_pitch_link")]
            + self.geometry[2:]
        )
        self.limits = model.jnt_range[[int(model.joint(n).id) for n in G1_DDS_JOINT_NAMES]].copy()
        self.kp, self.kd = kp.copy(), kd.copy()
        self.torque_limits = np.asarray(G1_HARD_TORQUE_LIMITS)
        self.navigation = G1SonicNavigation(
            model_root,
            "cpu.first_touch.0",
            SonicNavigationConfig(
                maximum_frames=300,
                planner_seed=30300,
                model_variant="low_latency",
                experimental_maximum_speed_mps=1.5,
                inference_threads=1,
                record_default_planner_events=True,
            ),
        )
        self.navigation.backend.qualification.require_eligible()
        if not np.array_equal(self.navigation.backend.kp, kp) or not np.array_equal(
            self.navigation.backend.kd, kd
        ):
            raise ValueError("same original PD gains required")
        self.foundation, self.policy, self.decoder = foundation, policy, decoder
        self.profile = profile
        self.pulse = OriginalIkPlannerPulse(0.08)
        self.ik = FootHeadingCpuReconstruction(model, heading)
        self.early_impact_enabled = early_impact_enabled
        self.early_evidence = (
            EarlyImpactEvidence(early_evidence_root) if early_evidence_root is not None else None
        )
        self.teacher: EarlyImpactController | ImpactFeedbackTeacher | None
        if early_impact_enabled:
            self.teacher = EarlyImpactController(
                model,
                world_hash=world_hash,
                kp=kp,
                kd=kd,
                reuse_owned_updates=reuse_owned_early_updates,
            )
        else:
            self.teacher = (
                ImpactFeedbackTeacher(model, world_hash=world_hash, kp=kp, kd=kd)
                if impact_enabled
                else None
            )
        self.tracker: BatchedSonicTracker | None = None
        self.body: dict[str, list[np.ndarray]] = {k: [] for k in BODY_SHAPES}
        self.previous = np.zeros(12)
        self.previous_forces = np.zeros(6)
        self.next_frame = self.attempted_steps = self.integrated_steps = 0
        self.failed = False
        defaults = np.asarray(G1SonicRunupController.default_angles, dtype=np.float64)
        if not np.array_equal(defaults, self.navigation.backend.default_angles):
            raise ValueError("same independently declared initial joint defaults required")
        x, y, vx = course
        d = self.data
        d.qpos[:7] = (0, 0, 0.793, 1, 0, 0, 0)
        d.qpos[self.qi] = defaults
        d.qpos[self.bq : self.bq + 7] = (x, y, 0.13, 1, 0, 0, 0)
        d.qvel[self.bv : self.bv + 6] = (vx, 0, 0, 0, vx / 0.11, 0)
        q, v = self.canonical()
        self.initial_reference = NativeFirstTouchInitialState(course, defaults)
        self.initial_reference.verify(list(course), q, v)
        mujoco.mj_forward(model, d)
        self.warnings = tuple(int(w.number) for w in d.warning)

    def canonical(self) -> tuple[np.ndarray, np.ndarray]:
        d = self.data
        return np.concatenate(
            (d.qpos[:7], d.qpos[self.qi], d.qpos[self.bq : self.bq + 7])
        ), np.concatenate((d.qvel[:6], d.qvel[self.vi], d.qvel[self.bv : self.bv + 6]))

    def step(self, frame: int) -> CausalImpactFrame:
        if (
            type(frame) is not int
            or frame != self.next_frame
            or not 0 <= frame < 300
            or self.failed
            or self.attempted_steps + 10 > 3000
            or mujoco.get_mjcb_control() is not None
        ):
            self.failed = True
            raise ValueError(
                "one unchanged next causal frame within exact3000-step budget required"
            )
        self.failed = True  # Any error leaves this body terminal, including pre-integration errors.
        d, model = self.data, self.model
        q, v = self.canonical()
        ball_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, d, mujoco.mjtObj.mjOBJ_BODY, self.ball, ball_velocity, 0)
        current = dict(
            joint_position_rad=d.qpos[self.qi],
            joint_velocity_rad_s=d.qvel[self.vi],
            root_pose_xyzw_m=np.concatenate(
                (d.xpos[self.pelvis], d.xquat[self.pelvis][[1, 2, 3, 0]])
            ),
            root_velocity_world=root_velocity_world(model, d, self.pelvis, "body-com"),
            ball_position_before_step_m=d.xpos[self.ball],
            ball_linear_velocity_before_step_m_s=ball_velocity[3:],
            foot_geometry_position_before_step_m=d.xpos[self.geometry],
        )
        values = {k: a.copy()[None] for k, a in current.items()}
        for k, a in current.items():
            self.body[k].append(a.copy()[None])
        gap = d.xpos[self.ball] - d.xpos[self.pelvis]
        requested, baseline_command = self.profile.navigation(
            gap_x=float(gap[0]),
            gap_y=float(gap[1]),
            ball_vx=float(ball_velocity[3]),
            before_contact=self.ik.contact is None,
        )
        residual, command = self.pulse.compose(frame, baseline_command)
        values.update(
            canonical_qpos=q[None],
            canonical_qvel=v[None],
            approach_requested_navigation_command=np.asarray(requested)[None],
            navigation_command=np.asarray(command)[None],
            ik_planner_pulse_baseline_command=np.asarray(baseline_command)[None],
            ik_planner_pulse_requested_residual=np.asarray(residual)[None],
            ik_planner_pulse_applied_command=np.asarray(command)[None],
        )
        observation = TeamMotorObservation(
            agent_id=self.navigation.agent_id,
            frame=frame,
            time_sec=frame * 0.02,
            intent="other",
            prospective_owner=False,
            qpos=tuple(float(x) for x in q),
            qvel=tuple(float(x) for x in v),
            target_position_m=(0.0, 0.0, 0.0),
            navigation_command=command,
            navigation_envelope=self.navigation.navigation_envelope,
        )
        if frame == 0:
            self.navigation.start_from_observation(observation)
        self.navigation.prepare_batched_proposal(observation)
        if self.tracker is None:
            self.tracker = BatchedSonicTracker(
                self.foundation,
                self.navigation.backend.reference[None],
                low_latency_legacy_encoder_layout=True,
                capture_neural_observation=True,
            )
            self.tracker.reset(q[None], v[None])
        else:
            self.tracker.observe(q[None], v[None])
            if frame % 20 == 0:
                self.tracker.refresh_unexecuted_reference(
                    frame, self.navigation.backend.reference[None], unchanged_lookahead_frames=10
                )
        self.tracker.update(frame, q[None], v[None])
        for key, a in self.tracker.neural_observation().items():
            values["foundation_neural_" + key] = a.cpu().numpy().copy()
        proposal = self.navigation.commit_batched_action(self.tracker.action[0].numpy())
        nav = np.asarray(proposal.target_rad).copy()
        increment = self.ik.before_step(d, frame, d.xpos, nav)
        nominal = nav.copy()
        if self.ik.side >= 0:
            ids = slice(6 * self.ik.side, 6 * self.ik.side + 6)
            nominal[ids] += increment[ids]
        previous = self.decoder.delta_at_frame(
            self.policy,
            self.body,
            frame=frame,
            nominal_target=nominal,
            baseline=nominal[:12],
            limits=self.limits[:12],
            previous=self.previous,
            previous_contact_forces=self.previous_forces,
        )
        self.previous = previous.copy()
        target = nominal.copy()
        target[:12] += previous
        if self.ik.last_record is None:
            raise ValueError("complete own-state IK/heading observation required")
        for key, a in self.ik.last_record.items():
            values[key] = a.copy()[None]
        values.update(
            foundation_navigation_joint_target_rad=nav[None],
            taskspace_joint_increment_rad=(nominal - nav)[None],
            pre_motor_joint_target_rad=nominal[None],
            joint_target_rad=target[None],
            motor_delta_rad=previous[None],
            recurrent_residual_hidden_state=self.decoder.hidden_state.copy()[None],
        )
        for key, a in self.decoder.sampled_transition.items():
            values[key] = a.copy()[None]
        values["motor_pre_projection_requested_rad"] = (
            np.zeros(12)
            if frame < 30
            else CAP_RAD
            * np.tanh(self.decoder.sampled_transition["recurrent_sampling_latent_action"])
        )[None]
        plan = (
            None
            if self.teacher is None
            else self.teacher.plan(
                d,
                target,
                frame=frame,
                tracker=self.tracker,
                decoder=self.decoder,
                ik=self.ik,
                policy=self.policy,
                body_prefix={k: np.asarray(a, dtype=np.float64) for k, a in self.body.items()},
                previous_motor_delta_rad=previous,
            )
        )
        offset = np.zeros(29) if plan is None else plan.offset_nm
        if self.early_evidence is not None:
            if type(plan) is not EarlyImpactCommand:
                raise ValueError("owned early command required before recording/action")
            self.early_evidence.record(frame, plan)
        forces = np.zeros(6)
        pd, torques, actual, heights, applied = [], [], [], [], []
        for _ in range(10):
            raw = (target - d.qpos[self.qi]) * self.kp - d.qvel[self.vi] * self.kd
            parent = np.clip(raw, -self.torque_limits, self.torque_limits)
            tau = (
                np.clip(raw + offset, -self.torque_limits, self.torque_limits)
                if np.any(offset)
                else parent.copy()
            )
            d.ctrl[self.ai] = tau
            pd.append(raw.copy())
            torques.append(tau.copy())
            applied.append(tau - parent)
            old_time = float(d.time)
            self.attempted_steps += 1
            mujoco.mj_step(model, d)
            self.integrated_steps += 1
            if (
                not np.isclose(d.time, old_time + 0.002, atol=1e-12, rtol=0)
                or tuple(int(w.number) for w in d.warning) != self.warnings
                or any(
                    not np.isfinite(a).all()
                    for a in (d.qpos, d.qvel, d.qacc, d.actuator_force, d.qfrc_constraint)
                )
            ):
                raise ValueError("causal replay numerical warning/reset/nonfinite physics")
            actual.append(d.actuator_force[self.ai].copy())
            heights.append(float(d.xpos[self.pelvis, 2]))
            for i in range(d.ncon):
                contact = d.contact[i]
                g1, g2 = int(contact.geom1), int(contact.geom2)
                if self.bg not in (g1, g2):
                    continue
                body = int(model.geom_bodyid[g2 if g1 == self.bg else g1])
                if body in self.contacts:
                    w = np.zeros(6)
                    mujoco.mj_contactForce(model, d, i, w)
                    slot = self.contacts.index(body)
                    forces[slot] = max(forces[slot], float(np.linalg.norm(w[:3])))
        self.ik.after_step(frame, forces)
        self.previous_forces = forces.copy()
        values.update(
            pd_requested_torque_nm=np.asarray(pd)[None],
            torque_nm=np.asarray(torques)[None],
            actual_actuator_force_nm=np.asarray(actual)[None],
            pelvis_z_per_substep_m=np.asarray(heights)[None],
            force_n=forces[None],
            ball_position_after_step_m=d.xpos[self.ball].copy()[None],
        )
        if plan is not None:
            prefix = "early_impact_feedback_" if self.early_impact_enabled else "impact_feedback_"
            values.update(
                {
                    prefix + "requested_offset_nm": offset.copy()[None],
                    prefix + "applied_offset_nm": np.asarray(applied)[None],
                    prefix + "forecast_steps": np.asarray(
                        [[plan.report["forecast_substeps"]]], dtype=np.int64
                    ),
                }
            )
        self.next_frame += 1
        self.failed = False
        return CausalImpactFrame(frame, values, plan)

    def review(self) -> dict[str, Any]:
        if (
            self.failed
            or self.next_frame != 300
            or self.integrated_steps != 3000
            or self.attempted_steps != 3000
        ):
            raise ValueError("whole300-frame causal replay required before review")
        return dict(
            native_attempted_steps=self.attempted_steps,
            native_integrated_steps=self.integrated_steps,
            forecast_attempted_steps=0
            if self.teacher is None
            else self.teacher.forecaster.attempted_steps,
            forecast_integrated_steps=0
            if self.teacher is None
            else self.teacher.forecaster.integrated_steps,
            frames=300,
            current_body_policy_recomputed=True,
            source_future_trace_or_targets_consumed=False,
            shared_frozen_controller_algorithms=True,
            independently_implemented_neural_policy=False,
            initial_state_contract=self.initial_reference.contract(),
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
