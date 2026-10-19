"""Reintegrate a complete recorded early-impact SIM body, not its neural policy.

This separate physical check recomputes PD from recorded targets and its own
state. It does not call the producer, forecaster or causal controller. Sensor
history/NN/selection reconstruction remains a separate prerequisite; this is
never sufficient teacher qualification or hardware authorization.
"""

from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.physics.g1_pd_torque_forecast import G1PDTorqueForecaster
from rosclaw_soccer.rsi.native_first_touch_initial_state import NativeFirstTouchInitialState
from rosclaw_soccer.rsi.original_motor_component_audit import OriginalMotorComponentAudit
from rosclaw_soccer.sim.root_velocity_reference import root_velocity_world


class RecordedEarlyEpisode:
    def __init__(
        self,
        model: Any,
        *,
        world_hash: str,
        kp: np.ndarray,
        kd: np.ndarray,
        course: tuple[float, float, float],
        defaults: np.ndarray,
    ) -> None:
        checked = G1PDTorqueForecaster(
            model, expected_world_hash=world_hash, kp=kp, kd=kd, max_forecast_steps=3000
        )
        self.model, self.data = model, mujoco.MjData(model)
        self.qi, self.vi, self.ai = checked._qi, checked._vi, checked._ai
        self.ball, self.bg, self.pelvis = checked._ball, checked._bg, checked._pelvis
        self.bq, self.bv = checked._bq, checked._bv
        self.limits, self.caps = checked._limits, checked._offset_caps
        self.kp, self.kd = kp.copy(), kd.copy()
        self.initial = NativeFirstTouchInitialState(course, defaults)
        self.course = list(course)
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
        self.attempted_steps = self.integrated_steps = 0
        self.failed = self.used = False

    def verify(
        self,
        trace: dict[str, np.ndarray],
        *,
        motor_audit: OriginalMotorComponentAudit | None = None,
    ) -> dict[str, Any]:
        return self._verify(trace, motor_audit=motor_audit, postcontact=False)

    def verify_postcontact_release(
        self,
        trace: dict[str, np.ndarray],
        *,
        motor_audit: OriginalMotorComponentAudit | None = None,
    ) -> dict[str, Any]:
        """Distinct actual post-touch fields; no fabricated early-action capture.

        Allows a truncated-at-H300 final lease. The formal M0 outcome still
        judges late contact separately; legitimate censoring is not corruption.
        """
        return self._verify(trace, motor_audit=motor_audit, postcontact=True)

    def verify_postcontact_exit(
        self,
        trace: dict[str, np.ndarray],
        *,
        motor_audit: OriginalMotorComponentAudit | None = None,
    ) -> dict[str, Any]:
        """Actual exit fields; recorded search counters are NOT reexecution.

        Search causality, raw candidates, ranking and selected80ms alignment
        require their separate readback. This only reintegrates the body.
        """
        return self._verify(trace, motor_audit=motor_audit, postcontact=True, exit_profile=True)

    def _verify(
        self,
        trace: dict[str, np.ndarray],
        *,
        motor_audit: OriginalMotorComponentAudit | None,
        postcontact: bool,
        exit_profile: bool = False,
    ) -> dict[str, Any]:
        if self.used:
            raise ValueError("once-only callback-free whole-body physical replay required")
        self.used = self.failed = True
        if motor_audit is not None and (
            type(motor_audit) is not OriginalMotorComponentAudit
            or motor_audit.model is not self.model
            or motor_audit.failed
            or motor_audit.pending
            or motor_audit.next_frame != 0
        ):
            raise ValueError("fresh exact same-model original motor component auditor required")
        if mujoco.get_mjcb_control() is not None:
            raise ValueError("once-only callback-free whole-body physical replay required")
        prefix = (
            "postcontact_exit_"
            if exit_profile
            else "postcontact_release_"
            if postcontact
            else "early_impact_feedback_"
        )
        recorded_forecast_steps = 0
        if postcontact:
            forbidden = (
                "early_impact_feedback_",
                "impact_feedback_",
                "postcontact_release_" if exit_profile else "postcontact_exit_",
            )
            if any(k.startswith(forbidden) for k in trace):
                raise ValueError("distinct post-contact profile without old impact fields required")
            steps = trace.get(prefix + "forecast_steps")
            if (
                not isinstance(steps, np.ndarray)
                or steps.shape != (300, 1, 1)
                or steps.dtype != np.int64
                or (not exit_profile and np.any(steps))
                or (
                    exit_profile
                    and (
                        np.any(steps < 0)
                        or np.any(steps > 560)
                        or np.any(steps % 40)
                        or steps.sum() > 560
                        or np.count_nonzero(steps) > 1
                    )
                )
            ):
                raise ValueError("post-contact control requires its exact bounded forecast profile")
            recorded_forecast_steps = int(steps.sum())
            if exit_profile:
                searched = np.flatnonzero(steps[:, 0, 0])
                if len(searched):
                    f = int(searched[0])
                    if not 30 <= f <= 294 or f // 20 != (f + 3) // 20 or steps[f, 0, 0] < 80:
                        raise ValueError("one bounded no-replan postcontact exit search required")
        expected_shapes = dict(
            canonical_qpos=(300, 1, 43),
            canonical_qvel=(300, 1, 41),
            joint_target_rad=(300, 1, 29),
            pd_requested_torque_nm=(300, 1, 10, 29),
            torque_nm=(300, 1, 10, 29),
            actual_actuator_force_nm=(300, 1, 10, 29),
            pelvis_z_per_substep_m=(300, 1, 10),
            force_n=(300, 1, 6),
            ball_position_after_step_m=(300, 1, 3),
        )
        expected_shapes[prefix + "requested_offset_nm"] = (300, 1, 29)
        expected_shapes[prefix + "applied_offset_nm"] = (300, 1, 10, 29)
        for key, shape in expected_shapes.items():
            a = trace.get(key)
            if (
                not isinstance(a, np.ndarray)
                or a.shape != shape
                or a.dtype != np.float64
                or not np.isfinite(a).all()
            ):
                raise ValueError("whole finite typed recorded body required: " + key)
        actions = trace[prefix + "requested_offset_nm"][:, 0]
        frames = np.flatnonzero(np.any(actions != 0, axis=1))
        if (
            np.any(abs(actions) > self.caps)
            or np.max(abs(trace["joint_target_rad"])) > 10
            or (
                len(frames)
                and (
                    len(frames) > 3
                    or not 30 <= frames[0] <= (299 if postcontact else 294)
                    or not np.array_equal(frames, np.arange(frames[0], frames[-1] + 1))
                    or not np.array_equal(
                        actions[frames], np.tile(actions[frames[0]], (len(frames), 1))
                    )
                    or np.any(actions[:, 12:])
                )
            )
        ):
            raise ValueError("one bounded at-most60ms original leg action required")
        if postcontact and len(frames) and not np.any(trace["force_n"][frames[0] - 1, 0, :2] > 1):
            raise ValueError("nonzero post-contact lease requires a completed foot force bin")
        if exit_profile:
            searched = np.flatnonzero(trace[prefix + "forecast_steps"][:, 0, 0])
            if (len(frames) and (not len(searched) or frames[0] != searched[0])) or (
                len(searched) and not np.any(trace["force_n"][searched[0] - 1, 0, :2] > 1)
            ):
                raise ValueError("actual exit search/action must follow the same completed contact")
        q0, v0 = trace["canonical_qpos"][0, 0], trace["canonical_qvel"][0, 0]
        self.initial.verify(self.course, q0, v0)
        d, m = self.data, self.model
        d.qpos[:7], d.qpos[self.qi], d.qpos[self.bq : self.bq + 7] = q0[:7], q0[7:36], q0[36:]
        d.qvel[:6], d.qvel[self.vi], d.qvel[self.bv : self.bv + 6] = v0[:6], v0[6:35], v0[35:]
        mujoco.mj_forward(m, d)
        warnings = tuple(int(w.number) for w in d.warning)
        errors: dict[str, float] = {}

        def equal(key: str, actual: Any, expected: Any) -> None:
            a, b = np.asarray(actual), np.asarray(expected)
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("finite aligned actual physical measurement required: " + key)
            error = float(np.max(abs(a - b)))
            if error > 1e-8:
                raise ValueError("recorded versus independently integrated body differs: " + key)
            errors[key] = max(errors.get(key, 0.0), error)

        for f in range(300):
            q = np.concatenate((d.qpos[:7], d.qpos[self.qi], d.qpos[self.bq : self.bq + 7]))
            v = np.concatenate((d.qvel[:6], d.qvel[self.vi], d.qvel[self.bv : self.bv + 6]))
            equal("canonical_qpos", q, trace["canonical_qpos"][f, 0])
            equal("canonical_qvel", v, trace["canonical_qvel"][f, 0])
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.ball, velocity, 0)
            observed = dict(
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
            for key, value in observed.items():
                if key in trace:
                    equal(key, value, trace[key][f, 0])
            forces = np.zeros(6)
            target, offset = trace["joint_target_rad"][f, 0], actions[f]
            if motor_audit is not None:
                target = motor_audit.before_frame(
                    d, f, {key: value[f, 0] for key, value in trace.items()}
                )
            for s in range(10):
                raw = (target - d.qpos[self.qi]) * self.kp - d.qvel[self.vi] * self.kd
                parent = np.clip(raw, -self.limits, self.limits)
                tau = np.clip(raw + offset, -self.limits, self.limits) if np.any(offset) else parent
                equal("pd_requested_torque_nm", raw, trace["pd_requested_torque_nm"][f, 0, s])
                equal("torque_nm", tau, trace["torque_nm"][f, 0, s])
                equal(
                    "applied_offset_nm",
                    tau - parent,
                    trace[prefix + "applied_offset_nm"][f, 0, s],
                )
                d.ctrl[self.ai] = tau
                old_time = float(d.time)
                self.attempted_steps += 1
                mujoco.mj_step(m, d)
                self.integrated_steps += 1
                if (
                    not np.isclose(d.time, old_time + 0.002, rtol=0, atol=1e-12)
                    or tuple(int(w.number) for w in d.warning) != warnings
                    or any(
                        not np.isfinite(a).all()
                        for a in (d.qpos, d.qvel, d.qacc, d.actuator_force, d.qfrc_constraint)
                    )
                ):
                    raise ValueError("independent physical integration warning/reset/nonfinite")
                equal(
                    "actual_actuator_force_nm",
                    d.actuator_force[self.ai],
                    trace["actual_actuator_force_nm"][f, 0, s],
                )
                equal(
                    "pelvis_z_per_substep_m",
                    d.xpos[self.pelvis, 2],
                    trace["pelvis_z_per_substep_m"][f, 0, s],
                )
                for n in range(d.ncon):
                    c = d.contact[n]
                    g1, g2 = int(c.geom1), int(c.geom2)
                    if self.bg in (g1, g2):
                        body = int(m.geom_bodyid[g2 if g1 == self.bg else g1])
                        if body in self.contacts:
                            wrench = np.zeros(6)
                            mujoco.mj_contactForce(m, d, n, wrench)
                            i = self.contacts.index(body)
                            forces[i] = max(forces[i], float(np.linalg.norm(wrench[:3])))
            equal("force_n", forces, trace["force_n"][f, 0])
            equal(
                "ball_position_after_step_m",
                d.xpos[self.ball],
                trace["ball_position_after_step_m"][f, 0],
            )
            if motor_audit is not None:
                motor_audit.after_frame(f, forces)
        result = dict(
            schema="soccer.rsi.recorded_early_h300_physics.v1",
            actual_physical_steps=self.integrated_steps,
            maximum_absolute_errors=errors,
            recorded_targets_consumed=True,
            neural_targets_recomputed=False,
            independent_search_recomputed=False,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        if motor_audit is not None:
            result["schema"] = "soccer.rsi.recorded_early_h300_motor_physics.v1"
            result["motor_review"] = motor_audit.review()
            result["original_ik_and_recurrent_motor_recomputed"] = True
        if postcontact:
            result["schema"] = (
                "soccer.rsi.recorded_postcontact_h300_motor_physics.v1"
                if motor_audit is not None
                else "soccer.rsi.recorded_postcontact_h300_physics.v1"
            )
            result["actual_postcontact_action_fields_consumed"] = True
            result["early_action_arrays_fabricated_or_aliased"] = False
        if exit_profile:
            result["schema"] = (
                "soccer.rsi.recorded_postcontact_exit_h300_motor_physics.v1"
                if motor_audit is not None
                else "soccer.rsi.recorded_postcontact_exit_h300_physics.v1"
            )
            result["recorded_forecast_steps_not_reexecuted"] = recorded_forecast_steps
            result["independent_exit_search_qualified"] = False
        self.failed = False
        return result
