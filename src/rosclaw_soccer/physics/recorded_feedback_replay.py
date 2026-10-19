"""Independent integration of recorded40ms or80ms feedback, SIM_ONLY.

The next targets are recorded, NOT recomputed by a neural controller here. This
checks physical/control trajectory fidelity, not causal policy qualification,
task gain, a teacher, or hardware authorization. Every attempted step is charged.
"""

from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.physics.contact_wrench import read_geom_contact_wrenches
from rosclaw_soccer.physics.g1_pd_torque_forecast import G1PDTorqueForecaster
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes


class RecordedFeedbackReplay:
    def __init__(
        self,
        model: Any,
        *,
        world_hash: str,
        kp: np.ndarray,
        kd: np.ndarray,
        maximum_steps: int,
        early80: bool = False,
    ) -> None:
        horizon = 40 if early80 else 20
        if (
            type(early80) is not bool
            or type(maximum_steps) is not int
            or not horizon <= maximum_steps <= (3360 if early80 else 1680)
            or maximum_steps % horizon
        ):
            raise ValueError("explicit bounded whole-window recorded replay budget required")
        self.early80, self.horizon = early80, horizon
        self.correction_steps = 30 if early80 else 10
        self.failed = False
        checked = G1PDTorqueForecaster(
            model, expected_world_hash=world_hash, kp=kp, kd=kd, max_forecast_steps=20
        )
        self.model, self.scratch = model, mujoco.MjData(model)
        self.qi, self.vi, self.ai = checked._qi, checked._vi, checked._ai
        self.bq, self.bv, self.bg, self.pelvis = (
            checked._bq,
            checked._bv,
            checked._bg,
            checked._pelvis,
        )
        self.support = checked._support
        self.kp, self.kd = kp.copy(), kd.copy()
        self.limits = np.asarray(G1_HARD_TORQUE_LIMITS)
        self.caps = np.minimum(12.0, 0.2 * self.limits)
        self.feet = [int(model.body(s + "_ankle_roll_link").id) for s in ("left", "right")]
        self.contacts = (
            self.feet
            + [
                int(model.body(n).id)
                for n in (
                    "left_ankle_pitch_link",
                    "right_ankle_pitch_link",
                    "left_knee_link",
                    "right_knee_link",
                )
            ]
            if early80
            else []
        )
        self.maximum_steps = maximum_steps
        self.attempted_steps = self.integrated_steps = self.attempted_windows = (
            self.completed_windows
        ) = 0

    def verify(
        self,
        donor: Any,
        *,
        first_target_rad: np.ndarray,
        recorded: dict[str, np.ndarray],
        donor_hash: str,
    ) -> dict[str, Any]:
        if (
            not isinstance(donor, mujoco.MjData)
            or donor.model is not self.model
            or donor is self.scratch
            or mujoco.get_mjcb_control() is not None
            or self.failed
        ):
            raise ValueError("original caller-owned same-world callback-free state required")
        if self.attempted_steps + self.horizon > self.maximum_steps:
            raise RuntimeError("recorded feedback replay budget exhausted")
        prefixes = [f"next_{j}_" for j in range(1, 4)] if self.early80 else ["next_"]
        targets = [first_target_rad, *(recorded[p + "joint_target_rad"] for p in prefixes)]
        if self.early80:
            schedule = recorded.get("requested_offset_nm")
            if (
                not isinstance(schedule, np.ndarray)
                or schedule.shape != (40, 29)
                or schedule.dtype != np.float64
                or not np.isfinite(schedule).all()
            ):
                raise ValueError("whole finite80ms requested action schedule required")
            offset = schedule[0]
        else:
            offset = recorded["offset_nm"]
        for a in (*targets, offset):
            if (
                not isinstance(a, np.ndarray)
                or a.shape != (29,)
                or a.dtype != np.float64
                or not np.isfinite(a).all()
            ):
                raise ValueError("whole finite canonical target/action required")
        if np.any(abs(offset) > self.caps) or any(np.max(abs(target)) > 10 for target in targets):
            raise ValueError("bounded recorded targets and correction required")
        for key, shape, dtype in (
            ("canonical_qpos_after_substep", (self.horizon, 43), np.float64),
            ("canonical_qvel_after_substep", (self.horizon, 41), np.float64),
            ("requested_pd_torque_nm", (self.horizon, 29), np.float64),
            ("applied_control_nm", (self.horizon, 29), np.float64),
            ("actual_actuator_force_nm", (self.horizon, 29), np.float64),
            ("pelvis_height_solver_epoch_m", (self.horizon,), np.float64),
            ("robot_force_on_ball_world_n", (self.horizon, 3), np.float64),
            ("foot_contact_peak_force_n_after_substep", (self.horizon, 2), np.float64),
            ("nonfoot_robot_contact_over1n_after_substep", (self.horizon,), np.bool_),
            ("completed_foot_ground_normal_n", (self.horizon, 2), np.float64),
            *[(p + "canonical_qpos", (43,), np.float64) for p in prefixes],
            *[(p + "canonical_qvel", (41,), np.float64) for p in prefixes],
        ):
            a = recorded[key]
            if (
                not isinstance(a, np.ndarray)
                or a.shape != shape
                or a.dtype != dtype
                or not np.isfinite(a).all()
            ):
                raise ValueError("complete finite recorded branch required before physics")
        if self.early80:
            schedule, peaks = (
                recorded["requested_offset_nm"],
                recorded["completed_contact_peak_force_n"],
            )
            if (
                schedule.shape != (40, 29)
                or schedule.dtype != np.float64
                or not np.isfinite(schedule).all()
                or not np.array_equal(schedule[:30], np.tile(offset, (30, 1)))
                or np.any(schedule[30:])
                or peaks.shape != (40, 6)
                or peaks.dtype != np.float64
                or not np.isfinite(peaks).all()
                or np.any(peaks < 0)
            ):
                raise ValueError("whole60ms-then-zero schedule and causal forces required")
        end_expected = recorded["end_integration_state"]
        if (
            not isinstance(end_expected, np.ndarray)
            or end_expected.shape
            != (mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION),)
            or end_expected.dtype != np.float64
            or not np.isfinite(end_expected).all()
        ):
            raise ValueError("same whole recorded end integration state required")
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        initial = np.empty(mujoco.mj_stateSize(self.model, spec))
        mujoco.mj_getState(self.model, donor, initial, spec)
        if not np.isfinite(initial).all() or hash_bytes(initial.tobytes()) != donor_hash:
            raise ValueError("exact original whole donor integration state required")
        mujoco.mj_copyData(self.scratch, self.model, donor)
        d = self.scratch
        self.attempted_windows += 1
        warnings = tuple(int(w.number) for w in d.warning)
        max_errors: dict[str, float] = {}
        contact_peaks = np.zeros(6)

        def compare(key: str, actual: Any, expected: Any) -> None:
            a, e = np.asarray(actual), np.asarray(expected)
            if (
                a.shape != e.shape
                or a.dtype != e.dtype
                or not np.isfinite(a).all()
                or not np.isfinite(e).all()
            ):
                raise ValueError("same finite typed complete recorded arrays required: " + key)
            error = float(np.any(a != e)) if a.dtype == np.bool_ else float(np.max(abs(a - e)))
            max_errors[key] = max(max_errors.get(key, 0.0), error)
            if error > 1e-8:
                raise ValueError("independent recorded feedback physics differs: " + key)

        try:
            for step in range(self.horizon):
                if step and step % 10 == 0:
                    prefix = prefixes[step // 10 - 1]
                    compare(
                        prefix + "canonical_qpos",
                        np.concatenate(
                            (d.qpos[:7], d.qpos[self.qi], d.qpos[self.bq : self.bq + 7])
                        ),
                        recorded[prefix + "canonical_qpos"],
                    )
                    compare(
                        prefix + "canonical_qvel",
                        np.concatenate(
                            (d.qvel[:6], d.qvel[self.vi], d.qvel[self.bv : self.bv + 6])
                        ),
                        recorded[prefix + "canonical_qvel"],
                    )
                    contact_peaks.fill(0)
                target = targets[step // 10]
                raw = (target - d.qpos[self.qi]) * self.kp - d.qvel[self.vi] * self.kd
                requested = raw + offset if step < self.correction_steps and np.any(offset) else raw
                tau = np.clip(requested, -self.limits, self.limits)
                d.ctrl[self.ai] = tau
                old_time = float(d.time)
                self.attempted_steps += 1
                mujoco.mj_step(self.model, d)
                self.integrated_steps += 1
                if (
                    not np.isclose(d.time, old_time + 0.002, rtol=0, atol=1e-12)
                    or tuple(int(w.number) for w in d.warning) != warnings
                    or any(
                        not np.isfinite(a).all()
                        for a in (d.qpos, d.qvel, d.qacc, d.qfrc_constraint, d.actuator_force)
                    )
                ):
                    raise ValueError(
                        "independent replay numerical warning/reset/nonfinite dynamics"
                    )
                force, foot, nonfoot = np.zeros(3), np.zeros(2), False
                for w in read_geom_contact_wrenches(self.model, d, geom_id=self.bg):
                    body = int(self.model.geom_bodyid[w.other_geom_id])
                    if body == 0:
                        continue
                    magnitude = float(np.linalg.norm(w.force_world_n))
                    force += np.asarray(w.force_world_n)
                    if body in self.feet:
                        side = self.feet.index(body)
                        foot[side] = max(foot[side], magnitude)
                    elif magnitude > 1:
                        nonfoot = True
                for key, actual in dict(
                    canonical_qpos_after_substep=np.concatenate(
                        (d.qpos[:7], d.qpos[self.qi], d.qpos[self.bq : self.bq + 7])
                    ),
                    canonical_qvel_after_substep=np.concatenate(
                        (d.qvel[:6], d.qvel[self.vi], d.qvel[self.bv : self.bv + 6])
                    ),
                    requested_pd_torque_nm=raw,
                    applied_control_nm=tau,
                    actual_actuator_force_nm=d.actuator_force[self.ai],
                    pelvis_height_solver_epoch_m=np.asarray(d.xpos[self.pelvis, 2]),
                    robot_force_on_ball_world_n=force,
                    foot_contact_peak_force_n_after_substep=foot,
                    nonfoot_robot_contact_over1n_after_substep=np.asarray(nonfoot),
                    completed_foot_ground_normal_n=self.support.read(d),
                ).items():
                    compare(key, actual, recorded[key][step])
                if self.early80:
                    for i in range(d.ncon):
                        g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
                        if self.bg not in (g1, g2):
                            continue
                        body = int(self.model.geom_bodyid[g2 if g1 == self.bg else g1])
                        if body in self.contacts:
                            wrench = np.zeros(6)
                            mujoco.mj_contactForce(self.model, d, i, wrench)
                            channel = self.contacts.index(body)
                            contact_peaks[channel] = max(
                                contact_peaks[channel], float(np.linalg.norm(wrench[:3]))
                            )
                    compare(
                        "completed_contact_peak_force_n",
                        contact_peaks,
                        recorded["completed_contact_peak_force_n"][step],
                    )
            end = np.empty_like(initial)
            mujoco.mj_getState(self.model, d, end, spec)
            compare("end_integration_state", end, recorded["end_integration_state"])
            self.completed_windows += 1
            return dict(
                **{f"all{self.horizon}_substeps_physically_replayed": True},
                maximum_absolute_errors=max_errors,
                replay_end_integration_state_hash=hash_bytes(end.tobytes()),
                recorded_next_target_used=True,
                next_neural_target_independently_recomputed=False,
                full_h300_teacher_qualified=False,
                hardware_authorized=False,
                promotion_authorized=False,
            )
        except BaseException:
            if self.early80:
                self.failed = True
            raise
        finally:
            check = np.empty_like(initial)
            mujoco.mj_getState(self.model, donor, check, spec)
            if check.tobytes() != initial.tobytes():
                raise ValueError("independent replay mutated its original donor")
