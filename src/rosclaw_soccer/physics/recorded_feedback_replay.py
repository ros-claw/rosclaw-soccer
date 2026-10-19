"""Independent integration of a recorded40ms feedback branch, SIM_ONLY.

The next target is recorded, NOT recomputed by a neural controller here. This
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
        self, model: Any, *, world_hash: str, kp: np.ndarray, kd: np.ndarray, maximum_steps: int
    ) -> None:
        if type(maximum_steps) is not int or not 20 <= maximum_steps <= 1680 or maximum_steps % 20:
            raise ValueError("explicit at-most1680-step whole-window replay budget required")
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
        ):
            raise ValueError("original caller-owned same-world callback-free state required")
        if self.attempted_steps + 20 > self.maximum_steps:
            raise RuntimeError("recorded feedback replay budget exhausted")
        next_target = recorded["next_joint_target_rad"]
        offset = recorded["offset_nm"]
        for a in (first_target_rad, next_target, offset):
            if (
                not isinstance(a, np.ndarray)
                or a.shape != (29,)
                or a.dtype != np.float64
                or not np.isfinite(a).all()
            ):
                raise ValueError("whole finite canonical target/action required")
        if (
            np.any(abs(offset) > self.caps)
            or np.max(abs(first_target_rad)) > 10
            or np.max(abs(next_target)) > 10
        ):
            raise ValueError("bounded recorded targets and correction required")
        for key, shape, dtype in (
            ("canonical_qpos_after_substep", (20, 43), np.float64),
            ("canonical_qvel_after_substep", (20, 41), np.float64),
            ("requested_pd_torque_nm", (20, 29), np.float64),
            ("applied_control_nm", (20, 29), np.float64),
            ("actual_actuator_force_nm", (20, 29), np.float64),
            ("pelvis_height_solver_epoch_m", (20,), np.float64),
            ("robot_force_on_ball_world_n", (20, 3), np.float64),
            ("foot_contact_peak_force_n_after_substep", (20, 2), np.float64),
            ("nonfoot_robot_contact_over1n_after_substep", (20,), np.bool_),
            ("completed_foot_ground_normal_n", (20, 2), np.float64),
            ("next_canonical_qpos", (43,), np.float64),
            ("next_canonical_qvel", (41,), np.float64),
        ):
            a = recorded[key]
            if (
                not isinstance(a, np.ndarray)
                or a.shape != shape
                or a.dtype != dtype
                or not np.isfinite(a).all()
            ):
                raise ValueError("complete finite recorded branch required before physics")
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
            for step in range(20):
                if step == 10:
                    compare(
                        "next_canonical_qpos",
                        np.concatenate(
                            (d.qpos[:7], d.qpos[self.qi], d.qpos[self.bq : self.bq + 7])
                        ),
                        recorded["next_canonical_qpos"],
                    )
                    compare(
                        "next_canonical_qvel",
                        np.concatenate(
                            (d.qvel[:6], d.qvel[self.vi], d.qvel[self.bv : self.bv + 6])
                        ),
                        recorded["next_canonical_qvel"],
                    )
                target = first_target_rad if step < 10 else next_target
                raw = (target - d.qpos[self.qi]) * self.kp - d.qvel[self.vi] * self.kd
                requested = raw + offset if step < 10 and np.any(offset) else raw
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
            end = np.empty_like(initial)
            mujoco.mj_getState(self.model, d, end, spec)
            compare("end_integration_state", end, recorded["end_integration_state"])
            self.completed_windows += 1
            return dict(
                all20_substeps_physically_replayed=True,
                maximum_absolute_errors=max_errors,
                replay_end_integration_state_hash=hash_bytes(end.tobytes()),
                recorded_next_target_used=True,
                next_neural_target_independently_recomputed=False,
                full_h300_teacher_qualified=False,
                hardware_authorized=False,
                promotion_authorized=False,
            )
        finally:
            check = np.empty_like(initial)
            mujoco.mj_getState(self.model, donor, check, spec)
            if check.tobytes() != initial.tobytes():
                raise ValueError("independent replay mutated its original donor")
