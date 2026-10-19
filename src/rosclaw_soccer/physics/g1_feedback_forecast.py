"""40ms owned physics with a recomputed full next-frame feedback target.

The first correction lasts20ms only. Foundation, original IK/contact release
and recurrent residual then respond to the branch's actual body and contact
state. No future trace, state write to the donor, or hardware control occurs.
"""

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.physics.contact_wrench import read_geom_contact_wrenches
from rosclaw_soccer.physics.g1_pd_torque_forecast import G1PDTorqueForecaster
from rosclaw_soccer.rsi.feedback_branch_control import G1FeedbackBranchControl
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes


@dataclass(frozen=True)
class G1FeedbackForecast:
    donor_integration_state_hash: str
    end_integration_state: np.ndarray
    arrays: dict[str, np.ndarray]
    next_target_record: dict[str, np.ndarray]


class G1FeedbackForecaster:
    def __init__(
        self,
        model: Any,
        *,
        world_hash: str,
        kp: np.ndarray,
        kd: np.ndarray,
        max_forecast_steps: int,
    ) -> None:
        if (
            type(max_forecast_steps) is not int
            or not 20 <= max_forecast_steps <= 600000
            or max_forecast_steps % 20
        ):
            raise ValueError("whole40ms bounded feedback forecast budget required")
        # Reuse exact model/gain/callback admission, without calling its physics.
        checked = G1PDTorqueForecaster(
            model, expected_world_hash=world_hash, kp=kp, kd=kd, max_forecast_steps=20
        )
        self.model, self._scratch = model, mujoco.MjData(model)
        self._qi, self._vi, self._ai = checked._qi.copy(), checked._vi.copy(), checked._ai.copy()
        self._bq, self._bv = checked._bq, checked._bv
        self._ball, self._bg, self._pelvis = checked._ball, checked._bg, checked._pelvis
        self._support = checked._support
        self.kp, self.kd = kp.copy(), kd.copy()
        self._limits = np.asarray(G1_HARD_TORQUE_LIMITS)
        self._caps = np.minimum(12.0, 0.2 * self._limits)
        self._contact_bodies = [
            int(model.body(n).id)
            for n in (
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_ankle_pitch_link",
                "right_ankle_pitch_link",
                "left_knee_link",
                "right_knee_link",
            )
        ]
        self.max_forecast_steps = max_forecast_steps
        self.attempted_steps = self.integrated_steps = self.forecast_attempts = 0
        self.next_target_attempts = self.next_target_completed = 0

    def forecast(
        self,
        donor: Any,
        current_target_rad: np.ndarray,
        *,
        control: G1FeedbackBranchControl,
        first_offset_nm: np.ndarray,
    ) -> G1FeedbackForecast:
        if (
            not isinstance(donor, mujoco.MjData)
            or donor.model is not self.model
            or donor is self._scratch
            or mujoco.get_mjcb_control() is not None
            or type(control) is not G1FeedbackBranchControl
            or control.model is not self.model
            or control.consumed
            or not np.isclose(donor.time, control.frame * 0.02, rtol=0, atol=1e-8)
        ):
            raise ValueError(
                "caller-owned current simulation state and unused matching feedback branch required"
            )
        for a in (current_target_rad, first_offset_nm):
            if (
                not isinstance(a, np.ndarray)
                or a.shape != (29,)
                or a.dtype != np.float64
                or not np.isfinite(a).all()
            ):
                raise ValueError("finite canonical target and bounded impulse action required")
        if np.max(abs(current_target_rad)) > 10 or np.any(abs(first_offset_nm) > self._caps):
            raise ValueError("bounded current held target and20ms action required")
        if self.attempted_steps + 20 > self.max_forecast_steps:
            raise RuntimeError("feedback physical budget exhausted; no implicit extension")
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        before = np.empty(mujoco.mj_stateSize(self.model, spec))
        mujoco.mj_getState(self.model, donor, before, spec)
        if not np.isfinite(before).all():
            raise ValueError("finite whole donor state required")
        # Full copy preserves the actual last-solver cached body/velocity clock.
        d = self._scratch
        mujoco.mj_copyData(d, self.model, donor)
        self.forecast_attempts += 1
        values: dict[str, list[np.ndarray]] = {
            name: []
            for name in (
                "canonical_qpos_after_substep",
                "canonical_qvel_after_substep",
                "requested_pd_torque_nm",
                "applied_control_nm",
                "actual_actuator_force_nm",
                "pelvis_height_solver_epoch_m",
                "robot_force_on_ball_world_n",
                "foot_contact_peak_force_n_after_substep",
                "nonfoot_robot_contact_over1n_after_substep",
                "completed_foot_ground_normal_n",
            )
        }
        peaks = np.zeros(6)
        target = current_target_rad.copy()
        record = None
        warnings = tuple(int(w.number) for w in d.warning)
        try:
            for step in range(20):
                if step == 10:
                    self.next_target_attempts += 1
                    target, record = control.next_target(d, peaks.copy())
                    self.next_target_completed += 1
                raw = (target - d.qpos[self._qi]) * self.kp - d.qvel[self._vi] * self.kd
                requested = raw + first_offset_nm if step < 10 and np.any(first_offset_nm) else raw
                tau = np.clip(requested, -self._limits, self._limits)
                d.ctrl[self._ai] = tau
                self.attempted_steps += 1
                old_time = float(d.time)
                mujoco.mj_step(self.model, d)
                self.integrated_steps += 1
                if (
                    not np.isclose(d.time, old_time + 0.002, rtol=0, atol=1e-12)
                    or tuple(int(w.number) for w in d.warning) != warnings
                    or any(
                        not np.isfinite(a).all()
                        for a in (d.qpos, d.qvel, d.qacc, d.actuator_force, d.qfrc_constraint)
                    )
                ):
                    raise ValueError("numerical warning/reset/nonfinite feedback dynamics")
                force = np.zeros(3)
                foot_force = np.zeros(2)
                nonfoot = False
                for wrench in read_geom_contact_wrenches(self.model, d, geom_id=self._bg):
                    other_body = int(self.model.geom_bodyid[wrench.other_geom_id])
                    magnitude = float(np.linalg.norm(wrench.force_world_n))
                    if other_body != 0:
                        force += np.asarray(wrench.force_world_n)
                        if other_body in self._contact_bodies[:2]:
                            foot_side = self._contact_bodies[:2].index(other_body)
                            foot_force[foot_side] = max(foot_force[foot_side], magnitude)
                        elif magnitude > 1:
                            nonfoot = True
                if step < 10:
                    for c in range(d.ncon):
                        g1, g2 = int(d.contact[c].geom1), int(d.contact[c].geom2)
                        if self._bg not in (g1, g2):
                            continue
                        body = int(self.model.geom_bodyid[g2 if g1 == self._bg else g1])
                        if body in self._contact_bodies:
                            local_wrench = np.zeros(6)
                            mujoco.mj_contactForce(self.model, d, c, local_wrench)
                            i = self._contact_bodies.index(body)
                            peaks[i] = max(peaks[i], float(np.linalg.norm(local_wrench[:3])))
                for name, value in dict(
                    canonical_qpos_after_substep=np.concatenate(
                        (d.qpos[:7], d.qpos[self._qi], d.qpos[self._bq : self._bq + 7])
                    ),
                    canonical_qvel_after_substep=np.concatenate(
                        (d.qvel[:6], d.qvel[self._vi], d.qvel[self._bv : self._bv + 6])
                    ),
                    requested_pd_torque_nm=raw,
                    applied_control_nm=tau,
                    actual_actuator_force_nm=d.actuator_force[self._ai],
                    pelvis_height_solver_epoch_m=float(d.xpos[self._pelvis, 2]),
                    robot_force_on_ball_world_n=force,
                    foot_contact_peak_force_n_after_substep=foot_force,
                    nonfoot_robot_contact_over1n_after_substep=nonfoot,
                    completed_foot_ground_normal_n=self._support.read(d),
                ).items():
                    values[name].append(np.asarray(value).copy())
            end = np.empty_like(before)
            mujoco.mj_getState(self.model, d, end, spec)
            assert record is not None
            arrays = {k: np.asarray(v) for k, v in values.items()}
            arrays["first_control_peak_force_n"] = peaks
            return G1FeedbackForecast(hash_bytes(before.tobytes()), end, arrays, record)
        finally:
            check = np.empty_like(before)
            mujoco.mj_getState(self.model, donor, check, spec)
            if check.tobytes() != before.tobytes():
                raise ValueError("feedback forecast mutated donor state")
