"""Bounded80ms causal SIM branch with three recomputed feedback decisions.

An optional torque correction lasts60ms, followed by20ms of unmodified
feedback. This is a diagnostic, not a qualified teacher or hardware policy.
There is deliberately no future-trace or future-contact-frame input.
"""

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.physics.contact_wrench import read_geom_contact_wrenches
from rosclaw_soccer.physics.g1_feedback_forecast import G1FeedbackForecaster
from rosclaw_soccer.rsi.feedback_branch_control import G1FeedbackBranchControl
from rosclaw_soccer.sim.contracts import hash_bytes


@dataclass(frozen=True)
class G1EarlyFeedbackForecast:
    donor_integration_state_hash: str
    end_integration_state: np.ndarray
    arrays: dict[str, np.ndarray]
    next_target_records: tuple[dict[str, np.ndarray], ...]
    next_neural_records: tuple[dict[str, np.ndarray], ...]


class G1EarlyFeedbackForecaster:
    """Own physics and controller state; never cross a planner refresh."""

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
            or not 40 <= max_forecast_steps <= 600000
            or max_forecast_steps % 40
        ):
            raise ValueError("whole80ms bounded forecast budget required")
        # Static admission only: this object never executes the old40ms path.
        self._checked = G1FeedbackForecaster(
            model, world_hash=world_hash, kp=kp, kd=kd, max_forecast_steps=40
        )
        self.model = model
        self._scratch = mujoco.MjData(model)
        self.max_forecast_steps = max_forecast_steps
        self.attempted_steps = self.integrated_steps = self.forecast_attempts = 0
        self.next_target_attempts = self.next_target_completed = 0
        self._failed = False

    def forecast(
        self,
        donor: Any,
        current_target_rad: np.ndarray,
        *,
        control: G1FeedbackBranchControl,
        early_offset_nm: np.ndarray,
    ) -> G1EarlyFeedbackForecast:
        c = self._checked
        if (
            self._failed
            or not isinstance(donor, mujoco.MjData)
            or donor.model is not self.model
            or donor is self._scratch
            or mujoco.get_mjcb_control() is not None
            or type(control) is not G1FeedbackBranchControl
            or control.model is not self.model
            or control.consumed
            or type(control.frame) is not int
            or not 30 <= control.frame <= 294
            or control.frame // 20 != (control.frame + 3) // 20
            or not np.isclose(donor.time, control.frame * 0.02, rtol=0, atol=1e-8)
        ):
            raise ValueError("unused causal80ms branch without planner refresh required")
        for a in (current_target_rad, early_offset_nm):
            if (
                not isinstance(a, np.ndarray)
                or a.shape != (29,)
                or a.dtype != np.float64
                or not np.isfinite(a).all()
            ):
                raise ValueError("finite canonical target and bounded60ms torque required")
        if np.max(abs(current_target_rad)) > 10 or np.any(abs(early_offset_nm) > c._caps):
            raise ValueError("bounded canonical target/torque required")
        if self.attempted_steps + 40 > self.max_forecast_steps:
            raise RuntimeError("80ms physics budget exhausted; no implicit extension")
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        before = np.empty(mujoco.mj_stateSize(self.model, spec))
        mujoco.mj_getState(self.model, donor, before, spec)
        if not np.isfinite(before).all():
            raise ValueError("finite whole donor integration state required")
        d = self._scratch
        mujoco.mj_copyData(d, self.model, donor)
        self.forecast_attempts += 1
        values: dict[str, list[np.ndarray]] = {
            name: []
            for name in (
                "canonical_qpos_after_substep",
                "canonical_qvel_after_substep",
                "requested_pd_torque_nm",
                "requested_offset_nm",
                "applied_control_nm",
                "actual_actuator_force_nm",
                "pelvis_height_solver_epoch_m",
                "robot_force_on_ball_world_n",
                "foot_contact_peak_force_n_after_substep",
                "nonfoot_robot_contact_over1n_after_substep",
                "completed_foot_ground_normal_n",
                "completed_contact_peak_force_n",
            )
        }
        target = current_target_rad.copy()
        peaks = np.zeros(6)
        records = []
        neural_records = []
        warnings = tuple(int(w.number) for w in d.warning)
        try:
            for step in range(40):
                if step and step % 10 == 0:
                    self.next_target_attempts += 1
                    target, record = control.next_target(d, peaks.copy())
                    records.append({k: v.copy() for k, v in record.items()})
                    neural_records.append(
                        {
                            k: v.cpu().numpy().copy()
                            for k, v in control.tracker.neural_observation().items()
                        }
                    )
                    self.next_target_completed += 1
                    peaks.fill(0)
                    if step < 30:
                        # Each fork starts from the branch's actual updated NN,
                        # IK and residual state, NOT its original donor state.
                        control = G1FeedbackBranchControl(
                            self.model,
                            frame=control.frame + 1,
                            tracker=control.tracker,
                            decoder=control.decoder,
                            ik=control.ik,
                            policy=control.policy,
                            body_prefix=control.body,
                            previous_motor_delta_rad=record["motor_delta_rad"],
                        )
                raw = (target - d.qpos[c._qi]) * c.kp - d.qvel[c._vi] * c.kd
                offset = early_offset_nm if step < 30 else np.zeros(29)
                requested = raw + offset if np.any(offset) else raw
                tau = np.clip(requested, -c._limits, c._limits)
                d.ctrl[c._ai] = tau
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
                    raise ValueError("numerical warning/reset/nonfinite early-feedback dynamics")
                force, foot_force = np.zeros(3), np.zeros(2)
                nonfoot = False
                for wrench in read_geom_contact_wrenches(self.model, d, geom_id=c._bg):
                    body = int(self.model.geom_bodyid[wrench.other_geom_id])
                    magnitude = float(np.linalg.norm(wrench.force_world_n))
                    if body != 0:
                        force += np.asarray(wrench.force_world_n)
                        if body in c._contact_bodies[:2]:
                            side = c._contact_bodies[:2].index(body)
                            foot_force[side] = max(foot_force[side], magnitude)
                        elif magnitude > 1:
                            nonfoot = True
                for i in range(d.ncon):
                    geom_a, geom_b = int(d.contact[i].geom1), int(d.contact[i].geom2)
                    if c._bg not in (geom_a, geom_b):
                        continue
                    body = int(self.model.geom_bodyid[geom_b if geom_a == c._bg else geom_a])
                    if body in c._contact_bodies:
                        wrench6 = np.zeros(6)
                        mujoco.mj_contactForce(self.model, d, i, wrench6)
                        j = c._contact_bodies.index(body)
                        peaks[j] = max(peaks[j], float(np.linalg.norm(wrench6[:3])))
                for name, value in dict(
                    canonical_qpos_after_substep=np.concatenate(
                        (d.qpos[:7], d.qpos[c._qi], d.qpos[c._bq : c._bq + 7])
                    ),
                    canonical_qvel_after_substep=np.concatenate(
                        (d.qvel[:6], d.qvel[c._vi], d.qvel[c._bv : c._bv + 6])
                    ),
                    requested_pd_torque_nm=raw,
                    requested_offset_nm=offset,
                    applied_control_nm=tau,
                    actual_actuator_force_nm=d.actuator_force[c._ai],
                    pelvis_height_solver_epoch_m=float(d.xpos[c._pelvis, 2]),
                    robot_force_on_ball_world_n=force,
                    foot_contact_peak_force_n_after_substep=foot_force,
                    nonfoot_robot_contact_over1n_after_substep=nonfoot,
                    completed_foot_ground_normal_n=c._support.read(d),
                    completed_contact_peak_force_n=peaks,
                ).items():
                    values[name].append(np.asarray(value).copy())
            end = np.empty_like(before)
            mujoco.mj_getState(self.model, d, end, spec)
            return G1EarlyFeedbackForecast(
                hash_bytes(before.tobytes()),
                end,
                {k: np.asarray(v) for k, v in values.items()},
                tuple(records),
                tuple(neural_records),
            )
        except BaseException:
            self._failed = True  # No retry after partial integration/controller failure.
            raise
        finally:
            check = np.empty_like(before)
            mujoco.mj_getState(self.model, donor, check, spec)
            if check.tobytes() != before.tobytes():
                self._failed = True
                raise ValueError("early feedback forecast mutated donor state")
