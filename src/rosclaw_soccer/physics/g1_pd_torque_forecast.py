"""Owned ten-substep PD+torque forecasts for a future SIM-only contact teacher.

Not a policy, controller, actuator bridge or task-success verifier. Every
forecast advances its own copied MuJoCo state and charges a physical-step budget.
The caller's state, ball and floating root are never written or commanded.
"""

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.physics.contact_wrench import read_geom_contact_wrenches
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.completed_support import CompletedGroundSupport
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes


@dataclass(frozen=True)
class G1PDTorqueForecast:
    donor_integration_state_hash: str
    end_integration_state: np.ndarray[Any, Any]
    canonical_qpos_after_substep: np.ndarray[Any, Any]
    canonical_qvel_after_substep: np.ndarray[Any, Any]
    requested_pd_torque_nm: np.ndarray[Any, Any]
    applied_control_nm: np.ndarray[Any, Any]
    actual_actuator_force_nm: np.ndarray[Any, Any]
    pelvis_height_solver_epoch_m: np.ndarray[Any, Any]
    robot_force_on_ball_world_n: np.ndarray[Any, Any]
    completed_foot_ground_normal_n: np.ndarray[Any, Any]
    nonfoot_robot_contact_over_1n: bool


class G1PDTorqueForecaster:
    """Exact held-target PD law, independent candidate copies and explicit budget."""

    def __init__(
        self,
        model: Any,
        *,
        expected_world_hash: str,
        kp: np.ndarray[Any, Any],
        kd: np.ndarray[Any, Any],
        max_forecast_steps: int,
        coupled_stiffness_nm_per_rad: np.ndarray[Any, Any] | None = None,
    ) -> None:
        if (
            not isinstance(model, mujoco.MjModel)
            or (model.nq, model.nv, model.nu) != (43, 41, 29)
            or model.opt.timestep != 0.002
            or model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER
            or model.nplugin
            or mujoco.get_mjcb_control() is not None
        ):
            raise ValueError("canonical callback-free 2ms Euler native G1 simulation required")
        binary = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model, None, binary)
        if hash_bytes(binary.tobytes()) != expected_world_hash:
            raise ValueError("forecast must bind the exact declared compiled contact world")
        for a in (kp, kd):
            if (
                not isinstance(a, np.ndarray)
                or a.shape != (29,)
                or a.dtype != np.float64
                or not np.isfinite(a).all()
                or np.any(a < 0)
                or np.max(a) > 2000
            ):
                raise ValueError("finite explicit canonical PD gains required")
        if (
            np.any(kp <= 0)
            or type(max_forecast_steps) is not int
            or not 10 <= max_forecast_steps <= 600000
            or max_forecast_steps % 10
        ):
            raise ValueError(
                "positive stiffness and explicit bounded whole-frame forecast budget required"
            )
        self._model = model
        self._kp, self._kd = kp.copy(), kd.copy()
        self._coupled_stiffness: np.ndarray[Any, Any] | None = None
        if coupled_stiffness_nm_per_rad is not None:
            matrix = coupled_stiffness_nm_per_rad
            if (
                not isinstance(matrix, np.ndarray)
                or matrix.shape != (29, 29)
                or matrix.dtype != np.float64
                or not np.isfinite(matrix).all()
                or not np.array_equal(matrix, matrix.T)
            ):
                raise ValueError("finite symmetric explicit SIM stiffness matrix required")
            relative = matrix / np.sqrt(kp)[:, None] / np.sqrt(kp)[None, :]
            eigenvalues = np.linalg.eigvalsh(relative)
            if eigenvalues.min() < 0.5 - 1e-10 or eigenvalues.max() > 1.0 + 1e-10:
                raise ValueError("explicit matrix must contract original stiffness by at most half")
            if not np.array_equal(matrix, np.diag(kp)):
                self._coupled_stiffness = matrix.copy()
                self._coupled_stiffness.flags.writeable = False
        self._scratch = mujoco.MjData(model)
        joints = [int(model.joint(n).id) for n in G1_DDS_JOINT_NAMES]
        self._qi = [int(model.jnt_qposadr[j]) for j in joints]
        self._vi = [int(model.jnt_dofadr[j]) for j in joints]
        self._ai = []
        for joint in joints:
            ids = np.flatnonzero(
                (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                & (model.actuator_trnid[:, 0] == joint)
            )
            if len(ids) != 1:
                raise ValueError("one original actuator for every canonical hinge required")
            self._ai.append(int(ids[0]))
        if set(self._vi) != set(range(6, 35)):
            raise ValueError("only canonical actuated joints may receive torque")
        self._ball = int(model.body("ball").id)
        self._bg = int(model.geom("ball_geom").id)
        bj = int(model.body_jntadr[self._ball])
        self._bq, self._bv = int(model.jnt_qposadr[bj]), int(model.jnt_dofadr[bj])
        self._pelvis = int(model.body("pelvis").id)
        self._feet = [int(model.body(s + "_ankle_roll_link").id) for s in ("left", "right")]
        grounds = frozenset(
            int(i)
            for i in np.flatnonzero(
                (model.geom_bodyid == 0) & (model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
            )
        )
        self._support = CompletedGroundSupport(
            model,
            pelvis_body=self._pelvis,
            left_foot_geoms=frozenset(
                int(i) for i in np.flatnonzero(model.geom_bodyid == self._feet[0])
            ),
            right_foot_geoms=frozenset(
                int(i) for i in np.flatnonzero(model.geom_bodyid == self._feet[1])
            ),
            ground_geoms=grounds,
        )
        self._limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
        self._offset_caps = np.minimum(12.0, 0.2 * self._limits)
        self.max_forecast_steps = max_forecast_steps
        self.attempted_steps = 0
        self.integrated_steps = 0
        self.forecast_attempts = 0

    def forecast(
        self, data: Any, target_rad: np.ndarray[Any, Any], offset_nm: np.ndarray[Any, Any]
    ) -> G1PDTorqueForecast:
        if (
            not isinstance(data, mujoco.MjData)
            or data.model is not self._model
            or data is self._scratch
            or mujoco.get_mjcb_control() is not None
            or not np.isfinite(data.qpos).all()
            or not np.isfinite(data.qvel).all()
        ):
            raise ValueError("finite caller-owned bound simulation state and no callback required")
        for a in (target_rad, offset_nm):
            if (
                not isinstance(a, np.ndarray)
                or a.shape != (29,)
                or a.dtype != np.float64
                or not np.isfinite(a).all()
            ):
                raise ValueError("finite canonical target and torque vector required")
        if np.max(abs(target_rad)) > 10 or np.any(abs(offset_nm) > self._offset_caps):
            raise ValueError("bounded virtual target and declared research torque offset required")
        if self.attempted_steps + 10 > self.max_forecast_steps:
            raise RuntimeError("forecast physical-step budget exhausted; no implicit extension")
        model, d = self._model, self._scratch
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        donor = np.empty(mujoco.mj_stateSize(model, state_spec), dtype=np.float64)
        mujoco.mj_getState(model, data, donor, state_spec)
        if not np.isfinite(donor).all():
            raise ValueError("finite complete donor integration state required")
        mujoco.mj_copyData(d, model, data)
        self.forecast_attempts += 1
        q, v, pd, control, actuator, pelvis, ballforce, support = [], [], [], [], [], [], [], []
        nonfoot = False
        bad_warnings = tuple(
            int(w)
            for w in (
                mujoco.mjtWarning.mjWARN_BADQPOS,
                mujoco.mjtWarning.mjWARN_BADQVEL,
                mujoco.mjtWarning.mjWARN_BADQACC,
                mujoco.mjtWarning.mjWARN_BADCTRL,
            )
        )
        warning_counts = tuple(int(d.warning[w].number) for w in bad_warnings)
        for _ in range(10):
            if self._coupled_stiffness is None:
                raw = (target_rad - d.qpos[self._qi]) * self._kp - d.qvel[self._vi] * self._kd
            else:
                raw = (
                    self._coupled_stiffness @ (target_rad - d.qpos[self._qi])
                    - d.qvel[self._vi] * self._kd
                )
            # Preserve the old PD arithmetic exactly for the all-zero variant.
            requested = raw + offset_nm if np.any(offset_nm) else raw
            tau = np.clip(requested, -self._limits, self._limits)
            d.ctrl[self._ai] = tau
            self.attempted_steps += 1
            old_time = float(d.time)
            mujoco.mj_step(model, d)
            self.integrated_steps += 1
            if (
                not np.isclose(d.time, old_time + 0.002, rtol=0, atol=1e-12)
                or tuple(int(d.warning[w].number) for w in bad_warnings) != warning_counts
                or any(
                    not np.isfinite(a).all()
                    for a in (
                        d.qpos,
                        d.qvel,
                        d.qacc,
                        d.actuator_force,
                        d.qfrc_constraint,
                    )
                )
            ):
                raise ValueError("forecast numerical warning/reset or nonfinite dynamics")
            pd.append(raw.copy())
            control.append(tau.copy())
            actuator.append(d.actuator_force[self._ai].copy())
            q.append(
                np.concatenate((d.qpos[:7], d.qpos[self._qi], d.qpos[self._bq : self._bq + 7]))
            )
            v.append(
                np.concatenate((d.qvel[:6], d.qvel[self._vi], d.qvel[self._bv : self._bv + 6]))
            )
            pelvis.append(float(d.xpos[self._pelvis, 2]))
            force = np.zeros(3)
            for wrench in read_geom_contact_wrenches(model, d, geom_id=self._bg):
                body = int(model.geom_bodyid[wrench.other_geom_id])
                if body == 0:
                    continue
                force += np.asarray(wrench.force_world_n)
                if body not in self._feet and np.linalg.norm(wrench.force_world_n) > 1:
                    nonfoot = True
            ballforce.append(force)
            support.append(self._support.read(d))
        end = np.empty_like(donor)
        mujoco.mj_getState(model, d, end, state_spec)
        # Reading/forecasting may not alter any part of the donor integration state.
        check = np.empty_like(donor)
        mujoco.mj_getState(model, data, check, state_spec)
        if donor.tobytes() != check.tobytes():
            raise ValueError("forecast mutated live donor state")
        return G1PDTorqueForecast(
            hash_bytes(donor.tobytes()),
            end,
            np.asarray(q),
            np.asarray(v),
            np.asarray(pd),
            np.asarray(control),
            np.asarray(actuator),
            np.asarray(pelvis),
            np.asarray(ballforce),
            np.asarray(support),
            nonfoot,
        )
