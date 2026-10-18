"""Causal bounded SIM gain shaping with per-interval actual-state witnesses.

Not learned, promoted, passive-by-proof or authorized for real execution.
Gain slew bounds are NOT actuator torque slew guarantees. Foundation and
parent actor feedback must still run in the real closed-loop experiment.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.physics.contact_wrench import read_geom_contact_wrenches
from rosclaw_soccer.physics.g1_joint_task_kinematics import G1JointTaskKinematicSampler
from rosclaw_soccer.physics.g1_pd_torque_forecast import G1PDTorqueForecaster
from rosclaw_soccer.rsi.contact_exit_torque_mpc import pressure_side
from rosclaw_soccer.rsi.directional_contact_impedance import vertical_compliance_stiffness
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def slew_stiffness(
    kp: np.ndarray, previous_relative: np.ndarray, desired_stiffness: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Convex contraction step with <=0.1 whitened spectral-norm change."""
    if (
        kp.shape != (29,)
        or kp.dtype != np.float64
        or not np.isfinite(kp).all()
        or np.any(kp < 1e-6)
        or np.any(kp > 2000)
    ):
        raise ValueError("finite positive canonical stiffness required")
    scale = np.sqrt(kp)
    for matrix in (previous_relative, desired_stiffness):
        if (
            matrix.shape != (29, 29)
            or matrix.dtype != np.float64
            or not np.isfinite(matrix).all()
            or not np.array_equal(matrix, matrix.T)
        ):
            raise ValueError("finite exactly symmetric gain state required")
    desired = desired_stiffness / scale[:, None] / scale[None, :]
    desired = (desired + desired.T) * 0.5
    identity = np.eye(29)
    unchanged = desired_stiffness == np.diag(kp)
    desired[unchanged] = identity[unchanged]
    for matrix in (previous_relative, desired):
        eig = np.linalg.eigvalsh(matrix)
        if eig.min() < 0.5 - 1e-10 or eig.max() > 1 + 1e-10:
            raise ValueError("all gain states must remain bounded contractions")
    delta = desired - previous_relative
    distance = float(np.max(abs(np.linalg.eigvalsh(delta))))
    relative = desired.copy() if distance <= 0.1 else previous_relative + (0.1 / distance) * delta
    # Canonical exact identity allows a true return to the original PD law.
    if np.array_equal(desired_stiffness, np.diag(kp)) and distance <= 0.1:
        relative = np.eye(29)
    step = float(np.max(abs(np.linalg.eigvalsh(relative - previous_relative))))
    if step > 0.1 + 1e-10:
        raise ValueError("relative gain change exceeds the declared frame bound")
    result = np.diag(kp)
    changed = relative != np.eye(29)
    weighted = (scale[:, None] * scale[None, :]) * relative
    result[changed] = weighted[changed]
    return result, relative, step


@dataclass(frozen=True)
class DirectionalCompliancePlan:
    stiffness_nm_per_rad: np.ndarray[Any, Any]
    identity: bool
    report: dict[str, Any]
    expected_end_state: np.ndarray[Any, Any] | None


class DirectionalContactCompliance:
    def __init__(
        self,
        model: Any,
        *,
        world_hash: str,
        kp: np.ndarray,
        kd: np.ndarray,
        max_forecast_steps: int = 700,
        entry_frame: int = 0,
    ) -> None:
        if (
            type(max_forecast_steps) is not int
            or not 10 <= max_forecast_steps <= 3000
            or max_forecast_steps % 10
            or type(entry_frame) is not int
            or not 0 <= entry_frame < 300
        ):
            raise ValueError("bounded explicit frame and forecast allocation required")
        # The original validator allocates no integration or policy inference.
        self._validator = G1PDTorqueForecaster(
            model, expected_world_hash=world_hash, kp=kp, kd=kd, max_forecast_steps=10
        )
        self._sampler = G1JointTaskKinematicSampler(model, expected_world_hash=world_hash)
        self._model, self._world = model, world_hash
        self._kp, self._kd = kp.copy(), kd.copy()
        self._kp.flags.writeable = self._kd.flags.writeable = False
        self._relative = np.eye(29)
        self._entry_frame = self._next_frame = entry_frame
        self._maximum_steps = max_forecast_steps
        self._reserved = self._attempted = self._integrated = self._attempts = 0
        self._bg = int(model.geom("ball_geom").id)
        self._feet = [int(model.body(s + "_ankle_roll_link").id) for s in ("left", "right")]

    def counters(self) -> dict[str, int]:
        return dict(
            frames_planned=self._next_frame - self._entry_frame,
            forecast_attempts=self._attempts,
            reserved_forecast_steps=self._reserved,
            attempted_forecast_steps=self._attempted,
            integrated_forecast_steps=self._integrated,
            maximum_forecast_steps=self._maximum_steps,
        )

    def contract(self) -> dict[str, Any]:
        import rosclaw_soccer.physics.g1_joint_task_kinematics as kinematics
        import rosclaw_soccer.physics.g1_pd_torque_forecast as forecast
        import rosclaw_soccer.rsi.contact_exit_torque_mpc as pressure
        import rosclaw_soccer.rsi.directional_contact_impedance as shaping

        value = dict(
            schema="soccer.rsi.directional_contact_compliance.v1",
            world_hash=self._world,
            source_hashes={
                k: hash_bytes(Path(p).read_bytes())
                for k, p in dict(
                    controller=__file__,
                    shaping=shaping.__file__,
                    forecast=forecast.__file__,
                    kinematics=kinematics.__file__,
                    pressure=pressure.__file__,
                ).items()
            },
            kp=self._kp.tolist(),
            kd=self._kd.tolist(),
            entry_frame=self._entry_frame,
            horizon_substeps=10,
            dt_s=0.002,
            control_dt_s=0.02,
            contraction_fraction=0.5,
            relative_stiffness_eigenvalue_bounds=[0.5, 1.0],
            maximum_relative_matrix_change_per_frame=0.1,
            maximum_contact_age_frames=60,
            forecast_budget_steps=self._maximum_steps,
            pressure_gate="three completed bins AND current solved foot force AND gap<=.01m",
            stiffness_slew_is_not_torque_slew=True,
            tangential_compliance_identity_applies_to_instantaneous_target_only=True,
            damping_changed=False,
            physical_hard_torque_limits_changed=False,
            trained_policy=False,
            teacher_qualified=False,
            passivity_proven=False,
            floating_base_stability_proven=False,
            activation_ceiling="SIM_ONLY",
            hardware_authorized=False,
            promotion_authorized=False,
        )
        return dict(value, contract_hash=hash_json(value))

    def plan(
        self,
        data: Any,
        target: np.ndarray,
        *,
        frame: int,
        first_foot_contact_frame: int | None,
        force_history: np.ndarray,
        prior_nonfoot_contact: bool,
    ) -> DirectionalCompliancePlan:
        if (
            not isinstance(data, mujoco.MjData)
            or data.model is not self._model
            or type(frame) is not int
            or frame != self._next_frame
            or frame >= 300
            or not np.isclose(data.time, frame * 0.02, atol=1e-8, rtol=0)
            or not isinstance(target, np.ndarray)
            or target.shape != (29,)
            or target.dtype != np.float64
            or not np.isfinite(target).all()
            or np.max(abs(target)) > 10
            or not isinstance(force_history, np.ndarray)
            or force_history.shape != (min(frame, 3), 6)
            or (
                first_foot_contact_frame is not None
                and (
                    type(first_foot_contact_frame) is not int
                    or not 0 <= first_foot_contact_frame < frame
                )
            )
        ):
            raise ValueError("one causal current body/target per sequential frame required")
        if first_foot_contact_frame is None and np.any(force_history[:, :2] > 1):
            raise ValueError("observed foot contact requires its causal first-foot timestamp")
        sample = self._sampler.sample(data)
        foot_forces = np.zeros(2)
        for wrench in read_geom_contact_wrenches(self._model, data, geom_id=self._bg):
            body = int(self._model.geom_bodyid[wrench.other_geom_id])
            if body in self._feet:
                j = self._feet.index(body)
                foot_forces[j] = max(foot_forces[j], float(np.linalg.norm(wrench.force_world_n)))
        side = pressure_side(
            force_history,
            sample.foot_ball_signed_gap_m,
            foot_forces,
            prior_nonfoot_contact=prior_nonfoot_contact,
        )
        if first_foot_contact_frame is None or frame - first_foot_contact_frame > 60:
            side = None
        jac = sample.foot_material_point_jacobian_m_per_rad[0 if side is None else side]
        desired, shaping = vertical_compliance_stiffness(self._kp, jac, side)
        matrix, relative, step = slew_stiffness(self._kp, self._relative, desired)
        self._relative = relative.copy()
        self._next_frame += 1  # A failed forecast cannot silently be retried as the same frame.
        identity = bool(np.array_equal(matrix, np.diag(self._kp)))
        end = None
        used = 0
        if not identity:
            if self._reserved + 10 > self._maximum_steps:
                raise RuntimeError("directional compliance forecast budget exhausted")
            self._reserved += 10
            f = G1PDTorqueForecaster(
                self._model,
                expected_world_hash=self._world,
                kp=self._kp,
                kd=self._kd,
                max_forecast_steps=10,
                coupled_stiffness_nm_per_rad=matrix,
            )
            self._attempts += 1
            try:
                result = f.forecast(data, target, np.zeros(29))
                end = result.end_integration_state.copy()
            finally:
                self._attempted += f.attempted_steps
                self._integrated += f.integrated_steps
                used = f.integrated_steps
        report = dict(
            schema="soccer.rsi.directional_compliance_plan.v1",
            frame=frame,
            mode="COMPLY" if side is not None else "INACTIVE" if identity else "RESTORE",
            side=side,
            identity=identity,
            shaping=shaping,
            relative_matrix_step_norm=step,
            relative_stiffness_eigenvalues=np.linalg.eigvalsh(relative).tolist(),
            stiffness_matrix_hash=hash_bytes(matrix.tobytes()),
            forecast_substeps=used,
            expected_end_state_hash=None if end is None else hash_bytes(end.tobytes()),
            hardware_authorized=False,
            promotion_authorized=False,
        )
        report["report_hash"] = hash_json(report)
        matrix.flags.writeable = False
        return DirectionalCompliancePlan(matrix, identity, report, end)

    def verify_actual_interval(self, data: Any, plan: DirectionalCompliancePlan) -> None:
        if plan.expected_end_state is None:
            return
        if not isinstance(data, mujoco.MjData) or data.model is not self._model:
            raise ValueError("actual state must belong to bound SIM model")
        end = np.empty_like(plan.expected_end_state)
        mujoco.mj_getState(self._model, data, end, mujoco.mjtState.mjSTATE_INTEGRATION)
        if end.tobytes() != plan.expected_end_state.tobytes():
            raise ValueError("directional gain forecast and actual full interval differ")
