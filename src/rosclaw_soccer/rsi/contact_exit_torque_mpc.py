"""SIM-only short-horizon torque proposals for persistent foot/ball loading.

The numerical teacher is not learned, promoted or a hardware execution path.
Forecasts see only the current body and held current PD target. Full native
Foundation feedback and independent outcome/retention tests remain mandatory.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.physics.contact_wrench import read_geom_contact_wrenches
from rosclaw_soccer.physics.g1_joint_task_kinematics import (
    G1JointTaskKinematics,
    G1JointTaskKinematicSampler,
)
from rosclaw_soccer.physics.g1_pd_torque_forecast import G1PDTorqueForecaster
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json


def pressure_side(
    history: np.ndarray,
    gaps: np.ndarray,
    solver_foot_force: np.ndarray,
    *,
    prior_nonfoot_contact: bool,
) -> int | None:
    """Three completed peak bins AND last solved contact, not a duration claim."""
    if (
        not isinstance(history, np.ndarray)
        or history.ndim != 2
        or history.shape[1:] != (6,)
        or not 0 <= len(history) <= 3
        or history.dtype != np.float64
        or not np.isfinite(history).all()
        or np.any(history < 0)
        or type(prior_nonfoot_contact) is not bool
    ):
        raise ValueError("finite completed causal contact history required")
    for a in (gaps, solver_foot_force):
        if (
            not isinstance(a, np.ndarray)
            or a.shape != (2,)
            or a.dtype != np.float64
            or not np.isfinite(a).all()
        ):
            raise ValueError("two current surface gaps and last-solver foot forces required")
    if np.any(solver_foot_force < 0):
        raise ValueError("nonnegative current foot force magnitudes required")
    if prior_nonfoot_contact or np.any(history[:, 2:] > 1) or len(history) != 3:
        return None
    eligible = np.all(history[:, :2] > 1, axis=0) & (gaps <= 0.01) & (solver_foot_force > 1)
    if not np.any(eligible):
        return None
    return int(np.argmax(np.where(eligible, solver_foot_force, -1)))


def torque_candidates(
    sample: G1JointTaskKinematics, side: int, previous: np.ndarray
) -> list[tuple[str, np.ndarray]]:
    caps = np.minimum(12.0, 0.2 * np.asarray(G1_HARD_TORQUE_LIMITS))
    if (
        type(side) is not int
        or side not in (0, 1)
        or not isinstance(previous, np.ndarray)
        or previous.shape != (29,)
        or previous.dtype != np.float64
        or not np.isfinite(previous).all()
        or np.any(abs(previous) > caps)
    ):
        raise ValueError("explicit selected foot and bounded previous torque correction required")
    jac = sample.foot_material_point_jacobian_m_per_rad[side]
    up = jac.T @ np.array([0.0, 0.0, 80.0])
    forward = jac.T @ np.array([40.0, 0.0, 0.0])
    com_force = (
        np.clip(sample.foot_origin_world_m[1 - side] - sample.robot_subtree_com_world_m, -0.1, 0.1)
        * 400
    )
    com_force[2] = 0
    transfer = sample.com_jacobian_m_per_rad.T @ com_force
    result = []
    seen = set()
    for name, wish in (
        ("decay", np.zeros(29)),
        ("lift", up),
        ("lift_forward", up + forward),
        ("lift_transfer", up + transfer),
        ("lift_forward_transfer", up + forward + transfer),
        ("hold", previous),
    ):
        if wish.shape != (29,) or not np.isfinite(wish).all():
            raise ValueError("finite actual-Jacobian torque basis required")
        offset = np.clip(np.clip(wish, previous - 2.0, previous + 2.0), -caps, caps)
        offset[offset == 0] = 0  # Canonicalize signed zero before deduplicating physical controls.
        identity = offset.tobytes()
        if identity not in seen:
            result.append((name, offset))
            seen.add(identity)
    return result


@dataclass(frozen=True)
class ContactExitTorquePlan:
    offset_nm: np.ndarray[Any, Any]
    report: dict[str, Any]
    expected_end_state: np.ndarray[Any, Any] | None


class ContactExitTorqueMPC:
    def __init__(
        self,
        model: Any,
        *,
        world_hash: str,
        kp: np.ndarray,
        kd: np.ndarray,
        forecast_budget_steps: int = 6000,
    ) -> None:
        self._model = model
        self._forecaster = G1PDTorqueForecaster(
            model,
            expected_world_hash=world_hash,
            kp=kp,
            kd=kd,
            max_forecast_steps=forecast_budget_steps,
        )
        self._sampler = G1JointTaskKinematicSampler(model, expected_world_hash=world_hash)
        self._metric = mujoco.MjData(model)
        self._bg = int(model.geom("ball_geom").id)
        self._feet = [int(model.body(s + "_ankle_roll_link").id) for s in ("left", "right")]
        self._robot_mass = float(model.body_subtreemass[model.body("pelvis").id])
        if not np.isfinite(self._robot_mass) or self._robot_mass <= 0:
            raise ValueError("finite robot subtree mass required")
        self._world_hash = world_hash

    def counters(self) -> dict[str, int]:
        f = self._forecaster
        return dict(
            forecast_attempts=f.forecast_attempts,
            attempted_forecast_steps=f.attempted_steps,
            integrated_forecast_steps=f.integrated_steps,
            maximum_forecast_steps=f.max_forecast_steps,
        )

    def contract(self) -> dict[str, Any]:
        import rosclaw_soccer.physics.g1_joint_task_kinematics as kinematic_module
        import rosclaw_soccer.physics.g1_pd_torque_forecast as forecast_module

        value = dict(
            schema="soccer.rsi.contact_exit_torque_mpc.v1",
            world_hash=self._world_hash,
            source_hashes=dict(
                teacher=hash_bytes(Path(__file__).read_bytes()),
                forecast=hash_bytes(Path(forecast_module.__file__).read_bytes()),
                kinematics=hash_bytes(Path(kinematic_module.__file__).read_bytes()),
            ),
            horizon_substeps=10,
            dt_s=0.002,
            maximum_offset_nm=12.0,
            maximum_hard_limit_fraction=0.2,
            maximum_offset_change_per_control_frame_nm=2.0,
            maximum_contact_age_frames=60,
            pressure_gate=(
                "three completed peak bins plus current last-solver force and gap <= 0.01m"
            ),
            foot_virtual_lift_force_n=80.0,
            foot_virtual_forward_force_n=40.0,
            com_virtual_force_gain_n_per_m=400.0,
            com_virtual_force_max_displacement_m=0.1,
            maximum_eligible_controls=6,
            maximum_forecasts_per_frame=7,
            forecast_budget_steps=self._forecaster.max_forecast_steps,
            cost_weights=dict(
                down_impulse=1.0,
                clearance_gap=0.25,
                opposite_support=0.1,
                rotation_deviation=0.1,
                forward_velocity_loss=0.25,
                lateral_velocity=0.1,
                joint_velocity_deviation=0.01,
                offset=0.01,
            ),
            trained_policy=False,
            teacher_qualified=False,
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
        previous_offset_nm: np.ndarray,
        prior_nonfoot_contact: bool,
    ) -> ContactExitTorquePlan:
        if (
            not isinstance(data, mujoco.MjData)
            or data.model is not self._model
            or type(frame) is not int
            or not 0 <= frame < 300
            or not np.isclose(data.time, frame * 0.02, rtol=0, atol=1e-8)
            or (
                first_foot_contact_frame is not None
                and (
                    type(first_foot_contact_frame) is not int
                    or not 0 <= first_foot_contact_frame < frame
                )
            )
        ):
            raise ValueError(
                "current control-frame clock and causal observed first-foot time required"
            )
        if not isinstance(force_history, np.ndarray) or force_history.shape != (min(frame, 3), 6):
            raise ValueError("exact recent completed control bins required")
        if (
            not isinstance(target, np.ndarray)
            or target.shape != (29,)
            or target.dtype != np.float64
            or not np.isfinite(target).all()
            or np.max(abs(target)) > 10
        ):
            raise ValueError("finite canonical current PD target required")
        sample = self._sampler.sample(data)
        forces = np.zeros(2)
        for w in read_geom_contact_wrenches(self._model, data, geom_id=self._bg):
            body = int(self._model.geom_bodyid[w.other_geom_id])
            if body in self._feet:
                observed_side = self._feet.index(body)
                forces[observed_side] = max(
                    forces[observed_side], float(np.linalg.norm(w.force_world_n))
                )
        side = pressure_side(
            force_history,
            sample.foot_ball_signed_gap_m,
            forces,
            prior_nonfoot_contact=prior_nonfoot_contact,
        )
        if first_foot_contact_frame is None and np.any(force_history[:, :2] > 1):
            raise ValueError("observed foot contact requires its causal first-contact timestamp")
        if first_foot_contact_frame is None or frame - first_foot_contact_frame > 60:
            side = None
        # Candidate validation is shared even for inactive/ramp-out calls.
        basis = torque_candidates(sample, 0 if side is None else side, previous_offset_nm)
        if side is None and not np.any(previous_offset_nm):
            inactive = dict(
                schema="soccer.rsi.contact_exit_torque_plan.v1",
                frame=frame,
                mode="INACTIVE",
                side=None,
                choices=[],
                selected_index=None,
                forecast_substeps=0,
                expected_end_state_hash=None,
                no_safe_choice_fallback=False,
                hardware_authorized=False,
                promotion_authorized=False,
            )
            inactive["report_hash"] = hash_json(inactive)
            return ContactExitTorquePlan(
                np.zeros(29),
                inactive,
                None,
            )
        if side is None:
            basis = basis[:1]
        before = self._forecaster.integrated_steps
        zero = np.zeros(29)
        reference = self._forecaster.forecast(data, target, zero)
        evaluations = {zero.tobytes(): reference}
        for _, offset in basis:
            if offset.tobytes() not in evaluations:
                evaluations[offset.tobytes()] = self._forecaster.forecast(data, target, offset)
        metric_side = 0 if side is None else side
        reference_q = reference.canonical_qpos_after_substep[-1]
        reference_v = reference.canonical_qvel_after_substep[-1]
        caps = np.minimum(12.0, 0.2 * np.asarray(G1_HARD_TORQUE_LIMITS))
        choices = []
        for name, offset in basis:
            f = evaluations[offset.tobytes()]
            mujoco.mj_setState(
                self._model,
                self._metric,
                f.end_integration_state,
                mujoco.mjtState.mjSTATE_INTEGRATION,
            )
            end = self._sampler.sample(self._metric)
            q = f.canonical_qpos_after_substep[-1]
            v = f.canonical_qvel_after_substep[-1]
            angle = 2 * float(np.arccos(np.clip(abs(np.dot(q[3:7], reference_q[3:7])), -1, 1)))
            down = float(np.maximum(-f.robot_force_on_ball_world_n[:, 2], 0).sum() * 0.002)
            gap = float(end.foot_ball_signed_gap_m[metric_side])
            support = float(
                np.mean(f.completed_foot_ground_normal_n[:, 1 - metric_side])
                / (self._robot_mass * 9.81)
            )
            cost = (
                down / (self._robot_mass * 9.81 * 0.02)
                + 0.25 * (max(0.0, 0.02 - gap) / 0.02) ** 2
                + 0.1 * max(0.0, 0.4 - support) ** 2
                + 0.1 * (angle / 0.05) ** 2
                + 0.25 * max(0.0, float(reference_v[35] - v[35])) ** 2
                + 0.1 * float(v[36]) ** 2
                + 0.01 * float(np.mean((v[6:35] - reference_v[6:35]) ** 2))
                + 0.01 * float(np.mean((offset / caps) ** 2))
            )
            safe = (
                not f.nonfoot_robot_contact_over_1n
                and float(f.pelvis_height_solver_epoch_m.min()) >= 0.65
            )
            if not np.isfinite(cost):
                raise ValueError("finite bounded forecast score required")
            choices.append(
                dict(
                    name=name,
                    offset_nm=offset.tolist(),
                    forecast_cost=cost,
                    downward_robot_impulse_n_s=down,
                    foot_ball_gap_m=gap,
                    opposite_support_weight_fraction=support,
                    rotation_delta_from_zero_rad=angle,
                    ball_forward_velocity_m_s=float(v[35]),
                    ball_lateral_velocity_m_s=float(v[36]),
                    predicted_safe=safe,
                    donor_state_hash=f.donor_integration_state_hash,
                    end_state_hash=hash_bytes(f.end_integration_state.tobytes()),
                )
            )
        safe_choices = [i for i, c in enumerate(choices) if c["predicted_safe"]]
        selected = (
            min(safe_choices, key=lambda i: choices[i]["forecast_cost"]) if safe_choices else 0
        )
        offset = basis[selected][1].copy()
        expected = evaluations[offset.tobytes()].end_integration_state.copy()
        report = dict(
            schema="soccer.rsi.contact_exit_torque_plan.v1",
            frame=frame,
            mode="PRESSURE_EXIT" if side is not None else "RAMP_OUT",
            side=side,
            metric_side=metric_side,
            choices=choices,
            selected_index=selected,
            no_safe_choice_fallback=not bool(safe_choices),
            forecast_substeps=self._forecaster.integrated_steps - before,
            expected_end_state_hash=hash_bytes(expected.tobytes()),
            hardware_authorized=False,
            promotion_authorized=False,
        )
        report["report_hash"] = hash_json(report)
        return ContactExitTorquePlan(offset, report, expected)

    def verify_actual_interval(self, data: Any, plan: ContactExitTorquePlan) -> None:
        if plan.expected_end_state is None:
            return
        if not isinstance(data, mujoco.MjData) or data.model is not self._model:
            raise ValueError("actual state must belong to the bound model")
        actual = np.empty_like(plan.expected_end_state)
        mujoco.mj_getState(self._model, data, actual, mujoco.mjtState.mjSTATE_INTEGRATION)
        if actual.tobytes() != plan.expected_end_state.tobytes():
            raise ValueError("selected forecast and actual full integration state differ")
