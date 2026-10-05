"""One-control-tick MuJoCo interventions on an owned recorded-state replay.

SIM_ONLY. Foundation targets are frozen for a 20 ms branch; this is not an
adaptive future policy rollout, counterfactual terminal return or authority.
Callers authenticate original reports, traces and independent reviews.
"""

from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes


def intervention_actions(
    actual: Any, previous: Any, nominal: Any, joint_limits: Any, *, radius: float = 0.003
) -> np.ndarray[Any, Any]:
    arrays = [np.asarray(v) for v in (actual, previous, nominal, joint_limits)]
    if (
        type(radius) is not float
        or not 0 < radius <= 0.012
        or [v.shape for v in arrays] != [(12,), (12,), (12,), (12, 2)]
        or any(v.dtype.kind not in "fiu" or not np.isfinite(v).all() for v in arrays)
    ):
        raise ValueError("finite declared twelve-joint intervention box required")
    actual, previous, nominal, limits = [np.array(v, dtype=np.float64, copy=True) for v in arrays]
    # Match the original controller: slew first, then the final joint/cap
    # shield. A moving nominal target can make final deltas differ by more
    # than the pre-shield slew. Do not force a bad Foundation baseline into
    # range: zero remains allowed, as in the unchanged source controller.
    final_low = np.maximum(-0.16, np.minimum(0.0, limits[:, 0] - nominal))
    final_high = np.minimum(0.16, np.maximum(0.0, limits[:, 1] - nominal))
    low = np.clip(previous - 0.012, final_low, final_high)
    high = np.clip(previous + 0.012, final_low, final_high)
    if (
        np.any(limits[:, 0] >= limits[:, 1])
        or np.any(np.abs(previous) > 0.1600000001)
        or np.any(low > high)
        or np.any(actual < low - 1e-10)
        or np.any(actual > high + 1e-10)
    ):
        raise ValueError("original actual action and nonempty unchanged box required")
    result = np.repeat(actual[None], 25, axis=0)
    for joint in range(12):
        for side, sign in enumerate((-1, 1)):
            row = 1 + 2 * joint + side
            result[row, joint] = np.clip(actual[joint] + sign * radius, low[joint], high[joint])
    return result


class CounterfactualMotorReplay:
    def __init__(
        self, snapshot: Path, trace: dict[str, Any], physics: dict[str, Any], *, world_hash: str
    ) -> None:
        import mujoco

        if hash_bytes(snapshot.read_bytes()) != world_hash:
            raise ValueError("exact original compiled simulation world required")
        self._model = mujoco.MjModel.from_binary_path(str(snapshot))
        m = self._model
        if (
            (m.nq, m.nv, m.nu) != (43, 41, 29)
            or m.opt.timestep != 0.002
            or m.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER
            or m.opt.solver != mujoco.mjtSolver.mjSOL_NEWTON
            or m.opt.iterations != 100
            or physics.get("canonical_joint_names") != list(G1_DDS_JOINT_NAMES)
        ):
            raise ValueError("unchanged canonical MuJoCo replay and joint order required")
        self._kp, self._kd = [
            np.array(physics[k], dtype=np.float64, copy=True)
            for k in ("canonical_pd_kp", "canonical_pd_kd")
        ]
        if (
            self._kp.shape != (29,)
            or self._kd.shape != (29,)
            or not np.isfinite(self._kp).all()
            or not np.isfinite(self._kd).all()
            or np.any(self._kp <= 0)
            or np.any(self._kd < 0)
        ):
            raise ValueError("finite original PD controller required")

        def identifier(kind: Any, name: str) -> int:
            index = mujoco.mj_name2id(m, kind, name)
            if index < 0:
                raise ValueError("original model element missing")
            return int(index)

        joints = [identifier(mujoco.mjtObj.mjOBJ_JOINT, n) for n in G1_DDS_JOINT_NAMES]
        self._qi = np.asarray([m.jnt_qposadr[j] for j in joints])
        self._vi = np.asarray([m.jnt_dofadr[j] for j in joints])
        self._limits = m.jnt_range[joints[:12]].copy()
        self._ai = []
        for j in joints:
            matches = np.flatnonzero(m.actuator_trnid[:, 0] == j)
            if len(matches) != 1:
                raise ValueError("one canonical actuator per joint required")
            self._ai.append(int(matches[0]))
        self._pelvis = identifier(mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self._ball = identifier(mujoco.mjtObj.mjOBJ_BODY, "ball")
        self._bg = identifier(mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        bj = m.body_jntadr[self._ball]
        self._bq, self._bv = m.jnt_qposadr[bj], m.jnt_dofadr[bj]
        if (
            m.geom_size[self._bg, 0] != 0.11
            or m.body_mass[self._ball] != 0.43
            or np.any(m.dof_damping[self._bv : self._bv + 6] != 0)
        ):
            raise ValueError("unchanged physical ball required")
        self._contacts = [
            identifier(mujoco.mjtObj.mjOBJ_BODY, n)
            for n in (
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_ankle_pitch_link",
                "right_ankle_pitch_link",
                "left_knee_link",
                "right_knee_link",
            )
        ]
        shapes = dict(
            canonical_qpos=(300, 1, 43),
            canonical_qvel=(300, 1, 41),
            joint_target_rad=(300, 1, 29),
            pre_motor_joint_target_rad=(300, 1, 29),
            motor_delta_rad=(300, 1, 12),
            torque_nm=(300, 1, 10, 29),
            actual_actuator_force_nm=(300, 1, 10, 29),
            pelvis_z_per_substep_m=(300, 1, 10),
            force_n=(300, 1, 6),
            ball_position_before_step_m=(300, 1, 3),
            ball_position_after_step_m=(300, 1, 3),
        )
        self._trace = {}
        for k, shape in shapes.items():
            v = np.asarray(trace.get(k))
            if (
                v.shape != shape
                or v.dtype.kind not in "fiu"
                or not np.isfinite(v).all()
                or np.max(np.abs(v)) > 1e6
            ):
                raise ValueError("complete finite original recorded trace required")
            self._trace[k] = np.array(v, dtype=np.float64, copy=True)[:, 0]
        self._data = mujoco.MjData(m)
        q, v = self._trace["canonical_qpos"][0], self._trace["canonical_qvel"][0]
        self._data.qpos[:7], self._data.qpos[self._qi], self._data.qpos[self._bq : self._bq + 7] = (
            q[:7],
            q[7:36],
            q[36:],
        )
        self._data.qvel[:6], self._data.qvel[self._vi], self._data.qvel[self._bv : self._bv + 6] = (
            v[:6],
            v[6:35],
            v[35:],
        )
        mujoco.mj_forward(m, self._data)
        self._frame = 0
        self.actual_recorded_replay_substeps = 0
        self.actual_branch_substeps = 0

    @staticmethod
    def _equal(a: Any, b: Any) -> None:
        if not np.allclose(a, b, atol=1e-8, rtol=0):
            raise ValueError("physical prefix or baseline clone differs from recorded truth")

    def _canonical(self, data: Any) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        return (
            np.concatenate(
                (data.qpos[:7], data.qpos[self._qi], data.qpos[self._bq : self._bq + 7])
            ),
            np.concatenate(
                (data.qvel[:6], data.qvel[self._vi], data.qvel[self._bv : self._bv + 6])
            ),
        )

    def _before(self) -> None:
        q, v = self._canonical(self._data)
        self._equal(q, self._trace["canonical_qpos"][self._frame])
        self._equal(v, self._trace["canonical_qvel"][self._frame])
        self._equal(
            self._data.xpos[self._ball], self._trace["ball_position_before_step_m"][self._frame]
        )

    def _step(self, data: Any, target: np.ndarray[Any, Any]) -> dict[str, Any]:
        import mujoco

        force, torques, actual, pelvis = np.zeros(6), [], [], []
        for _ in range(10):
            torque = np.clip(
                (target - data.qpos[self._qi]) * self._kp - data.qvel[self._vi] * self._kd,
                -np.asarray(G1_HARD_TORQUE_LIMITS),
                np.asarray(G1_HARD_TORQUE_LIMITS),
            )
            data.ctrl[self._ai] = torque
            torques.append(torque.copy())
            mujoco.mj_step(self._model, data)
            actual.append(data.actuator_force[self._ai].copy())
            pelvis.append(float(data.xpos[self._pelvis, 2]))
            for i in range(data.ncon):
                contact = data.contact[i]
                a, b = int(contact.geom1), int(contact.geom2)
                if self._bg not in (a, b):
                    continue
                body = int(self._model.geom_bodyid[b if a == self._bg else a])
                if body in self._contacts:
                    value = np.zeros(6)
                    mujoco.mj_contactForce(self._model, data, i, value)
                    if not np.isfinite(value).all():
                        raise ValueError("nonfinite physical contact force")
                    j = self._contacts.index(body)
                    force[j] = max(force[j], float(np.linalg.norm(value[:3])))
        q, v = self._canonical(data)
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise ValueError("nonfinite branch dynamics")
        return dict(
            torque_nm=np.asarray(torques),
            actual_actuator_force_nm=np.asarray(actual),
            pelvis_z_per_substep_m=np.asarray(pelvis),
            force_n=force,
            ball_position_after_step_m=data.xpos[self._ball].copy(),
            canonical_qpos_after=q,
            canonical_qvel_after=v,
        )

    def _compare(self, result: dict[str, Any]) -> None:
        for k in (
            "torque_nm",
            "actual_actuator_force_nm",
            "pelvis_z_per_substep_m",
            "force_n",
            "ball_position_after_step_m",
        ):
            self._equal(result[k], self._trace[k][self._frame])
        if self._frame < 299:
            self._equal(
                result["canonical_qpos_after"], self._trace["canonical_qpos"][self._frame + 1]
            )
            self._equal(
                result["canonical_qvel_after"], self._trace["canonical_qvel"][self._frame + 1]
            )

    def advance_recorded(self) -> None:
        if self._frame >= 300:
            raise ValueError("recorded episode complete")
        self._before()
        result = self._step(self._data, self._trace["joint_target_rad"][self._frame])
        self.actual_recorded_replay_substeps += 10
        self._compare(result)
        self._frame += 1

    def fork(self, *, radius: float = 0.003) -> dict[str, Any]:
        import mujoco

        if not 30 <= self._frame < 300:
            raise ValueError("aligned learner frame required")
        self._before()
        nominal = self._trace["pre_motor_joint_target_rad"][self._frame]
        baseline = self._trace["motor_delta_rad"][self._frame]
        previous = self._trace["motor_delta_rad"][self._frame - 1]
        actions = intervention_actions(
            baseline, previous, nominal[:12], self._limits, radius=radius
        )
        expected = nominal.copy()
        expected[:12] += baseline
        self._equal(expected, self._trace["joint_target_rad"][self._frame])
        results = []
        for index, action in enumerate(actions):
            clone = mujoco.MjData(self._model)
            mujoco.mj_copyData(clone, self._model, self._data)
            target = nominal.copy()
            target[:12] += action
            result = self._step(clone, target)
            self.actual_branch_substeps += 10
            if index == 0:
                self._compare(result)
            results.append(result)
        # Independent branches must leave the recorded continuation unchanged.
        self._before()
        return dict(
            frame=self._frame,
            actions=actions,
            results=results,
            baseline_clone_matches_recorded_physics=True,
            full_policy_future_not_simulated=True,
            horizon_sec=0.02,
            new_foundation_calls=0,
            duplicate_projected_actions_not_independent=True,
            slew_is_pre_final_joint_shield_not_final_delta_guarantee=True,
            activation_ceiling="SIM_ONLY",
            promotion_authorized=False,
            hardware_authorized=False,
        )
