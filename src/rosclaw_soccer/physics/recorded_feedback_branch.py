"""Independently replay a recorded80ms SIM branch from an owned real prefix.

Targets are recorded, not recomputed neural decisions. Reuses the unchanged
native PD primitive; no time reset, mj_forward at the fork, model fitting or
hardware/runtime interface. Branches do NOT automatically share the execution
controller's safety-cancellation law.
The caller authenticates and reexecutes the recorded prefix; clock/counter
checks here are not a substitute for that source and physics evidence.
"""

from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.physics.g1_pd_torque_forecast import G1PDTorqueForecaster
from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
from rosclaw_soccer.sim.contracts import hash_bytes


class RecordedFeedbackBranch:
    def __init__(self, donor: G1TargetEpisode, *, world_hash: str) -> None:
        if (
            type(donor) is not G1TargetEpisode
            or donor.failed
            or not 31 <= donor.next_frame <= 294
            or donor.next_frame // 20 != (donor.next_frame + 3) // 20
            or donor.attempted_steps != donor.next_frame * 10
            or donor.integrated_steps != donor.attempted_steps
            or not np.isclose(donor.data.time, donor.next_frame * 0.02, atol=1e-8, rtol=0)
        ):
            raise ValueError(
                "completed owned partial native prefix and no-replan80ms window required"
            )
        self.donor = donor
        self.native = G1PDTorqueForecaster(
            donor.model,
            expected_world_hash=world_hash,
            kp=donor.kp,
            kd=donor.kd,
            max_forecast_steps=40,
        )
        self.owned = mujoco.MjData(donor.model)
        mujoco.mj_copyData(self.owned, donor.model, donor.data)
        self.used = False

    def verify(self, raw: dict[str, np.ndarray], *, current_target: np.ndarray) -> dict[str, Any]:
        if self.used:
            raise ValueError("single-use recorded branch; never retry charged native steps")
        self.used = True
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        before = np.empty(mujoco.mj_stateSize(self.donor.model, spec))
        mujoco.mj_getState(self.donor.model, self.donor.data, before, spec)
        fields = {
            "canonical_qpos_after_substep": (43,),
            "canonical_qvel_after_substep": (41,),
            "requested_pd_torque_nm": (29,),
            "requested_offset_nm": (29,),
            "applied_control_nm": (29,),
            "actual_actuator_force_nm": (29,),
            "pelvis_height_solver_epoch_m": (),
            "robot_force_on_ball_world_n": (3,),
            "completed_foot_ground_normal_n": (2,),
        }
        for key, shape in fields.items():
            value = raw[key]
            if (
                type(value) is not np.ndarray
                or value.shape != (40, *shape)
                or value.dtype != np.float64
                or not np.isfinite(value).all()
            ):
                raise ValueError("complete finite recorded branch physical tensors required")
        end = raw["end_integration_state"]
        if (
            type(end) is not np.ndarray
            or end.shape != before.shape
            or end.dtype != np.float64
            or not np.isfinite(end).all()
            or not np.array_equal(
                raw["requested_offset_nm"][:30],
                np.repeat(raw["requested_offset_nm"][:1], 30, axis=0),
            )
            or np.any(raw["requested_offset_nm"][30:])
        ):
            raise ValueError("complete finite end state and unconditional60ms offset law required")
        targets = [current_target] + [raw[f"next_{j}_joint_target_rad"] for j in range(1, 4)]
        for target in targets:
            if (
                type(target) is not np.ndarray
                or target.shape != (29,)
                or target.dtype != np.float64
                or not np.isfinite(target).all()
                or np.max(abs(target)) > 10
            ):
                raise ValueError("finite recorded canonical current/next targets required")
        errors = {key: 0.0 for key in fields if key != "requested_offset_nm"}
        for j, target in enumerate(targets):
            offsets = raw["requested_offset_nm"][j * 10 : (j + 1) * 10]
            if not np.array_equal(offsets, np.repeat(offsets[:1], 10, axis=0)):
                raise ValueError("original held-offset native frame required")
            result = self.native.forecast(self.owned, target, offsets[0].copy())
            mujoco.mj_copyData(self.owned, self.donor.model, self.native._scratch)
            for key in errors:
                actual = getattr(result, key)
                expected = raw[key][j * 10 : (j + 1) * 10]
                if actual.tobytes() != expected.tobytes():
                    raise ValueError("independent native recorded branch differs: " + key)
            if j == 3 and not np.array_equal(result.end_integration_state, end):
                raise ValueError("whole recorded branch end integration state differs")
        after = np.empty_like(before)
        mujoco.mj_getState(self.donor.model, self.donor.data, after, spec)
        if not np.array_equal(before, after):
            raise ValueError("recorded branch verifier mutated its donor")
        if self.native.attempted_steps != 40 or self.native.integrated_steps != 40:
            raise ValueError("exact forty newly integrated native steps required")
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.donor.model, self.owned, mujoco.mjtObj.mjOBJ_BODY, self.donor.ball, velocity, 0
        )
        return dict(
            schema="soccer.research.recorded_feedback_branch_physics.v1",
            actual_new_native_steps=self.native.integrated_steps,
            maximum_absolute_errors=errors,
            complete_end_integration_state_exact=True,
            donor_integration_hash=hash_bytes(before.tobytes()),
            donor_unchanged=True,
            cached_terminal_ball_velocity_m_s=velocity[3:].tolist(),
            controller_law="RECORDED_UNCONDITIONAL_60MS_PLANNING_BRANCH",
            background_neural_policy_reexecuted=False,
            hardware_authorized=False,
            teacher_qualified=False,
        )
