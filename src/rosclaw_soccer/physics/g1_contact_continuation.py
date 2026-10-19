"""Owned 40ms held-target predictions; not Foundation feedback or task truth.

Compose the unchanged 20ms PD forecast primitive using its complete integration
state, without another forward solve between chunks. This copying assumption
must be checked against uninterrupted PD integration on the declared world.
The API has no future target, contact timestamp, trace, or task outcome input.
"""

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.physics.g1_pd_torque_forecast import (
    G1PDTorqueForecast,
    G1PDTorqueForecaster,
)


@dataclass(frozen=True)
class G1ContactContinuationForecast:
    first: G1PDTorqueForecast
    second: G1PDTorqueForecast

    def substep_arrays(self) -> dict[str, np.ndarray[Any, Any]]:
        return {
            name: np.concatenate((getattr(self.first, name), getattr(self.second, name)))
            for name in (
                "canonical_qpos_after_substep",
                "canonical_qvel_after_substep",
                "requested_pd_torque_nm",
                "applied_control_nm",
                "actual_actuator_force_nm",
                "pelvis_height_solver_epoch_m",
                "robot_force_on_ball_world_n",
                "completed_foot_ground_normal_n",
            )
        }


class G1ContactContinuationForecaster:
    """Fixed two control frames, explicit shared physical budget, no live writes."""

    def __init__(
        self,
        model: Any,
        *,
        expected_world_hash: str,
        kp: np.ndarray[Any, Any],
        kd: np.ndarray[Any, Any],
        max_forecast_steps: int,
    ) -> None:
        if (
            type(max_forecast_steps) is not int
            or not 20 <= max_forecast_steps <= 600000
            or max_forecast_steps % 20
        ):
            raise ValueError("explicit whole40ms forecast budget required")
        self._primitive = G1PDTorqueForecaster(
            model,
            expected_world_hash=expected_world_hash,
            kp=kp,
            kd=kd,
            max_forecast_steps=max_forecast_steps,
        )
        self._model = model
        self._continuation = mujoco.MjData(model)
        self.forecast_attempts = 0

    @property
    def attempted_steps(self) -> int:
        return self._primitive.attempted_steps

    @property
    def integrated_steps(self) -> int:
        return self._primitive.integrated_steps

    def forecast(
        self, data: Any, current_target_rad: np.ndarray[Any, Any], offset_nm: np.ndarray[Any, Any]
    ) -> G1ContactContinuationForecast:
        if data is self._continuation:
            raise ValueError("caller-owned donor required")
        if self.attempted_steps + 20 > self._primitive.max_forecast_steps:
            raise RuntimeError("whole40ms budget exhausted; no partial reservation or retry")
        # The primitive validates all numeric inputs before it charges a step.
        first = self._primitive.forecast(data, current_target_rad, offset_nm)
        self.forecast_attempts += 1
        mujoco.mj_setState(
            self._model,
            self._continuation,
            first.end_integration_state,
            mujoco.mjtState.mjSTATE_INTEGRATION,
        )
        second = self._primitive.forecast(self._continuation, current_target_rad, offset_nm)
        return G1ContactContinuationForecast(first, second)
