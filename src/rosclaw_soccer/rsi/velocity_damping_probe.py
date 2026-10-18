"""Fixed bounded proprioceptive torque probe, not a trained motor policy.

Only for local offline replay. Ground-truth contact/constraint forces are not
inputs. Comparing the same gain/cap at two feedback rates tests temporal
authority, not learning, task success, or a deployable recovery controller.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_json

MODES = ("CONTROL_TICK_HOLD", "PHYSICS_SUBSTEP_FEEDBACK")


@dataclass(frozen=True)
class VelocityDampingProbe:
    reference_velocity_rad_s: tuple[float, ...]
    mode: str

    def __post_init__(self) -> None:
        if (
            type(self.reference_velocity_rad_s) is not tuple
            or len(self.reference_velocity_rad_s) != 29
            or any(
                type(value) is not float or not math.isfinite(value) or abs(value) > 1000
                for value in self.reference_velocity_rad_s
            )
            or type(self.mode) is not str
            or self.mode not in MODES
        ):
            raise ValueError("fixed mode and finite canonical 29-joint velocity reference required")

    def torque_offset(self, velocity: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if (
            not isinstance(velocity, np.ndarray)
            or velocity.shape != (29,)
            or velocity.dtype != np.float64
            or not np.isfinite(velocity).all()
            or np.max(np.abs(velocity)) > 1000
        ):
            raise ValueError("finite float64 canonical joint velocity measurement required")
        cap = np.minimum(2.0, 0.04 * np.asarray(G1_HARD_TORQUE_LIMITS))
        return np.clip(-0.5 * (velocity - self.reference_velocity_rad_s), -cap, cap)

    def contract(self) -> dict[str, Any]:
        value = dict(
            schema="soccer.rsi.local_velocity_damping_probe.v1",
            mode=self.mode,
            reference_velocity_rad_s=list(self.reference_velocity_rad_s),
            gain_nm_per_rad_s=0.5,
            offset_cap_nm=np.minimum(2.0, 0.04 * np.asarray(G1_HARD_TORQUE_LIMITS)).tolist(),
            comparison_control_dt_s=0.02,
            comparison_physics_dt_s=0.002,
            activation_ceiling="SIM_ONLY",
            local_diagnostic_only=True,
            learned=False,
            trained_optimizer_updates=0,
            privileged_contact_or_constraint_forces_are_inputs=False,
            full_policy_or_recovery_qualified=False,
            runtime_execution_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        return dict(value, contract_hash=hash_json(value))
