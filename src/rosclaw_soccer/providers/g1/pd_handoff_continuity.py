"""SIM-only equivalent-target offset for a declared PD expert handoff.

Match the old controller at the entry state, then release a quintic offset.
This is not learned control, a stability certificate or actuator authority.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.transition_bridge import (
    G1TransitionBridgeConfig,
    G1VelocityMatchedTransitionBridge,
)
from rosclaw_soccer.sim.contracts import hash_json


class PDHandoffContinuityOffset:
    """One private 50Hz sequence; fixed .40s release, no gain/time search."""

    def __init__(
        self,
        *,
        joint_position: np.ndarray,
        joint_velocity: np.ndarray,
        previous_target: np.ndarray,
        old_kp: np.ndarray,
        old_kd: np.ndarray,
        contact_target: np.ndarray,
        new_kp: np.ndarray,
        new_kd: np.ndarray,
    ) -> None:
        values = tuple(
            self._vector(value)
            for value in (
                joint_position,
                joint_velocity,
                previous_target,
                old_kp,
                old_kd,
                contact_target,
                new_kp,
                new_kd,
            )
        )
        q, v, previous, op, od, target, np_, nd = values
        if np.any(op <= 0) or np.any(np_ <= 0) or np.any(od < 0) or np.any(nd < 0):
            raise ValueError("positive stiffness and nonnegative damping required")
        old_raw_torque = op * (previous - q) - od * v
        equivalent_target = q + (old_raw_torque + nd * v) / np_
        offset = equivalent_target - target
        if not np.isfinite(offset).all():
            raise ValueError("finite equivalent PD target required")
        self._entry = target.copy()
        self._bridge = G1VelocityMatchedTransitionBridge(
            entry_position=offset,
            entry_velocity=np.zeros(29),
            exit_position=np.zeros(29),
            exit_velocity=np.zeros(29),
            config=G1TransitionBridgeConfig(duration_sec=0.4),
        )
        self._next = 0
        self._faulted = False

    @staticmethod
    def _vector(value: np.ndarray) -> np.ndarray:
        if (
            type(value) is not np.ndarray
            or value.shape != (29,)
            or value.dtype != np.float64
            or not np.isfinite(value).all()
        ):
            raise ValueError("owned finite float64 29-joint vector required")
        return value.copy()

    @staticmethod
    def contract() -> dict[str, Any]:
        value = dict(
            schema="soccer.g1.pd_handoff_continuity_offset.v1",
            control_dt_s=0.02,
            release_frames=20,
            release_duration_s=0.4,
            entry_equation="newPD(new_target+offset,q,v)==oldPD(previous_target,q,v)",
            release="existing quintic zero-boundary-velocity bridge offset to zero",
            guarantee="entry-state raw torque equality only; later substeps not guaranteed",
            caller_must_record_actual_target_and_pd=True,
            caller_must_apply_original_hard_torque_bounds=True,
            neural_training=False,
            stability_proven=False,
            activation_ceiling="SIM_ONLY",
            promotion_authorized=False,
            hardware_authorized=False,
        )
        return dict(value, contract_hash=hash_json(value))

    def propose(
        self, *, relative_frame: int, contact_target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._faulted:
            raise ValueError("faulted PD handoff sequence")
        try:
            target = self._vector(contact_target)
            if (
                type(relative_frame) is not int
                or relative_frame != self._next
                or not 0 <= relative_frame < 300
                or (relative_frame == 0 and not np.array_equal(target, self._entry))
            ):
                raise ValueError("ordered relative frame and exact admitted entry target required")
            offset = self._bridge.sample(relative_frame * 0.02).position
            result = target + offset
            if not np.isfinite(result).all():
                raise ValueError("finite proposed PD target required")
        except Exception:
            self._faulted = True
            raise
        self._next += 1
        return result, offset.copy()
