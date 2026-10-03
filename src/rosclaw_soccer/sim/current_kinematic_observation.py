"""Opt-in current-state observation; never refresh or advance live dynamics.

After mj_step, qpos/qvel are integrated but derived Cartesian fields can still
describe the preceding forward stage. Recompute ONLY kinematics and velocity
in private MjData. No forward dynamics, contact solve, integration, controls,
runtime, transport or hardware is invoked. Force observations remain the last
completed frame and are not inferred from this view.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def observation_contract(reference: str, snapshot: str) -> dict[str, str] | None:
    from rosclaw_soccer.sim.contracts import hash_bytes
    from rosclaw_soccer.sim.root_velocity_reference import root_observation_contract

    if snapshot == "cached":
        return root_observation_contract(reference)
    if snapshot != "current-kinematic" or reference != "body-origin":
        raise ValueError("current kinematic observations require explicit regular-body origin")
    return {
        "root_velocity_reference": "body-origin",
        "velocity_order": "world-linear-then-world-angular",
        "derived_state_stage": "isolated-current-qpos-qvel-kinematics-no-live-refresh",
        "pose_reference": "regular-body-origin",
        "force_input": "previous-completed-frame",
        "kinematic_source_hash": hash_bytes(Path(__file__).read_bytes()),
    }


def snapshot_from_contract(contract: Any) -> tuple[str, str]:
    from rosclaw_soccer.sim.root_velocity_reference import reference_from_contract

    if type(contract) is dict and contract == observation_contract(
        "body-origin", "current-kinematic"
    ):
        return "body-origin", "current-kinematic"
    return reference_from_contract(contract), "cached"


@dataclass(frozen=True)
class KinematicObservation:
    time_sec: float
    body_position_m: np.ndarray[Any, Any]
    body_quaternion_wxyz: np.ndarray[Any, Any]
    body_origin_velocity_world: np.ndarray[Any, Any]


class CurrentKinematicObserver:
    """Private derived-state workspace, independent of the stepped simulator."""

    def __init__(self, model: Any) -> None:
        import mujoco

        self._model = model
        self._scratch = mujoco.MjData(model)

    def sample(self, data: Any) -> KinematicObservation:
        import mujoco

        if (
            data.model is not self._model
            or np.asarray(data.qpos).shape != (self._model.nq,)
            or np.asarray(data.qvel).shape != (self._model.nv,)
            or not np.isfinite(data.qpos).all()
            or not np.isfinite(data.qvel).all()
            or not np.isfinite(data.time)
        ):
            raise ValueError("finite complete current simulator state required")
        self._scratch.qpos[:] = data.qpos
        self._scratch.qvel[:] = data.qvel
        self._scratch.time = data.time
        mujoco.mj_kinematics(self._model, self._scratch)
        mujoco.mj_comPos(self._model, self._scratch)
        mujoco.mj_comVel(self._model, self._scratch)
        velocities = np.zeros((self._model.nbody, 6), dtype=np.float64)
        for body in range(self._model.nbody):
            angular_linear = np.zeros(6)
            mujoco.mj_objectVelocity(
                self._model, self._scratch, mujoco.mjtObj.mjOBJ_XBODY, body, angular_linear, 0
            )
            velocities[body] = angular_linear[[3, 4, 5, 0, 1, 2]]
        positions = self._scratch.xpos.copy()
        quaternions = self._scratch.xquat.copy()
        for array in (positions, quaternions, velocities):
            if not np.isfinite(array).all():
                raise ValueError("nonfinite isolated kinematic observation")
            array.flags.writeable = False
        return KinematicObservation(float(data.time), positions, quaternions, velocities)
