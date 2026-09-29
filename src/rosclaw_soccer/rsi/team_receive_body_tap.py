"""Read-only 50Hz per-player receiving context for causal sequence training."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.navigation_option import NavigationDelta, NavigationObservation


@dataclass
class TeamReceiveBodyTap:
    agent_id: str
    foundation_hash: str
    foundation_config_hash: str
    contract_hash: str = field(init=False)
    activation_ceiling: str = field(init=False, default="SIM_ONLY")
    needs_effector_velocities: bool = field(init=False, default=True)
    needs_full_proprioception: bool = field(init=False, default=True)
    needs_continuous_body_context: bool = field(init=False, default=True)
    _next_frame: int = field(init=False, default=0)
    _rows: list[tuple[float, ...]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.team_receive_body_tap.v1",
                "agent_id": self.agent_id,
                "foundation_hash": self.foundation_hash,
                "foundation_config_hash": self.foundation_config_hash,
                "needs_continuous_body_context": True,
                "needs_effector_velocities": True,
                "needs_full_proprioception": True,
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: NavigationObservation) -> NavigationDelta | None:
        if (
            not isinstance(observation, NavigationObservation)
            or observation.agent_id != self.agent_id
            or observation.frame != self._next_frame
            or len(observation.effector_positions) != 2
            or len(observation.effector_velocities) != 2
            or tuple(value[0] for value in observation.effector_positions)
            != ("left_foot", "right_foot")
            or tuple(value[0] for value in observation.effector_velocities)
            != ("left_foot", "right_foot")
            or observation.body_angular_velocity is None
            or len(observation.joint_positions_rad) != 29
            or len(observation.joint_velocities_radps) != 29
        ):
            raise ValueError("continuous same-player measured receive context required")
        observation.__post_init__()
        self._next_frame += 1
        self._rows.append(
            (
                float(observation.frame),
                observation.time_sec,
                *observation.body_pose,
                *observation.body_velocity,
                *observation.body_angular_velocity,
                *observation.ball_position,
                *observation.ball_velocity,
                *(value for foot in observation.effector_positions for value in foot[1:]),
                *(value for foot in observation.effector_velocities for value in foot[1:]),
                *observation.joint_positions_rad,
                *observation.joint_velocities_radps,
                float(observation.committed_receiver),
                float(observation.intent == "receive"),
            )
        )
        return None

    def arrays(self) -> dict[str, NDArray[np.float64]]:
        array = np.asarray(self._rows, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 93 or not np.isfinite(array).all():
            raise ValueError("complete finite continuous receive observations required")
        return {
            "frame": array[:, 0].copy(),
            "time_sec": array[:, 1].copy(),
            "body_pose": array[:, 2:9].copy(),
            "body_velocity": array[:, 9:12].copy(),
            "body_angular_velocity": array[:, 12:15].copy(),
            "ball_position": array[:, 15:18].copy(),
            "ball_velocity": array[:, 18:21].copy(),
            "foot_position": array[:, 21:27].reshape(-1, 2, 3).copy(),
            "foot_velocity": array[:, 27:33].reshape(-1, 2, 3).copy(),
            "joint_position": array[:, 33:62].copy(),
            "joint_velocity": array[:, 62:91].copy(),
            "committed_receiver": array[:, 91].copy(),
            "receive_intent": array[:, 92].copy(),
        }
