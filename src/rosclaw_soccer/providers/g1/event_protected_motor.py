"""Frozen numeric successor checkpoint, with one shared private filter history.

The binary event feature is caller-owned data, NOT verified physical evidence
or admission authority. The caller must independently establish causality and
enforce world/runtime guards. This adapter only produces SIM_ONLY proposals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.approach_router import load_bounded_reference_parameters
from rosclaw_soccer.providers.g1.goal_reference_motor import (
    G1FrozenGoalReferenceMotor,
    GoalReferenceMotorProposal,
)
from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class EventProtectedMotorProposal(GoalReferenceMotorProposal):
    successor_enabled: bool = False


class G1FrozenEventProtectedMotor(G1FrozenGoalReferenceMotor):
    """Numeric counterpart of the 137-input event-protected training model.

    Both branches consume the actual previous residual/reference state; a
    dormant successor must not evolve a separate fictional action history.
    The existing motor envelopes and 200-frame episode limit remain unchanged.
    """

    def __init__(
        self,
        weights: Path,
        *,
        expected_actor_hash: str,
        foundation_hash: str,
        reference_library_hash: str,
        course_transform_hash: str,
        default_angles: NDArray[Any],
        reference_center_rad: float = 0.0,
    ) -> None:
        self._successor_enabled = False
        super().__init__(
            weights,
            expected_actor_hash=expected_actor_hash,
            foundation_hash=foundation_hash,
            reference_library_hash=reference_library_hash,
            course_transform_hash=course_transform_hash,
            default_angles=default_angles,
            reference_center_rad=reference_center_rad,
        )
        self.contract_hash = str(
            hash_json(
                {
                    "schema": "rosclaw_soccer.g1_event_protected_motor.v1",
                    "base_envelope_contract": self.contract_hash,
                    "actor_hash": self.policy_hash,
                    "observation": "course_ball_133+target_ray_xy+previous_heading+binary_event",
                    "history": "one actual private residual/reference history across both branches",
                    "event": "caller-owned causal latch; no activation on first frame or reversal",
                    "activation_ceiling": "SIM_ONLY",
                }
            )
        )

    def _load_parameters(
        self, weights: Path, expected_hash: str, shapes: dict[str, tuple[int, ...]]
    ) -> tuple[dict[str, NDArray[Any]], str]:
        protected_shapes = {
            f"{prefix}.{name}": shape
            for prefix in ("anchor", "plastic")
            for name, shape in shapes.items()
        }
        protected_shapes["logstd"] = (30,)
        parameters, policy_hash = load_bounded_reference_parameters(
            weights, expected_hash, protected_shapes
        )
        if not (
            np.array_equal(parameters["logstd"], parameters["anchor.logstd"])
            and np.array_equal(parameters["logstd"], parameters["plastic.logstd"])
        ):
            raise ValueError("protected checkpoint must retain a single frozen exploration vector")
        self._successor_parameters = {name: parameters[f"plastic.{name}"] for name in shapes}
        return {name: parameters[f"anchor.{name}"] for name in shapes}, policy_hash

    def _network_parameters(self) -> dict[str, NDArray[Any]]:
        return self._successor_parameters if self._successor_enabled else self._parameters

    def begin_episode(self, *, target_position_xy: NDArray[Any]) -> None:
        self._successor_enabled = False
        super().begin_episode(target_position_xy=target_position_xy)

    def propose(
        self,
        *,
        frame: int,
        course_qpos: NDArray[Any],
        course_qvel: NDArray[Any],
        teacher_target: NDArray[Any],
        successor_enabled: bool = False,
    ) -> EventProtectedMotorProposal:
        if (
            type(successor_enabled) is not bool
            or (frame == 0 and successor_enabled)
            or (self._successor_enabled and not successor_enabled)
        ):
            self._faulted = True
            raise ValueError("binary monotone successor feature after the first frame required")
        self._successor_enabled = successor_enabled
        proposal = super().propose(
            frame=frame,
            course_qpos=course_qpos,
            course_qvel=course_qvel,
            teacher_target=teacher_target,
        )
        values = asdict(proposal)
        values["observation_hash"] = str(
            hash_json(
                {"base_observation_hash": proposal.observation_hash, "successor": successor_enabled}
            )
        )
        return EventProtectedMotorProposal(**values, successor_enabled=successor_enabled)
