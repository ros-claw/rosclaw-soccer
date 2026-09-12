"""Bounded streaming neural navigation proposals, with no simulator handles.

Research backend, not an adopted team champion. The caller owns phase admission,
physical clearance validation and the 500 Hz torque guard. A delayed reference
planner cannot promise instantaneous tracking of a collision-avoidance command.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rosclaw_soccer.providers.g1.sonic_history_handoff import (
    SonicHistoryHandoffReceipt,
    handoff_sonic_history,
)
from rosclaw_soccer.providers.g1.sonic_runup import (
    G1SonicRunupConfig,
    G1SonicRunupController,
    _resample_segments_30_to_50,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation, TeamMotorTarget


@dataclass(frozen=True)
class SonicNavigationConfig:
    maximum_frames: int = 600
    planner_seed: int = 30300
    replan_frames: int = 20
    lookahead_frames: int = 10

    def __post_init__(self) -> None:
        if (
            type(self.maximum_frames) is not int
            or not 50 <= self.maximum_frames <= 3000
            or type(self.planner_seed) is not int
            or not 0 <= self.planner_seed <= 2**31 - 3000
            or type(self.replan_frames) is not int
            or self.replan_frames != 20
            or type(self.lookahead_frames) is not int
            or self.lookahead_frames != 10
        ):
            raise ValueError(
                "bounded 50 Hz navigation horizon and qualified planning cadence required"
            )


@dataclass(frozen=True)
class SonicObservationStartReceipt:
    """Fresh measured initialization, not a history handoff or motion permit."""

    agent_id: str
    frame: int
    time_sec: float
    observation_hash: str
    reference_hash: str
    binding_hash: str
    activation_ceiling: str = "SIM_ONLY"


def future_reference_context(reference: np.ndarray, start: int) -> np.ndarray:
    """Four future 30 Hz samples, with short-hemisphere normalized quaternions.

    This interpolation is not a claim of native SONIC numerical equivalence.
    """
    if (
        type(start) is not int
        or start < 0
        or reference.ndim != 2
        or reference.shape[1] != 36
        or len(reference) <= start + 6
    ):
        raise ValueError("complete future reference context required")
    result = []
    for sample in start + np.arange(4) * 50 / 30:
        left = int(sample)
        fraction = float(sample - left)
        a, b = reference[left].copy(), reference[left + 1].copy()
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("nonfinite future reference")
        if a[3:7] @ b[3:7] < 0:
            b[3:7] *= -1
        value = (1 - fraction) * a + fraction * b
        norm = float(np.linalg.norm(value[3:7]))
        if not 0.5 <= norm <= 1.5:
            raise ValueError("invalid future reference quaternion")
        value[3:7] /= norm
        result.append(value)
    return np.asarray(result)


class _StreamingBackend(G1SonicRunupController):
    def __init__(self, model_root: Path, config: SonicNavigationConfig) -> None:
        self.navigation = config
        self.command = (0.0, 0.0, 0.0)
        self.facing = 0.0
        self.planner_calls = 0
        self.events: list[dict[str, object]] = []
        super().__init__(
            model_root,
            G1SonicRunupConfig(model_variant="sonic_v1_1", execution_duration_sec=4.5),
        )

    def plan(self, context: np.ndarray, frame: int) -> np.ndarray:
        vx, vy, yaw_rate = self.command
        speed = math.hypot(vx, vy)
        direction = (
            (vx / speed, vy / speed)
            if speed > 1e-8
            else (math.cos(self.facing), math.sin(self.facing))
        )
        heading = self.facing + yaw_rate * 0.4
        feed = {
            "context_mujoco_qpos": context[None].astype(np.float32),
            "target_vel": np.asarray([speed], dtype=np.float32),
            "mode": np.asarray([0 if speed < 0.02 else 1], dtype=np.int64),
            "movement_direction": np.asarray([[*direction, 0.0]], dtype=np.float32),
            "facing_direction": np.asarray(
                [[math.cos(heading), math.sin(heading), 0.0]], dtype=np.float32
            ),
            "random_seed": np.asarray(
                [self.navigation.planner_seed + self.planner_calls], dtype=np.int64
            ),
            "has_specific_target": np.zeros((1, 1), dtype=np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
            "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
            "allowed_pred_num_tokens": np.asarray(
                [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.int64
            ),
            "height": np.asarray([-1.0], dtype=np.float32),
        }
        output, count = self._planner.run(None, feed)
        count_array = np.asarray(count)
        if (
            count_array.shape != (1,)
            or count_array.dtype.kind not in "iu"
            or not 4 <= count_array[0] <= 64
        ):
            raise ValueError("invalid SONIC planner length")
        n = int(count_array[0])
        output = np.asarray(output)
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] < n or output.shape[2] != 36:
            raise ValueError("invalid SONIC planner output shape")
        segment = np.asarray(output[0, :n], dtype=np.float64)
        if not np.isfinite(segment).all():
            raise ValueError("nonfinite SONIC planner output")
        norms = np.linalg.norm(segment[:, 3:7], axis=1)
        if np.any(np.abs(norms - 1) > 0.01):
            raise ValueError("nonunit SONIC planner quaternion")
        self.events.append(
            {
                "frame": frame,
                "command": self.command,
                "reference_hash": hash_bytes(segment.tobytes()),
            }
        )
        self.planner_calls += 1
        return _resample_segments_30_to_50([segment], 0.02)

    def _generate_reference(self, initial_qpos: np.ndarray) -> np.ndarray:
        segment = self.plan(np.repeat(initial_qpos[None], 4, axis=0), 0)
        length = self.navigation.maximum_frames + 110
        return np.concatenate(
            (segment, np.repeat(segment[-1:], max(0, length - len(segment)), axis=0))
        )

    def navigation_tick(self, state: SimpleNamespace, frame: int) -> np.ndarray:
        if frame and frame % self.navigation.replan_frames == 0:
            start = frame + self.navigation.lookahead_frames
            segment = self.plan(future_reference_context(self.reference, start), frame)
            n = min(len(segment), len(self.reference) - start)
            self.reference[start : start + n] = segment[:n]
            self.reference[start + n :] = segment[n - 1]
        return self._update_from_reference(state, frame)


class G1SonicNavigation:
    """One fixed player and continuous option; any invalid call latches a fault.

    Only immutable observations cross the boundary. Construct another option
    explicitly for another episode; no implicit reset after a missed tick.
    No ownership, possession, physical step, or motor torque authority is held.
    """

    def __init__(
        self, model_root: Path, agent_id: str, config: SonicNavigationConfig | None = None
    ) -> None:
        import re

        if (
            not isinstance(agent_id, str)
            or re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", agent_id) is None
        ):
            raise ValueError("explicit navigation player identity required")
        self.agent_id = agent_id
        self.config = config or SonicNavigationConfig()
        self.backend = _StreamingBackend(model_root, self.config)
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.g1_sonic_navigation.v1",
                "agent_id": agent_id,
                "config": asdict(self.config),
                "foundation": self.backend.qualification.qualification_hash,
                "command": "post-clearance world vx vy <= .7 m/s, yaw rate <= 1.5 rad/s",
                "activation_ceiling": "SIM_ONLY",
            }
        )
        self._next_frame = 0
        self._faulted = False
        self._retired = False
        self._origin_frame = 0
        self._ready_from_handoff = False
        self._ready_from_observation = False
        self._boundary_observation_hash: str | None = None

    def start_from_observation(
        self, observation: TeamMotorObservation
    ) -> SonicObservationStartReceipt:
        """Initialize an unused option at the current global simulation frame.

        The backend cold-start pads from the current measurement; this is not
        evidence of earlier observed states. No stale history is imported. The
        next propose() must consume this same bound observation. The caller
        still owns admission and physical stepping.
        An invalid initialization latches off, never rearms an old option.
        """
        if self._faulted or self._retired:
            raise ValueError("navigation is faulted or retired")
        try:
            if (
                self._next_frame != 0
                or self._ready_from_handoff
                or self._ready_from_observation
                or len(self.backend._history) != 0
                or not isinstance(observation, TeamMotorObservation)
                or observation.agent_id != self.agent_id
                or observation.navigation_command is None
                or abs(observation.time_sec - observation.frame * 0.02) > 1e-6
            ):
                raise ValueError("unused navigation and current cleared observation required")
            q, v = np.asarray(observation.qpos), np.asarray(observation.qvel)
            if abs(float(np.linalg.norm(q[3:7])) - 1) > 0.01:
                raise ValueError("unit measured body quaternion required")
            self.backend.command = observation.navigation_command
            w, x, y, z = q[3:7]
            self.backend.facing = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            self.backend.reset(SimpleNamespace(qpos=q, qvel=v))
            observation_hash = hash_json(asdict(observation))
            reference_hash = hash_bytes(self.backend.reference.tobytes())
            receipt = SonicObservationStartReceipt(
                agent_id=self.agent_id,
                frame=observation.frame,
                time_sec=observation.time_sec,
                observation_hash=observation_hash,
                reference_hash=reference_hash,
                binding_hash=hash_json(
                    {
                        "kind": "fresh_observation_navigation_start.v1",
                        "navigation_contract": self.contract_hash,
                        "observation_hash": observation_hash,
                        "reference_hash": reference_hash,
                    }
                ),
            )
            self._origin_frame = observation.frame
            self._next_frame = observation.frame
            self._ready_from_observation = True
            self._boundary_observation_hash = observation_hash
            return receipt
        except Exception as error:
            self._faulted = True
            raise ValueError(
                "navigation observation initialization failed; option latched off"
            ) from error

    def propose(self, observation: TeamMotorObservation) -> TeamMotorTarget:
        if self._retired:
            raise ValueError("navigation retired after history handoff; a new option is required")
        if self._faulted:
            raise ValueError("navigation fault is latched; a new episode is required")
        try:
            if (
                observation.agent_id != self.agent_id
                or observation.frame != self._next_frame
                or observation.frame >= self._origin_frame + self.config.maximum_frames
                or abs(observation.time_sec - observation.frame * 0.02) > 1e-6
                or observation.navigation_command is None
            ):
                raise ValueError(
                    "consecutive timed navigation observations and clearance command required"
                )
            q, v = np.asarray(observation.qpos), np.asarray(observation.qvel)
            if abs(float(np.linalg.norm(q[3:7])) - 1) > 0.01:
                raise ValueError("unit measured body quaternion required")
            state = SimpleNamespace(qpos=q, qvel=v)
            self.backend.command = observation.navigation_command
            w, x, y, z = q[3:7]
            self.backend.facing = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            local_frame = observation.frame - self._origin_frame
            if local_frame == 0:
                if (
                    self._boundary_observation_hash is not None
                    and hash_json(asdict(observation)) != self._boundary_observation_hash
                ):
                    raise ValueError("navigation boundary observation commitment changed")
                if not (self._ready_from_handoff or self._ready_from_observation):
                    self.backend.reset(state)
            else:
                self.backend.observe(state)
            target = self.backend.navigation_tick(state, local_frame)
            proposal = TeamMotorTarget(
                tuple(float(v) for v in target),
                tuple(float(v) for v in self.backend.kp),
                tuple(float(v) for v in self.backend.kd),
            )
            self._next_frame += 1
            return proposal
        except Exception as error:
            self._faulted = True
            raise ValueError("navigation proposal failed; option latched off") from error

    def handoff_to(
        self,
        destination: G1SonicRunupController,
        *,
        observation: TeamMotorObservation,
        destination_observation: TeamMotorObservation,
        allow_planar_yaw_rotation: bool = False,
    ) -> SonicHistoryHandoffReceipt:
        """Commit the final measured state, transfer history and retire navigation.

        This replaces this tick's navigation proposal. The destination owns the
        new reference and next proposal; the caller still owns physical stepping.
        Any failed handoff latches navigation off, rather than retrying with an
        extra observation in the policy history.
        """
        if self._faulted or self._retired:
            raise ValueError("navigation is faulted or retired")
        try:
            if (
                observation.agent_id != self.agent_id
                or observation.frame != self._next_frame
                or not self._origin_frame + 1
                <= observation.frame
                < self._origin_frame + self.config.maximum_frames
                or abs(observation.time_sec - observation.frame * 0.02) > 1e-6
            ):
                raise ValueError("handoff requires the next timed observation for this player")
            self.backend.observe(
                SimpleNamespace(
                    qpos=np.asarray(observation.qpos), qvel=np.asarray(observation.qvel)
                )
            )
            receipt = handoff_sonic_history(
                source=self.backend,
                destination=destination,
                source_observation=observation,
                destination_observation=destination_observation,
                allow_planar_yaw_rotation=allow_planar_yaw_rotation,
            )
            self._retired = True
            self._next_frame += 1
            return receipt
        except Exception as error:
            self._faulted = True
            raise ValueError("navigation history handoff failed; option latched off") from error

    def start_from_history(
        self,
        source: G1SonicRunupController,
        *,
        source_observation: TeamMotorObservation,
        observation: TeamMotorObservation,
        allow_planar_yaw_rotation: bool = False,
    ) -> SonicHistoryHandoffReceipt:
        """Initialize a fresh navigation option on the existing global clock.

        The next propose() consumes this SAME boundary observation at local
        reference frame zero. It neither resets imported history nor adds a
        duplicate measured entry. This is not an implicit restart of a faulted
        option; construct a new instance for the next declared skill segment.
        """
        if self._faulted or self._retired:
            raise ValueError("navigation is faulted or retired")
        try:
            if (
                self._next_frame != 0
                or self._ready_from_handoff
                or self._ready_from_observation
                or len(self.backend._history) != 0
                or observation.agent_id != self.agent_id
                or observation.navigation_command is None
                or abs(observation.time_sec - observation.frame * 0.02) > 1e-6
            ):
                raise ValueError(
                    "new navigation and its current post-clearance observation required"
                )
            self.backend.command = observation.navigation_command
            q, v = np.asarray(observation.qpos), np.asarray(observation.qvel)
            w, x, y, z = q[3:7]
            self.backend.facing = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            self.backend.reset(SimpleNamespace(qpos=q, qvel=v))
            receipt = handoff_sonic_history(
                source=source,
                destination=self.backend,
                source_observation=source_observation,
                destination_observation=observation,
                allow_planar_yaw_rotation=allow_planar_yaw_rotation,
            )
            self._origin_frame = observation.frame
            self._next_frame = observation.frame
            self._ready_from_handoff = True
            self._boundary_observation_hash = hash_json(asdict(observation))
            return receipt
        except Exception as error:
            self._faulted = True
            raise ValueError(
                "navigation history initialization failed; option latched off"
            ) from error
