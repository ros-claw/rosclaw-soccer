"""Roster-scalable G1 pitch construction isolated from frozen field evidence."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.world.field import (
    G1TrainingGoalSpec,
    _add_goalkeeper_hand_envelopes,
    _attach_g1,
    _configure_ball_dof_damping,
    _require_stadium_model,
    _stadium_spec,
)

_PLAYER_ID = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_BODY_PREFIX = re.compile(r"^[a-z][a-z0-9_]{0,62}_$")


@dataclass(frozen=True)
class G1PitchPlayerSpec:
    """One independently addressed G1 in a shared soccer pitch.

    This object owns scene identity and initial placement only.  It carries no
    controller or hardware authority.
    """

    agent_id: str
    body_prefix: str
    origin_m: tuple[float, float, float]
    yaw_rad: float
    goalkeeper_gloves: bool = False
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.g1_pitch_player_spec.v1"

    def __post_init__(self) -> None:
        if not _PLAYER_ID.fullmatch(self.agent_id):
            raise ValueError("pitch player agent identity is invalid")
        if self.body_prefix and not _BODY_PREFIX.fullmatch(self.body_prefix):
            raise ValueError("pitch player body prefix is invalid")
        if (
            len(self.origin_m) != 3
            or not all(math.isfinite(value) for value in self.origin_m)
            or not -8.0 <= self.origin_m[0] <= 12.0
            or not -4.0 <= self.origin_m[1] <= 4.0
            or abs(self.origin_m[2]) > 1.0e-12
            or not math.isfinite(self.yaw_rad)
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("pitch player placement violates its SIM-only field envelope")

    @property
    def spec_hash(self) -> str:
        return str(hash_json(asdict(self)))


def build_g1_multi_player_stadium_model(
    asset_root: Path,
    *,
    players: tuple[G1PitchPlayerSpec, ...],
    spec: G1TrainingGoalSpec | None = None,
) -> Any:
    """Compile one pitch with two to ten separately actuated 29-DoF G1s."""

    roster = tuple(players)
    if not 2 <= len(roster) <= 10:
        raise ValueError("multi-player stadium requires between two and ten G1 players")
    if len({player.agent_id for player in roster}) != len(roster):
        raise ValueError("multi-player stadium agent identities must be unique")
    prefixes = tuple(player.body_prefix for player in roster)
    if prefixes.count("") != 1 or len(set(prefixes)) != len(prefixes):
        raise ValueError("multi-player stadium requires one source G1 and unique prefixes")
    if sum(player.goalkeeper_gloves for player in roster) > 2:
        raise ValueError("multi-player stadium supports at most one goalkeeper per team")

    goal = spec or G1TrainingGoalSpec()
    root = asset_root.expanduser().resolve()
    parent = _stadium_spec(root, goal)

    import mujoco

    for index, player in enumerate(roster):
        if not player.body_prefix:
            if player.goalkeeper_gloves:
                _add_goalkeeper_hand_envelopes(
                    parent,
                    body_prefix="",
                    geom_prefix="",
                    mujoco=mujoco,
                )
            continue
        _attach_g1(
            parent,
            root=root,
            frame_name=f"team_player_{index}_frame",
            prefix=player.body_prefix,
            origin_m=player.origin_m,
            yaw_rad=player.yaw_rad,
            mujoco=mujoco,
        )
        if player.goalkeeper_gloves:
            _add_goalkeeper_hand_envelopes(
                parent,
                body_prefix=player.body_prefix,
                geom_prefix=player.body_prefix,
                mujoco=mujoco,
            )
    model = parent.compile()
    _configure_ball_dof_damping(model, goal)
    _require_stadium_model(model)
    expected_actuators = 29 * len(roster)
    if model.nu != expected_actuators:
        raise ValueError(
            f"multi-player stadium has {model.nu} actuators, expected {expected_actuators}"
        )
    return model


__all__ = ["G1PitchPlayerSpec", "build_g1_multi_player_stadium_model"]
