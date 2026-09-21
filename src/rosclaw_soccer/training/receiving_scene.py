"""Immutable current scene evidence for private receiving forecasts, not authority."""

import math
import re
from dataclasses import dataclass


def _agent(value: object) -> bool:
    return type(value) is str and re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", value) is not None


@dataclass(frozen=True)
class ReceivingPeerState:
    agent_id: str
    position_xy_m: tuple[float, float]
    velocity_xy_mps: tuple[float, float]

    def __post_init__(self) -> None:
        if not _agent(self.agent_id):
            raise ValueError("named measured peer required")
        for values in (self.position_xy_m, self.velocity_xy_mps):
            if (
                type(values) is not tuple
                or len(values) != 2
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1e4
                    for v in values
                )
            ):
                raise ValueError("bounded immutable measured peer vectors required")


@dataclass(frozen=True)
class ReceivingSceneContext:
    """Current values only; no future policy decisions or simulator references.

    `navigation_overrides_present` deliberately over-approximates exceptional
    navigation paths. A restricted forecast must reject unsupported context; absence
    does not certify future stability, peer extrapolation, or contact predictions.
    """

    frame: int
    agent_id: str
    world_config_hash: str
    intent: str
    peers: tuple[ReceivingPeerState, ...]
    possession_agent_id: str | None
    receive_lease_agent_id: str | None
    receive_lease_active: bool
    navigation_overrides_present: bool
    post_receive_hold: bool
    receive_foot_lateral_offset_m: float

    def __post_init__(self) -> None:
        if (
            type(self.frame) is not int
            or not 0 <= self.frame < 1000
            or not _agent(self.agent_id)
            or type(self.world_config_hash) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.world_config_hash) is None
            or type(self.intent) is not str
            or re.fullmatch(r"[a-z][a-z_]{0,63}", self.intent) is None
            or type(self.peers) is not tuple
            or len(self.peers) > 31
            or type(self.receive_lease_active) is not bool
            or type(self.navigation_overrides_present) is not bool
            or type(self.post_receive_hold) is not bool
            or type(self.receive_foot_lateral_offset_m) not in (int, float)
            or not math.isfinite(self.receive_foot_lateral_offset_m)
            or not 0 <= self.receive_foot_lateral_offset_m <= 1
        ):
            raise ValueError("bounded current receiving scene required")
        for peer in self.peers:
            if not isinstance(peer, ReceivingPeerState):
                raise ValueError("typed immutable peers required")
            peer.__post_init__()
        ids = [peer.agent_id for peer in self.peers]
        if ids != sorted(set(ids)) or self.agent_id in ids:
            raise ValueError("sorted unique other agents required")
        for owner in (self.possession_agent_id, self.receive_lease_agent_id):
            if owner is not None and (not _agent(owner) or owner not in [self.agent_id, *ids]):
                raise ValueError("scene owner must be an observed agent")
        if self.receive_lease_active and self.receive_lease_agent_id is None:
            raise ValueError("active receive lease requires its owner")
