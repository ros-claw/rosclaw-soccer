"""Explicit simulation-only policy-action to G1 joint-target translation.

This numeric contract does not load a policy, command a simulator, replicate
motor dynamics or establish that an external policy transfers to the team.
The caller must separately qualify the observation, timing and motor backend.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.motor_option import TeamMotorTarget


def _vector(value: tuple[float, ...], *, bound: float, name: str) -> None:
    if (
        type(value) is not tuple
        or len(value) != 29
        or any(type(x) not in (float, int) or not math.isfinite(x) or abs(x) > bound for x in value)
    ):
        raise ValueError(f"{name} requires 29 bounded finite immutable values")


@dataclass(frozen=True)
class G1AffinePositionContract:
    """Names and units bind raw policy outputs to a non-authorizing target."""

    joint_names: tuple[str, ...]
    offset_rad: tuple[float, ...]
    scale_rad: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        if type(self.joint_names) is not tuple or self.joint_names != G1_DDS_JOINT_NAMES:
            raise ValueError("exact G1 DDS joint order required")
        if self.activation_ceiling != "SIM_ONLY":
            raise ValueError("simulation-only contract required")
        for value, bound, name in (
            (self.offset_rad, 10.0, "offset"),
            (self.scale_rad, 10.0, "scale"),
            (self.kp, 300.0, "kp"),
            (self.kd, 30.0, "kd"),
        ):
            _vector(value, bound=bound, name=name)
        if any(x <= 0 for x in self.scale_rad) or any(x < 0 for x in (*self.kp, *self.kd)):
            raise ValueError("positive scales and nonnegative gains required")

    @property
    def contract_hash(self) -> str:
        return str(hash_json({"schema": "soccer.g1.affine_position.v1", **asdict(self)}))

    def propose(
        self, action: tuple[float, ...], *, encoder_bias_rad: tuple[float, ...]
    ) -> TeamMotorTarget:
        """Return offset + scale * raw action - explicitly supplied encoder bias.

        The caller must maintain the raw action history separately. No clipping,
        substitution of gains, physics stepping or authority escalation occurs.
        TeamMotorTarget enforces the existing bounds on the resulting proposal.
        """
        _vector(action, bound=100.0, name="raw action")
        _vector(encoder_bias_rad, bound=1.0, name="encoder bias")
        target = tuple(
            float(offset + scale * raw - bias)
            for offset, scale, raw, bias in zip(
                self.offset_rad, self.scale_rad, action, encoder_bias_rad, strict=True
            )
        )
        return TeamMotorTarget(target, self.kp, self.kd)
