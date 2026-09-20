"""Bounded reference-command attenuation for frozen SONIC research proposals.

This scales an already-cleared navigation command; it cannot enlarge or reverse
it. Scaling alone is not a collision-clearance or balance certificate.
"""

import math
from dataclasses import asdict, dataclass, replace

from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class SonicCommandScaleSchedule:
    knots: tuple[float, ...]
    knot_frames: int = 20

    def __post_init__(self) -> None:
        if (
            type(self.knots) is not tuple
            or not 1 <= len(self.knots) <= 32
            or type(self.knot_frames) is not int
            or not 1 <= self.knot_frames <= 100
            or any(
                type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
                for value in self.knots
            )
        ):
            raise ValueError("finite [0,1] command attenuation knots required")

    @property
    def contract_hash(self) -> str:
        return str(hash_json({"schema": "soccer.sonic_command_scale.v1", **asdict(self)}))

    def at(self, local_frame: int) -> float:
        if type(local_frame) is not int or local_frame < 0:
            raise ValueError("nonnegative local command frame required")
        position = local_frame / self.knot_frames
        left = min(int(position), len(self.knots) - 1)
        right = min(left + 1, len(self.knots) - 1)
        fraction = min(position - left, 1.0)
        return float(
            min(1, max(0, (1 - fraction) * self.knots[left] + fraction * self.knots[right]))
        )


def future_attenuations(
    schedule: SonicCommandScaleSchedule, *, local_branch_frame: int
) -> tuple[SonicCommandScaleSchedule, ...]:
    """Propose weaker future commands without rewriting interpolation history.

    Includes the incumbent. Actual physics/controller prefix equality must
    still be checked by the simulator; these proposals are not safe actions.
    """
    if not isinstance(schedule, SonicCommandScaleSchedule):
        raise ValueError("typed command schedule required")
    schedule.__post_init__()
    if type(local_branch_frame) is not int or not 0 <= local_branch_frame <= 1000:
        raise ValueError("bounded local query frame required")
    locked = (
        0
        if local_branch_frame == 0
        else min(
            len(schedule.knots),
            (local_branch_frame - 1 + schedule.knot_frames - 1) // schedule.knot_frames + 1,
        )
    )
    if locked == len(schedule.knots):
        raise ValueError("no future command knots remain")
    return (schedule,) + tuple(
        replace(
            schedule,
            knots=schedule.knots[:locked]
            + tuple(value * factor for value in schedule.knots[locked:]),
        )
        for factor in (0.0, 0.25, 0.5, 0.75)
    )
