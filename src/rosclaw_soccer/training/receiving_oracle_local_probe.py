"""Named-coordinate local authority probes around a retained oracle schedule.

These are offline proposals, not a feedback teacher, learned policy, or motion
permission. Keep the original motor envelope and examination. Replay physical
controls before interpreting any candidate; an algebraic neighbor is not a
qualified skill. Repeated saturated proposals are rejected, not more samples.
"""

import math
from dataclasses import replace

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def local_oracle_proposals(
    schedule: ReceivingOracleSchedule,
    *,
    offsets: tuple[tuple[tuple[str, float], ...], ...],
) -> tuple[ReceivingOracleSchedule, ...]:
    """Return incumbent first, then explicit bounded named-joint perturbations.

    Offsets are normalized residual coordinates, not radians or absolute joint
    targets; the unchanged schedule maps one unit to 0.1 rad before filtering.
    At most 0.25 per coordinate is added to every existing knot, with explicit
    clipping to the original [-1, 1] range. Timing and all other joints remain
    identical. No inference of the affected leg from an agent's role is made.
    """
    if not isinstance(schedule, ReceivingOracleSchedule):
        raise ValueError("typed bounded oracle schedule required")
    schedule.__post_init__()
    if type(offsets) is not tuple or not 1 <= len(offsets) <= 32:
        raise ValueError("one to 32 explicit local proposals required")
    names = G1_DDS_JOINT_NAMES[: len(schedule.knots[0])]
    proposals = [schedule]
    seen = {schedule.contract_hash}
    for proposal in offsets:
        if type(proposal) is not tuple or not 1 <= len(proposal) <= 4:
            raise ValueError("one to four named local coordinates required")
        changes: dict[int, float] = {}
        for entry in proposal:
            if type(entry) is not tuple or len(entry) != 2:
                raise ValueError("named coordinate and normalized offset required")
            name, delta = entry
            if type(name) is not str or name not in names:
                raise ValueError("joint unavailable in this action substrate")
            index = names.index(name)
            if index in changes:
                raise ValueError("duplicate local coordinate")
            if (
                type(delta) not in (int, float)
                or not math.isfinite(delta)
                or not 0 < abs(delta) <= 0.25
            ):
                raise ValueError("finite nonzero local offset bounded by 0.25 required")
            changes[index] = float(delta)
        candidate = replace(
            schedule,
            knots=tuple(
                tuple(
                    min(1.0, max(-1.0, value + changes[i])) if i in changes else value
                    for i, value in enumerate(row)
                )
                for row in schedule.knots
            ),
        )
        if candidate.contract_hash in seen:
            raise ValueError(
                "duplicate or fully saturated proposal is not an independent candidate"
            )
        seen.add(candidate.contract_hash)
        proposals.append(candidate)
    return tuple(proposals)
