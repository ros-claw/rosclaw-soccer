"""Value-only attribution of a foot-to-foot transfer, never reception readiness.

The caller authenticates the stream and separately examines launch quality,
receiver control and task success. Floor contacts are excluded by the upstream
contract. Other environmental contacts break a clean transfer.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Literal

from rosclaw_soccer.skills.team.motor_option import TeamMotorPhysicsObservation


@dataclass(frozen=True)
class PassContactChainResult:
    clean_transfer_observed: bool
    reasons: tuple[str, ...]
    source_contact_sec: float | None
    receiver_contact_sec: float | None
    interrupted_at_sec: float | None
    successor_ready_verified: Literal[False] = field(default=False, init=False)
    training_authorized: Literal[False] = field(default=False, init=False)
    activation_ceiling: Literal["SIM_ONLY"] = field(default="SIM_ONLY", init=False)


def inspect_pass_contact_chain(
    observations: tuple[TeamMotorPhysicsObservation, ...],
    *,
    sender_id: str,
    receiver_id: str,
    request_time_sec: float,
) -> PassContactChainResult:
    """Require an uninterrupted, complete 500 Hz stream spanning the request.

    The first force-positive sender foot contact after the request starts this
    attempt. A foreign/body/environment contact ends it; later retries must be
    separate declared attempts. Receiver arrival requires a contact-free substep
    after the latest sender contact, preventing simultaneous contact from being
    called a pass. Unsafe bodies anywhere in the supplied stream reject the claim.
    """
    if (
        any(
            type(agent) is not str or re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", agent) is None
            for agent in (sender_id, receiver_id)
        )
        or sender_id == receiver_id
        or type(request_time_sec) not in (int, float)
        or not 0 <= request_time_sec <= 20
        or not math.isfinite(request_time_sec)
        or type(observations) is not tuple
        or not 2 <= len(observations) <= 10001
    ):
        raise ValueError("bounded distinct players, request and immutable stream required")
    previous: float | None = None
    for row in observations:
        if not isinstance(row, TeamMotorPhysicsObservation):
            raise ValueError("typed physics observations required")
        row.__post_init__()
        for contact in row.ball_contacts:
            contact.__post_init__()
        if (
            not row.contacts_complete
            or row.observer_agent_id != sender_id
            or row.time_sec > 20
            or previous is not None
            and abs(row.time_sec - previous - 0.002) > 1e-9
        ):
            raise ValueError("complete same-sender consecutive 500 Hz evidence required")
        previous = row.time_sec
    if not observations[0].time_sec <= request_time_sec < observations[-1].time_sec:
        raise ValueError("stream must span the request and its physical aftermath")
    source = arrival = interruption = None
    separated = False
    for row in observations:
        if row.time_sec <= request_time_sec + 1e-9:
            continue
        active = tuple(c for c in row.ball_contacts if c.normal_force_n > 0)
        own = any(c.agent_id == sender_id and c.is_foot for c in active)
        received = any(c.agent_id == receiver_id and c.is_foot for c in active)
        if source is None and not own:
            continue
        if own:
            source = row.time_sec
            separated = False
        if any(c.agent_id not in (sender_id, receiver_id) or not c.is_foot for c in active):
            interruption = row.time_sec
            break
        if received:
            if own or not separated:
                interruption = row.time_sec
            else:
                arrival = row.time_sec
            break
        if not active:
            separated = True
    reasons = []
    if not all(row.world_bodies_safe for row in observations):
        reasons.append("unsafe_episode")
    if source is None:
        reasons.append("no_sender_foot_contact")
    if interruption is not None:
        reasons.append("interrupted_or_simultaneous_transfer")
    if arrival is None:
        reasons.append("no_clean_receiver_arrival")
    return PassContactChainResult(not reasons, tuple(reasons), source, arrival, interruption)
