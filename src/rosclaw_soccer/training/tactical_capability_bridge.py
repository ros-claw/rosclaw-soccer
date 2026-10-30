"""Bridge physical capability evidence to 2v1 research proposals, without actuation."""

from dataclasses import dataclass
from typing import Any

from rosclaw_soccer.growth.tactical_2v1 import TacticalAction
from rosclaw_soccer.growth.tactical_2v1_actor import TwoVsOneTacticalDecision
from rosclaw_soccer.training.physical_skill_capability import choose_tactical_option


@dataclass(frozen=True)
class CapabilityFilteredProposal:
    action: TacticalAction | None
    route: str
    hardware_authorized: bool = False
    promotion_authorized: bool = False


def constrain_two_vs_one_proposal(
    decision: TwoVsOneTacticalDecision,
    reports: dict[str, dict[str, Any]],
    identities: dict[str, tuple[str, str, str, str]],
) -> CapabilityFilteredProposal:
    """Filter an actor's Q-values; None means replan/observe, not stand still.

    Reports must be independently authenticated before being passed here;
    their checksums are not signatures. This is a simulation-only research
    ranking, not a command or a permit to execute any physical motion.
    """
    if type(decision) is not TwoVsOneTacticalDecision:
        raise ValueError("typed tactical actor decision required")
    if not decision.accepted:
        return CapabilityFilteredProposal(None, "ACTOR_ABSTAIN_REPLAN")
    name = choose_tactical_option(
        {"pass": decision.q_pass, "shoot": decision.q_shoot}, reports, identities
    )
    if name is None:
        return CapabilityFilteredProposal(None, "NO_PHYSICAL_EVIDENCE_REPLAN")
    return CapabilityFilteredProposal(TacticalAction(name), "EXACT_PHYSICAL_EVIDENCE_SCREEN")
