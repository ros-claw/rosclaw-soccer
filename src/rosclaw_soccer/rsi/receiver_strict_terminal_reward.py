"""SIM research reward from completed physical pass/receive evidence."""


def strict_terminal_reward(outcome: dict) -> float:
    """Reward only both-clean healthy handoff; failures never net positive.

    A neutral no-contact episode earns zero, not a successful-receive label.
    This is a research target, not a formal exam or promotion decision.
    """
    required = (
        "strict_both_clean_handoff",
        "strict_receiver_clean_handoff",
        "passer_foot_precedes_receiver",
        "both_bodies_healthy",
        "original_handoff",
    )
    if any(type(outcome.get(k)) is not bool for k in required):
        raise ValueError("complete boolean physical outcomes required")
    for key in ("passer_nonfoot_substep", "receiver_nonfoot_substep"):
        value = outcome.get(key)
        if key not in outcome or (value is not None and (type(value) is not int or value < 0)):
            raise ValueError("measured nonfoot history required for both participants")
    passer_other = outcome["passer_nonfoot_substep"] is not None
    receiver_other = outcome["receiver_nonfoot_substep"] is not None
    clean = outcome["strict_both_clean_handoff"]
    if clean and (
        passer_other
        or receiver_other
        or not outcome["both_bodies_healthy"]
        or not outcome["original_handoff"]
        or not outcome["strict_receiver_clean_handoff"]
        or not outcome["passer_foot_precedes_receiver"]
    ):
        raise ValueError("clean success contradicts observed physical outcome")
    return float(2 * clean - passer_other - receiver_other - (not outcome["both_bodies_healthy"]))
