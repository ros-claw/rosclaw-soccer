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


def admit_strict_training(records: list[dict]) -> dict:
    """Count distinct physical conditions; repeated seeds are not conditions.

    This is a data sufficiency gate only. It neither runs an optimizer nor
    establishes generalization, physical retention, or policy promotion.
    """
    if not isinstance(records, list) or len(records) != 72:
        raise ValueError("all 72 reviewed sampled episodes required")
    keys, positive, negative = set(), set(), set()
    for row in records:
        context, side, seed = row.get("context"), row.get("side"), row.get("seed")
        if (
            type(context) is not int
            or type(side) is not float
            or side not in (-0.2, 0.2)
            or type(seed) is not int
            or seed not in (92117, 92143, 92171)
        ):
            raise ValueError("explicit center condition and sampling seed required")
        key = (context, side, seed)
        if key in keys:
            raise ValueError("duplicate sampled episode")
        keys.add(key)
        score = strict_terminal_reward(row["physical_clean_outcome"])
        if type(row.get("terminal_reward")) is not float or row["terminal_reward"] != score:
            raise ValueError("reward must match observed full physical outcome")
        if score > 0:
            positive.add((context, side))
        elif score < 0:
            negative.add((context, side))
    conditions = {(context, side) for context, side, _ in keys}
    if len(conditions) != 24 or any(
        (context, side, seed) not in keys
        for context, side in conditions
        for seed in (92117, 92143, 92171)
    ):
        raise ValueError("complete 24 conditions and all three seeds required")
    return {
        "conditions": len(conditions),
        "positive_conditions": len(positive),
        "negative_conditions": len(negative),
        "research_training_data_sufficient": len(positive) >= 4 and len(negative) >= 4,
        "optimizer_authorized": False,
        "promotion_authorized": False,
        "hardware_authorized": False,
    }
