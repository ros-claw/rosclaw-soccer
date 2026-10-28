"""Compare two replay-reviewed SIM passer ablations on matched conditions."""


def compare_passer_reviews(first: dict, exit_review: dict) -> dict:
    """Require identical historical baselines; count all paired outcomes.

    The returned research counts do not authorize a learned policy, hardware,
    or a claim about a fresh test distribution.
    """
    for report in (first, exit_review):
        if (
            report.get("status") != "PASSER_RESIDUAL_PAIR_REPLAY_REVIEWED"
            or report.get("conditions") != 24
            or any(
                report.get(key) is not False
                for key in ("hardware_authorized", "promotion_authorized", "goal_complete")
            )
        ):
            raise ValueError("complete unpromoted physical replay review required")
    indexed = []
    for report in (first, exit_review):
        rows = report.get("records")
        if not isinstance(rows, list) or len(rows) != 24:
            raise ValueError("24 reviewed pairs required")
        table = {(r["block"], r["context"], r["side"]): r for r in rows}
        if len(table) != 24 or any(
            r.get("historical_baseline_identical") is not True for r in rows
        ):
            raise ValueError("unique exact historical baselines required")
        indexed.append(table)
    old, new = indexed
    if old.keys() != new.keys() or any(old[k]["baseline"] != new[k]["baseline"] for k in old):
        raise ValueError("condition-matched physical baselines must be identical")
    metrics = {
        "strict_receiver_clean_handoff": lambda v: v["strict_receiver_clean_handoff"],
        "strict_both_clean_handoff": lambda v: v["strict_both_clean_handoff"],
        "original_handoff": lambda v: v["original_handoff"],
        "both_bodies_healthy": lambda v: v["both_bodies_healthy"],
        "precise_healthy_return": lambda v: v["original"]["return_0p1m_and_healthy"],
    }
    summary = {}
    for name, extract in metrics.items():
        pairs = [(extract(old[k]["teacher"]), extract(new[k]["teacher"])) for k in old]
        if any(type(value) is not bool for pair in pairs for value in pair):
            raise ValueError("strict boolean physical outcomes required")
        summary[name] = {
            "first_contact_exit": sum(a for a, _ in pairs),
            "full_contact_exit": sum(b for _, b in pairs),
            "rescues": sum(not a and b for a, b in pairs),
            "losses": sum(a and not b for a, b in pairs),
        }
    return {
        "conditions": 24,
        "summary": summary,
        "condition_keys": [list(k) for k in sorted(old)],
        "independent_fresh": False,
        "promotion_authorized": False,
        "hardware_authorized": False,
        "goal_complete": False,
    }
