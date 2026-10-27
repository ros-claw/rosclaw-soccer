"""Matched physical relay outcomes; no model selection or promotion."""

from collections import Counter
from typing import Any


def compare_balance_outcomes(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for parent, candidate in pairs:
        for row in (parent, candidate):
            if (
                type(row.get("return_0p1m_and_healthy")) is not bool
                or type(row.get("body_failure")) is not bool
                or type(row.get("outgoing_foot_strike")) is not bool
            ):
                raise ValueError("explicit paired measured relay outcomes required")
        p, c = parent["return_0p1m_and_healthy"], candidate["return_0p1m_and_healthy"]
        pf, cf = parent["body_failure"], candidate["body_failure"]
        counts.update(
            worlds=1,
            parent_precise_healthy=int(p),
            candidate_precise_healthy=int(c),
            precise_rescues=int(not p and c),
            precise_regressions=int(p and not c),
            parent_body_failure=int(pf),
            candidate_body_failure=int(cf),
            body_failure_rescues=int(pf and not cf),
            body_failure_regressions=int(not pf and cf),
            candidate_outgoing_strike=int(candidate["outgoing_foot_strike"]),
        )
    return dict(counts)
