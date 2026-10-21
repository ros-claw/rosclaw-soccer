"""Paired case-level retention is stricter than higher aggregate success."""

from typing import Any


def compare_router_frontier(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    if not reference or len(reference) != len(candidate):
        raise ValueError("same nonempty complete bank required")
    losses: list[list[int]] = []
    gains: list[list[int]] = []
    body_losses: list[int] = []
    body_gains: list[int] = []
    old_count, new_count = [0, 0, 0], [0, 0, 0]
    for i, (old, new) in enumerate(zip(reference, candidate, strict=True)):
        for r in (old, new):
            q = r.get("quality_by_kick")
            if (
                type(r.get("stable")) is not bool
                or not isinstance(q, list)
                or not 1 <= len(q) <= 3
                or any(type(v) is not bool for v in q)
                or (r["stable"] and len(q) != 3)
            ):
                raise ValueError("actual legal partial lifecycle outcomes required")
        body_losses += [i] if old["stable"] and not new["stable"] else []
        body_gains += [i] if new["stable"] and not old["stable"] else []
        for k in range(3):
            a = len(old["quality_by_kick"]) > k and old["quality_by_kick"][k]
            b = len(new["quality_by_kick"]) > k and new["quality_by_kick"][k]
            old_count[k] += a
            new_count[k] += b
            losses += [[i, k]] if a and not b else []
            gains += [[i, k]] if b and not a else []
    return dict(
        cases=len(reference),
        quality_losses=losses,
        quality_gains=gains,
        body_losses=body_losses,
        body_gains=body_gains,
        reference_quality=old_count,
        candidate_quality=new_count,
        reference_stable=sum(r["stable"] for r in reference),
        candidate_stable=sum(r["stable"] for r in candidate),
        retention_pass=not losses and not body_losses,
        has_actual_quality_growth=bool(gains),
        unique_declared_contexts=len(reference),
        statistical_independence_not_established=True,
        repeats_are_not_extra_contexts=True,
    )
