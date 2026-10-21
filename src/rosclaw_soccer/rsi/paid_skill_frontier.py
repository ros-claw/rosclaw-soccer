"""Per-case skill retention comparisons; counts alone cannot prove growth.

Pure arithmetic on declared same-bank outcomes. Callers must separately admit
complete audited reports and identical case declarations. Not a promotion gate.
"""

from typing import Any


def _outcomes(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = report.get("worlds", [])
    if (
        len(rows) != 16
        or any(type(r.get("case")) is not int for r in rows)
        or [r.get("case") for r in rows] != list(range(16))
    ):
        raise ValueError("all16 ordered cases required, including failures")
    for row in rows:
        quality = row.get("quality_by_kick")
        if (
            not isinstance(quality, list)
            or len(quality) > 3
            or any(type(x) is not bool for x in quality)
            or type(row.get("stable")) is not bool
        ):
            raise ValueError("explicit boolean outcomes and body status required")
    return rows


def compare_skill_frontier(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    bank = reference.get("bank", reference.get("course_bank"))
    if bank not in ("train", "dev") or candidate.get("bank", candidate.get("course_bank")) != bank:
        raise ValueError("same declared bank required; TRAIN and DEV cannot be pooled")
    old, new = _outcomes(reference), _outcomes(candidate)
    old_success, new_success = [], []
    for k in range(3):
        old_success.append(
            [
                i
                for i, r in enumerate(old)
                if len(r["quality_by_kick"]) > k and r["quality_by_kick"][k]
            ]
        )
        new_success.append(
            [
                i
                for i, r in enumerate(new)
                if len(r["quality_by_kick"]) > k and r["quality_by_kick"][k]
            ]
        )
    gained = [sorted(set(b) - set(a)) for a, b in zip(old_success, new_success, strict=True)]
    lost = [sorted(set(a) - set(b)) for a, b in zip(old_success, new_success, strict=True)]
    body_lost = [
        i for i, (a, b) in enumerate(zip(old, new, strict=True)) if a["stable"] and not b["stable"]
    ]
    return dict(
        bank=bank,
        reference_successes_by_kick=old_success,
        candidate_successes_by_kick=new_success,
        gained_by_kick=gained,
        lost_by_kick=lost,
        reference_counts=[len(x) for x in old_success],
        candidate_counts=[len(x) for x in new_success],
        body_regressions=body_lost,
        no_regression=not any(lost) and not body_lost,
        promotion_authorized=False,
    )
