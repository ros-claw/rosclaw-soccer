"""Development-only retention selection from integrity-checked learning audits.

This selects a research continuation, never activates a runtime policy. Counts
cannot compensate for losing a previously qualified physical course.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json


def assess_retention(audit: dict[str, Any]) -> dict[str, Any]:
    payload = dict(audit)
    digest = payload.pop("audit_hash", None)
    if digest != hash_json(payload):
        raise ValueError("learning audit integrity differs")
    if (
        audit.get("activation_ceiling") != "SIM_ONLY"
        or audit.get("promotion_eligible") is not False
    ):
        raise ValueError("research-only learning evidence required")
    outcomes = audit["evaluation"]
    baseline = "parent" if "parent" in outcomes else "zero"
    if set(outcomes) != {baseline, "candidate"}:
        raise ValueError("one paired parent/candidate examination required")

    def index(rows: list[dict[str, Any]]) -> dict[tuple[str, bool, float], dict[str, Any]]:
        result = {}
        for row in rows:
            if (
                row["role"] not in {"goalkeeper", "defender", "playmaker", "finisher"}
                or type(row["blue"]) is not bool
                or type(row["offset"]) not in {int, float}
                or not math.isfinite(row["offset"])
                or abs(row["offset"]) > 0.2
                or type(row["qualified_passes"]) is not int
                or row["qualified_passes"] < 0
                or any(
                    type(row[name]) is not bool
                    for name in (
                        "safe",
                        "exact_replay",
                        "foot_only_ball_control",
                        "full_match_passed",
                    )
                )
                or not row["exact_replay"]
                or (
                    row["qualified_passes"] > 0
                    and not (row["safe"] and row["foot_only_ball_control"])
                )
            ):
                raise ValueError("invalid physical retention course")
            key = (row["role"], row["blue"], row["offset"])
            if key in result:
                raise ValueError("duplicate retention course")
            result[key] = row
        if not result:
            raise ValueError("empty retention examination")
        return result

    before, after = index(outcomes[baseline]), index(outcomes["candidate"])
    if before.keys() != after.keys():
        raise ValueError("retention contexts differ")
    lost, gained = [], []
    for key, old in before.items():
        new = after[key]
        identity = {"role": key[0], "blue": key[1], "offset": key[2]}
        losses = [
            name
            for name in ("safe", "foot_only_ball_control", "full_match_passed")
            if old[name] and not new[name]
        ]
        if new["qualified_passes"] < old["qualified_passes"]:
            losses.append("qualified_passes")
        if losses:
            lost.append({**identity, "regressions": losses})
        if new["qualified_passes"] > old["qualified_passes"]:
            gained.append(identity)
    result = {
        "schema": "s232.role_retention.v1",
        "learning_audit_hash": digest,
        "retained_all_courses": not lost,
        "regressed_courses": lost,
        "new_pass_courses": gained,
        "research_continuation_accepted": not lost and bool(gained),
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "evaluation_boundary": "development examination; not an independent blind test",
    }
    result["decision_hash"] = hash_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess_retention(json.loads(args.audit.read_text()))
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
