"""Matched physical comparison of projected/raw previous-action semantics."""


def compare_projection_memory(old: dict, new: dict) -> dict:
    if (
        any(
            not isinstance(x, dict)
            or x.get("status") != "RECEIVER_LEG_PROJECTION_REPLAY_REVIEWED"
            or x.get("independent_fresh") is not False
            or x.get("promotion_authorized") is not False
            or x.get("hardware_authorized") is not False
            or x.get("zero_identity_count") != 8
            for x in (old, new)
        )
        or old.get("raw_action_memory_preserved") is not True
        or new.get("candidate_action_memory_reflects_projection") is not True
        or old.get("prior_hash") != new.get("prior_hash")
    ):
        raise ValueError("two audited same-parent raw-vs-executed-memory studies required")

    def index(review):
        records = review.get("records")
        if not isinstance(records, list) or len(records) != 8:
            raise ValueError("eight original conditions required")
        result = {(r["context"], r["side"]): r for r in records}
        if len(result) != 8 or any(
            r.get("zero_projection_matches_prior") is not True for r in records
        ):
            raise ValueError("unique original conditions and exact zero identity required")
        return result

    before, after = index(old), index(new)
    if before.keys() != after.keys():
        raise ValueError("exact same original physical conditions required")
    rows = []
    for key in sorted(before):
        if before[key]["baseline"] != after[key]["baseline"]:
            raise ValueError("the zero-projection histories must match")
        rows.append(
            dict(
                context=key[0],
                side=key[1],
                raw=before[key]["projected"],
                executed=after[key]["projected"],
            )
        )
    selectors = {
        "strict_clean": lambda r: r["outcome"]["strict_both_clean_handoff"],
        "healthy": lambda r: r["outcome"]["both_bodies_healthy"],
        "foot_after_pass": lambda r: r["outcome"]["passer_foot_precedes_receiver"],
        "precise": lambda r: r["precise_healthy_return"],
    }
    summary = {}
    for name, select in selectors.items():
        pairs = [(select(r["raw"]), select(r["executed"])) for r in rows]
        if any(type(x) is not bool or type(y) is not bool for x, y in pairs):
            raise ValueError("actual boolean physical outcomes required")
        summary[name] = dict(
            raw=sum(x for x, _ in pairs),
            executed=sum(y for _, y in pairs),
            gained=sum(not x and y for x, y in pairs),
            lost=sum(x and not y for x, y in pairs),
        )
    return dict(
        conditions=8,
        summary=summary,
        records=rows,
        independent_fresh=False,
        promotion_authorized=False,
        hardware_authorized=False,
        goal_complete=False,
    )
