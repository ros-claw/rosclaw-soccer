"""Compare two fixed receiver feedback campaigns on matched SIM starts."""

import argparse
import json
from pathlib import Path

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import report
from scripts.rsi_review_whole_receiver_identity import artifact_equal


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dense-campaign", "motion-campaign", "dense-review", "motion-review", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "dense-campaign-hash",
        "motion-campaign-hash",
        "dense-review-hash",
        "motion-review-hash",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    pins = (
        (a.dense_campaign, a.dense_campaign_hash),
        (a.motion_campaign, a.motion_campaign_hash),
        (a.dense_review, a.dense_review_hash),
        (a.motion_review, a.motion_review_hash),
    )
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() or digest(p) != h for p, h in pins)
    ):
        raise ValueError("exact completed campaign reviews and unused external output required")
    dense, motion, dr, mr = [json.loads(p.read_text()) for p, _ in pins]
    if (
        dense.get("status") != "DENSE_RECEIVER_CAMPAIGN_ALL_OWN_REPLAYED_NOT_PROMOTED"
        or motion.get("status") != "MOTION_RECEIVER_CAMPAIGN_ALL_OWN_REPLAYED_NOT_PROMOTED"
        or dr.get("status") != "DENSE_RECEIVER_ALL_SEEDS_DEVELOPMENT_REVIEWED"
        or mr.get("status") != "MOTION_RECEIVER_ALL_SEEDS_DEVELOPMENT_REVIEWED"
        or dr.get("campaign_hash") != a.dense_campaign_hash
        or mr.get("campaign_hash") != a.motion_campaign_hash
        or any(
            d.get(k) is not False
            for d in (dense, motion, dr, mr)
            for k in ("promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("two complete unpromoted replayed comparisons required")
    drecords = {(r["seed"], r["block"]): r for r in dense["records"]}
    mrecords = {(r["seed"], r["block"]): r for r in motion["records"]}
    expected = {(seed, block) for seed in (92401, 92402, 92403) for block in range(3)}
    if set(drecords) != expected or set(mrecords) != expected:
        raise ValueError("all fixed seeds and blocks required")
    outcomes = []
    drs = {(r["seed"], r["block"], r["context"], r["side"]): r for r in dr["records"]}
    mrs = {(r["seed"], r["block"], r["context"], r["side"]): r for r in mr["records"]}
    if len(drs) != 72 or set(drs) != set(mrs):
        raise ValueError("complete same-context per-seed outcome coverage required")
    frozen = []
    for key in sorted(expected):
        pair = []
        for campaign, item, status in (
            (dense, drecords[key], "NEURAL_RELAY_RECEIVER_DENSE_PAIR16_PENDING_OWN_REPLAY"),
            (motion, mrecords[key], "NEURAL_RELAY_RECEIVER_MOTION_PAIR16_PENDING_OWN_REPLAY"),
        ):
            root, audit = Path(item["root"]) / "source", Path(item["root"]) / "audit/complete.json"
            source, _, sh = report(root, audit, item["audit_hash"], status, worlds=16)
            expected_head = campaign["pilot_head"] if key == (92401, 0) else campaign["source_head"]
            if sh != item["source_hash"] or source.get("source_head") != expected_head:
                raise ValueError("pinned source and independent replay required")
            indexed = {
                (
                    r["course"]["context"],
                    r["course"]["lateral_m"],
                    r["course"]["arm"]["receiver_learned"],
                ): r
                for r in source["records"]
            }
            if len(indexed) != 16:
                raise ValueError("complete baseline and candidate paired worlds required")
            pair.append((root, indexed))
            frozen.extend(((root / "complete.json", sh), (audit, item["audit_hash"])))
        (droot, dindex), (mroot, mindex) = pair
        if set(dindex) != set(mindex):
            raise ValueError("paired task conditions differ")
        for context, side, learned in sorted(dindex):
            if not learned:
                continue
            if not all(
                artifact_equal(left_root, left, right_root, right, "initial")
                for left_root, left, right_root, right in (
                    (droot, dindex[(context, side, False)], mroot, mindex[(context, side, False)]),
                    (droot, dindex[(context, side, True)], mroot, mindex[(context, side, True)]),
                )
            ):
                raise ValueError("initial physical states must match across rewards")
            identity = (*key, context, side)
            d, m = drs[identity], mrs[identity]
            outcomes.append(
                dict(
                    seed=key[0],
                    block=key[1],
                    context=context,
                    side=side,
                    dense=d["candidate"],
                    motion=m["candidate"],
                )
            )
    if len(outcomes) != 72 or any(digest(p) != h for p, h in (*pins, *frozen)):
        raise ValueError("all unchanged matched evidence required")
    summary = {}
    for seed in (92401, 92402, 92403):
        group = [r for r in outcomes if r["seed"] == seed]
        summary[str(seed)] = {}
        for metric in (
            "both_bodies_healthy",
            "strict_both_clean_handoff",
            "passer_foot_precedes_receiver",
        ):
            pairs = [(r["dense"][metric], r["motion"][metric]) for r in group]
            summary[str(seed)][metric] = dict(
                dense=sum(x for x, _ in pairs),
                motion=sum(y for _, y in pairs),
                gained=sum(not x and y for x, y in pairs),
                lost=sum(x and not y for x, y in pairs),
            )
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="RECEIVER_FEEDBACK_CAMPAIGNS_MATCHED_DEVELOPMENT_COMPARISON",
            source_head=a.expected_head,
            dense_campaign_hash=a.dense_campaign_hash,
            motion_campaign_hash=a.motion_campaign_hash,
            dense_review_hash=a.dense_review_hash,
            motion_review_hash=a.motion_review_hash,
            matched_pairs=72,
            summary=summary,
            records=outcomes,
            independent_fresh_contexts=False,
            correlated_seeds_not_independent_conditions=True,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
