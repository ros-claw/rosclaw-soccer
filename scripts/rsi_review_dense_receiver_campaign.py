"""Compare all own-replayed SIM receiver candidates with paired zero residual."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.passer_curriculum_outcome import strict_pass_receive
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, measured, report
from scripts.rsi_review_whole_receiver_identity import artifact_equal


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--campaign-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    cp = a.campaign / "complete.json"
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or cp.is_symlink()
        or digest(cp) != a.campaign_hash
    ):
        raise ValueError("pinned complete campaign and unused external output required")
    campaign = json.loads(cp.read_text())
    if (
        campaign.get("status") != "DENSE_RECEIVER_CAMPAIGN_ALL_OWN_REPLAYED_NOT_PROMOTED"
        or len(campaign.get("records", [])) != 9
        or any(
            campaign.get(k) is not False
            for k in ("independent_fresh_contexts", "promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("all nine unpromoted own-replayed campaign blocks required")
    rows, seen, frozen = [], set(), []
    for item in campaign["records"]:
        seed, block = item["seed"], item["block"]
        if (seed, block) in seen or seed not in (92401, 92402, 92403) or block not in (0, 1, 2):
            raise ValueError("each declared seed and block must occur once")
        seen.add((seed, block))
        root = Path(item["root"]) / "source"
        audit = Path(item["root"]) / "audit/complete.json"
        source, _, sh = report(
            root,
            audit,
            item["audit_hash"],
            "NEURAL_RELAY_RECEIVER_DENSE_PAIR16_PENDING_OWN_REPLAY",
            worlds=16,
        )
        expected_head = (
            campaign["pilot_head"] if (seed, block) == (92401, 0) else campaign["source_head"]
        )
        if sh != item["source_hash"] or source.get("source_head") != expected_head:
            raise ValueError("exact pinned physical producer required")
        indexed = {
            (
                r["course"]["context"],
                r["course"]["lateral_m"],
                r["course"]["arm"]["receiver_learned"],
            ): r
            for r in source["records"]
        }
        if len(indexed) != 16:
            raise ValueError("complete paired physical coverage required")
        for context, side, learned in sorted(indexed):
            if not learned:
                continue
            base, candidate = indexed[(context, side, False)], indexed[(context, side, True)]
            if side not in (-0.35, 0.35) or not artifact_equal(
                root, base, root, candidate, "initial"
            ):
                raise ValueError("exact preregistered position and identical start required")
            result = {}
            for name, row in (("baseline", base), ("candidate", candidate)):
                path = bind(root, row, "physics")
                if path is None:
                    raise ValueError("full physical contact trace required")
                with np.load(path, allow_pickle=False) as raw:
                    values = strict_pass_receive(
                        raw["ball_forces"],
                        raw["body_failed"],
                        handoff=row["result"]["handoff_executed"],
                    )
                values["original"] = measured(root, row)
                result[name] = values
            rows.append(dict(seed=seed, block=block, context=context, side=side, **result))
        frozen.extend(((root / "complete.json", sh), (audit, item["audit_hash"])))
    if (
        len(rows) != 72
        or len(seen) != 9
        or digest(cp) != a.campaign_hash
        or any(digest(path) != expected for path, expected in frozen)
    ):
        raise ValueError("all unmodified paired world histories required")
    summary = {}
    for seed in (92401, 92402, 92403):
        group = [r for r in rows if r["seed"] == seed]
        summary[str(seed)] = {}
        for name in ("baseline", "candidate"):
            summary[str(seed)][name] = dict(
                conditions=len(group),
                strict_both_clean=sum(r[name]["strict_both_clean_handoff"] for r in group),
                healthy_both=sum(r[name]["both_bodies_healthy"] for r in group),
                receiver_foot_after_passer=sum(
                    r[name]["passer_foot_precedes_receiver"] for r in group
                ),
                receiver_nonfoot_contact=sum(
                    r[name]["receiver_nonfoot_substep"] is not None for r in group
                ),
                precise_healthy_return=sum(
                    r[name]["original"]["return_0p1m_and_healthy"] for r in group
                ),
            )
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="DENSE_RECEIVER_ALL_SEEDS_DEVELOPMENT_REVIEWED",
            source_head=a.expected_head,
            campaign_hash=a.campaign_hash,
            independent_fresh_contexts=False,
            correlated_seeds_not_independent_conditions=True,
            summary=summary,
            records=rows,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
