"""Review every independently replayed receiver candidate pair and its old baseline."""

import argparse
import json
from pathlib import Path

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import measured, report
from scripts.rsi_review_whole_receiver_identity import artifact_equal, handoff_equal


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--group",
        nargs=4,
        action="append",
        required=True,
        metavar=("SOURCE", "AUDIT", "OLD", "OLD_AUDIT"),
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-head", required=True)
    p.add_argument("--old-head", required=True)
    p.add_argument("--expected-head", required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if len(a.group) != 3 or a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("three independent paired blocks and unused external output required")
    rows, blocks, frozen = [], set(), []
    for source_name, audit_name, old_name, old_audit_name in a.group:
        root, audit, old, old_audit = map(Path, (source_name, audit_name, old_name, old_audit_name))
        declaration = json.loads((root / "declaration.json").read_text())
        block = declaration["development_context_block"]
        if block in blocks or block not in (0, 1, 2):
            raise ValueError("three unique development blocks required")
        blocks.add(block)
        ah, oh = digest(audit), digest(old_audit)
        current, _, sh = report(
            root, audit, ah, "NEURAL_RELAY_RECEIVER_LEARNED_PAIR16_PENDING_OWN_REPLAY", worlds=16
        )
        previous, _, ph = report(
            old,
            old_audit,
            oh,
            "NEURAL_RELAY_WHOLE_RECEIVER_IDENTITY16_PENDING_OWN_REPLAY",
            worlds=16,
        )
        if current["source_head"] != a.source_head or previous["source_head"] != a.old_head:
            raise ValueError("exact frozen paired and historical source required")
        trial = {
            (
                r["course"]["context"],
                r["course"]["lateral_m"],
                r["course"]["arm"]["receiver_learned"],
            ): r
            for r in current["records"]
        }
        historic = {
            (r["course"]["context"], r["course"]["lateral_m"]): r
            for r in previous["records"]
            if r["course"]["arm"]["receiver_whole"] is True
        }
        if len(trial) != 16 or len(historic) != 8:
            raise ValueError("all paired arms and historical whole baselines required")
        for condition, prior in sorted(historic.items()):
            baseline, candidate = trial[(*condition, False)], trial[(*condition, True)]
            same = all(
                baseline[k] == prior[k]
                for k in ("termination", "controls", "native_steps", "result")
            ) and handoff_equal(root, baseline, old, prior)
            for name in prior["artifacts"]:
                same = same and artifact_equal(root, baseline, old, prior, name)
            rows.append(
                dict(
                    block=block,
                    context=condition[0],
                    side=condition[1],
                    historical_baseline_identical=same,
                    baseline=measured(root, baseline),
                    candidate=measured(root, candidate),
                )
            )
        frozen.extend(
            (
                (root / "complete.json", sh),
                (audit, ah),
                (old / "complete.json", ph),
                (old_audit, oh),
            )
        )
    if blocks != {0, 1, 2} or len(rows) != 24 or any(digest(path) != h for path, h in frozen):
        raise ValueError("all exact unmodified paired sources required")
    metrics = ("handoff", "outgoing_foot_strike", "return_0p1m_and_healthy", "body_failure")
    comparisons = {
        name: dict(
            rescued=sum(not r["baseline"][name] and r["candidate"][name] for r in rows),
            lost=sum(r["baseline"][name] and not r["candidate"][name] for r in rows),
            baseline=sum(r["baseline"][name] for r in rows),
            candidate=sum(r["candidate"][name] for r in rows),
        )
        for name in metrics
    }
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="RECEIVER_LEARNED_PAIR_PHYSICAL_REPLAY_REVIEWED"
            if all(r["historical_baseline_identical"] for r in rows)
            else "RECEIVER_LEARNED_PAIR_BASELINE_MISMATCH",
            source_head=a.expected_head,
            producer_head=a.source_head,
            conditions=len(rows),
            comparisons=comparisons,
            records=rows,
            independent_new_conditions=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
