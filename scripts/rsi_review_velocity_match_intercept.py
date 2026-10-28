"""Compare paired causal velocity-match receiving against original physics."""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.paired_health_exposure import paired_health_at_common_horizon
from rosclaw_soccer.rsi.passer_curriculum_outcome import strict_pass_receive
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report
from scripts.rsi_review_whole_receiver_identity import artifact_equal
from scripts.rsi_watch_expanded_return import wait_unit


def summarize_pairs(rows: list[dict]) -> dict:
    def counts(pairs):
        return dict(
            baseline=sum(x for x, _ in pairs),
            candidate=sum(y for _, y in pairs),
            rescues=sum(not x and y for x, y in pairs),
            regressions=sum(x and not y for x, y in pairs),
        )

    metrics = (
        "strict_both_clean_handoff",
        "strict_receiver_clean_handoff",
        "passer_foot_precedes_receiver",
        "original_handoff",
    )
    summary = {
        metric: counts([(r["baseline"][metric], r["candidate"][metric]) for r in rows])
        for metric in metrics
    }
    summary["health_common"] = counts(
        [
            (
                r["health_common"]["raw_healthy_common"],
                r["health_common"]["executed_healthy_common"],
            )
            for r in rows
        ]
    )
    summary["unequal_exposure_pairs"] = sum(not r["health_common"]["equal_exposure"] for r in rows)
    nonfoot = [
        (
            r["baseline"]["receiver_nonfoot_substep"] is not None,
            r["candidate"]["receiver_nonfoot_substep"] is not None,
        )
        for r in rows
    ]
    summary["receiver_nonfoot"] = dict(
        baseline=sum(x for x, _ in nonfoot),
        candidate=sum(y for _, y in nonfoot),
        new_nonfoot=sum(not x and y for x, y in nonfoot),
        prevented_nonfoot=sum(x and not y for x, y in nonfoot),
    )
    summary["precision"] = counts(
        [(r["baseline_precision"], r["candidate_precision"]) for r in rows]
    )
    summary["changed_command_pairs"] = sum(r["maximum_command_delta"] > 1e-6 for r in rows)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    for key in ("producer-head", "expected-head", "wait-unit"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--timeout-sec", type=int, default=7200)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        re.fullmatch(r"rosclaw-receiver-velocity-match-b0-[0-9]{8}\.service", a.wait_unit) is None
        or not 1 <= a.timeout_sec <= 7200
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
    ):
        raise ValueError("exact bounded worker and unused external output required")
    wait_unit(a.wait_unit, time.monotonic() + a.timeout_sec)
    root, audit = a.source / "source", a.source / "audit" / "complete.json"
    ah = digest(audit)
    source, _, sh = report(
        root, audit, ah, "NEURAL_RELAY_RECEIVER_ADAPTIVE_INTERCEPT32_PENDING_OWN_REPLAY", worlds=32
    )
    declaration = json.loads((root / "declaration.json").read_text())
    if (
        source["source_head"] != a.producer_head
        or digest(root / "declaration.json") != source["declaration_hash"]
        or declaration.get("development_context_block") != 0
        or declaration.get("adaptive_intercept_is_unlearned_teacher") is not True
    ):
        raise ValueError("exact unlearned development study required")
    indexed = {}
    for r in source["records"]:
        arm = r["course"]["arm"]
        key = (
            r["course"]["context"],
            r["course"]["lateral_m"],
            arm["receiver_lateral_offset_m"],
            arm["receiver_adaptive_intercept"],
        )
        if key in indexed:
            raise ValueError("duplicate paired condition")
        indexed[key] = r
    if len(indexed) != 32:
        raise ValueError("32 complete paired development worlds required")
    rows = []
    for context, side, offset in sorted({key[:3] for key in indexed}):
        baseline = indexed[(context, side, offset, False)]
        candidate = indexed[(context, side, offset, True)]
        if not artifact_equal(root, baseline, root, candidate, "initial"):
            raise ValueError("paired initial physical state differs")
        paths = [
            bind(root, r, name)
            for r in (baseline, candidate)
            for name in ("physics", "receiver_velocity")
        ]
        if any(path is None for path in paths):
            raise ValueError("complete paired physical and control histories required")
        bp, bv, cp, cv = paths
        with (
            np.load(bp, allow_pickle=False) as b,
            np.load(cp, allow_pickle=False) as c,
            np.load(bv, allow_pickle=False) as va,
            np.load(cv, allow_pickle=False) as vb,
        ):
            n = min(len(va["control"]), len(vb["control"]))
            if (
                n < 1
                or not np.array_equal(va["control"][:n], np.arange(n))
                or not np.array_equal(vb["control"][:n], np.arange(n))
            ):
                raise ValueError("aligned actual control histories required")
            difference = vb["command"][:n] - va["command"][:n]
            if not np.isfinite(difference).all():
                raise ValueError("finite actual commands required")
            outcomes = [
                strict_pass_receive(
                    x["ball_forces"], x["body_failed"], handoff=r["result"]["handoff_executed"]
                )
                for x, r in ((b, baseline), (c, candidate))
            ]
            health = paired_health_at_common_horizon(b["body_failed"], c["body_failed"])
            rows.append(
                dict(
                    context=context,
                    side=side,
                    offset_m=offset,
                    baseline=outcomes[0],
                    candidate=outcomes[1],
                    health_common=health,
                    maximum_command_delta=float(np.abs(difference).max()),
                    baseline_precision=baseline["result"]["return_0p1m_and_healthy"],
                    candidate_precision=candidate["result"]["return_0p1m_and_healthy"],
                )
            )
    if len(rows) != 16 or digest(root / "complete.json") != sh or digest(audit) != ah:
        raise ValueError("unchanged complete paired source and independent replay required")
    summary = summarize_pairs(rows)
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="RECEIVER_VELOCITY_MATCH_PAIRED_DEVELOPMENT_REVIEWED",
            source_hash=sh,
            audit_hash=ah,
            source_head=a.expected_head,
            summary=summary,
            rows=rows,
            unlearned_sim_teacher=True,
            independent_fresh=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
