"""Compare own-replayed development pilot to its exact unsmoothed history."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.passer_curriculum_outcome import strict_pass_receive
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, measured, report
from scripts.rsi_review_whole_receiver_identity import artifact_equal, handoff_equal


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "audit", "prior", "prior-audit", "output"):
        p.add_argument("--" + arg, type=Path, required=True)
    for arg in ("expected-head", "source-head", "prior-head"):
        p.add_argument("--" + arg, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external output required")
    status = "NEURAL_RELAY_PRECONTACT_FULLBODY16_PENDING_OWN_REPLAY"
    audit_hash, prior_audit_hash = digest(a.audit), digest(a.prior_audit)
    source, _, source_hash = report(a.source, a.audit, audit_hash, status, worlds=16)
    prior, _, prior_hash = report(a.prior, a.prior_audit, prior_audit_hash, status, worlds=16)
    if source["source_head"] != a.source_head or prior["source_head"] != a.prior_head:
        raise ValueError("exact physical producers required")
    declaration = json.loads((a.source / "declaration.json").read_text())
    if (
        declaration.get("protocol") != "soccer.neural_relay_precontact_smooth.development.v16"
        or declaration.get("precontact_smooth_controls") != 10
    ):
        raise ValueError("preregistered bounded smooth motor required")

    def index(doc):
        return {
            (
                r["course"]["context"],
                r["course"]["lateral_m"],
                r["course"]["arm"]["precontact_fullbody"],
            ): r
            for r in doc["records"]
        }

    current, old = index(source), index(prior)
    if len(current) != 16 or current.keys() != old.keys():
        raise ValueError("all eight paired conditions required")
    rows = []
    for context, side, enabled in sorted(current):
        if not enabled:
            continue
        base, old_base = current[(context, side, False)], old[(context, side, False)]
        candidate, abrupt = current[(context, side, True)], old[(context, side, True)]
        baseline_equal = (
            all(
                base[k] == old_base[k]
                for k in ("termination", "controls", "native_steps", "result")
            )
            and handoff_equal(a.source, base, a.prior, old_base)
            and all(artifact_equal(a.source, base, a.prior, old_base, n) for n in base["artifacts"])
        )
        same_entry = handoff_equal(a.source, candidate, a.prior, abrupt)
        values = {}
        for name, root, row in (
            ("baseline", a.source, base),
            ("abrupt", a.prior, abrupt),
            ("smooth", a.source, candidate),
        ):
            path = bind(root, row, "physics")
            with np.load(path, allow_pickle=False) as f:
                result = strict_pass_receive(
                    f["ball_forces"], f["body_failed"], handoff=row["result"]["handoff_executed"]
                )
                targets = f["target_b"]
                t = row["handoff"]["control"] if row["handoff"] else None
                result["transition_jump_rad"] = (
                    float(np.max(np.abs(targets[t] - targets[t - 1])))
                    if t is not None and t > 0
                    else None
                )
            result["original"] = measured(root, row)
            values[name] = result
        rows.append(
            dict(
                context=context,
                side=side,
                baseline_exact=baseline_equal,
                same_candidate_handoff=same_entry,
                **values,
            )
        )
    if len(rows) != 8 or not all(r["baseline_exact"] and r["same_candidate_handoff"] for r in rows):
        raise ValueError("baseline or causal candidate entry differs")
    if (
        digest(a.audit) != audit_hash
        or digest(a.prior_audit) != prior_audit_hash
        or digest(a.source / "complete.json") != source_hash
        or digest(a.prior / "complete.json") != prior_hash
    ):
        raise ValueError("input changed during review")
    summary = {}
    for name in ("baseline", "abrupt", "smooth"):
        jumps = [
            r[name]["transition_jump_rad"]
            for r in rows
            if r[name]["transition_jump_rad"] is not None
        ]
        summary[name] = dict(
            both_healthy=sum(r[name]["both_bodies_healthy"] for r in rows),
            strict_clean_handoff=sum(r[name]["strict_both_clean_handoff"] for r in rows),
            median_transition_jump_rad=float(np.median(jumps)) if jumps else None,
        )
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="SMOOTH_PILOT_OWN_REPLAY_REVIEWED",
            source_head=a.expected_head,
            conditions=8,
            summary=summary,
            records=rows,
            source_hash=source_hash,
            prior_hash=prior_hash,
            independent_fresh=False,
            trained=False,
            promotion_authorized=False,
            goal_complete=False,
            hardware_authorized=False,
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
