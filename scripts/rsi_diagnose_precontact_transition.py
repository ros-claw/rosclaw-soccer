"""Measure paired physical transition and failure timing; SIM_ONLY, no training."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, action="append", required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    review = json.loads(a.review.read_text())
    if (
        len(a.source) != 3
        or review.get("conditions") != 24
        or review.get("status") != "PRECONTACT_FULLBODY_PAIR_REPLAY_REVIEWED"
        or a.output.exists()
    ):
        raise ValueError("three reviewed paired blocks and unused output required")
    rows = []
    frozen = [(a.review, digest(a.review))]
    for root in a.source:
        receipt = root / "complete.json"
        doc = json.loads(receipt.read_text())
        frozen.append((receipt, digest(receipt)))
        paired = {}
        for record in doc["records"]:
            c = record["course"]
            paired.setdefault((c["context"], c["lateral_m"]), {})[
                c["arm"]["precontact_fullbody"]
            ] = record
        if len(paired) != 8 or any(set(pair) != {True, False} for pair in paired.values()):
            raise ValueError("incomplete paired block")
        for (context, side), pair in sorted(paired.items()):
            base, candidate = pair[False], pair[True]
            t = candidate["handoff"]["control"]
            arrays = []
            for row in (base, candidate):
                artifact = row["artifacts"]["physics"]
                path = root / artifact["path"]
                if digest(path) != artifact["hash"]:
                    raise ValueError("physics artifact hash mismatch")
                frozen.append((path, artifact["hash"]))
                with np.load(path, allow_pickle=False) as f:
                    arrays.append(
                        {k: f[k].copy() for k in ("qpos", "qvel", "target_b", "body_failed")}
                    )
            b, c = arrays
            if t < 1 or min(len(b["qpos"]), len(c["qpos"])) <= t:
                raise ValueError("transition outside paired physics")
            prefix_exact = all(
                np.array_equal(b[k][:t], c[k][:t])
                for k in ("qpos", "qvel", "target_b", "body_failed")
            )
            failed = np.flatnonzero(c["body_failed"][:, 1])
            baseline_failed = np.flatnonzero(b["body_failed"][:, 1])
            rows.append(
                dict(
                    context=context,
                    side=side,
                    transition_control=t,
                    prefix_exact=prefix_exact,
                    candidate_target_jump=float(
                        np.max(np.abs(c["target_b"][t] - c["target_b"][t - 1]))
                    ),
                    baseline_target_jump=float(
                        np.max(np.abs(b["target_b"][t] - b["target_b"][t - 1]))
                    ),
                    first_candidate_b_failure=int(failed[0]) if len(failed) else None,
                    first_baseline_b_failure=int(baseline_failed[0])
                    if len(baseline_failed)
                    else None,
                    controls_to_b_failure=int(failed[0]) - t if len(failed) else None,
                )
            )
    if len(rows) != 24 or any(digest(path) != value for path, value in frozen):
        raise ValueError("source changed during diagnosis")
    jumps = [r["candidate_target_jump"] for r in rows]
    bjumps = [r["baseline_target_jump"] for r in rows]
    summary = dict(
        pairs=len(rows),
        exact_pretransition_prefixes=sum(r["prefix_exact"] for r in rows),
        candidate_b_failures=sum(r["first_candidate_b_failure"] is not None for r in rows),
        baseline_b_failures=sum(r["first_baseline_b_failure"] is not None for r in rows),
        median_candidate_transition_jump_rad=float(np.median(jumps)),
        median_baseline_same_control_jump_rad=float(np.median(bjumps)),
        median_controls_to_candidate_b_failure=float(
            np.median(
                [r["controls_to_b_failure"] for r in rows if r["controls_to_b_failure"] is not None]
            )
        )
        if any(r["controls_to_b_failure"] is not None for r in rows)
        else None,
    )
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="PAIRED_TRANSITION_DIAGNOSED",
            summary=summary,
            records=rows,
            review_hash=frozen[0][1],
            source_head=a.expected_head,
            trained=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
