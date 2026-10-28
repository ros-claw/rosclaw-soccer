"""Compare the complete first fixed block of two own-replayed receiver pilots."""

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
    for name in ("dense", "motion", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("dense-head", "motion-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external output required")
    sources, frozen = [], []
    for root, status, head in (
        (a.dense, "NEURAL_RELAY_RECEIVER_DENSE_PAIR16_PENDING_OWN_REPLAY", a.dense_head),
        (a.motion, "NEURAL_RELAY_RECEIVER_MOTION_PAIR16_PENDING_OWN_REPLAY", a.motion_head),
    ):
        source_root, audit = root / "source", root / "audit/complete.json"
        ah = digest(audit)
        source, _, sh = report(source_root, audit, ah, status, worlds=16)
        if source["source_head"] != head:
            raise ValueError("exact own-replayed producer required")
        dec = json.loads((source_root / "declaration.json").read_text())
        if dec.get("development_context_block") != 0:
            raise ValueError("entire fixed development pilot block required")
        indexed = {
            (
                r["course"]["context"],
                r["course"]["lateral_m"],
                r["course"]["arm"]["receiver_learned"],
            ): r
            for r in source["records"]
        }
        if len(indexed) != 16:
            raise ValueError("complete two-arm world coverage required")
        sources.append((source_root, indexed))
        frozen.extend(((source_root / "complete.json", sh), (audit, ah)))
    (dr, di), (mr, mi) = sources
    if set(di) != set(mi):
        raise ValueError("matching physical task conditions required")
    rows = []
    for context, side, learned in sorted(di):
        if not learned:
            continue
        base, other = di[(context, side, False)], mi[(context, side, False)]
        if (
            side not in (-0.35, 0.35)
            or not all(
                base[k] == other[k] for k in ("termination", "controls", "native_steps", "result")
            )
            or not handoff_equal(dr, base, mr, other)
            or not all(artifact_equal(dr, base, mr, other, key) for key in base["artifacts"])
            or not artifact_equal(
                dr, di[(context, side, True)], mr, mi[(context, side, True)], "initial"
            )
        ):
            raise ValueError("zero baseline or paired physical start changed")
        outcomes = {}
        for name, root, row in (
            ("baseline", dr, base),
            ("dense", dr, di[(context, side, True)]),
            ("motion", mr, mi[(context, side, True)]),
        ):
            path = bind(root, row, "physics")
            if path is None:
                raise ValueError("full native force and failure history required")
            with np.load(path, allow_pickle=False) as raw:
                v = strict_pass_receive(
                    raw["ball_forces"],
                    raw["body_failed"],
                    handoff=row["result"]["handoff_executed"],
                )
            v["original"] = measured(root, row)
            outcomes[name] = v
        rows.append(dict(context=context, side=side, **outcomes))
    if len(rows) != 8 or any(digest(p) != h for p, h in frozen):
        raise ValueError("all unchanged pilot histories required")
    summary = {}
    for name in ("baseline", "dense", "motion"):
        summary[name] = dict(
            healthy_both=sum(r[name]["both_bodies_healthy"] for r in rows),
            strict_both_clean=sum(r[name]["strict_both_clean_handoff"] for r in rows),
            receiver_foot_after_pass=sum(r[name]["passer_foot_precedes_receiver"] for r in rows),
            receiver_nonfoot=sum(r[name]["receiver_nonfoot_substep"] is not None for r in rows),
            precise_healthy_return=sum(
                r[name]["original"]["return_0p1m_and_healthy"] for r in rows
            ),
        )
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="RECEIVER_FEEDBACK_PILOT_MATCHED_DEVELOPMENT_COMPARISON",
            source_head=a.expected_head,
            dense_head=a.dense_head,
            motion_head=a.motion_head,
            conditions=8,
            seed=92401,
            summary=summary,
            records=rows,
            all_seeds_or_blocks=False,
            independent_fresh_contexts=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
