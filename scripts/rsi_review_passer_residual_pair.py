"""Compare complete independently replayed passer interventions with old baselines."""

import argparse
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
    p.add_argument(
        "--group",
        nargs=4,
        action="append",
        required=True,
        metavar=("SOURCE", "AUDIT", "OLD", "OLD_AUDIT"),
    )
    for key in ("source-head", "old-head", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if len(a.group) != 3 or a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("three paired physical studies and unused output required")
    rows, blocks, frozen = [], set(), []
    for source_name, audit_name, old_name, old_audit_name in a.group:
        root, audit, old, old_audit = map(Path, (source_name, audit_name, old_name, old_audit_name))
        ah, oh = digest(audit), digest(old_audit)
        current, _, sh = report(
            root, audit, ah, "NEURAL_RELAY_PASSER_RESIDUAL_PAIR16_PENDING_OWN_REPLAY", worlds=16
        )
        previous, _, ph = report(
            old,
            old_audit,
            oh,
            "NEURAL_RELAY_WHOLE_RECEIVER_IDENTITY16_PENDING_OWN_REPLAY",
            worlds=16,
        )
        if current["source_head"] != a.source_head or previous["source_head"] != a.old_head:
            raise ValueError("exact frozen source histories required")
        current_rows = {
            (
                r["course"]["context"],
                r["course"]["lateral_m"],
                r["course"]["arm"]["passer_residual_fraction"],
            ): r
            for r in current["records"]
        }
        old_rows = {
            (r["course"]["context"], r["course"]["lateral_m"]): r
            for r in previous["records"]
            if r["course"]["arm"]["receiver_whole"] is False
        }
        if len(current_rows) != 16 or len(old_rows) != 8:
            raise ValueError("complete paired source and historical baseline required")
        block = None
        group_blocks = set()
        for key, prior in sorted(old_rows.items()):
            base, teacher = current_rows[(*key, 1.0)], current_rows[(*key, 0.5)]
            block = next(
                (
                    i
                    for i, group in enumerate(((2, 3, 5, 7), (8, 9, 11, 13), (15, 16, 17, 18)))
                    if key[0] in group
                ),
                None,
            )
            if block is None:
                raise ValueError("explicit consumed original development condition required")
            group_blocks.add(block)
            same = all(
                base[k] == prior[k] for k in ("termination", "controls", "native_steps", "result")
            ) and handoff_equal(root, base, old, prior)
            same = same and all(
                artifact_equal(root, base, old, prior, name) for name in prior["artifacts"]
            )
            outcomes = {}
            for name, row in (("baseline", base), ("teacher", teacher)):
                path = bind(root, row, "physics")
                if path is None:
                    raise ValueError("complete native contact history required")
                with np.load(path, allow_pickle=False) as f:
                    outcomes[name] = strict_pass_receive(
                        f["ball_forces"],
                        f["body_failed"],
                        handoff=row["result"]["handoff_executed"],
                    )
                outcomes[name]["original"] = measured(root, row)
            trace = bind(root, teacher, "passer_nn")
            if trace is None:
                raise ValueError("actual teacher action history required")
            with np.load(trace, allow_pickle=False) as t:
                active = t["BLEND_active"]
                actual = t["BLEND_action"]
                proposed = t["NN_action"]
                if (
                    active.dtype != np.bool_
                    or actual.shape != proposed.shape
                    or actual.shape != (teacher["controls"], 29)
                ):
                    raise ValueError("complete actual teacher and original proposals required")
                changed = np.any(actual != proposed, axis=1)
                if (changed & ~active).any() or np.any(np.diff(active.astype(np.int8)) > 0):
                    raise ValueError("teacher must stop permanently after actual passer foot event")
                changed_controls = int(changed.sum())
            rows.append(
                dict(
                    block=block,
                    context=key[0],
                    side=key[1],
                    historical_baseline_identical=same,
                    teacher_changed_controls=changed_controls,
                    **outcomes,
                )
            )
        if len(group_blocks) != 1 or block in blocks:
            raise ValueError("unique three development blocks required")
        blocks.add(block)
        frozen.extend(
            (
                (root / "complete.json", sh),
                (audit, ah),
                (old / "complete.json", ph),
                (old_audit, oh),
            )
        )
    if blocks != {0, 1, 2} or len(rows) != 24 or any(digest(p) != h for p, h in frozen):
        raise ValueError("all exact independently replayed pairs required")
    summary = dict(
        strict_baseline=sum(r["baseline"]["strict_receiver_clean_handoff"] for r in rows),
        strict_teacher=sum(r["teacher"]["strict_receiver_clean_handoff"] for r in rows),
        strict_rescues=sum(
            not r["baseline"]["strict_receiver_clean_handoff"]
            and r["teacher"]["strict_receiver_clean_handoff"]
            for r in rows
        ),
        strict_losses=sum(
            r["baseline"]["strict_receiver_clean_handoff"]
            and not r["teacher"]["strict_receiver_clean_handoff"]
            for r in rows
        ),
        original_handoff_baseline=sum(r["baseline"]["original_handoff"] for r in rows),
        original_handoff_teacher=sum(r["teacher"]["original_handoff"] for r in rows),
        original_handoff_rescues=sum(
            not r["baseline"]["original_handoff"] and r["teacher"]["original_handoff"] for r in rows
        ),
        original_handoff_losses=sum(
            r["baseline"]["original_handoff"] and not r["teacher"]["original_handoff"] for r in rows
        ),
        precise_old_success_losses=sum(
            r["baseline"]["original"]["return_0p1m_and_healthy"]
            and not r["teacher"]["original"]["return_0p1m_and_healthy"]
            for r in rows
        ),
        new_body_failure=sum(
            r["baseline"]["both_bodies_healthy"] and not r["teacher"]["both_bodies_healthy"]
            for r in rows
        ),
        teacher_active_worlds=sum(r["teacher_changed_controls"] > 0 for r in rows),
    )
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="PASSER_RESIDUAL_PAIR_REPLAY_REVIEWED"
            if all(r["historical_baseline_identical"] for r in rows)
            else "PASSER_RESIDUAL_BASELINE_MISMATCH",
            source_head=a.expected_head,
            producer_head=a.source_head,
            conditions=24,
            summary=summary,
            records=rows,
            teacher_not_learned=True,
            passer_nonfoot_not_evaluated=True,
            independent_fresh=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
