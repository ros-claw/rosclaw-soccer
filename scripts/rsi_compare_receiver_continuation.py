"""Compare replayed zero, parent and continued motor on matched SIM starts."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.passer_curriculum_outcome import strict_pass_receive
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, measured, report
from scripts.rsi_review_whole_receiver_identity import artifact_equal, handoff_equal
from scripts.rsi_watch_passer_residual_pair import worker_ready


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("continuation", "parent", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("continuation-head", "parent-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--wait-unit")
    p.add_argument("--block", type=int, choices=(0, 1, 2), default=0)
    p.add_argument("--timeout-sec", type=int, default=7200)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or not 1 <= a.timeout_sec <= 7200
        or a.wait_unit is not None
        and (
            not a.wait_unit.startswith("rosclaw-receiver-continuation-eval-")
            or not a.wait_unit.endswith(".service")
            or "/" in a.wait_unit
        )
    ):
        raise ValueError("bounded SIM worker and unused external output required")
    if a.wait_unit:
        deadline = time.monotonic() + a.timeout_sec
        while True:
            q = subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "show",
                    a.wait_unit,
                    "-p",
                    "LoadState",
                    "-p",
                    "ActiveState",
                    "-p",
                    "Result",
                    "-p",
                    "ExecMainStatus",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            state = dict(line.split("=", 1) for line in q.stdout.splitlines() if "=" in line)
            if worker_ready(a.continuation, state):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("physics observation expired; do not restart it")
            time.sleep(15)
    sources, frozen = [], []
    for root, status, head in (
        (
            a.continuation,
            "NEURAL_RELAY_RECEIVER_CONTINUATION_PAIR16_PENDING_OWN_REPLAY",
            a.continuation_head,
        ),
        (a.parent, "NEURAL_RELAY_RECEIVER_DENSE_PAIR16_PENDING_OWN_REPLAY", a.parent_head),
    ):
        source, audit = root / "source", root / "audit/complete.json"
        ah = digest(audit)
        data, _, sh = report(source, audit, ah, status, worlds=16)
        dec = json.loads((source / "declaration.json").read_text())
        if data["source_head"] != head or dec.get("development_context_block") != a.block:
            raise ValueError("exact fixed developmental pilot producers required")
        entries = {
            (
                r["course"]["context"],
                r["course"]["lateral_m"],
                r["course"]["arm"]["receiver_learned"],
            ): r
            for r in data["records"]
        }
        if len(entries) != 16:
            raise ValueError("complete paired factorial required")
        sources.append((source, entries, dec))
        frozen.extend(((source / "complete.json", sh), (audit, ah)))
    (cr, ci, cd), (pr, pi, pd) = sources
    cp = Path(cd["receiver_candidate_audit"])
    if digest(cp) != cd["receiver_candidate_audit_hash"]:
        raise ValueError("exact candidate audit required")
    candidate_audit = json.loads(cp.read_text())
    if candidate_audit.get("parent_hash") != pd["receiver_checkpoint_hash"]:
        raise ValueError("exact actual learned parent checkpoint required")
    rows = []
    for context, side, learned in sorted(ci):
        if not learned:
            continue
        cb, pb = ci[(context, side, False)], pi[(context, side, False)]
        if (
            side not in (-0.35, 0.35)
            or set(ci) != set(pi)
            or any(cb[k] != pb[k] for k in ("termination", "controls", "native_steps", "result"))
            or not handoff_equal(cr, cb, pr, pb)
            or not all(artifact_equal(cr, cb, pr, pb, key) for key in pb["artifacts"])
            or not artifact_equal(
                cr, ci[(context, side, True)], pr, pi[(context, side, True)], "initial"
            )
        ):
            raise ValueError("zero reference and candidate physical initial conditions must match")
        outcomes = {}
        for name, root, trial in (
            ("zero", cr, cb),
            ("parent", pr, pi[(context, side, True)]),
            ("continuation", cr, ci[(context, side, True)]),
        ):
            physical = bind(root, trial, "physics")
            if physical is None:
                raise ValueError("complete native physics trace required")
            with np.load(physical, allow_pickle=False) as raw:
                outcome = strict_pass_receive(
                    raw["ball_forces"],
                    raw["body_failed"],
                    handoff=trial["result"]["handoff_executed"],
                )
            outcome["original"] = measured(root, trial)
            outcomes[name] = outcome
        rows.append(dict(context=context, side=side, **outcomes))
    if (
        len(rows) != 8
        or any(digest(path) != h for path, h in frozen)
        or digest(cp) != cd["receiver_candidate_audit_hash"]
    ):
        raise ValueError("all unchanged matched physics sources required")
    summary = {
        name: dict(
            healthy_both=sum(row[name]["both_bodies_healthy"] for row in rows),
            strict_both_clean=sum(row[name]["strict_both_clean_handoff"] for row in rows),
            receiver_nonfoot=sum(row[name]["receiver_nonfoot_substep"] is not None for row in rows),
            receiver_foot_after_pass=sum(
                row[name]["passer_foot_precedes_receiver"] for row in rows
            ),
            precise_healthy_return=sum(
                row[name]["original"]["return_0p1m_and_healthy"] for row in rows
            ),
        )
        for name in ("zero", "parent", "continuation")
    }
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="RECEIVER_CONTINUATION_MATCHED_DEVELOPMENT_COMPARISON",
            source_head=a.expected_head,
            conditions=8,
            development_context_block=a.block,
            summary=summary,
            records=rows,
            independent_fresh_contexts=False,
            all_seeds_or_blocks=False,
            physical_retention_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
