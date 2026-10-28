"""Review actual center-curriculum receiver identity and contact outcomes."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.passer_curriculum_outcome import strict_pass_receive
from rosclaw_soccer.rsi.relay_whole_identity_comparison import compare_receiver_traces
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report
from scripts.rsi_review_whole_receiver_identity import artifact_equal, handoff_equal


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", nargs=2, action="append", required=True, metavar=("SOURCE", "AUDIT"))
    p.add_argument("--producer-head", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if len(a.group) != 3 or a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("three complete independent replays and unused external output required")
    rows, seen, pins = [], set(), []
    blocks = ((2, 3, 5, 7), (8, 9, 11, 13), (15, 16, 17, 18))
    for root_name, audit_name in a.group:
        root, audit = Path(root_name), Path(audit_name)
        ah = digest(audit)
        source, _, sh = report(
            root, audit, ah, "NEURAL_RELAY_RECEIVER_CENTER_IDENTITY16_PENDING_OWN_REPLAY", worlds=16
        )
        declaration = json.loads((root / "declaration.json").read_text())
        block = declaration["development_context_block"]
        if (
            block not in (0, 1, 2)
            or block in seen
            or source["source_head"] != a.producer_head
            or declaration["protocol"]
            != "soccer.neural_relay_receiver_center_identity.development.v13"
            or digest(root / "declaration.json") != source["declaration_hash"]
        ):
            raise ValueError("exact distinct producer and declared center blocks required")
        seen.add(block)
        cases = {
            (
                r["course"]["context"],
                r["course"]["lateral_m"],
                r["course"]["arm"]["receiver_whole"],
            ): r
            for r in source["records"]
        }
        if set(cases) != {
            (context, side, whole)
            for context in blocks[block]
            for side in (-0.2, 0.2)
            for whole in (False, True)
        }:
            raise ValueError("complete fixed center courses and both motor arms required")
        for context in blocks[block]:
            for side in (-0.2, 0.2):
                base, whole = cases[(context, side, False)], cases[(context, side, True)]
                same = all(
                    base[k] == whole[k]
                    for k in ("termination", "controls", "native_steps", "result")
                ) and handoff_equal(root, base, root, whole)
                names = ("initial", "physics", "passer_nn", "receiver_shadow", "receiver_return_nn")
                same = same and all(artifact_equal(root, base, root, whole, name) for name in names)
                bp, wp = (
                    bind(root, base, "receiver_velocity"),
                    bind(root, whole, "receiver_velocity"),
                )
                physics = bind(root, base, "physics")
                if bp is None or wp is None or physics is None:
                    raise ValueError("actual velocity traces and physical histories required")
                with np.load(bp, allow_pickle=False) as b, np.load(wp, allow_pickle=False) as w:
                    trace = compare_receiver_traces(dict(b), dict(w))
                with np.load(physics, allow_pickle=False) as f:
                    outcomes = strict_pass_receive(
                        f["ball_forces"],
                        f["body_failed"],
                        handoff=base["result"]["handoff_executed"],
                    )
                rows.append(
                    dict(
                        block=block,
                        context=context,
                        side=side,
                        identity_exact=same
                        and all(
                            trace[k]
                            for k in (
                                "common_fields_exact",
                                "action_is_actual_onnx_reference",
                                "extra_joint_delta_exact_zero",
                            )
                        ),
                        receiver_trace=trace,
                        outcomes=outcomes,
                        termination=base["termination"],
                    )
                )
        pins.extend(((root / "complete.json", sh), (audit, ah)))
    if seen != {0, 1, 2} or len(rows) != 24 or any(digest(path) != h for path, h in pins):
        raise ValueError("all 24 unchanged physical pairs required")
    identity = all(row["identity_exact"] for row in rows)
    summary = {
        name: sum(r["outcomes"][name] for r in rows)
        for name in (
            "original_handoff",
            "both_bodies_healthy",
            "strict_receiver_clean_handoff",
            "strict_both_clean_handoff",
        )
    }
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="CENTER_RECEIVER_IDENTITY_REVIEWED"
            if identity
            else "CENTER_RECEIVER_IDENTITY_REJECTED",
            source_head=a.expected_head,
            producer_head=a.producer_head,
            conditions=24,
            identity_passed=identity,
            summary=summary,
            records=rows,
            trained=False,
            independent_fresh=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
