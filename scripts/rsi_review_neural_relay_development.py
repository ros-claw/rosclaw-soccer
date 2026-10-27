"""Review all audited development relays, including ordered native contacts.

This does not prove ball causation, strict-foot capture, referee legality,
new policy learning, whole-game success or hardware safety.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.relay_contact_order import relay_contact_order
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source-root", "audit-root", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("source-hash", "audit-hash", "expected-head"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    source_file = args.source_root / "complete.json"
    audit_file = args.audit_root / "complete.json"
    declaration_file = args.source_root / "declaration.json"
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() for p in (source_file, audit_file, declaration_file))
        or any((p / "failed.json").exists() for p in (args.source_root, args.audit_root))
        or digest(source_file) != args.source_hash
        or digest(audit_file) != args.audit_hash
    ):
        raise ValueError("unused output and exact closed evidence required")
    source, audit, declaration = (
        json.loads(p.read_text()) for p in (source_file, audit_file, declaration_file)
    )
    if (
        source.get("status") != "NEURAL_RELAY_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
        or audit.get("status") != "ALL_NEURAL_RELAY_OWN_HISTORY_NATIVE_REPLAY_MATCH"
        or declaration.get("protocol") != "soccer.neural_relay_development_contexts.v4"
        or source.get("declaration_hash") != digest(declaration_file)
        or audit.get("producer_hash") != args.source_hash
        or audit.get("counts") != source.get("counts")
        or audit.get("worlds") != 32
        or len(source.get("records", [])) != 32
        or len(audit.get("records", [])) != 32
        or any(
            r.get(key) is not False
            for r in (source, audit)
            for key in ("goal_complete", "promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("full independently replayed development relay required")
    rows = []
    for index, (record, own) in enumerate(zip(source["records"], audit["records"], strict=True)):
        if record["index"] != index or any(
            record[key] != own[key] for key in ("index", "course", "termination", "result")
        ):
            raise ValueError("source and own relay outcomes differ")
        artifact = record["artifacts"]["physics"]
        path = args.source_root / artifact["path"]
        if (
            path.is_symlink()
            or path.resolve().parent != (args.source_root / f"trial{index}").resolve()
            or digest(path) != artifact["hash"]
        ):
            raise ValueError("exact confined native physics stream required")
        with np.load(path, allow_pickle=False) as data:
            order = relay_contact_order(
                data["ball_forces"], record["handoff"]["control"] if record["handoff"] else None
            )
        rows.append(
            {
                "index": index,
                "course": record["course"],
                "termination": record["termination"],
                "handoff": record["result"]["handoff_executed"],
                "outgoing_foot_strike": record["result"]["outgoing_foot_strike"],
                "return_0p1m_and_healthy": record["result"]["return_0p1m_and_healthy"],
                "max_target_jump_rad": record["result"]["maximum_B_target_change_rad"],
                **order,
            }
        )
    groups = defaultdict(list)
    for row in rows:
        arm = row["course"]["arm"]
        groups[(arm["start"], arm["memory"])].append(row)
    summaries = []
    for (start, memory), members in sorted(groups.items()):
        if len(members) != 8:
            raise ValueError("all eight fixed paired courses per arm required")
        summaries.append(
            {
                "start": start,
                "memory": memory,
                "worlds": len(members),
                "handoffs": sum(r["handoff"] for r in members),
                "ordered_source_then_receiver": sum(
                    r["ordered_source_then_receiver_before_handoff"] for r in members
                ),
                "body_failures": sum(r["termination"] == "BODY_FAILURE" for r in members),
                "outgoing_foot_strikes": sum(r["outgoing_foot_strike"] for r in members),
                "precise_healthy_returns": sum(r["return_0p1m_and_healthy"] for r in members),
                "ordered_precise_healthy_returns": sum(
                    r["return_0p1m_and_healthy"]
                    and r["ordered_source_then_receiver_before_handoff"]
                    for r in members
                ),
            }
        )
    clean(args.expected_head)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        args.output,
        {
            "status": "DEVELOPMENT_RELAY_REVIEW_NOT_GAME_OR_PROMOTION",
            "source_hash": args.source_hash,
            "audit_hash": args.audit_hash,
            "source_head": args.expected_head,
            "summary": summaries,
            "rows": rows,
            "ordered_contacts_prove_causation": False,
            "strict_foot_capture_regraded": False,
            "hardware_authorized": False,
            "promotion_authorized": False,
        },
    )


if __name__ == "__main__":
    main()
