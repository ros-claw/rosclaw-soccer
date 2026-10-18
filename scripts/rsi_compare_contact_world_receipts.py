"""Compare completed numerical qualifications across explicit physical worlds.

No new simulation, label repair or promotion. Logical shared reports are not
rebuilt here; whole witness hashes and their prior independent reviews bind
the original per-episode evidence, whose final campaign audit remains separate.
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.contact_motor_response import precontact_response
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.first_touch_qualification import require_first_touch_qualification
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once


def sealed(path, key="report_hash"):
    value = json.loads(path.read_text())
    if value[key] != hash_json({k: v for k, v in value.items() if k != key}):
        raise ValueError("sealed source document required")
    return value


def load(path):
    receipt = sealed(path, "receipt_hash")
    previous = "GENESIS"
    state = None
    with sqlite3.connect((path.parent / "campaign.sqlite").as_uri() + "?mode=ro", uri=True) as db:
        db.execute("BEGIN")
        for seq, link, raw, chain in db.execute("SELECT * FROM events ORDER BY seq"):
            event = json.loads(raw)
            if link != previous or chain != hash_json(dict(seq=seq, previous=link, body=event)):
                raise ValueError("whole source campaign ledger chain differs")
            previous, state = chain, event["state"]
    if state is None or state["completed"].get(receipt["job"]) != receipt:
        raise ValueError("receipt must already be committed by its original campaign")
    for witness in receipt["witnesses"].values():
        if hash_bytes(Path(witness["path"]).read_bytes()) != witness["sha256"]:
            raise ValueError("whole original witness bytes differ")
    with np.load(receipt["witnesses"]["trace"]["path"], allow_pickle=False) as archive:
        trace = dict(archive)
    outcome = label_first_touch(trace)
    review = sealed(Path(receipt["witnesses"]["independent_review"]["path"]))
    foundation = sealed(Path(receipt["witnesses"]["foundation_review"]["path"]))
    alignment = require_first_touch_qualification(review, foundation, outcome)
    if (
        outcome.to_dict() != receipt["outcome"]
        or alignment != receipt["qualification_alignment"]
        or review["reviewed_report_hash"] != receipt["physical_report_hash"]
        or foundation["physical_report_hash"] != receipt["physical_report_hash"]
        or foundation["physical_trace_hash"] != receipt["witnesses"]["trace"]["sha256"]
    ):
        raise ValueError("primitive label or independent source/trace binding differs")
    return receipt, trace, outcome, previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-receipt", type=Path, required=True)
    parser.add_argument("--new-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh comparison artifact required")
    a, old, ao, at = load(args.old_receipt)
    b, new, bo, bt = load(args.new_receipt)
    ac, bc = (json.loads(Path(r["witnesses"]["commitment"]["path"]).read_text()) for r in (a, b))
    if (
        a["witnesses"]["proof_payload"]["sha256"] != b["witnesses"]["proof_payload"]["sha256"]
        or any(
            ac[key] != bc[key] for key in ("foundation_hash", "late_actor_hash", "step_model_hash")
        )
        or any("cerebellar_model_hash" in r for r in (a, b))
    ):
        raise ValueError("exact same complete parent, Foundation and sampling policy required")
    if a["candidate"] != b["candidate"] or any(
        old[k][0].tobytes() != new[k][0].tobytes() for k in ("canonical_qpos", "canonical_qvel")
    ):
        raise ValueError("same explicit candidate and initial body/ball state required")
    common = sorted(set(old) & set(new))
    rows = []
    for key in common:
        x, y = old[key], new[key]
        if x.dtype != y.dtype or x.shape != y.shape or x.shape[0] != 300:
            raise ValueError("identical whole-trace layouts required")
        changed = [i for i in range(300) if x[i].tobytes() != y[i].tobytes()]
        rows.append(dict(array=key, first_exact_difference_frame=changed[0] if changed else None))
    physical_first = min(
        (
            row["first_exact_difference_frame"]
            for row in rows
            if row["array"]
            in ("canonical_qpos", "canonical_qvel", "force_n", "actual_actuator_force_nm")
            and row["first_exact_difference_frame"] is not None
        ),
        default=None,
    )
    result = dict(
        schema="soccer.rsi.paired_contact_world_receipts.v1",
        old_receipt_hash=a["receipt_hash"],
        new_receipt_hash=b["receipt_hash"],
        old_snapshot_ledger_tip=at,
        new_snapshot_ledger_tip=bt,
        old_outcome=ao.to_dict(),
        new_outcome=bo.to_dict(),
        array_comparison=rows,
        all_common_array_prefix_byte_equal_frames=min(
            (
                r["first_exact_difference_frame"]
                for r in rows
                if r["first_exact_difference_frame"] is not None
            ),
            default=300,
        ),
        first_physical_difference_frame=physical_first,
        precontact_response=precontact_response(
            old, new, ao.first_foot_contact_frame, bo.first_foot_contact_frame
        ),
        whole_witness_bytes_verified=True,
        shared_logical_report_seals_rebuilt_here=False,
        new_physical_substeps=0,
        new_optimizer_updates=0,
        policy_parameters_changed=False,
        skill_improvement_claimed=False,
        hardware_authorized=False,
        reader_source_hash=hash_bytes(Path(__file__).read_bytes()),
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output, result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
