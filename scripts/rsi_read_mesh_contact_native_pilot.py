"""Independent terminal receipt/ledger readback, zero new integration steps."""

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    status = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            args.unit,
            "-p",
            "MainPID",
            "-p",
            "ActiveState",
            "-p",
            "Result",
        ],
        text=True,
    )
    if set(status.splitlines()) != {"MainPID=0", "ActiveState=inactive", "Result=success"}:
        raise ValueError("finished worker required; partial output is not completion")
    if args.output.exists():
        raise ValueError("unused terminal audit output required")
    config = json.loads(args.config.read_text())
    source, core, root = (Path(config[k]) for k in ("source_root", "core_root", "output_root"))
    sys.path[:0] = [str(source / "src"), str(source), str(core / "src")]
    import numpy as np

    from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
    from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
    from rosclaw_soccer.rsi.first_touch_qualification import require_first_touch_qualification
    from rosclaw_soccer.rsi.fixed_physical_payload_reader import FixedPhysicalPayloadReader
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once

    def sealed(path, key="report_hash"):
        value = json.loads(path.read_text())
        if value[key] != hash_json({k: v for k, v in value.items() if k != key}):
            raise ValueError("source document seal changed")
        return value

    declaration, complete = (sealed(root / n) for n in ("declaration.json", "complete.json"))
    if (
        complete["declaration_hash"] != declaration["report_hash"]
        or complete["new_native_bodies"] != 3
        or complete["native_physical_substeps"] != 9000
        or complete["independent_cpu_substeps"] != 9000
        or complete["independent_foundation_calls"] != 900
        or complete["new_optimizer_updates"] != 0
    ):
        raise ValueError("exact completed three-body accounting required")
    if any(k not in declaration or declaration[k] != v for k, v in config.items()):
        raise ValueError("same frozen configuration required")
    pins = dict(declaration["input_file_hashes"])
    pins.update(declaration["prepared_pins"])
    for name, expected in pins.items():
        if digest(Path(name)) != expected:
            raise ValueError("source/input pin changed")
    for name, expected in declaration["source_heads"].items():
        if (
            subprocess.check_output(["git", "-C", name, "rev-parse", "HEAD"], text=True).strip()
            != expected
        ):
            raise ValueError("source checkout head changed")
    previous = "GENESIS"
    events = []
    with sqlite3.connect((root / "campaign.sqlite").as_uri() + "?mode=ro", uri=True) as db:
        db.execute("BEGIN")
        for seq, link, body, value in db.execute("SELECT * FROM events ORDER BY seq"):
            document = json.loads(body)
            if (
                seq != len(events) + 1
                or link != previous
                or value != hash_json(dict(seq=seq, previous=link, body=document))
            ):
                raise ValueError("whole durable event chain required")
            previous = value
            events.append(document)
    if not events:
        raise ValueError("nonempty durable ledger required")
    final = events[-1]["state"]
    names = ("old64-option3", "new96-option3", "unsafe104-option7")
    if (
        final["phase"] != "CONTACT_WORLD_DIAGNOSTIC_COMPLETE_NO_PROMOTION"
        or final["active"] is not None
        or final["attempts"] != 3
        or set(final["completed"]) != set(names)
    ):
        raise ValueError("all three charged attempts must finish in original order")
    rows = []
    for index, name in enumerate(names):
        receipt = sealed(root / f"receipt-{index:04d}.json", "receipt_hash")
        if (
            receipt != complete["results"][index]
            or receipt != final["completed"][name]
            or receipt["job"] != name
        ):
            raise ValueError("completion, ledger and actual ordered receipts must agree")
        witnesses = dict(receipt["witnesses"])
        witnesses.update({k: receipt[k] for k in ("intent_witness", "encoder_witness")})
        for witness in witnesses.values():
            if digest(Path(witness["path"])) != witness["sha256"]:
                raise ValueError("whole physical witness file changed")
        reader = FixedPhysicalPayloadReader(
            Path(witnesses["proof_payload"]["path"]),
            expected_file_hash=witnesses["proof_payload"]["sha256"],
        )
        report = _sealed(Path(witnesses["report"]["path"]), physical_payload_reader=reader)
        commitment = json.loads(Path(witnesses["commitment"]["path"]).read_text())
        if hash_json(commitment) != report["commitment_hash"]:
            raise ValueError("actual commitment hash must bind the sealed physical report")
        expected_world = config["mesh_task_contact_world_hash"]
        for doc in (report, commitment):
            if (
                doc["compiled_model_hash"] != expected_world
                or doc.get("mesh_task_contact_protocol", {}).get("expected_world_hash")
                != expected_world
            ):
                raise ValueError("new world must bind actual report and commitment")
        if (
            report["report_hash"] != receipt["physical_report_hash"]
            or digest(Path(witnesses["world"]["path"])) != expected_world
            or report["physical_trace_hash"] != witnesses["trace"]["sha256"]
        ):
            raise ValueError("actual report/world/trace identity differs")
        with np.load(witnesses["trace"]["path"], allow_pickle=False) as archive:
            trace = dict(archive)
        outcome = label_first_touch(trace)
        review = sealed(Path(witnesses["independent_review"]["path"]))
        foundation = sealed(Path(witnesses["foundation_review"]["path"]))
        alignment = require_first_touch_qualification(review, foundation, outcome)
        if (
            outcome.to_dict() != receipt["outcome"]
            or alignment != receipt["qualification_alignment"]
            or review["reviewed_report_hash"] != report["report_hash"]
            or foundation["physical_report_hash"] != report["report_hash"]
            or foundation["physical_trace_hash"] != witnesses["trace"]["sha256"]
        ):
            raise ValueError("raw outcome and independently reconstructed body/Foundation differ")
        intended = declaration["jobs"][index]
        context = intended["context"]
        intent = sealed(Path(witnesses["intent_witness"]["path"]))
        expected_intent = dict(
            job=name,
            context_hash=receipt["planning_context_hash"],
            option_index=intended["option_index"],
            candidate_hash=hash_json(intended["candidate"]),
            world_hash=expected_world,
            declaration_hash=declaration["report_hash"],
        )
        if {k: v for k, v in intent.items() if k != "report_hash"} != expected_intent:
            raise ValueError("candidate, context and world must be fixed before body execution")
        starts = [
            i
            for i, event in enumerate(events)
            if event["event"] == "ATTEMPT_STARTED"
            and event["state"].get("active") == dict(job=name, attempt=index)
        ]
        if len(starts) != 1 or starts[0] < 1:
            raise ValueError("one charged body start required per declared job")
        prior = events[starts[0] - 1]
        if (
            prior["event"] != "NEW_WORLD_INTENT_COMMITTED_BEFORE_BODY"
            or prior["state"].get("next_intent_hash") != intent["report_hash"]
            or prior["state"].get("active") is not None
        ):
            raise ValueError("intent must be journaled before the charged body start")
        if receipt["candidate"] != intended["candidate"]:
            raise ValueError("actual candidate differs from the frozen job")
        if (
            report["course"] != context["course"]
            or report["seed"] != context["source_seed"]
            or report["lane"] != context["source_lane"]
            or not np.array_equal(
                trace["canonical_qpos"][0, 0, 36:39], [*context["course"][:2], 0.13]
            )
        ):
            raise ValueError("declared initial course must enter physical trace")
        rows.append(
            dict(
                job=name,
                receipt_hash=receipt["receipt_hash"],
                outcome=outcome.to_dict(),
                world_hash=expected_world,
                physical_report_hash=report["report_hash"],
            )
        )
    report = dict(
        schema="soccer.rsi.mesh_contact_terminal_readback.v1",
        complete_hash=complete["report_hash"],
        ledger_tip=previous,
        rows=rows,
        reader_source_hash=digest(Path(__file__)),
        whole_input_and_witness_files_verified=True,
        shared_payload_logical_seals_rebuilt=True,
        prior_independent_replays_authenticated=True,
        new_physical_replays=0,
        new_optimizer_updates=0,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    report["report_hash"] = hash_json(report)
    write_once(args.output, report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
