"""Authenticate all eighteen consumed endpoints; never admit a learner or teacher."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.contact_demonstration_inventory import ENDPOINTS, inventory
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.first_touch_qualification import require_first_touch_qualification
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-report", type=Path, required=True)
    parser.add_argument("--study", nargs=3, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or len(args.study) != 4:
        raise ValueError("unused inventory output and all four original complete audits required")
    pins = {}

    def digest(path):
        with Path(path).open("rb") as stream:
            return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()

    def sealed(path, key="report_hash"):
        d = json.loads(Path(path).read_text())
        if d[key] != hash_json({k: v for k, v in d.items() if k != key}):
            raise ValueError("source logical seal differs")
        pins[str(path)] = digest(path)
        return d

    pins[str(Path(__file__))] = digest(__file__)
    from rosclaw_soccer.rsi import contact_demonstration_inventory

    pins[contact_demonstration_inventory.__file__] = digest(
        contact_demonstration_inventory.__file__
    )
    dense = sealed(args.dense_report)
    if dense.get("schema") != "soccer.rsi.contact_exit_metrics_readback.v1":
        raise ValueError("original full dense contact readback required")
    for path, h in dense["input_file_hashes"].items():
        if digest(path) != h:
            raise ValueError("original dense source or witness changed")
        pins[path] = h
    expected_studies = {
        "soccer.rsi.mesh_original_public_terminal_readback.v1": 4,
        "soccer.rsi.mesh_contact_catalog_terminal_readback.v1": 16,
        "soccer.rsi.mesh_option4_teacher_screen_terminal_readback.v1": 18,
        "soccer.rsi.mesh_surface_swing_terminal_readback.v1": 7,
    }
    seen, receipts = set(), {}
    for complete_path, audit_path, reader_path in args.study:
        complete, audit = sealed(complete_path), sealed(audit_path)
        schema = audit.get("schema")
        if (
            schema not in expected_studies
            or schema in seen
            or audit.get("source_native_bodies") != expected_studies[schema]
            or len(complete.get("results", [])) != expected_studies[schema]
            or audit.get("complete_hash") != complete["report_hash"]
            or audit.get("reader_source_hash") != digest(reader_path)
            or any(
                audit.get(k) is not True
                for k in (
                    "whole_input_and_witness_files_verified",
                    "shared_payload_logical_seals_rebuilt",
                    "prior_independent_replays_authenticated",
                )
            )
        ):
            raise ValueError("whole distinct original studies and exact reader provenance required")
        pins[reader_path] = digest(reader_path)
        seen.add(schema)
        for receipt in complete["results"]:
            if receipt["receipt_hash"] != hash_json(
                {k: v for k, v in receipt.items() if k != "receipt_hash"}
            ) or not any(
                r["receipt_hash"] == receipt["receipt_hash"] and r["outcome"] == receipt["outcome"]
                for r in audit["rows"]
            ):
                raise ValueError("original full receipt and terminal audit membership required")
            receipts[receipt["receipt_hash"]] = receipt
    for row in dense["rows"]:
        receipt = receipts[row["source_receipt_hash"]]
        family, course = ENDPOINTS[row["case_id"]]
        expected_group = (
            int(course.removeprefix("raw-"))
            if course.startswith("raw-")
            else 64 + int(course.removeprefix("context-"))
        )
        if (
            receipt["group"] != expected_group
            or (receipt["candidate"] is None) is not (family == "none")
        ):
            raise ValueError("dense endpoint cannot be retagged as another family or course")
        for witness in receipt["witnesses"].values():
            if digest(witness["path"]) != witness["sha256"]:
                raise ValueError("original receipt witness bytes changed")
            pins[witness["path"]] = witness["sha256"]
        with np.load(receipt["witnesses"]["trace"]["path"], allow_pickle=False) as archive:
            if ("mesh_surface_swing_record" in archive.files) is not (family == "mesh-surface"):
                raise ValueError("actual captured control family differs from demonstration label")
            outcome = label_first_touch(dict(archive))
        review = sealed(receipt["witnesses"]["independent_review"]["path"])
        foundation = sealed(receipt["witnesses"]["foundation_review"]["path"])
        if (
            row["original_outcome"] != receipt["outcome"]
            or outcome.to_dict() != receipt["outcome"]
            or require_first_touch_qualification(review, foundation, outcome)
            != receipt["qualification_alignment"]
        ):
            raise ValueError("original primitive and independent replay qualifications differ")
    result = inventory(dense["rows"], dense_report_hash=dense["report_hash"])
    result.pop("report_hash")
    result.update(input_file_hashes=pins, whole_original_study_membership_verified=True)
    if any(digest(path) != h for path, h in pins.items()):
        raise ValueError("source bytes changed while reading")
    result["report_hash"] = hash_json(result)
    write_once(args.output, result)
    print(result["report_hash"], flush=True)


if __name__ == "__main__":
    main()
