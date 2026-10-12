"""Reuse independently audited raw episodes under a new outcome contract.

No optimizer, no new physics, no private exam. This creates collection labels,
not a complete learning bank; missing neural sequences must still be rebuilt.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi import first_touch_outcome, first_touch_recovery
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch, outcome_contract
from rosclaw_soccer.rsi.first_touch_recovery import FirstTouchRecoveryLedger, file_hash
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once


def sealed(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("ordinary source receipt required")
    value = json.loads(path.read_bytes())
    if type(value) is not dict:
        raise ValueError("ordinary source receipt required")
    if value["report_hash"] != hash_json({k: v for k, v in value.items() if k != "report_hash"}):
        raise ValueError("invalid source receipt")
    return value


def envelope(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("ordinary envelope required")
    with gzip.open(path, "rt") as stream:
        value = json.load(stream)
    if type(value) is not dict:
        raise ValueError("ordinary source envelope required")
    if value.get("schema") != "rosclaw.growth.shared_json_payload.v1" or value[
        "envelope_hash"
    ] != hash_json({k: v for k, v in value.items() if k != "envelope_hash"}):
        raise ValueError("complete immutable envelope required")
    return value


def recover(bank: Path, audit_path: Path, audit_hash: str, root: Path) -> dict[str, Any]:
    audit = sealed(audit_path)
    if audit["report_hash"] != audit_hash:
        raise ValueError("exact independent archive audit required")
    for name, expected in audit["input_file_hashes"].items():
        if file_hash(Path(name)) != expected:
            raise ValueError("adopted archive audit input changed")
    original = sealed(bank / "commitment.json")["public_training_plan"]
    plan = PublicFirstTouchTrainingPlan(
        tuple(original["seeds"]), original["behavior_model_hash"], original["stream"]
    )
    if hash_json(original) != hash_json(plan.contract()):
        raise ValueError("exact original public declaration required")
    ledger = FirstTouchRecoveryLedger(root, plan)
    outcomes: Counter[str] = Counter()
    adopted = []
    pins = {
        str(audit_path.resolve()): file_hash(audit_path),
        str(Path(__file__).resolve()): file_hash(Path(__file__)),
    }
    for module in (first_touch_outcome, first_touch_recovery):
        if module.__file__ is None:
            raise ValueError("file-backed outcome and recovery modules required")
        path = Path(module.__file__).resolve()
        pins[str(path)] = file_hash(path)
    for entry in audit["records"]:
        group = entry["index"]
        if group == -1:
            continue
        if type(group) is not int or not 0 <= group < len(plan.draws()):
            raise ValueError("declared public group required")
        folder = bank / f"sample-{group}"
        report_envelope = envelope(folder / "report.json.gz")
        sampling_envelope = envelope(bank / "models" / f"sample-{group}.json.gz")
        report, view = report_envelope["stripped_document"], sampling_envelope["stripped_document"]
        review = sealed(folder / "independent_review.json")
        foundation = sealed(folder / "foundation_review.json")
        body = json.loads((folder / "commitment.json").read_bytes())
        draw = plan.draws()[group]
        if (
            report["report_hash"] != entry["physical_report_hash"]
            or review["report_hash"] != entry["review_hash"]
            or foundation["report_hash"] != entry["foundation_review_hash"]
            or review["reviewed_report_hash"] != report["report_hash"]
            or foundation["physical_report_hash"] != report["report_hash"]
            or report["physical_trace_hash"] != file_hash(folder / "physical_trace.npz")
            or foundation["physical_trace_hash"] != report["physical_trace_hash"]
            or report["commitment_hash"] != hash_json(body)
            or report["course"] != list(draw.course)
            or report["seed"] != draw.seed
            or report["lane"] != draw.lane
            or report["compiled_model_hash"] != file_hash(folder / "compiled_model.mjb")
            or report["step_model_hash"] != view["model_hash"]
            or report_envelope["payload_hash"] != sampling_envelope["payload_hash"]
            or report["partition"] != "DECLARED_PUBLIC_TRAINING"
            or report["approach_mode"] != "legacy"
            or report["swing_mode"] != "legacy"
            or hash_json(report["public_training_declaration"]) != hash_json(plan.contract())
            or view["seed"] != draw.sampling_seed
            or review["physical_substeps"] != 3000
            or foundation["recomputed_executed_calls"] != 300
        ):
            raise ValueError("physical/model/course/complete audit linkage mismatch")
        for receipt in (report, review, foundation, view):
            if any(
                receipt[k] is not False for k in ("promotion_authorized", "hardware_authorized")
            ):
                raise ValueError("SIM_ONLY ceiling required")
        with np.load(folder / "physical_trace.npz", allow_pickle=False) as arrays:
            label = label_first_touch({k: arrays[k] for k in arrays.files})
        if label.task_pass != entry["high_quality"]:
            raise ValueError("new labels must not change the frozen formal task result")
        inputs = {
            str(p.resolve()): file_hash(p)
            for p in (
                folder / "report.json.gz",
                folder / "commitment.json",
                folder / "compiled_model.mjb",
                folder / "physical_trace.npz",
                folder / "independent_review.json",
                folder / "foundation_review.json",
                bank / "models" / f"sample-{group}.json.gz",
            )
        }
        pins.update(inputs)
        receipt = dict(
            draw=asdict(draw),
            label=label.to_dict(),
            physical_report_hash=report["report_hash"],
            outcome_contract_hash=outcome_contract()["contract_hash"],
            adopted_audit_hash=audit_hash,
            input_file_hashes=inputs,
            archive_envelope_integrity_checked=True,
            whole_shared_model_logical_payload_reverified_here=False,
            new_physics_steps=0,
            training_sequence_rebuilt_here=False,
            training_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        receipt["report_hash"] = hash_json(receipt)
        labelpath = root / f"label-{group}.json"
        write_once(labelpath, receipt)
        ledger.record_episode(
            group,
            dict(
                trace=folder / "physical_trace.npz",
                replay=folder / "independent_review.json",
                foundation=folder / "foundation_review.json",
                sampling_view=bank / "models" / f"sample-{group}.json.gz",
                label=labelpath,
            ),
        )
        outcomes[label.event_status] += 1
        adopted.append(group)
    reopened = FirstTouchRecoveryLedger(root, plan)
    status = reopened.status()
    for name, expected in pins.items():
        if file_hash(Path(name)) != expected:
            raise ValueError("archive changed during recovery")
    result = dict(
        campaign="rsi_m0_first_touch_mechanism_recovery",
        stage="G0_PARTIAL",
        outcome_contract=outcome_contract(),
        source_audit_hash=audit_hash,
        status=status,
        adopted_groups=adopted,
        event_counts=dict(outcomes),
        input_file_hashes=pins,
        adopted_training_physical_episodes=len(adopted),
        new_physics_steps=0,
        new_optimizer_updates=0,
        complete_learning_bank=False,
        automatic_reopen_verified=True,
        private_fresh_accessed=False,
        training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(root / "relabel_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = recover(args.bank, args.audit, args.audit_hash, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "input_file_hashes"}, indent=2))


if __name__ == "__main__":
    main()
