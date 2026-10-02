"""Consolidate ALL successes of the actual qualified parent, not failed-child labels.

Reuses Core AnchorOutputMemory, including its exact-conflict and capacity gates.
This independently reconstructs existing executions; it runs no new physics.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import rosclaw.growth.anchor_consolidation as consolidation_module
from rosclaw.growth.anchor_consolidation import consolidate_unique
from rosclaw.growth.anchor_kernel import AnchorKernelGuard
from rosclaw.growth.anchor_output_memory import AnchorOutputMemory

from rosclaw_soccer.rsi.failure_curriculum_evidence import _outcome, _sealed
from rosclaw_soccer.rsi.output_memory_step_motor import encoder_identity, validate_model
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_audit_memory_learning_rollouts import ordered_audits
from scripts.rsi_collect_failed_step_courses import qualified_memory_failure_rows
from scripts.rsi_fit_protected_phase_step_motor import gpu_observations


def audit_success(job: dict[str, Any]) -> tuple[dict[str, Any], Any, Any]:
    row, commitment = job["row"], job["commitment"]
    seed, lane = row["seed"], row["lane"]
    folder = Path(job["root"]) / f"seed{seed}-lane{lane}-candidate-actor"
    raw = _sealed(folder / "report.json")
    decoded: list[Any] = []
    checked = _outcome(folder, raw["contact_motor_policy_hash"], commitment, decoder_sink=decoded)
    outcome = checked["outcome"]
    if (
        raw["report_hash"] != row["candidate"]["report_hash"]
        or raw["contact_motor_policy"]["step_motor_proof"]["model"]["model_hash"]
        != commitment["model_hash"]
        or any(outcome[k] != row["candidate"][k] for k in outcome)
        or outcome["high_quality"] is not True
        or len(decoded) != 1
    ):
        raise ValueError("complete independently reconstructed qualified parent success required")
    x, phase, _ = gpu_observations(folder, raw)
    if x.shape != (270, 134) or phase.shape != (270,):
        raise ValueError("every active current parent motor frame must be reconstructed")
    decoder = decoded[0]
    states = np.column_stack((np.stack([decoder.features(v)[:134] for v in x]), phase))
    predictions = np.stack([decoder.raw_mean(v, int(p)) for v, p in zip(x, phase, strict=True)])
    model = raw["contact_motor_policy"]["step_motor_proof"]["model"]
    guard = AnchorKernelGuard(model["output_memory"]["observations"], bandwidth=1e-4)
    gates = guard.gates(states)
    record = dict(
        seed=seed,
        lane=lane,
        report_hash=raw["report_hash"],
        frames=len(x),
        legacy_zero_gate_frames=int(np.sum(gates == 0)),
        legacy_unprotected_frames=int(np.sum(gates > 0)),
        legacy_gate_quantiles=np.quantile(gates, [0, 0.5, 1]).tolist(),
        backend="IsaacLab",
    )
    print(f"CURRENT_PARENT_SUCCESS_CONSOLIDATED seed={seed} lane={lane}", flush=True)
    return record, states, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-model", "bank-root", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--audit-workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    parent = json.loads(args.parent_model.read_text())
    validate_model(parent)
    bank = _sealed(args.bank_root / "validation_summary.json")
    review = _sealed(args.bank_root / "independent_review.json")
    qualified_memory_failure_rows(bank, review)
    if bank["commitment"]["model_hash"] != parent["model_hash"]:
        raise ValueError("actual qualified parent policy required")
    inherited = AnchorOutputMemory.from_dict(parent["output_memory"])
    if inherited.encoder_hash != encoder_identity(parent["frozen_parent"]):
        raise ValueError("immutable current parent encoder required")
    successes = [row for row in bank["rows"] if row["candidate"]["high_quality"]]
    jobs = [
        dict(root=str(args.bank_root), row=row, commitment=bank["commitment"]) for row in successes
    ]
    records, states, predictions = [], [], []
    for record, x, y in ordered_audits(audit_success, jobs, args.audit_workers):
        records.append(record)
        states.append(x)
        predictions.append(y)
    if len(records) != review["candidate_high_quality"]:
        raise ValueError("all current parent successful trajectories must enter consolidation")
    evidence = dict(
        schema="soccer.rsi.current_parent_success_consolidation_evidence.v1",
        parent_model_hash=parent["model_hash"],
        encoder_hash=inherited.encoder_hash,
        inherited_memory_hash=parent["output_memory"]["memory_hash"],
        bank_summary_hash=bank["report_hash"],
        bank_review_hash=review["report_hash"],
        selection="ALL_QUALIFIED_CURRENT_PARENT_SUCCESSES_IN_SEALED_ORDER",
        records=records,
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        consolidation_source_hash=hash_bytes(Path(consolidation_module.__file__).read_bytes()),
        promotion_authorized=False,
        hardware_authorized=False,
    )
    expanded, mapping = consolidate_unique(
        inherited,
        np.concatenate(states),
        np.concatenate(predictions),
        parent_policy_hash=parent["model_hash"],
        evidence_hash=hash_json(evidence),
    )
    encoded = expanded.to_dict()
    restored = AnchorOutputMemory.from_dict(encoded)
    for x, y in zip(np.concatenate(states), np.concatenate(predictions), strict=True):
        if not np.array_equal(
            restored.blend(x, np.zeros(12), encoder_hash=inherited.encoder_hash), y
        ):
            raise ValueError("all current parent outputs must reload exactly")
    if (
        _sealed(args.bank_root / "validation_summary.json")["report_hash"] != bank["report_hash"]
        or _sealed(args.bank_root / "independent_review.json")["report_hash"]
        != review["report_hash"]
        or parent != json.loads(args.parent_model.read_text())
    ):
        raise ValueError("parent or evidence drift")
    manifest = dict(
        **evidence,
        memory_hash=encoded["memory_hash"],
        inherited_frames=len(parent["output_memory"]["observations"]),
        added_frames=sum(r["frames"] for r in records),
        recorded_frames=len(encoded["observations"]),
        consolidation_mapping=mapping,
        independent_contexts=len(records),
        actual_physical_executions_added=0,
        existing_success_reports_reconstructed=len(records),
        exact_current_parent_output_reload=True,
        qualification="OFFLINE_CONSOLIDATION_NOT_TRAJECTORY_RETENTION_OR_NEW_LEARNING",
        physical_policy_execution_qualified=False,
    )
    manifest["report_hash"] = hash_json(manifest)
    args.output_root.mkdir(parents=True, exist_ok=False)
    write_once(args.output_root / "memory.json", encoded)
    write_once(args.output_root / "manifest.json", manifest)
    print(
        {k: v for k, v in manifest.items() if k not in ("records", "consolidation_mapping")},
        flush=True,
    )


if __name__ == "__main__":
    main()
