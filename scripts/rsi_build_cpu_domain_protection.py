"""Replay ALL consumed CPU parent successes and preserve the intact GPU bank.

Existing executions only: no new rollout, optimizer, reward/world change or
activation. A failed audit aborts the complete bank; rows are never skipped.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from rosclaw.growth.domain_anchor_bank import DomainAnchorGuard, build_domain_anchor_bank

from rosclaw_soccer.rsi import domain_memory_protection as protection_module
from rosclaw_soccer.rsi.cpu_motor_transfer_evidence import audit_cpu_transfer
from rosclaw_soccer.rsi.current_memory_motor import validate_model as validate_actor
from rosclaw_soccer.rsi.domain_memory_protection import FLAGS, SCHEMA, validate_protection
from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
from rosclaw_soccer.rsi.output_memory_step_motor import CompiledOutputMemoryMotor, make_preview
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_audit_memory_learning_rollouts import ordered_audits
from scripts.rsi_compressed_bank_storage import capacity_check
from scripts.rsi_fit_protected_phase_step_motor import cpu_features


def checked_success_rows(
    summary: dict[str, Any], commitment: dict[str, Any], parent_hash: str
) -> list[dict[str, Any]]:
    """Reject an incomplete declaration before allocating any audit worker."""
    if (
        summary.get("schema") != "soccer.rsi.cpu_retained_parent_full_coverage.v1"
        or commitment.get("schema") != "soccer.rsi.cpu_retained_parent_full_coverage_commitment.v1"
        or summary.get("report_hash")
        != hash_json({k: v for k, v in summary.items() if k != "report_hash"})
        or commitment.get("report_hash")
        != hash_json({k: v for k, v in commitment.items() if k != "report_hash"})
        or summary.get("commitment_hash") != commitment["report_hash"]
        or commitment.get("model_hash") != parent_hash
        or commitment.get("partition") != "TRAIN_CONSUMED_CPU_DOMAIN"
        or summary.get("independent_contexts") != 52
        or len(summary["rows"]) != 52
        or [r["index"] for r in summary["rows"]] != list(range(52))
        or [[r["seed"], r["lane"]] for r in summary["rows"]] != commitment["courses"]
        or len({(r["seed"], r["lane"]) for r in summary["rows"]}) != 52
        or summary.get("all_physical_substeps_replayed") != 156000
        or commitment.get("physics_change") is not False
        or any(
            item.get(k) is not False
            for item in (summary, commitment)
            for k in (
                "fresh_exam_authorized",
                "learning_authorized",
                "promotion_authorized",
                "hardware_authorized",
            )
        )
    ):
        raise ValueError("complete sealed consumed CPU parent bank required")
    for row in summary["rows"]:
        outcome = row["outcome"]
        if (
            outcome.get("report_hash")
            != hash_json({k: v for k, v in outcome.items() if k != "report_hash"})
            or outcome.get("report_hash") != row["review_hash"]
            or outcome.get("reviewed_report_hash") != row["report_hash"]
            or outcome.get("physical_substeps") != 3000
            or type(outcome.get("high_quality")) is not bool
            or any(
                outcome.get(k) is not True
                for k in (
                    "actual_mujoco_dynamics_replayed",
                    "actual_pd_torque_reconstructed",
                    "neural_target_reconstructed",
                )
            )
            or any(
                outcome.get(k) is not False for k in ("promotion_authorized", "hardware_authorized")
            )
        ):
            raise ValueError("all original physical reviews must remain complete")
    successes = [r for r in summary["rows"] if r["outcome"]["high_quality"] is True]
    if not successes or len(successes) != summary["high_quality"]:
        raise ValueError("ALL parent successes required")
    return successes


def audit_success(job: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    row = job["row"]
    folder, runner = Path(row["folder"]), Path(job["runner"])
    raw = _sealed(folder / "report.json")
    if (
        raw["step_model_hash"] != job["parent_hash"]
        or raw["compiled_model_hash"] != job["world_hash"]
        or raw["report_hash"] != row["report_hash"]
        or (raw["seed"], raw["lane"]) != (row["seed"], row["lane"])
    ):
        raise ValueError("original parent, world and consumed course required")
    reviewed = audit_cpu_transfer(folder, runner)
    # Audit source is part of the stored review. If it changes, this protocol
    # must be explicitly revised, not silently accept field-only equality.
    if reviewed != row["outcome"] or reviewed["high_quality"] is not True:
        raise ValueError("complete independent replay must match the original success review")
    x, phase, reread = cpu_features(folder, row)
    if reread != raw or x.shape != (270, 134) or phase.shape != (270,):
        raise ValueError("every causal motor frame required")
    policy = raw["executed_motor_policy"]
    model = policy["step_motor_proof"]["model"]
    if model["model_hash"] != job["parent_hash"] or make_preview(model) != policy:
        raise ValueError("exact original frozen NN decoder required")
    decoder = CompiledOutputMemoryMotor(policy)
    states = np.column_stack((np.stack([decoder.features(v)[:134] for v in x]), phase))
    record = {k: row[k] for k in ("index", "seed", "lane", "report_hash", "review_hash")}
    record.update(frames=270, state_hash=hash_json(states.tolist()))
    print("ALL_CPU_SUCCESS_REPLAYED_FOR_DOMAIN_GUARD", row["index"], flush=True)
    return record, states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("actor-container", "cpu-bank", "cpu-runner", "output-root"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--system-volume-root", type=Path, default=Path.home())
    args = parser.parse_args()
    if args.output_root.exists():
        raise ValueError("fresh output directory required; never overwrite evidence")
    actor = load_json_artifact(args.actor_container)["initial_actor"]
    validate_actor(actor)
    if actor["generation"] != 0:
        raise ValueError("zero-addition immutable current parent actor required")
    summary, commitment = [
        _sealed(args.cpu_bank / name) for name in ("validation_summary.json", "commitment.json")
    ]
    successes = checked_success_rows(summary, commitment, actor["parent_model_hash"])
    capacity = capacity_check(args.output_root.parent, args.system_volume_root, 2 * 1024**3)
    inputs = {
        str(path.resolve()): hash_bytes(path.read_bytes())
        for path in (
            args.actor_container,
            args.cpu_runner,
            Path(__file__),
            Path(protection_module.__file__),
        )
    }
    declaration = dict(
        schema="soccer.rsi.multi_domain_protection_reconstruction_commitment.v1",
        parent_hash=actor["parent_model_hash"],
        actor_hash=actor["model_hash"],
        cpu_summary_hash=summary["report_hash"],
        cpu_commitment_hash=commitment["report_hash"],
        ordered_success_contexts=[[r["index"], r["seed"], r["lane"]] for r in successes],
        input_file_hashes=inputs,
        capacity=capacity,
        workers=args.workers,
        actual_physical_executions_added=0,
        fresh_exam_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    declaration["report_hash"] = hash_json(declaration)
    args.output_root.mkdir(parents=False, exist_ok=False)
    write_once(args.output_root / "commitment.json", declaration)
    jobs = [
        dict(
            row=row,
            runner=str(args.cpu_runner),
            parent_hash=actor["parent_model_hash"],
            world_hash=commitment["compiled_world_hash"],
        )
        for row in successes
    ]
    records, chunks = [], []
    try:
        for record, states in ordered_audits(audit_success, jobs, args.workers):
            records.append(record)
            chunks.append(states)
    except Exception as error:
        failure = dict(
            stage="COMPLETE_PARENT_SUCCESS_REPLAY",
            error_type=type(error).__name__,
            error=str(error),
            complete_batch=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        failure["report_hash"] = hash_json(failure)
        write_once(args.output_root / "failure.json", failure)
        raise
    legacy = actor["baseline"]
    memory, manifest = legacy["memory"], legacy["consolidation_manifest"]
    states = np.concatenate(chunks)
    bank = build_domain_anchor_bank(
        [
            dict(
                domain_id="inherited-gpu-memory",
                source_evidence_hash=manifest["report_hash"],
                context_ids=["intact-legacy-memory"],
                context_rows=[len(memory["observations"])],
                observations=memory["observations"],
            ),
            dict(
                domain_id="cpu-retained-parent",
                source_evidence_hash=summary["report_hash"],
                context_ids=[f"seed{r['seed']}-lane{r['lane']}" for r in records],
                context_rows=[270] * len(records),
                observations=states,
            ),
        ],
        parent_hash=actor["parent_model_hash"],
        encoder_hash=memory["encoder_hash"],
    )
    guard = DomainAnchorGuard(bank)
    if np.any(guard.gates(states) != 0):
        raise ValueError("every declared CPU state must be exactly protected")
    result = dict(
        schema=SCHEMA,
        source_hash=hash_bytes(Path(protection_module.__file__).read_bytes()),
        selection="ALL_SEALED_CPU_SUCCESSES_PLUS_INTACT_INHERITED_GPU_MEMORY",
        bank=bank,
        cpu_summary=summary,
        cpu_commitment=commitment,
        cpu_records=records,
        reconstruction_commitment_hash=declaration["report_hash"],
        actual_physical_executions_added=0,
        actual_cpu_physical_substeps_replayed=len(records) * 3000,
        **dict.fromkeys(FLAGS, False),
    )
    result["report_hash"] = hash_json(result)
    validate_protection(result, actor)
    if (
        any(hash_bytes(Path(path).read_bytes()) != digest for path, digest in inputs.items())
        or _sealed(args.cpu_bank / "validation_summary.json") != summary
    ):
        raise ValueError("immutable reconstruction input changed")
    write_once(args.output_root / "protection.json.gz", result)
    print(
        "COMPLETE_MULTI_DOMAIN_PROTECTION_RECONSTRUCTED",
        result["report_hash"],
        bank["row_count"],
        flush=True,
    )


if __name__ == "__main__":
    main()
