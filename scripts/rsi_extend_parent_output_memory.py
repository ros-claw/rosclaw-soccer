"""Extend frozen later-parent memory from every reviewed consumed success.

All old records are retained. This is training evidence, not a policy promotion.
Neither frame repetitions nor multiple backends count as new course identities.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from rosclaw.growth.anchor_output_memory import AnchorOutputMemory

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.kernel_guarded_step_execution import CompiledKernelStepMotor, make_preview
from rosclaw_soccer.rsi.kernel_guarded_step_network import validate_model
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_fit_protected_phase_step_motor import gpu_features
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("memory-root", "parent-model", "bank-root", "output-root"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    parent = json.loads(args.parent_model.read_text())
    validate_model(parent)
    saved = json.loads((args.memory_root / "memory.json").read_text())
    memory = AnchorOutputMemory.from_dict(saved)
    inherited = _sealed(args.memory_root / "manifest.json")
    bank = _sealed(args.bank_root / "validation_summary.json")
    review = _sealed(args.bank_root / "independent_review.json")
    if (
        inherited["memory_hash"] != saved["memory_hash"]
        or memory.parent_policy_hash != parent["model_hash"]
        or bank["commitment"]["model_hash"] != parent["model_hash"]
        or bank["commitment"]["partition"] != "TRAIN_CONSUMED"
        or bank["independent_contexts"] != 52
        or len(bank["rows"]) != 52
        or review["source_summary_hash"] != bank["report_hash"]
        or review["physical_reports_reviewed"] != 156
        or review["motor_frames_reconstructed"] != 31200
        or any(
            r.get(k) is not False
            for r in (bank, review)
            for k in ("promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("unchanged memory and fully reviewed consumed parent bank required")
    decoder = CompiledKernelStepMotor(make_preview(parent))
    states, predictions, records = [], [], []
    for row in bank["rows"]:
        if not row["candidate"]["high_quality"]:
            continue
        seed, lane = row["seed"], row["lane"]
        folder = args.bank_root / f"seed{seed}-lane{lane}-candidate-actor"
        x, phase, raw = gpu_features(folder)
        if (
            raw["report_hash"] != row["candidate"]["report_hash"]
            or raw["contact_motor_policy"]["step_motor_proof"]["model"] != parent
        ):
            raise ValueError("reviewed successful parent execution changed")
        states.append(np.column_stack((np.stack([decoder.features(v)[:134] for v in x]), phase)))
        predictions.append(
            np.stack([decoder.raw_mean(v, int(p)) for v, p in zip(x, phase, strict=True)])
        )
        records.append(
            dict(
                seed=seed,
                lane=lane,
                backend="IsaacLab",
                report_hash=raw["report_hash"],
                frames=len(x),
            )
        )
        print(f"CURRENT_PARENT_SUCCESS_RECONSTRUCTED seed={seed} lane={lane}", flush=True)
    if len(records) != bank["candidate_high_quality"] or not records:
        raise ValueError("complete declared successful trajectories required")
    evidence = dict(
        schema="soccer.rsi.full_parent_memory_extension.v1",
        predecessor_manifest_hash=inherited["report_hash"],
        predecessor_memory_hash=saved["memory_hash"],
        parent_model_hash=parent["model_hash"],
        bank_summary_hash=bank["report_hash"],
        bank_review_hash=review["report_hash"],
        records=records,
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        promotion_authorized=False,
        hardware_authorized=False,
    )
    expanded = memory.extend(
        np.concatenate(states),
        np.concatenate(predictions),
        parent_policy_hash=parent["model_hash"],
        evidence_hash=hash_json(evidence),
    )
    encoded = expanded.to_dict()
    contexts = {(r["seed"], r["lane"]) for r in inherited["records"] + records}
    manifest = dict(
        **evidence,
        memory_hash=encoded["memory_hash"],
        inherited_frames=len(saved["observations"]),
        added_frames=sum(r["frames"] for r in records),
        recorded_frames=len(encoded["observations"]),
        independent_contexts=len(contexts),
        physical_policy_execution_qualified=False,
    )
    manifest["report_hash"] = hash_json(manifest)
    args.output_root.mkdir(parents=True, exist_ok=False)
    write_once(args.output_root / "memory.json", encoded)
    write_once(args.output_root / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
