"""Learn guarded motor heads from audited data; protect every old HQ state."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.cpu_motor_transfer_evidence import audit_cpu_transfer
from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.kernel_guarded_step_network import fit_update, initial_model
from rosclaw_soccer.rsi.stochastic_step_execution import features_at_frame
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_protected_phase_bank_validation import review
from scripts.rsi_fit_protected_phase_step_motor import gpu_features
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "bank-physics-root",
        "bank-path",
        "cpu-bank-root",
        "cpu-source",
        "rollout-root",
        "encoder-template",
        "output-root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    # A rejected old candidate does not invalidate its separately audited warm parent.
    independent = review(args.bank_physics_root, args.bank_path)
    summary = _sealed(args.bank_physics_root / "validation_summary.json")
    cpu = _sealed(args.cpu_bank_root / "rollout_manifest.json")
    old_template = json.loads(args.encoder_template.read_text())
    encoder = {k: old_template[k] for k in ("base_model", "frozen_random_features")}
    warm = encoder["base_model"]
    anchors, records = [], []
    for row in summary["rows"]:
        if not row["warm"]["high_quality"]:
            continue
        folder = args.bank_physics_root / f"seed{row['seed']}-lane{row['lane']}-warm-actor"
        x, _, raw = gpu_features(folder)
        base = raw["contact_motor_policy"]["step_motor_proof"]["model"]["base_model"]
        if base != warm or raw["report_hash"] != row["warm"]["report_hash"]:
            raise ValueError("successful warm anchor differs from frozen neural parent")
        anchors.append(x)
        records.append(
            dict(
                backend="IsaacLab",
                seed=row["seed"],
                lane=row["lane"],
                report_hash=raw["report_hash"],
                active_frames=270,
            )
        )
    for (seed, lane), row in zip(cpu["commitment"]["courses"], cpu["baselines"], strict=True):
        if not row["high_quality"]:
            continue
        folder = args.cpu_bank_root / f"seed{seed}-lane{lane}-greedy"
        checked = audit_cpu_transfer(folder, args.cpu_source)
        raw = _sealed(folder / "report.json")
        base = raw["executed_motor_policy"]["step_motor_proof"]["model"]["base_model"]
        if (
            checked["high_quality"] is not True
            or raw["report_hash"] != row["report_hash"]
            or base != warm
            or checked["reviewed_report_hash"] != row["report_hash"]
        ):
            raise ValueError("CPU warm successful trajectory changed")
        with np.load(folder / "physical_trace.npz", allow_pickle=False) as body:
            x = np.stack(
                [
                    features_at_frame(
                        body,
                        frame=f,
                        nominal_target=body["pre_motor_joint_target_rad"][f, 0],
                        previous=body["motor_delta_rad"][f - 1, 0],
                        previous_contact_forces=body["force_n"][f - 1, 0],
                    )
                    for f in range(30, 300)
                ]
            )
        anchors.append(x)
        records.append(
            dict(
                backend="MuJoCo",
                seed=seed,
                lane=lane,
                report_hash=raw["report_hash"],
                active_frames=270,
            )
        )
    if not anchors:
        raise ValueError("at least one actual successful predecessor required")
    batch = _sealed(args.rollout_root / "rollout_manifest.json")
    data_path = args.rollout_root / "rollouts.npz"
    if (
        batch["partition"] != "TRAIN_CONSUMED"
        or batch["physical_rollout_count"] != 64
        or batch["frame_sample_count"] != 17280
        or batch["independent_contexts"] != 4
        or batch["backend_count"] != 2
        or batch["data_hash"] != hash_bytes(data_path.read_bytes())
        or batch["promotion_authorized"] is not False
        or batch["hardware_authorized"] is not False
    ):
        raise ValueError("audited consumed dual-backend physical rollout bank required")
    args.output_root.mkdir(parents=True, exist_ok=False)
    anchor_data = np.concatenate(anchors)
    path = args.output_root / "anchors.npz"
    np.savez_compressed(path, observation=anchor_data)
    manifest: dict[str, Any] = dict(
        schema="soccer.rsi.successful_state_anchor_bank.v1",
        partition="TRAIN_CONSUMED",
        records=records,
        protected_active_frames=len(anchor_data),
        parent_model_hash=warm["model_hash"],
        physics_review_hash=independent["report_hash"],
        cpu_bank_hash=cpu["report_hash"],
        data_hash=hash_bytes(path.read_bytes()),
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        promotion_authorized=False,
        hardware_authorized=False,
    )
    manifest["report_hash"] = hash_json(manifest)
    write_once(args.output_root / "anchor_manifest.json", manifest)
    zero = initial_model(encoder, anchor_data, anchor_bank_hash=manifest["report_hash"])
    write_once(args.output_root / "zero_model.json", zero)
    with np.load(data_path, allow_pickle=False) as arrays:
        updated = fit_update(zero, arrays, batch_hash=batch["report_hash"])
    write_once(args.output_root / "model.json", updated)
    print(
        json.dumps(dict(model_hash=updated["model_hash"], receipt=updated["learning_receipt"])),
        flush=True,
    )


if __name__ == "__main__":
    main()
