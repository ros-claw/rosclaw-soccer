"""Combine audited GPU/CPU consumed rollouts and fit protected causal motor heads."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.contact_motor_evidence import audit_motor_execution
from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.protected_phase_step_network import fit_update, initial_model
from rosclaw_soccer.rsi.step_motor_phase_context import phase_sequence
from rosclaw_soccer.rsi.stochastic_step_execution import features_at_frame
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once


def gpu_features(folder: Path) -> tuple[Any, Any, dict[str, Any]]:
    raw = _sealed(folder / "report.json")
    audit_motor_execution(folder, raw)
    ids = [raw["taskspace_joint_order"].index(n) for n in G1_DDS_JOINT_NAMES]
    with (
        np.load(folder / "body_trace.npz", allow_pickle=False) as body,
        np.load(folder / "late_swing_action_trace.npz", allow_pickle=False) as swing,
        np.load(folder / "contact_motor_trace.npz", allow_pickle=False) as motor,
        np.load(folder / "trace.npz", allow_pickle=False) as physics,
    ):
        forces = physics["ball_body_contact_force_peak_n"][:, 0].copy()
        body_arrays = {key: body[key] for key in body.files}
        targets = swing["executed_taskspace_joint_target_rad"]
        deltas = motor["applied_joint_delta_rad"]
        x = np.stack(
            [
                features_at_frame(
                    body_arrays,
                    frame=f,
                    nominal_target=targets[f, 0, ids],
                    previous=deltas[f - 1, 0],
                    previous_contact_forces=forces[f - 1],
                )
                for f in range(30, 300)
            ]
        )
    return x, phase_sequence(forces)[30:], raw


def cpu_features(folder: Path, record: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    raw = _sealed(folder / "report.json")
    review = _sealed(folder / "review.json")
    path = folder / "physical_trace.npz"
    if (
        raw["report_hash"] != record["report_hash"]
        or review["report_hash"] != record["review_hash"]
        or review["reviewed_report_hash"] != raw["report_hash"]
        or review["actual_mujoco_dynamics_replayed"] is not True
        or review["actual_pd_torque_reconstructed"] is not True
        or review["neural_target_reconstructed"] is not True
        or raw["physical_trace_hash"] != hash_bytes(path.read_bytes())
    ):
        raise ValueError("CPU physical replay artifact changed")
    with np.load(path, allow_pickle=False) as body:
        forces = body["force_n"][:, 0].copy()
        x = np.stack(
            [
                features_at_frame(
                    body,
                    frame=f,
                    nominal_target=body["pre_motor_joint_target_rad"][f, 0],
                    previous=body["motor_delta_rad"][f - 1, 0],
                    previous_contact_forces=forces[f - 1],
                )
                for f in range(30, 300)
            ]
        )
    return x, phase_sequence(forces)[30:], raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "gpu-rollout-root",
        "gpu-exploration-root",
        "cpu-rollout-root",
        "pilot-root",
        "warm-model",
        "output-root",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    warm = json.loads(args.warm_model.read_text())
    chunks: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    manifests: list[str] = []
    for backend, root in (("IsaacLab", args.gpu_rollout_root), ("MuJoCo", args.cpu_rollout_root)):
        manifest = _sealed(root / "rollout_manifest.json")
        path = root / "rollouts.npz"
        if (
            manifest["partition"] != "TRAIN_CONSUMED"
            or manifest["physical_rollout_count"] != 32
            or manifest["frame_sample_count"] != 8640
            or manifest["promotion_authorized"] is not False
            or manifest["hardware_authorized"] is not False
            or manifest["data_hash"] != hash_bytes(path.read_bytes())
        ):
            raise ValueError("complete audited consumed physical bank required")
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in data.files}
        phase_parts = []
        for index, record in enumerate(manifest["records"]):
            if backend == "IsaacLab":
                folder = args.gpu_exploration_root / (
                    f"seed{record['seed']}-lane{record['lane']}-sample-{record['sample']}-actor"
                )
                x, phase, raw = gpu_features(folder)
                base_hash = raw["contact_motor_policy"]["step_motor_proof"]["model"][
                    "training_sampling"
                ]["base_model_hash"]
            else:
                folder = Path(record["folder"])
                x, phase, raw = cpu_features(folder, record)
                base_hash = raw["executed_motor_policy"]["step_motor_proof"]["model"]["base_model"][
                    "training_sampling"
                ]["base_model_hash"]
            ids = slice(index * 270, (index + 1) * 270)
            if (
                base_hash != warm["model_hash"]
                or raw["report_hash"] != record["report_hash"]
                or not np.array_equal(x, arrays["observation"][ids])
                or not np.all(arrays["trajectory_index"][ids] == index)
            ):
                raise ValueError("rollout features, frozen actor or group binding changed")
            phase_parts.append(phase)
            records.append({**record, "backend": backend, "group": len(records)})
            print(f"AUDITED_PHASE_ROLLOUT backend={backend} group={index}", flush=True)
        arrays["phase_index"] = np.concatenate(phase_parts)
        arrays["trajectory_index"] += len(chunks) * 32
        chunks.append(arrays)
        manifests.append(manifest["report_hash"])
    combined = {k: np.concatenate([chunk[k] for chunk in chunks]) for k in chunks[0]}
    anchors, phases, anchor_records = [], [], []
    pilot = _sealed(args.pilot_root / "pilot_summary.json")
    for seed, lane in ((20261177, 0), (20262104, 6)):
        folder = args.pilot_root / f"seed{seed}-lane{lane}-step-neural-actor"
        x, phase, raw = gpu_features(folder)
        row = next(r for r in pilot["rows"] if (r["seed"], r["lane"]) == (seed, lane))
        if (
            raw["contact_motor_policy"]["step_motor_proof"]["model"] != warm
            or row["neural"]["report_hash"] != raw["report_hash"]
            or row["neural"]["high_quality"] is not True
        ):
            raise ValueError("protected predecessor is not frozen warm actor")
        anchors.append(x)
        phases.append(phase)
        anchor_records.append(dict(seed=seed, lane=lane, report_hash=raw["report_hash"]))
    args.output_root.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output_root / "anchors.npz",
        observation=np.concatenate(anchors),
        phase_index=np.concatenate(phases),
    )
    bank = dict(
        schema="soccer.rsi.protected_phase_anchor_bank.v1",
        records=anchor_records,
        protected_frames=540,
        phase_source="previous_completed_force",
        data_hash=hash_bytes((args.output_root / "anchors.npz").read_bytes()),
        base_model_hash=warm["model_hash"],
        promotion_authorized=False,
    )
    bank["report_hash"] = hash_json(bank)
    write_once(args.output_root / "anchor_manifest.json", bank)
    np.savez_compressed(args.output_root / "rollouts.npz", **combined)
    batch: dict[str, Any] = dict(
        schema="soccer.rsi.protected_phase_rollout_bank.v1",
        partition="TRAIN_CONSUMED",
        source_manifests=manifests,
        records=records,
        physical_rollout_count=64,
        independent_contexts=4,
        backend_count=2,
        frame_sample_count=17280,
        data_hash=hash_bytes((args.output_root / "rollouts.npz").read_bytes()),
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        promotion_authorized=False,
        hardware_authorized=False,
    )
    batch["report_hash"] = hash_json(batch)
    write_once(args.output_root / "rollout_manifest.json", batch)
    zero = initial_model(
        warm, np.concatenate(anchors), np.concatenate(phases), anchor_bank_hash=bank["report_hash"]
    )
    write_once(args.output_root / "zero_model.json", zero)
    updated = fit_update(zero, combined, batch_hash=batch["report_hash"])
    write_once(args.output_root / "model.json", updated)
    print(json.dumps(updated["learning_receipt"]), flush=True)


if __name__ == "__main__":
    main()
