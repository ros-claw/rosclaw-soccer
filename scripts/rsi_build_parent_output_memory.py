"""Record later-parent predictions on inherited and newly successful states.

This builds training memory, not a deployed motor policy. Old warm states use
the current parent's reconstructed predictions (its existing gate is exactly
zero there); new states come from independently reviewed actual execution.
All causal phase context is included; neither course identity nor outcome is
part of the memory observation.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import rosclaw.growth.anchor_output_memory as memory_module
from rosclaw.growth.anchor_output_memory import AnchorOutputMemory

from rosclaw_soccer.rsi import step_motor_features, step_motor_phase_context
from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.kernel_guarded_step_execution import CompiledKernelStepMotor, make_preview
from rosclaw_soccer.rsi.kernel_guarded_step_network import validate_model
from rosclaw_soccer.rsi.step_motor_phase_context import phase_sequence
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_fit_protected_phase_step_motor import gpu_features
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "parent-model",
        "anchor-root",
        "warm-physics-root",
        "cpu-physics-root",
        "pilot-root",
        "pilot-review",
        "output-root",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    model = json.loads(args.parent_model.read_text())
    validate_model(model)
    decoder = CompiledKernelStepMotor(make_preview(model))
    anchors = _sealed(args.anchor_root / "anchor_manifest.json")
    old_review = _sealed(args.warm_physics_root / "independent_review.json")
    bank = _sealed(args.warm_physics_root / "validation_summary.json")
    cpu = _sealed(args.cpu_physics_root / "rollout_manifest.json")
    path = args.anchor_root / "anchors.npz"
    pilot = _sealed(args.pilot_root / "validation_summary.json")
    review = _sealed(args.pilot_review)
    if (
        anchors["partition"] != "TRAIN_CONSUMED"
        or anchors["report_hash"] != model["anchor_bank_hash"]
        or anchors["data_hash"] != hash_bytes(path.read_bytes())
        or anchors["protected_active_frames"] != 8910
        or len(anchors["records"]) != 33
        or anchors["physics_review_hash"] != old_review["report_hash"]
        or old_review["source_summary_hash"] != bank["report_hash"]
        or anchors["cpu_bank_hash"] != cpu["report_hash"]
        or pilot["commitment"]["partition"] != "CONSUMED_PILOT"
        or pilot["commitment"]["model_hash"] != model["model_hash"]
        or review["source_summary_hash"] != pilot["report_hash"]
        or review["actual_reports_reviewed"] != 12
        or review["actual_motor_actions_reconstructed"] != 2400
        or any(
            obj.get(k) is not False
            for obj in (anchors, pilot, review)
            for k in ("promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError(
            "complete sealed inherited anchors and independently reviewed current parent required"
        )
    encoder = hash_json(
        dict(
            schema="soccer.rsi.frozen_motor_memory_context.v1",
            warm_model_hash=model["encoder"]["base_model"]["model_hash"],
            feature_source_hash=hash_bytes(Path(step_motor_features.__file__).read_bytes()),
            phase_source_hash=hash_bytes(Path(step_motor_phase_context.__file__).read_bytes()),
            metric="concat(clipped_normalized_134,causal_phase_index)",
        )
    )
    with np.load(path, allow_pickle=False) as data:
        x = data["observation"].copy()
    if x.shape != (8910, 134):
        raise ValueError("complete inherited raw causal observation bank required")

    def encode(features: Any, phases: Any) -> np.ndarray[Any, Any]:
        phi = np.stack([decoder.features(row)[:134] for row in features])
        return np.column_stack((phi, phases))

    phases = []
    records = []
    for record in anchors["records"]:
        seed, lane = record["seed"], record["lane"]
        if record["backend"] == "IsaacLab":
            folder = args.warm_physics_root / f"seed{seed}-lane{lane}-warm-actor"
            row = next(r for r in bank["rows"] if (r["seed"], r["lane"]) == (seed, lane))
            if (
                not row["warm"]["high_quality"]
                or row["warm"]["report_hash"] != record["report_hash"]
            ):
                raise ValueError("inherited success evidence changed")
            raw = _sealed(folder / "report.json")
            trace_path = folder / "trace.npz"
            if raw["trace_hash"] != hash_bytes(trace_path.read_bytes()):
                raise ValueError("inherited contact trace changed")
            with np.load(trace_path, allow_pickle=False) as trace:
                force = trace["ball_body_contact_force_peak_n"][:, 0].copy()
        else:
            folder = args.cpu_physics_root / f"seed{seed}-lane{lane}-greedy"
            raw = _sealed(folder / "report.json")
            checked = _sealed(folder / "review.json")
            trace_path = folder / "physical_trace.npz"
            if (
                not checked["high_quality"]
                or checked["reviewed_report_hash"] != raw["report_hash"]
                or raw["physical_trace_hash"] != hash_bytes(trace_path.read_bytes())
            ):
                raise ValueError("inherited CPU success evidence changed")
            with np.load(trace_path, allow_pickle=False) as trace:
                force = trace["force_n"][:, 0].copy()
        if raw["report_hash"] != record["report_hash"] or record["active_frames"] != 270:
            raise ValueError("inherited physical observation identity changed")
        phases.append(phase_sequence(force)[30:])
        records.append(
            dict(**record, reference_kind="CURRENT_PARENT_RECONSTRUCTION_ON_INHERITED_SUCCESS")
        )
    phase = np.concatenate(phases)
    states = encode(x, phase)
    if not np.array_equal(states[:, :134], np.asarray(model["anchor_guard"]["anchors"])):
        raise ValueError("memory metric differs from complete frozen success bank")
    means = np.stack([decoder.raw_mean(row, int(p)) for row, p in zip(x, phase, strict=True)])
    initial = AnchorOutputMemory(
        states,
        means,
        bandwidth=1e-4,
        encoder_hash=encoder,
        parent_policy_hash=model["model_hash"],
        evidence_hash=anchors["report_hash"],
    )
    additions, outputs = [], []
    for row in pilot["rows"]:
        if not row["online"]["high_quality"]:
            continue
        seed, lane = row["seed"], row["lane"]
        folder = args.pilot_root / f"seed{seed}-lane{lane}-online-actor"
        features, contact_phase, raw = gpu_features(folder)
        if (
            raw["report_hash"] != row["online"]["report_hash"]
            or raw["contact_motor_policy"]["step_motor_proof"]["model"] != model
        ):
            raise ValueError("new successful parent execution changed")
        additions.append(encode(features, contact_phase))
        outputs.append(
            np.stack(
                [decoder.raw_mean(v, int(p)) for v, p in zip(features, contact_phase, strict=True)]
            )
        )
        records.append(
            dict(
                backend="IsaacLab",
                seed=seed,
                lane=lane,
                report_hash=raw["report_hash"],
                active_frames=270,
                reference_kind="ACTUAL_REVIEWED_CURRENT_PARENT_SUCCESS",
            )
        )
    if not additions:
        raise ValueError("no newly executed parent success to append")
    combined = initial.extend(
        np.concatenate(additions),
        np.concatenate(outputs),
        parent_policy_hash=model["model_hash"],
        evidence_hash=review["report_hash"],
    )
    blob = combined.to_dict()
    restored = AnchorOutputMemory.from_dict(blob)
    for state, expected in zip(blob["observations"], blob["predictions"], strict=True):
        if not np.array_equal(restored.blend(state, np.zeros(12), encoder_hash=encoder), expected):
            raise ValueError("recorded parent prediction does not reload exactly")
    args.output_root.mkdir(parents=True, exist_ok=False)
    write_once(args.output_root / "initial_memory.json", initial.to_dict())
    write_once(args.output_root / "memory.json", blob)
    manifest = dict(
        schema="soccer.rsi.parent_prediction_memory_bank.v1",
        partition="TRAIN_CONSUMED",
        parent_model_hash=model["model_hash"],
        encoder_hash=encoder,
        records=records,
        inherited_frames=8910,
        newly_executed_success_frames=sum(len(v) for v in additions),
        recorded_frames=len(blob["observations"]),
        independent_contexts=len({(r["seed"], r["lane"]) for r in records}),
        memory_hash=blob["memory_hash"],
        predecessor_memory_hash=initial.to_dict()["memory_hash"],
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        core_source_hash=hash_bytes(Path(memory_module.__file__).read_bytes()),
        pilot_review_hash=review["report_hash"],
        exact_recorded_output_reload=True,
        physical_policy_execution_qualified=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    manifest["report_hash"] = hash_json(manifest)
    write_once(args.output_root / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
