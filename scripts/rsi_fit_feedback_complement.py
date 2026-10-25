"""Fit a plastic motor subspace from actual successful corrections, no physics."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "nominal-audit",
        "nominal-root",
        "perturbed-audit",
        "perturbed-root",
        "backbone-model",
        "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("nominal-hash", "perturbed-hash", "backbone-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins
    from scripts.rsi_verified_teacher_sources import admit_teacher_sources

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("new model output required")
    pins, sources = admit_teacher_sources(
        (
            ("nominal", a.nominal_audit, a.nominal_hash, a.nominal_root),
            ("perturbed", a.perturbed_audit, a.perturbed_hash, a.perturbed_root),
        )
    )
    if (
        pins.get(str(a.backbone_model)) != a.backbone_hash
        or digest(a.backbone_model) != a.backbone_hash
    ):
        raise ValueError("exact independently verified frozen backbone required")
    backbone = json.loads(a.backbone_model.read_text())
    import numpy as np

    from rosclaw_soccer.rsi.feedback_motion_complement import fit_feedback_complement
    from rosclaw_soccer.sim.contracts import hash_json

    artifact = a.backbone_model.parent / backbone["artifact"]["path"]
    if pins.get(str(artifact)) != digest(artifact):
        raise ValueError("bound frozen backbone artifact required")
    with np.load(artifact, allow_pickle=False) as raw:
        basis = raw["basis_hg"].copy()
    samples, records, selected = [], [], []
    for kind, root, producer, audit in sources:
        for i, result in enumerate(producer["results"]):
            rows, arrays = {}, {}
            for arm in ("B", "C"):
                ref = result["arms"][arm]
                path = root / ref["record"]
                if (
                    not path.resolve().is_relative_to(root.resolve())
                    or pins.get(str(path)) != ref["hash"]
                    or digest(path) != ref["hash"]
                ):
                    raise ValueError("bound paired actual record required")
                row = json.loads(path.read_text())
                if hash_json(row["outcome"]) != hash_json(audit["results"][i][arm]):
                    raise ValueError("paired outcome differs from native readback")
                rows[arm] = row
            c, b = rows["C"], rows["B"]
            qualifies = bool(
                c.get("changed_controls", 0)
                and c["outcome"]["task_pass"]
                and c["healthy_full_horizon"]
            )
            records.append(
                dict(
                    kind=kind,
                    course_index=i,
                    included=qualifies,
                    C_task_pass=c["outcome"]["task_pass"],
                    C_healthy=c["healthy_full_horizon"],
                    source_C_hash=result["arms"]["C"]["hash"],
                )
            )
            if not qualifies:
                continue
            entry = c["entry_control"]
            end = min(entry + 48, c["outcome"]["first_foot_contact_frame"] + 10, 300)
            if entry != b["entry_control"] or end <= entry:
                raise ValueError("causally matched precontact/swing interval required")
            for arm, row in rows.items():
                path = root / result["arms"][arm]["record"]
                raw_path = path.parent / row["artifact"]["path"]
                if pins.get(str(raw_path)) != digest(raw_path):
                    raise ValueError("bound actual control trace required")
                with np.load(raw_path, allow_pickle=False) as raw:
                    arrays[arm] = raw["joint_targets"][entry:end].copy()
            samples.append(arrays["C"] - arrays["B"])
            selected.append(
                dict(
                    kind=kind, source_context=i, start=entry, end_exclusive=end, samples=end - entry
                )
            )
    if len(samples) < 4:
        raise ValueError("multiple actual successful correction trajectories required")
    learned, fit = fit_feedback_complement(np.concatenate(samples).astype(np.float64), basis)
    source_pins(pins)
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    artifact = a.output / "feedback-complement.npz"
    np.savez_compressed(artifact, basis_hg=learned, frozen_backbone_basis_hg=basis)
    write_once(
        a.output / "complete.json",
        dict(
            status="ACTUAL_SUCCESSFUL_MOTOR_FEEDBACK_COMPLEMENT_FITTED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            backbone_model_hash=a.backbone_hash,
            fit=fit,
            all_context_records=records,
            selected_actual_intervals=selected,
            actual_paired_states=len(selected),
            original_context_clusters=len({r["source_context"] for r in selected}),
            artifact=dict(path=artifact.name, hash=digest(artifact)),
            new_physical_steps=0,
            neural_optimizer_updates=0,
            physics_qualified=False,
            prior_perturbation_bank_is_now_consumed_development=True,
            requires_new_disjoint_local_validation=True,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
