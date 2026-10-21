"""Append only executed successful expert takeovers to the unchanged TRAIN curriculum."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "pilot", "audit"):
        p.add_argument("--" + name + "-root", type=Path, required=True)
        p.add_argument("--" + name + "-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused data bank and clean frozen curation source required")
    pins = {}
    for name in ("original", "pilot", "audit"):
        root = getattr(a, name + "_root")
        rp = root / "complete.json"
        if rp.is_symlink() or digest(rp) != getattr(a, name + "_hash"):
            raise ValueError("exact predeclared closed source receipts required")
        d = json.loads(rp.read_text())
        for path, h in d["input_file_hashes"].items():
            if path in pins and pins[path] != h:
                raise ValueError("source lineage conflict")
            pins[path] = h
        pins[str(rp)] = digest(rp)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_full_entry_student import load_full_entry_data
        from rosclaw_soccer.rsi.paid_interventional_data import (
            DATA_STATUS,
            aggregate_handoff,
            load_recovery_witness,
        )

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
        parent, baseline, old = load_full_entry_data(a.original_root, a.original_hash, policy)
        pr, ar = load_recovery_witness(
            a.pilot_root,
            a.pilot_hash,
            a.audit_root,
            a.audit_hash,
            original_hash=a.original_hash,
            scene_hash=baseline["body_scene_hash"],
        )
        identity = dict(
            source_head=a.expected_head,
            original_root=str(a.original_root),
            original_hash=a.original_hash,
            pilot_root=str(a.pilot_root),
            pilot_hash=a.pilot_hash,
            audit_root=str(a.audit_root),
            audit_hash=a.audit_hash,
            bank="train",
            Fresh_fit_rows=0,
            DEV_fit_rows=0,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        write_once(
            a.output / "declaration.json",
            dict(
                schema="paid.interventional_dagger.TRAIN_features.v1",
                **identity,
                label_source="actual_executed_expert_on_successful_recovery_only",
                excluded_prefix="student_prefix_not_relabeled_as_teacher",
                canonical_repeat=0,
                original_rows_immutable=True,
                optimizer_updates=0,
                own_native_steps=0,
            ),
        )
        z, groups = aggregate_handoff(parent, old, a.pilot_root, pr, ar, a.pilot_hash)
        for g in groups:
            pins[g["source_path"]] = g["source_hash"]
        original_archive = Path(baseline["feature_path"])
        pins[str(original_archive)] = baseline["feature_hash"]
        archive = a.output / "interventional-features.npz"
        np.savez_compressed(archive, **z)
        failures = []
        for arm in pr["arms"]:
            for repeat, rows in enumerate(arm["repeats"]):
                for row in rows:
                    failures.append(
                        dict(
                            arm=arm["arm"],
                            case=row["case"],
                            repeat=repeat,
                            healthy_success=row["healthy_success"],
                            quality=row["quality"],
                            kick_raw_hash=row["kick"]["raw_hash"],
                            postcontact=row["kick"]["postcontact"],
                            selected_for_positive_labels=repeat == 0
                            and any(
                                g["arm"] == arm["arm"] and g["case"] == row["case"] for g in groups
                            ),
                        )
                    )
        write_once(
            a.output / "source-outcomes.json",
            dict(
                records=failures,
                repeats_are_not_extra_contexts=True,
                failed_branches_retained=True,
                counterfactual_labels_used_for_training=False,
            ),
        )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status=DATA_STATUS,
                **identity,
                declaration_hash=digest(a.output / "declaration.json"),
                input_file_hashes=pins,
                feature_path=str(archive),
                feature_hash=digest(archive),
                **{
                    k: baseline[k]
                    for k in ("parent_hash", "parent_checkpoint_hash", "body_scene_hash")
                },
                rows=len(z["features"]),
                original_rows=len(old["features"]),
                added_rows=len(z["features"]) - len(old["features"]),
                teacher_rows=int(z["teacher"].sum()),
                protected_rows=int((~z["teacher"]).sum()),
                active_rows=int(z["active"].sum()),
                new_groups=groups,
                old_rowwise_parent_verification_rows=len(old["features"]),
                additional_R2_forward_rows=sum(g["whole_source_rows"] for g in groups),
                additional_parent_forward_rows=sum(
                    g["whole_source_rows"] - g["switch_control"] for g in groups
                ),
                original_arrays_exact_prefix=True,
                source_outcomes_hash=digest(a.output / "source-outcomes.json"),
                optimizer_updates=0,
                own_native_steps=0,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                automatic_retry=False,
                optimizer_updates=0,
                own_native_steps=0,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
