"""Close a second-generation TRAIN dataset from all-trial-audited current-state teachers."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for key in ("previous-root", "replan-root", "audit-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("previous-hash", "producer-head", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--wait-for-completion", action="store_true")
    a = p.parse_args()

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean frozen second-generation curation required")

    clean()
    if a.output.exists():
        raise ValueError("unused curation output required")
    deadline = time.monotonic() + 7200
    while not all((root / "complete.json").exists() for root in (a.replan_root, a.audit_root)):
        clean()
        if any((root / "failed.json").exists() for root in (a.replan_root, a.audit_root)):
            raise ValueError(
                "failed teacher/audit: preserve without implicit retry or positive labels"
            )
        if not a.wait_for_completion or time.monotonic() > deadline:
            raise TimeoutError("complete own-prefix search and all-trial audit required")
        time.sleep(10)
    previous_path, producer_path, audit_path = [
        root / "complete.json" for root in (a.previous_root, a.replan_root, a.audit_root)
    ]
    if (
        any(path.is_symlink() for path in (previous_path, producer_path, audit_path))
        or digest(previous_path) != a.previous_hash
    ):
        raise ValueError("exact ordinary previous dataset and source receipts required")
    previous, producer, audit = [
        json.loads(f.read_text()) for f in (previous_path, producer_path, audit_path)
    ]
    producer_hash, audit_hash = digest(producer_path), digest(audit_path)
    decl_path = a.replan_root / "declaration.json"
    decl = json.loads(decl_path.read_text())
    if (
        producer["source_head"] != a.producer_head
        or audit["source_head"] != a.producer_head
        or producer["declaration_hash"] != digest(decl_path)
        or decl["source_head"] != a.producer_head
    ):
        raise ValueError("exact declared producer and independent audit source required")
    args = decl["source_arguments"]
    student_path = Path(args["student_root"]) / "complete.json"
    if student_path.is_symlink() or digest(student_path) != args["student_hash"]:
        raise ValueError("exact current student source identity required")
    student = json.loads(student_path.read_text())
    if (
        student["data_hash"] != a.previous_hash
        or student["status"]
        != "PAID_INTERVENTIONAL_DAGGER_STUDENT_TRAINED_NOT_PHYSICALLY_QUALIFIED"
        or student["config"]["seed_index"] != 0
        or any(
            student[k] != previous[k]
            for k in ("body_scene_hash", "parent_hash", "parent_checkpoint_hash")
        )
    ):
        raise ValueError("teacher must recover the current student's actual curriculum and body")
    pins = {}
    for r in (previous, producer, audit):
        for path, h in r["input_file_hashes"].items():
            if path in pins and pins[path] != h:
                raise ValueError("conflicting dataset/expert lineage")
            pins[path] = h
    for path in (previous_path, producer_path, audit_path, decl_path, student_path):
        pins[str(path)] = digest(path)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    clean()
    a.output.mkdir()
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_interventional_data import load_interventional_data
        from rosclaw_soccer.rsi.paid_replan_data import admitted_replan_windows, aggregate_replan

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
        parent, baseline, old = load_interventional_data(a.previous_root, a.previous_hash, policy)
        windows = admitted_replan_windows(producer, audit, producer_hash)
        identity = dict(
            source_head=a.expected_head,
            schema="paid.interventional_dagger.TRAIN_features.v2",
            previous_root=str(a.previous_root),
            previous_hash=a.previous_hash,
            replan_root=str(a.replan_root),
            replan_hash=producer_hash,
            audit_root=str(a.audit_root),
            audit_hash=audit_hash,
            original_root=baseline["original_root"],
            original_hash=baseline["original_hash"],
            source_student_hash=args["student_hash"],
            bank="train",
            Fresh_fit_rows=0,
            DEV_fit_rows=0,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        write_once(a.output / "declaration.json", dict(**identity, input_file_hashes=pins))

        def read(link):
            path = a.replan_root / link["raw_path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(a.replan_root.resolve())
                or digest(path) != link["raw_hash"]
                or audit["input_file_hashes"].get(str(path)) != link["raw_hash"]
            ):
                raise ValueError("exact independently audited executed expert raw required")
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        z, groups = aggregate_replan(parent, old, windows, read, source_hash=producer_hash)
        archive = a.output / "features.npz"
        np.savez_compressed(archive, **z)
        outcomes = []
        admitted = {(w["switch_control"], w["case"]) for w in windows}
        for search in producer["searches"]:
            pairs = [
                ("student-control", search["baseline"]),
                ("fixed-teacher-control", search["fixed_teacher"]),
            ]
            pairs += [
                (f"generation{g['generation']}-slot{c['slot']}", c["rows"])
                for g in search["history"]
                for c in g["candidates"]
            ]
            pairs += [(f"selected-repeat{i}", rows) for i, rows in enumerate(search["selected"])]
            for phase, rows in pairs:
                for row in rows:
                    outcomes.append(
                        dict(
                            switch_control=search["switch_control"],
                            phase=phase,
                            case=row["case"],
                            quality=row["quality"],
                            healthy_success=row["healthy_success"],
                            admitted_positive=phase == "selected-repeat0"
                            and (search["switch_control"], row["case"]) in admitted,
                        )
                    )
        if len(outcomes) != 112:
            raise ValueError("all charged success and failure memories required")
        write_once(
            a.output / "source-outcomes.json",
            dict(
                records=outcomes,
                failed_branches_retained=True,
                counterfactual_labels_used_for_training=False,
                repeats_are_not_extra_contexts=True,
            ),
        )
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_REPLANNED_INTERVENTIONAL_TRAIN_FEATURES_CAUSALLY_RECOMPUTED",
                **identity,
                declaration_hash=digest(a.output / "declaration.json"),
                input_file_hashes=pins,
                feature_path=str(archive),
                feature_hash=digest(archive),
                new_groups=groups,
                rows=len(z["features"]),
                previous_rows=len(old["features"]),
                original_rows=baseline["original_rows"],
                added_rows=len(z["features"]) - len(old["features"]),
                teacher_rows=int(z["teacher"].sum()),
                protected_rows=int((~z["teacher"]).sum()),
                active_rows=int(z["active"].sum()),
                source_outcomes_hash=digest(a.output / "source-outcomes.json"),
                **{
                    k: baseline[k]
                    for k in ("body_scene_hash", "parent_hash", "parent_checkpoint_hash")
                },
                previous_arrays_exact_prefix=True,
                additional_R2_forward_rows=sum(g["whole_source_rows"] for g in groups),
                additional_parent_forward_rows=sum(
                    g["whole_source_rows"] - g["switch_control"] for g in groups
                ),
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
                own_native_steps=0,
                optimizer_updates=0,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
