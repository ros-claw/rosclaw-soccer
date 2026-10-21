"""TRAIN-only acquired-parent retention plus two independently audited teachers.

No optimizer or physics execution. Recompute own causal parent on every row;
protect all 43 actual parent successes, not merely the original 36 R2 successes.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "parent-root", "exam-root", "audit-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("data-hash", "parent-hash", "exam-hash", "audit-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and frozen clean feature producer required")
    pins = {}
    reports = []
    for root, expected in (
        (a.data_root, a.data_hash),
        (a.parent_root, a.parent_hash),
        (a.exam_root, a.exam_hash),
        (a.audit_root, a.audit_hash),
    ):
        path = root / "complete.json"
        if root.is_symlink() or (root / "failed.json").exists() or digest(path) != expected:
            raise ValueError("exact closed source commitment required")
        report = json.loads(path.read_text())
        if any(report.get(k) is not False for k in ("hardware_authorized", "promotion_authorized")):
            raise ValueError("SIM-only nonpromoting evidence required")
        for key, value in {**report["input_file_hashes"], str(path): expected}.items():
            if key in pins and pins[key] != value:
                raise ValueError("source lineage conflict")
            pins[key] = value
        reports.append(report)
    data, training, exam, audit = reports
    dp = a.exam_root / "declaration.json"
    declaration = json.loads(dp.read_text())
    from rosclaw_soccer.rsi.paid_student_evidence import qualify_student_worlds
    from rosclaw_soccer.rsi.paid_successor_curriculum import validate_train_curriculum

    qualify_student_worlds(
        exam,
        declaration,
        audit,
        report_path=str(a.exam_root / "complete.json"),
        report_hash=a.exam_hash,
        declaration_hash=digest(dp),
    )
    validate_train_curriculum(data, json.loads((a.data_root / "declaration.json").read_text()))
    if (
        data["curriculum_protocol"] != "successor-six-teacher-v2"
        or training["config"]["protocol"] != "successor-contact-student-v2"
        or training["config"]["seed_index"] != 0
        or declaration["bank"] != "train"
        or declaration["student_hash"] != a.parent_hash
        or Path(declaration["student_root"]).resolve() != a.parent_root.resolve()
        or exam["summary"] != dict(quality_by_kick=[16, 14, 13], stable=16)
        or data["body_scene_hash"] != training["body_scene_hash"]
        or data["base_wrapper_checkpoint_hash"] != training["base_wrapper_checkpoint_hash"]
    ):
        raise ValueError(
            "predeclared four-teacher seed0 acquired parent and six-teacher TRAIN data required"
        )
    pins[str(dp)] = digest(dp)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            schema="soccer.progressive_contact_curriculum.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            bank="train",
            DEV_episodes_fit=0,
            parent_hash=a.parent_hash,
            teacher_cases=[8, 9],
            protected_successes_by_kick=[16, 14, 13],
            target_units="joint_radian_correction_relative_to_own_learned_parent",
            final_R2_relative_bound_radians=0.2,
            internal_difference_bound_radians=0.4,
            features="own_normalized160+own_causal128",
            runtime_case_lookup=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    neural_rows = 0
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_progressive_features import progressive_features
        from rosclaw_soccer.rsi.paid_successor_student_actor import load_successor_student

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policy = next(Path(path) for path in pins if path.endswith("/ckp/policy_30000.onnx"))
        actor, _ = load_successor_student(
            policy,
            checkpoint=Path(training["checkpoint_path"]),
            checkpoint_hash=training["checkpoint_hash"],
            training_receipt=a.parent_root / "complete.json",
            training_receipt_hash=a.parent_hash,
            scene_hash=training["body_scene_hash"],
        )
        actor.requires_grad_(False).eval()
        groups, xs, ys, ps, roles = [], [], [], [], []
        trajectories = []

        def add(path, expected, *, case, skill, teacher):
            nonlocal neural_rows
            if path.is_symlink() or pins.get(str(path)) != expected or digest(path) != expected:
                raise ValueError("independently audited exact TRAIN raw required")
            with np.load(path, allow_pickle=False) as archive:
                raw = {key: archive[key].copy() for key in archive.files}
            x, y, prior = progressive_features(
                actor.base,
                actor.residual,
                raw,
                actor.action_scale,
                recorded_policy="base" if teacher or skill == 0 else "parent",
                protect=not teacher,
            )
            n = len(x)
            if not teacher and np.count_nonzero(y):
                raise ValueError("acquired protection must have exact zero parent-relative target")
            group = len(xs)
            xs.append(x)
            ys.append(y)
            ps.append(prior)
            roles.append(np.full(n, teacher, np.bool_))
            groups.append(np.full(n, group, np.int32))
            trajectories.append(
                dict(
                    case=case,
                    skill=skill,
                    role="teacher" if teacher else "protected_acquired_success",
                    source_path=str(path),
                    source_hash=expected,
                    rows=n,
                    start=neural_rows,
                    stop=neural_rows + n,
                )
            )
            neural_rows += n

        teachers = [
            r for r in data["trajectories"] if r["role"] == "teacher" and r["case"] in (8, 9)
        ]
        if sorted(r["case"] for r in teachers) != [8, 9]:
            raise ValueError("exact two additional TRAIN teachers required")
        for row in teachers:
            add(
                Path(row["source_path"]),
                row["source_hash"],
                case=row["case"],
                skill=2,
                teacher=True,
            )
        protected = [0, 0, 0]
        for i, world in enumerate(exam["worlds"]):
            if not world["stable"]:
                raise ValueError(
                    "physically failed parent cannot supply acquired-success supervision"
                )
            for k, quality in enumerate(world["quality_by_kick"]):
                if quality:
                    row = world["kicks"][k]["result"]
                    path = a.exam_root / row["raw_path"]
                    if not path.resolve().is_relative_to(a.exam_root.resolve()):
                        raise ValueError("confined parent raw required")
                    add(path, row["raw_hash"], case=i, skill=k, teacher=False)
                    protected[k] += 1
        if protected != [16, 14, 13] or len(xs) != 45:
            raise ValueError("all 43 acquired successes and both teachers required")
        x, y, prior, teacher, group = map(np.concatenate, (xs, ys, ps, roles, groups))
        active = np.abs(y).max(axis=1) > 1e-6
        if not active.any() or active[~teacher].any():
            raise ValueError("nontrivial teacher learning and exact acquired retention required")
        feature_path = a.output / "train-features.npz"
        np.savez_compressed(
            feature_path,
            features=x,
            target=y,
            parent_delta=prior,
            teacher=teacher,
            group=group,
            active=active,
        )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_PROGRESSIVE_TRAIN_PARENT_FEATURES_CAUSALLY_RECOMPUTED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                feature_path=str(feature_path),
                feature_hash=digest(feature_path),
                trajectories=trajectories,
                parent_hash=a.parent_hash,
                parent_checkpoint_hash=training["checkpoint_hash"],
                body_scene_hash=training["body_scene_hash"],
                base_wrapper_checkpoint_hash=training["base_wrapper_checkpoint_hash"],
                base_wrapper_training_receipt_hash=training["base_wrapper_training_receipt_hash"],
                rows=neural_rows,
                CPU_base_feature_recompute_rows=neural_rows,
                CPU_parent_head_recompute_rows=neural_rows,
                teacher_rows=int(teacher.sum()),
                protected_acquired_rows=int((~teacher).sum()),
                active_rows=int(active.sum()),
                negative_support_rows=int((~active).sum()),
                protected_successes_by_kick=protected,
                bank="train",
                DEV_episodes_fit=0,
                optimizer_updates=0,
                own_native_steps=0,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                CPU_base_feature_recompute_rows=neural_rows,
                own_native_steps=0,
                hardware_authorized=False,
                automatic_retry=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
