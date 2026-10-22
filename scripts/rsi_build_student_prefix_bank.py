"""Aggregate qualified learner-prefix corrections without dropping teacher failures.

65 original histories + all36 teacher continuations + nine actual learned
successful new skills =110 whole sequences. No optimizer or physics execution.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("base-root", "prefix-root", "prefix-audit-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("prefix-source-head", "prefix-auditor-head", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from rosclaw_soccer.rsi.contact_consolidation_data import (
        BANK_STATUS,
        CORRECTION_BANK_STATUS,
        consolidation_pools,
        contact_loss_weights,
        own_reference_sequence,
    )
    from rosclaw_soccer.rsi.student_prefix_feasibility import prefix_feasibility_summary
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_student_prefix_admission import admit_prefix_study

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean pinned actual learner-prefix bank source required")

    clean()
    bp, rp, ap, dp = (
        a.base_root / "complete.json",
        a.prefix_root / "complete.json",
        a.prefix_audit_root / "complete.json",
        a.prefix_root / "declaration.json",
    )
    if (
        a.output.exists()
        or any(x.is_symlink() for x in (bp, rp, ap, dp))
        or any(
            (r / "failed.json").exists() for r in (a.base_root, a.prefix_root, a.prefix_audit_root)
        )
    ):
        raise ValueError("ordinary complete separately audited sources and unused output required")
    base, source, audit, declaration = [json.loads(x.read_text()) for x in (bp, rp, ap, dp)]
    if (
        base["status"] != BANK_STATUS
        or len(base["records"]) != 65
        or source["declaration_hash"] != digest(dp)
        or audit["status"] != "STUDENT_PREFIX36_ALL_OWN_NATIVE_TEACHER_FEASIBILITY_AUDIT_PASS"
        or audit["source_head"] != a.prefix_auditor_head
        or source["source_head"] != a.prefix_source_head
        or audit["source_hash"] != digest(rp)
        or audit["worlds"] != 36
        or audit["counts"]["native"] != source["actual_native_steps"]
        or audit["counts"]["neural"] != source["actual_neural_decisions"]
        or audit["summary"] != prefix_feasibility_summary(audit["results"])
        or audit["summary"]["all_observed_regressions_recoverable"] is not True
        or any(
            v[k] is not False
            for v in (source, audit)
            for k in ("hardware_authorized", "promotion_authorized", "student_success_claim")
        )
    ):
        raise ValueError(
            "all36 independently replayed teacher outcomes and recoverable old losses required"
        )
    pins, warm, students, courses = admit_prefix_study(
        a.base_root, [Path(s["exam_root"]) for s in declaration["students"]]
    )
    if courses != declaration["courses"] or students != declaration["students"]:
        raise ValueError("same original preregistered student-prefix sources required")
    for path, h in audit["input_file_hashes"].items():
        if path in pins and pins[path] != h:
            raise ValueError("conflicting whole-history source")
        pins[path] = h
    for path in (bp, rp, ap, dp):
        pins[str(path)] = digest(path)
    for row in base["records"]:
        path = a.base_root / row["raw_path"]
        if (
            path.is_symlink()
            or path.parent.resolve() != a.base_root.resolve()
            or digest(path) != row["raw_hash"]
        ):
            raise ValueError("exact original complete sequence required")
        pins[str(path)] = row["raw_hash"]
    episodes = []
    for course, row, own in zip(courses, source["records"], audit["results"], strict=True):
        if (
            row["index"] != course["index"]
            or own["index"] != course["index"]
            or own["quality"] != row["healthy_quality"]
        ):
            raise ValueError("all36 aligned outcomes including failures required")
        role = (
            "retention_own_state_correction"
            if course["context"] in (5, 8, 17)
            else "new_skill_teacher_continuation"
        )
        episodes.append((a.prefix_root, row, role, course["seed"]))
    for s in students:
        root = Path(s["exam_root"]) / "native-exam"
        native = json.loads((root / "complete.json").read_text())
        for context in (4, 20, 21):
            row = next(r for r in native["records"] if r["context"] == context and r["repeat"] == 0)
            if not row["healthy_quality"]:
                raise ValueError("all nine actual learned successful skill anchors required")
            episodes.append((root, row, "retained_new_neural_skill", s["seed"]))
    if len(episodes) != 45:
        raise ValueError("all36 outcomes and nine actual neural skill anchors required")
    for root, row, _, _ in episodes:
        for art in row["artifacts"].values():
            path = root / art["path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve())
                or digest(path) != art["hash"]
            ):
                raise ValueError("exact confined physical history required")
            pins[str(path)] = art["hash"]
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(x): digest(x) for x in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="dual256.student_prefix_correction_bank110.v2",
            source_head=a.expected_head,
            input_file_hashes=pins,
            base_data_hash=digest(bp),
            prefix_audit_hash=digest(ap),
            records=110,
            actual_old_W6_anchors=14,
            actual_learned_new_skill_anchors=9,
            all_teacher_continuations=36,
            teacher_failures_retained=True,
            teacher_hidden_state_supplied_to_learner=False,
            Fresh_fit_rows=0,
            DEV_fit_rows=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    records, queries = [], 0
    try:
        import numpy as np

        from scripts.rsi_stroke_search_common import open_motor

        for row in base["records"]:
            path = a.output / f"sequence{row['index']}.npz"
            shutil.copyfile(a.base_root / row["raw_path"], path)
            records.append(dict(row, raw_path=path.name, raw_hash=digest(path)))
        bench, teacher = open_motor(pins, warm, {c["index"]: c["ctx"] for c in courses})
        scale = teacher.parent.action_scale.numpy().copy()
        for root, row, role, seed in episodes:
            raw = {}
            for name, art in row["artifacts"].items():
                with np.load(root / art["path"], allow_pickle=False) as z:
                    raw[name] = {k: z[k].copy() for k in z.files}
            kick = len(raw["kick"]["NN_obs"])
            stand = len(raw["stand"]["NN_obs"]) if "stand" in raw else 0
            eligible = row["healthy_quality"]
            if eligible and (stand != 100 or row["result"]["body_failed"]):
                raise ValueError("positive correction requires complete healthy neural recovery")
            obs = np.concatenate([raw[k]["NN_obs"] for k in ("kick", "stand") if k in raw])
            target = np.concatenate([raw[k]["NN_action"] for k in ("kick", "stand") if k in raw])
            nominal, prior = own_reference_sequence(teacher.parent, obs)
            queries += len(obs)
            weights = contact_loss_weights(
                kick, stand, row["result"]["postcontact"]["first_foot_contact_substep"]
            )
            index = len(records)
            path = a.output / f"sequence{index}.npz"
            np.savez_compressed(
                path,
                observations=obs,
                targets=target,
                action_scale=scale,
                reference_nominal=nominal,
                reference_delta=prior,
                loss_weights=weights,
            )
            records.append(
                dict(
                    index=index,
                    role=role,
                    seed=seed,
                    bc_weight=1.0 if eligible else 0.0,
                    rows=len(obs),
                    kick_rows=kick,
                    stand_rows=stand,
                    task_success=bool(eligible),
                    raw_path=path.name,
                    raw_hash=digest(path),
                    source_root=str(root),
                    source_context=row["context"],
                    source_prefix_controls=row.get("student_prefix_controls"),
                    measured_failure=None if eligible else row["result"],
                )
            )
        if len(records) != 110 or bench.native_steps != 0:
            raise ValueError("complete bank110 without physics integration required")
        consolidation_pools(records, correction=True)
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status=CORRECTION_BANK_STATUS,
                source_head=a.expected_head,
                declaration_hash=digest(a.output / "declaration.json"),
                input_file_hashes=pins,
                records=records,
                base_data_hash=digest(bp),
                prefix_audit_hash=digest(ap),
                warm_student_hashes={str(s["seed"]): s["training_hash"] for s in students},
                parent_lineage_hash=base["parent_lineage_hash"],
                context_witness_hash=base["context_witness_hash"],
                whole_actual_sequences=True,
                teacher_hidden_state_supplied_to_learner=False,
                learner_hidden_width=256,
                observation_width=160,
                own_reference_queries=queries,
                physical_steps=0,
                optimizer_updates=0,
                PPO_actor_rows=0,
                DEV_fit_rows=0,
                Fresh_fit_rows=0,
                student_success_claim=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                completed_sequences=len(records),
                own_reference_queries=queries,
                physical_steps=0,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
