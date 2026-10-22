"""Independently replay all36 student-prefix -> teacher continuation worlds."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("producer-head", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--failed-predecessor-root", type=Path)
    p.add_argument("--failed-predecessor-hash")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_student_prefix_admission import admit_prefix_study

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean independently pinned student-prefix native audit required")

    clean()
    rp, dp = a.root / "complete.json", a.root / "declaration.json"
    if (
        a.output.exists()
        or any(p.is_symlink() for p in (rp, dp))
        or (a.root / "failed.json").exists()
    ):
        raise ValueError("complete unmodified source and unused own-native audit output required")
    r, d = [json.loads(p.read_text()) for p in (rp, dp)]
    if (
        r["status"] != "STUDENT_PREFIX36_TEACHER_PROBED_PENDING_OWN_NATIVE_AUDIT"
        or r["source_head"] != a.producer_head
        or d["source_head"] != a.producer_head
        or r["declaration_hash"] != digest(dp)
        or r["input_file_hashes"] != d["input_file_hashes"]
        or len(r["records"]) != 36
        or d["worlds"] != 36
        or d["source_contexts"] != 18
        or d["boundaries_per_source"] != 2
        or r["actual_native_steps"] > 144000
        or r["actual_neural_decisions"] > 25200
        or any(
            v[k] is not False
            for v in (r, d)
            for k in ("hardware_authorized", "promotion_authorized", "student_success_claim")
        )
        or r["teacher_optimization_steps"] != 0
        or r["optimizer_updates"] != 0
    ):
        raise ValueError("exact bounded36 whole-world diagnostic contract required")
    roots = [Path(s["exam_root"]) for s in d["students"]]
    bank_paths = [
        Path(p)
        for p, h in d["input_file_hashes"].items()
        if p.endswith("/complete.json") and h == d["data_hash"]
    ]
    if len(bank_paths) != 1:
        raise ValueError("unique original bank65 required")
    pins, warm, students, courses = admit_prefix_study(bank_paths[0].parent, roots)
    if courses != d["courses"] or students != d["students"] or warm != d["warm_start_hash"]:
        raise ValueError("regenerated complete course declaration differs")
    for path, h in d["input_file_hashes"].items():
        if path in pins and pins[path] != h:
            raise ValueError("conflicting own replay lineage")
        pins[path] = h
    for path in (rp, dp):
        pins[str(path)] = digest(path)
    if bool(a.failed_predecessor_root) != bool(a.failed_predecessor_hash):
        raise ValueError("exact preserved audit failure binding required")
    if a.failed_predecessor_root is not None:
        failed = a.failed_predecessor_root / "failed.json"
        if failed.is_symlink() or digest(failed) != a.failed_predecessor_hash:
            raise ValueError("exact original zero-native-step audit failure required")
        failure = json.loads(failed.read_text())
        if (
            failure["completed_worlds"] != 0
            or failure["counts"] != dict(native=0, neural=1, kinematic_intent=0, shadow_neural=0)
            or failure["error"]
            != "ValueError('one causal observation per real neural control required')"
        ):
            raise ValueError("recovery only fixes duplicate causal observer before physical replay")
        pins[str(failed)] = a.failed_predecessor_hash
    for row in r["records"]:
        for art in row["artifacts"].values():
            path = a.root / art["path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(a.root.resolve())
                or digest(path) != art["hash"]
            ):
                raise ValueError("confined exact physical source trajectory required")
            pins[str(path)] = art["hash"]
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(x): digest(x) for x in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    counts = dict(native=0, neural=0, kinematic_intent=0, shadow_neural=0)
    results = []
    try:
        import numpy as np

        from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
        from rosclaw_soccer.rsi.student_prefix_feasibility import (
            prefix_feasibility_summary,
            verify_physical_prefix,
        )
        from scripts.rsi_stroke_search_common import open_motor
        from scripts.rsi_stroke_world_audit import replay_stroke_world

        bench, teacher = open_motor(pins, warm, {c["index"]: c["ctx"] for c in courses})
        actors = {}
        for s in students:
            actor, _ = load_dual_candidate_for_sim_audit(
                teacher.parent,
                parent_lineage_hash=json.loads(
                    (Path(s["training_root"]) / "complete.json").read_text()
                )["parent_lineage_hash"],
                root=Path(s["training_root"]),
                expected_receipt_hash=s["training_hash"],
                checkpoint_kind="contact-consolidated",
            )
            actors[s["seed"]] = actor
        for c, row in zip(courses, r["records"], strict=True):
            if (
                row["index"] != c["index"]
                or row["seed"] != c["seed"]
                or row["context"] != c["context"]
                or row["student_prefix_controls"] != c["prefix_controls"]
                or row["baseline_quality"] != c["baseline_quality"]
            ):
                raise ValueError("same complete course identities and original outcomes required")
            raw = {}
            for name, art in row["artifacts"].items():
                with np.load(a.root / art["path"], allow_pickle=False) as z:
                    raw[name] = {k: z[k].copy() for k in z.files}
            with np.load(c["baseline_kick_path"], allow_pickle=False) as z:
                baseline = {k: z[k].copy() for k in z.files}
            n = c["prefix_controls"]
            verify_physical_prefix(baseline, raw["kick"], n)
            before = dict(counts)
            result, stand, good, events, _ = replay_stroke_world(
                bench,
                teacher,
                c["ctx"],
                c["center"],
                np.asarray(c["latent"], np.float64),
                raw,
                counts,
                shadow_student=actors[c["seed"]],
                prefix_controls=n,
                student_source=baseline,
            )
            if (
                result != row["result"]
                or stand != row["stand"]
                or good != row["healthy_quality"]
                or counts["native"] - before["native"] != row["native_steps"]
                or counts["neural"] - before["neural"] != row["actor_decisions"]
                or counts["shadow_neural"] - before["shadow_neural"]
                != row["shadow_student_queries"]
            ):
                raise ValueError("whole own-neural/native result or actual inference count differs")
            contacts = a.output / f"world{c['index']}-contacts.json"
            write_once(contacts, dict(events=events))
            out = dict(
                index=c["index"],
                seed=c["seed"],
                context=c["context"],
                prefix_controls=n,
                quality=good,
                baseline_quality=c["baseline_quality"],
                contacts_path=contacts.name,
                contacts_hash=digest(contacts),
            )
            results.append(out)
            print(json.dumps(dict(out, own_native=counts["native"])), flush=True)
        if (
            counts["native"] != r["actual_native_steps"]
            or counts["neural"] != r["actual_neural_decisions"]
            or counts["kinematic_intent"] != 0
        ):
            raise ValueError("all actual teacher/student/world accounting must match")
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="STUDENT_PREFIX36_ALL_OWN_NATIVE_TEACHER_FEASIBILITY_AUDIT_PASS",
                source_head=a.expected_head,
                input_file_hashes=pins,
                source_hash=digest(rp),
                counts=counts,
                results=results,
                summary=prefix_feasibility_summary(results),
                worlds=36,
                source_contexts=18,
                teacher_successes=sum(x["quality"] for x in results),
                rescues=sum(x["quality"] and not x["baseline_quality"] for x in results),
                regressions=sum(not x["quality"] and x["baseline_quality"] for x in results),
                student_success_claim=False,
                teacher_generalization_proven=False,
                optimizer_updates=0,
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
                counts=counts,
                completed_worlds=len(results),
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
