"""Can the qualified teacher repair a body state reached by an actual student?

Three seeds, six consumed contexts, two fixed handover boundaries: 36 complete
worlds. The student physically controls the original prefix, never teleports.
Not DAgger training, not generalization, no automatic candidate promotion.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("data-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--exam-roots", type=Path, nargs=3, required=True)
    p.add_argument("--expected-head", required=True)
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
            raise ValueError("frozen clean actual student-prefix probe source required")

    clean()
    if a.output.exists():
        raise ValueError("unused bounded diagnostic output required")
    pins, warm, students, courses = admit_prefix_study(a.data_root, a.exam_roots)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(x): digest(x) for x in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="dual256.actual_student_prefix_teacher_feasibility36.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            warm_start_hash=warm,
            students=students,
            courses=courses,
            data_hash=digest(a.data_root / "complete.json"),
            worlds=36,
            source_contexts=18,
            boundaries_per_source=2,
            maximum_native_steps=144000,
            maximum_neural_queries=25200,
            source_prefix_physics_unchanged=True,
            teacher_optimization_steps=0,
            Fresh_fit_rows=0,
            optimizer_updates=0,
            student_success_claim=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    bench, records, queries = None, [], 0
    try:
        import numpy as np

        from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
        from scripts.rsi_stroke_search_common import collect, make_episode, open_motor

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
        for course in courses:
            make_episode(bench, course["ctx"])
        if bench.native_steps != 0:
            raise ValueError("entire declared bank must preflight before first physical step")
        write_once(
            a.output / "preflight.json", dict(worlds=36, physical_steps=0, all_cases_admitted=True)
        )
        for c in courses:
            with np.load(c["baseline_kick_path"], allow_pickle=False) as z:
                baseline = {k: z[k].copy() for k in z.files}
            row, _ = collect(
                bench,
                teacher,
                c["ctx"],
                c["center"],
                np.asarray(c["latent"], np.float64),
                a.output / f"course{c['index']}",
                shadow_student=actors[c["seed"]],
                student_prefix_controls=c["prefix_controls"],
                student_source=baseline,
            )
            row.update(index=c["index"], seed=c["seed"], baseline_quality=c["baseline_quality"])
            records.append(row)
            queries += row["actor_decisions"]
            write_once(a.output / f"world{c['index']}.json", row)
            print(
                json.dumps(
                    dict(
                        index=c["index"],
                        seed=c["seed"],
                        context=c["context"],
                        prefix=c["prefix_controls"],
                        quality=row["healthy_quality"],
                        baseline=c["baseline_quality"],
                        native=bench.native_steps,
                    )
                ),
                flush=True,
            )
            if bench.native_steps > 144000 or queries > 25200:
                raise ValueError("bounded whole-world probe budget exceeded")
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="STUDENT_PREFIX36_TEACHER_PROBED_PENDING_OWN_NATIVE_AUDIT",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                records=records,
                actual_native_steps=bench.native_steps,
                actual_neural_decisions=queries,
                source_contexts=18,
                worlds=36,
                teacher_optimization_steps=0,
                optimizer_updates=0,
                teacher_own_state_qualification_pending=True,
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
                completed_worlds=len(records),
                actual_native_steps=bench.native_steps if bench else 0,
                actual_neural_decisions=queries,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
