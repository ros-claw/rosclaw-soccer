"""Fixed best stroke under48 new public TRAIN perturbations, paired with baseline."""

import argparse
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("search-root", "search-audit-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("search-audit-hash", "expected-head"):
        p.add_argument("--" + n, required=True)
    a = p.parse_args()
    from rosclaw_soccer.rsi.stroke_robustness import paired_robustness_summary, robustness_courses
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_stroke_search_common import collect, make_episode, open_motor
    from scripts.rsi_stroke_teacher_admission import admit_teacher

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean frozen paired robustness source required")

    clean()
    if a.output.exists():
        raise ValueError("unused robustness output required")
    pins, training_hash, contexts, centers, latent = admit_teacher(
        a.search_root, a.search_audit_root, a.search_audit_hash
    )
    courses = robustness_courses(contexts)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="dual256.TRAIN.fixed_stroke_paired48.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            training_hash=training_hash,
            search_receipt_hash=digest(a.search_root / "complete.json"),
            search_audit_hash=a.search_audit_hash,
            courses=courses,
            centers={str(i): c for i, c in centers.items()},
            latent=latent,
            maximum_worlds=96,
            maximum_native_steps=384000,
            maximum_neural_decisions=38400,
            teacher_parameters_frozen=True,
            planning_centers_frozen=True,
            paired_arms=["baseline", "teacher"],
            local_minimum_successes=39,
            local_minimum_per_context=12,
            teacher_body_failures_allowed=0,
            positive_net_gain_required=True,
            no_per_context_net_regression=True,
            student_success_claim=False,
            PPO_actor_rows=0,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    rows = []
    bench = None
    try:
        import json

        import numpy as np

        bench, actor = open_motor(pins, training_hash, contexts)
        for ctx in courses:
            make_episode(bench, ctx)
        if bench.native_steps != 0:
            raise ValueError("entire bank must preflight before physics")
        write_once(a.output / "preflight.json", dict(courses=48, valid=True, physics_steps=0))
        z = np.asarray(latent, np.float64)
        for ctx in courses:
            pair = []
            for arm, proposal in (("baseline", np.zeros((2, 6))), ("teacher", z)):
                tag = f"course{ctx['course_index']}-{arm}"
                row, blobs = collect(
                    bench, actor, ctx, centers[ctx["context_id"]], proposal, a.output / tag
                )
                row.update(course_index=ctx["course_index"], arm=arm, assisted=arm == "teacher")
                write_once(a.output / tag / "complete.json", row)
                rows.append(row)
                pair.append(blobs)
                print(
                    json.dumps(
                        dict(
                            world=len(rows),
                            course=ctx["course_index"],
                            arm=arm,
                            quality=row["healthy_quality"],
                            native=bench.native_steps,
                        )
                    ),
                    flush=True,
                )
            aa, bb = pair[0]["kick"], pair[1]["kick"]
            n = min(80, len(aa["qpos"]), len(bb["qpos"]))
            if n < 80 and len(aa["qpos"]) != len(bb["qpos"]):
                raise ValueError("pre-intervention termination differs")
            for k in (
                "NN_obs",
                "NN_h_before",
                "NN_c_before",
                "NN_action",
                "qpos",
                "qvel",
                "forces",
                "torque",
            ):
                if not np.array_equal(aa[k][:n], bb[k][:n]):
                    raise ValueError("matched physical/NN prefix differs")
            if bench.native_steps > 384000 or sum(r["actor_decisions"] for r in rows) > 38400:
                raise ValueError("fixed paired robustness work budget exceeded")
        summary = paired_robustness_summary(rows)
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="FIXED_STROKE_PAIRED48_CLOSED_PENDING_OWN_NATIVE_AUDIT",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                training_hash=training_hash,
                records=rows,
                summary=summary,
                actual_native_steps=bench.native_steps,
                actual_neural_decisions=sum(r["actor_decisions"] for r in rows),
                matched_prefix_verified=True,
                actor_updates=0,
                critic_updates=0,
                student_success_claim=False,
                PPO_actor_rows=0,
                promotion_authorized=False,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
        print(json.dumps(summary), flush=True)
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                completed_worlds=len(rows),
                actual_native_steps=bench.native_steps if bench else 0,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
