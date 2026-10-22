"""Own physical replay of all96 fixed-teacher perturbation and baseline worlds."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("producer-head", "expected-head"):
        p.add_argument("--" + n, required=True)
    a = p.parse_args()
    from rosclaw_soccer.rsi.stroke_robustness import paired_robustness_summary, robustness_courses
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_stroke_search_common import open_motor
    from scripts.rsi_stroke_teacher_admission import admit_teacher
    from scripts.rsi_stroke_world_audit import replay_stroke_world, verify_matched_prefix

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean frozen paired robustness audit source required")

    clean()
    rp, dp = a.root / "complete.json", a.root / "declaration.json"
    if (
        a.output.exists()
        or any(p.is_symlink() for p in (rp, dp))
        or (a.root / "failed.json").exists()
    ):
        raise ValueError("complete ordinary source and unused independent output required")
    r, d = [json.loads(p.read_text()) for p in (rp, dp)]
    if (
        r["status"] != "FIXED_STROKE_PAIRED48_CLOSED_PENDING_OWN_NATIVE_AUDIT"
        or r["declaration_hash"] != digest(dp)
        or r["input_file_hashes"] != d["input_file_hashes"]
        or d["protocol"] != "dual256.TRAIN.fixed_stroke_paired48.v1"
        or any(
            x["source_head"] != a.producer_head
            or x["hardware_authorized"] is not False
            or x["promotion_authorized"] is not False
            or x["student_success_claim"] is not False
            or x["PPO_actor_rows"] != 0
            for x in (r, d)
        )
        or d["maximum_worlds"] != 96
        or d["maximum_native_steps"] != 384000
        or d["maximum_neural_decisions"] != 38400
        or d["teacher_parameters_frozen"] is not True
        or d["planning_centers_frozen"] is not True
        or d["paired_arms"] != ["baseline", "teacher"]
        or d["local_minimum_successes"] != 39
        or d["local_minimum_per_context"] != 12
        or d["teacher_body_failures_allowed"] != 0
        or d["positive_net_gain_required"] is not True
        or d["no_per_context_net_regression"] is not True
        or r["matched_prefix_verified"] is not True
    ):
        raise ValueError("fixed preregistered paired qualification contract required")
    pins = dict(r["input_file_hashes"])

    def receipt(h):
        ps = [Path(p) for p, v in pins.items() if v == h and p.endswith("/complete.json")]
        if len(ps) != 1:
            raise ValueError("unique exact admitted teacher source required")
        return ps[0]

    sp, ap = receipt(d["search_receipt_hash"]), receipt(d["search_audit_hash"])
    _, training_hash, contexts, centers, latent = admit_teacher(
        sp.parent, ap.parent, d["search_audit_hash"]
    )
    courses = robustness_courses(contexts)
    if (
        d["courses"] != courses
        or d["latent"] != latent
        or d["centers"] != {str(i): c for i, c in centers.items()}
        or r["training_hash"] != training_hash
        or d["training_hash"] != training_hash
        or r["summary"] != paired_robustness_summary(r["records"])
    ):
        raise ValueError("course generation, fixed teacher, or source candidate changed")
    for p in (rp, dp):
        pins[str(p)] = digest(p)
    for index, rec in enumerate(r["records"]):
        course = courses[index // 2]
        arm = ("baseline", "teacher")[index % 2]
        expected = latent if arm == "teacher" else [[0.0] * 6 for _ in range(2)]
        if (
            rec["tag"] != f"course{index // 2}-{arm}"
            or rec["center"] != centers[course["context_id"]]
            or rec["latent"] != expected
        ):
            raise ValueError("every recorded intervention must match its preregistered course")
        for art in rec["artifacts"].values():
            p = a.root / art["path"]
            if (
                p.is_symlink()
                or not p.resolve().is_relative_to(a.root.resolve())
                or digest(p) != art["hash"]
            ):
                raise ValueError("confined exact complete trajectory required")
            pins[str(p)] = art["hash"]
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_hash=digest(rp),
            source_head=a.expected_head,
            producer_head=a.producer_head,
            worlds=96,
            input_file_hashes=pins,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    counts = dict(native=0, neural=0, kinematic_intent=0)
    verified = []
    try:
        import numpy as np

        from rosclaw_soccer.rsi.run_contact_geometry import summarize_contacts

        bench, actor = open_motor(pins, training_hash, contexts)
        previous = None
        for index, rec in enumerate(r["records"]):
            raw = {}
            for name, art in rec["artifacts"].items():
                with np.load(a.root / art["path"], allow_pickle=False) as z:
                    raw[name] = {k: z[k].copy() for k in z.files}
            before = dict(counts)
            result, ss, good, events, goal = replay_stroke_world(
                bench,
                actor,
                courses[index // 2],
                rec["center"],
                np.asarray(rec["latent"], np.float64),
                raw,
                counts,
            )
            if (
                result != rec["result"]
                or ss != rec["stand"]
                or good != rec["healthy_quality"]
                or counts["native"] - before["native"] != rec["native_steps"]
                or counts["neural"] - before["neural"] != rec["actor_decisions"]
            ):
                raise ValueError("own physical outcome or work differs")
            if index % 2:
                verify_matched_prefix(previous, raw)
                previous = None
            else:
                previous = raw
            p = a.output / (rec["tag"] + "-contacts.json")
            write_once(p, dict(events=events, summary=summarize_contacts(events, goal)))
            verified.append(
                dict(
                    tag=rec["tag"],
                    context=rec["context"],
                    course_index=rec["course_index"],
                    arm=rec["arm"],
                    quality=good,
                    contact_path=p.name,
                    contact_hash=digest(p),
                )
            )
            print(
                json.dumps(
                    dict(
                        world=len(verified),
                        tag=rec["tag"],
                        quality=good,
                        own_native=counts["native"],
                    )
                ),
                flush=True,
            )
        if (
            counts["native"] != r["actual_native_steps"]
            or counts["neural"] != r["actual_neural_decisions"]
            or counts["native"] > 384000
            or counts["neural"] > 38400
            or counts["kinematic_intent"] != 0
        ):
            raise ValueError("all bounded independent work accounting must match")
        summary = paired_robustness_summary(r["records"])
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="FIXED_STROKE_PAIRED48_ALL96_OWN_NATIVE_AUDIT_PASS",
                source_head=a.expected_head,
                source_hash=digest(rp),
                input_file_hashes=pins,
                training_hash=training_hash,
                worlds=96,
                counts=counts,
                results=verified,
                summary=summary,
                independent_whole_world_replay=True,
                shared_proposal_and_scoring_implementation=True,
                actor_updates=0,
                critic_updates=0,
                PPO_actor_rows=0,
                student_success_claim=False,
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
                counts=counts,
                completed_worlds=len(verified),
                automatic_retry=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
