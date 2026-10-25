"""Fixed single-finisher contact-authority pilot; no optimizer or promotion.

Every legal failed task remains a measured result. Only integrity/runner
failures stop the experiment, without restarting a charged physical attempt.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("plan-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--variant", choices=("position", "match-half", "match-full"), required=True)
    a = p.parse_args()
    from rosclaw_soccer.rsi.native_contact_authority_metrics import contact_authority_metrics
    from rosclaw_soccer.rsi.native_foot_contact_teacher import FootContactTeacherConfig
    from rosclaw_soccer.training.native_team_courses import native_team_course_bank
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists() or a.plan.is_symlink() or digest(a.plan) != a.plan_hash:
        raise ValueError("new pilot output and exact preregistered plan required")
    plan = json.loads(a.plan.read_text())
    base_path = Path(plan["parent_plan"])
    if base_path.is_symlink() or digest(base_path) != plan["parent_plan_hash"]:
        raise ValueError("exact qualified native parent plan required")
    base = json.loads(base_path.read_text())
    courses = native_team_course_bank()
    if (
        plan["schema"] != "soccer.native_contact_authority_pilot.v1"
        or plan["source_head"] != a.expected_head
        or preflight_receiving_courses(courses) != base["course_bank_hash"]
        or [c.course_id for c in courses] != base["courses"]
        or plan["hardware_authorized"] is not False
    ):
        raise ValueError("whole fixed five-course SIM authority pilot required")
    from dataclasses import asdict

    import numpy as np

    parameters = asdict(
        FootContactTeacherConfig(
            strength=0.5 if a.variant == "match-half" else 1.0,
            velocity_match_gain=0.0 if a.variant == "position" else 1.0,
        )
    )
    motor = Path(plan["motor_root"])
    if subprocess.check_output(
        ["git", "-C", str(motor), "rev-parse", "HEAD"], text=True
    ).strip() != plan["motor_head"] or subprocess.check_output(
        ["git", "-C", str(motor), "status", "--porcelain"], text=True
    ):
        raise ValueError("unchanged exact teacher motor source required")
    zero_root = Path(plan["zero_root"])
    path = zero_root / "complete.json"
    if path.is_symlink() or digest(path) != plan["zero_hash"]:
        raise ValueError("completed actual zero-teacher identity proof required")
    zero = json.loads(path.read_text())
    zero_declaration = json.loads((zero_root / "declaration.json").read_text())
    if (
        zero["status"] != "NATIVE8_CONTACT_TEACHER_AUTHORITY_MEASURED_NOT_LEARNED"
        or zero["source_head"] != plan["motor_head"]
        or zero_declaration["protocol"] != "native8_single_player_foot_contact_teacher.v9"
        or zero_declaration["foot_contact_teacher"]["parameters"]["strength"] != 0
        or zero["optimizer_updates"] != 0
        or not zero["fresh_reexecution_exact"]
        or len(zero["records"]) != 4
        or not all(r["healthy_full_horizon"] for r in zero["records"])
        or not all(all(r["teacher_parent_identity"].values()) for r in zero["records"])
        or not all(all(r["teacher_target_authority"].values()) for r in zero["records"])
        or any(r["teacher_active_controls"] != 0 for r in zero["records"])
    ):
        raise ValueError("zero strength must preserve full actual parent physics")
    pins = dict(zero["input_file_hashes"])
    pins[str(path)] = plan["zero_hash"]
    pins[str(a.plan)] = a.plan_hash
    pins[str(base_path)] = plan["parent_plan_hash"]
    for row in zero["records"]:
        path = zero_root / row["artifact"]["path"]
        if path.is_symlink() or digest(path) != row["artifact"]["hash"]:
            raise ValueError("zero identity arrays changed")
        pins[str(path)] = row["artifact"]["hash"]
    # Preflight EVERY declared parent before the first child is allocated.
    for course in courses:
        ref = plan["parent_courses"][course.course_id]
        path = Path(ref["root"]) / "complete.json"
        if path.is_symlink() or digest(path) != ref["hash"]:
            raise ValueError("entire fixed same-course parent bank required")
        pins[str(path)] = ref["hash"]
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            schema=plan["schema"],
            source_head=a.expected_head,
            plan_hash=a.plan_hash,
            variant=a.variant,
            parameters=parameters,
            maximum_worlds=20,
            maximum_native_steps=120000,
            independent_geometries=5,
            task_failures_are_data=True,
            optimizer_updates=0,
            hardware_authorized=False,
        ),
    )
    results = []
    try:
        for course in courses:
            ref = plan["parent_courses"][course.course_id]
            config_path = a.output / ("config-" + course.course_id + ".json")
            write_once(
                config_path,
                dict(
                    schema="soccer.native_foot_contact_teacher.v1",
                    agent="red_finisher",
                    parameters=parameters,
                    parent_root=ref["root"],
                    parent_hash=ref["hash"],
                ),
            )
            output = a.output / course.course_id
            command = [
                sys.executable,
                "-m",
                "scripts.rsi_probe_native_team_intent",
                "--intercept-course",
                "--foot-track-course",
                "--clearance-course",
                "--ball-course",
                course.course_id,
                "--foot-contact-config",
                str(config_path),
                "--foot-contact-config-hash",
                digest(config_path),
                "--expected-head",
                plan["motor_head"],
                "--output",
                str(output),
            ]
            for option, key in (
                ("core-source-root", "core_root"),
                ("core-expected-head", "core_head"),
                ("parent", "parent_root"),
                ("parent-hash", "parent_hash"),
                ("own", "own_root"),
                ("own-hash", "own_hash"),
                ("previous", "previous_root"),
                ("previous-hash", "previous_hash"),
            ):
                command += ["--" + option, plan[key] if key.startswith("core_") else base[key]]
            foot = base["foot_courses"][course.course_id]
            command += ["--foot-reference", foot["root"], "--foot-reference-hash", foot["hash"]]
            subprocess.run(command, cwd=motor, check=True)
            report = json.loads((output / "complete.json").read_text())
            if not report["fresh_reexecution_exact"] or len(report["records"]) != 4:
                raise ValueError("all physical source/re-execution pairs required")
            check_pins(report["input_file_hashes"])
            measures = {}
            for row in report["records"]:
                path = output / row["artifact"]["path"]
                if (
                    path.parent.resolve() != output.resolve()
                    or path.is_symlink()
                    or digest(path) != row["artifact"]["hash"]
                ):
                    raise ValueError("physical authority arrays changed")
                if not all(row["teacher_target_authority"].values()):
                    raise ValueError("actual applied target authority failed")
                if row["arm"] == "role-foot-clearance" and not all(
                    row["teacher_parent_identity"].values()
                ):
                    raise ValueError("original qualified parent changed")
                if row["repeat"] == 0:
                    with np.load(path) as data:
                        measures[row["arm"]] = contact_authority_metrics(dict(data))
                    measures[row["arm"]]["all_bodies_healthy"] = row["healthy_full_horizon"]
            result = dict(
                course=course.course_id,
                variant=a.variant,
                source_hash=digest(output / "complete.json"),
                measures=measures,
                native_steps=report["native_steps"],
            )
            write_once(a.output / ("stage-" + course.course_id + ".json"), result)
            results.append(result)
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="NATIVE8_SINGLE_FINISHER_CONTACT_AUTHORITY_MEASURED_NOT_LEARNED",
                source_head=a.expected_head,
                plan_hash=a.plan_hash,
                variant=a.variant,
                input_file_hashes=pins,
                results=results,
                native_steps=sum(r["native_steps"] for r in results),
                optimizer_updates=0,
                private_fresh_accessed=False,
                promotion_authorized=False,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(error=repr(error), charged_outputs_retained=True, blind_retry=False),
        )
        raise


if __name__ == "__main__":
    main()
