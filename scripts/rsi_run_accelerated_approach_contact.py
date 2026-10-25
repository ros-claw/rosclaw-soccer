"""Fixed paired eight-body B / B+C contact study, no training or promotion.

First course executes serially as a real admission. Remaining four courses
run with three bounded child processes. Failed physical tasks are retained;
integrity/runner failures stop this declaration without retries.
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("plan-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from rosclaw_soccer.rsi.native_contact_authority_metrics import contact_authority_metrics
    from rosclaw_soccer.training.native_team_courses import native_team_course_bank
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists() or a.plan.is_symlink() or digest(a.plan) != a.plan_hash:
        raise ValueError("new study output and exact full declared plan required")
    plan = json.loads(a.plan.read_text())
    base_path = Path(plan["parent_plan"])
    if base_path.is_symlink() or digest(base_path) != plan["parent_plan_hash"]:
        raise ValueError("exact qualified parent course plan required")
    base = json.loads(base_path.read_text())
    courses = native_team_course_bank()
    if (
        plan["schema"] != "soccer.accelerated_approach_contact_pilot.v1"
        or plan["source_head"] != a.expected_head
        or preflight_receiving_courses(courses) != base["course_bank_hash"]
        or [c.course_id for c in courses] != base["courses"]
        or plan["hardware_authorized"] is not False
        or set(plan["parent_courses"]) != set(base["courses"])
    ):
        raise ValueError("whole fixed five-course eight-body pilot required")
    pins = {str(a.plan): a.plan_hash, str(base_path): plan["parent_plan_hash"]}
    for course in courses:
        ref = plan["parent_courses"][course.course_id]
        path = Path(ref["root"]) / "complete.json"
        if path.is_symlink() or digest(path) != ref["hash"]:
            raise ValueError("every same-course parent must exist before worker allocation")
        pins[str(path)] = ref["hash"]
    approach = Path(plan["approach_config"])
    if approach.is_symlink() or digest(approach) != plan["approach_config_hash"]:
        raise ValueError("exact experimental locomotion evidence binding required")
    pins[str(approach)] = plan["approach_config_hash"]
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            schema=plan["schema"],
            source_head=a.expected_head,
            plan_hash=a.plan_hash,
            maximum_worlds=30,
            maximum_native_steps=180000,
            maximum_parallel_children=3,
            first_course_before_remaining_workers=True,
            all_legal_physical_failures_retained=True,
            optimizer_updates=0,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )
    import numpy as np

    def execute(course):
        ref = plan["parent_courses"][course.course_id]
        config = a.output / f"contact-{course.course_id}.json"
        write_once(
            config,
            dict(
                schema="soccer.native_foot_contact_teacher.v1",
                agent="red_finisher",
                parameters={"strength": 1.0, "velocity_match_gain": 1.0},
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
            str(config),
            "--foot-contact-config-hash",
            digest(config),
            "--experimental-approach-config",
            str(approach),
            "--experimental-approach-config-hash",
            plan["approach_config_hash"],
            "--expected-head",
            a.expected_head,
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
        subprocess.run(command, check=True)
        report = json.loads((output / "complete.json").read_text())
        if (
            report["status"] != "NATIVE8_ACCELERATED_APPROACH_CONTACT_MEASURED_NOT_LEARNED"
            or not report["fresh_reexecution_exact"]
            or len(report["records"]) != 6
        ):
            raise ValueError("all three arms with true source/fresh re-execution required")
        check_pins(report["input_file_hashes"])
        measures = {}
        for row in report["records"]:
            path = output / row["artifact"]["path"]
            if (
                path.parent.resolve() != output.resolve()
                or path.is_symlink()
                or digest(path) != row["artifact"]["hash"]
            ):
                raise ValueError("physical intervention arrays changed")
            if not all(row["teacher_target_authority"].values()) or not all(
                row["experimental_approach_authority"].values()
            ):
                raise ValueError("actual command or target authority failed")
            if row["arm"] == "role-foot-clearance" and not all(
                row["teacher_parent_identity"].values()
            ):
                raise ValueError("historically qualified parent physics changed")
            if row["repeat"] == 0:
                with np.load(path) as saved:
                    data = dict(saved)
                result = contact_authority_metrics(data)
                result.update(
                    all_bodies_healthy=row["healthy_full_horizon"],
                    selected_maximum_actual_speed_mps=row["players"][3]["actual_max_speed_mps"],
                    any_player_warmup_nonfoot_contact=bool(
                        (data["ball_forces"][:500, :, 2] > 1).any()
                    ),
                    approach_fallback_controls=int(data["approach_parent_fallback"][:, 3].sum()),
                    maximum_selected_command_norm=float(
                        np.linalg.norm(data["commands"][:, 3, :2], axis=1).max()
                    ),
                    maximum_selected_scale=float(data["approach_scale"][:, 3].max()),
                    terminal_ball_speed_mps=float(data["ball_speed_after_mps"][-1]),
                )
                measures[row["arm"]] = result
        result = dict(
            course=course.course_id,
            source_hash=digest(output / "complete.json"),
            measures=measures,
            native_steps=report["native_steps"],
            all_arms_retained=True,
            official_exam_pass_proven=False,
        )
        write_once(a.output / f"stage-{course.course_id}.json", result)
        return result

    results = []
    try:
        results.append(execute(courses[0]))
        with ThreadPoolExecutor(max_workers=3) as workers:
            results.extend(workers.map(execute, courses[1:]))
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="NATIVE8_B_AND_B_PLUS_C_CONTACT_PILOT_MEASURED_NOT_LEARNED",
                source_head=a.expected_head,
                plan_hash=a.plan_hash,
                input_file_hashes=pins,
                results=results,
                native_steps=sum(r["native_steps"] for r in results),
                optimizer_updates=0,
                physical_teacher_qualified=False,
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
