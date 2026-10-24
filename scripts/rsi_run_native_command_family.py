"""Execute all5 predeclared native command exams after one exact training job.

Real bounded succession, not an observer. Waits for the named existing job;
never launches duplicate training or restarts partial/failed physics.
"""

import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("plan-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, choices=(926061, 926062, 926063), required=True)
    a = p.parse_args()
    from rosclaw_soccer.training.native_team_courses import native_team_course_bank
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import digest
    from scripts.rsi_native_fresh_common import clean

    clean(a.expected_head)
    if a.plan.is_symlink() or digest(a.plan) != a.plan_hash or a.output.is_symlink():
        raise ValueError("ordinary exact family plan and local evidence directory required")
    plan = json.loads(a.plan.read_text())
    courses = native_team_course_bank()
    bank_hash = preflight_receiving_courses(courses)
    if (
        plan["schema"] != "soccer.native_command_family.v1"
        or plan["source_head"] != a.expected_head
        or plan["courses"] != [c.course_id for c in courses]
        or plan["course_bank_hash"] != bank_hash
        or plan["hardware_authorized"] is not False
        or plan["training_expected_updates"] not in (512, 1024)
    ):
        raise ValueError("complete fixed five-course SIM-only exam family required")
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_once(
            a.output / "declaration.json",
            dict(
                schema=plan["schema"],
                source_head=a.expected_head,
                seed=a.seed,
                plan_hash=a.plan_hash,
                course_bank_hash=bank_hash,
                optimizer_updates=0,
                maximum_physics_worker_executions=5,
                maximum_world_executions=20,
                maximum_native_steps=120000,
                blind_retry=False,
                observer_only=False,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
        try:
            training = Path(plan["training_root_template"].format(seed=a.seed))
            training_unit = plan["training_unit_template"].format(seed=a.seed)
            deadline = time.monotonic() + 900
            while not (training / "complete.json").exists():
                result = subprocess.run(
                    [
                        "systemctl",
                        "--user",
                        "show",
                        training_unit,
                        "-p",
                        "LoadState",
                        "-p",
                        "ActiveState",
                        "-p",
                        "Result",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10,
                )
                state = dict(
                    line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
                )
                if (
                    (training / "failed.json").exists()
                    or state.get("LoadState") != "loaded"
                    or state.get("ActiveState") not in ("active", "activating")
                    or state.get("Result") != "success"
                    or time.monotonic() >= deadline
                ):
                    raise ValueError("training absent, failed or wait expired; no automatic retry")
                time.sleep(2)
            training_hash = digest(training / "complete.json")
            fitted = json.loads((training / "complete.json").read_text())
            if (
                fitted["source_head"] != plan["training_source_head"]
                or fitted["plan_hash"] != plan["training_plan_hash"]
                or fitted["seed"] != a.seed
                or fitted["actual_BC_actor_optimizer_updates"] != 512
                or fitted["optimizer_updates"] != plan["training_expected_updates"]
                or fitted["fresh_physical_task_gain_not_claimed"] is not True
            ):
                raise ValueError("completed training differs from exact family commitment")
            write_once(
                a.output / "training-bound.json",
                dict(root=str(training), hash=training_hash, policy=fitted["policy"]),
            )
            residual = None
            if "residual_initializations" in plan:
                item = plan["residual_initializations"][str(a.seed)]
                root = Path(item["root"])
                path = root / "complete.json"
                if path.is_symlink() or digest(path) != item["hash"]:
                    raise ValueError("exact explicit residual initialization required")
                residual = json.loads(path.read_text())
                if (
                    residual["training_hash"] != training_hash
                    or residual["optimizer_updates"] != 0
                    or residual["seed"] != a.seed
                ):
                    raise ValueError("own encoder and zero-initialized head required")
                residual_path = root / residual["policy"]["path"]
                if (
                    residual_path.is_symlink()
                    or residual_path.parent.resolve() != root.resolve()
                    or digest(residual_path) != residual["policy"]["hash"]
                ):
                    raise ValueError("exact confined residual head required")
            results = []
            for course in courses:
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
                    "--command-training",
                    str(training),
                    "--command-training-hash",
                    training_hash,
                    "--expected-head",
                    a.expected_head,
                    "--output",
                    str(output),
                ]
                if residual is not None:
                    command += [
                        "--command-residual",
                        str(residual_path),
                        "--command-residual-hash",
                        residual["policy"]["hash"],
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
                    command += ["--" + option, plan[key]]
                reference = plan["foot_courses"][course.course_id]
                command += [
                    "--foot-reference",
                    reference["root"],
                    "--foot-reference-hash",
                    reference["hash"],
                ]
                if not output.exists():
                    subprocess.run(command, check=True)
                complete = output / "complete.json"
                if not complete.is_file() or (output / "failed.json").exists():
                    raise ValueError("partial physics retained; no blind restart")
                report = json.loads(complete.read_text())
                declaration = json.loads((output / "declaration.json").read_text())
                if (
                    report["source_head"] != a.expected_head
                    or len(report["records"]) != 4
                    or not report["fresh_reexecution_exact"]
                    or declaration["protocol"]
                    != (
                        "native8_parent_role_residual_contrast.v7"
                        if residual is not None
                        else "native8_learned_role_command_contrast.v6"
                    )
                    or declaration["imported_command_training_hash"] != training_hash
                    or declaration["ball_course"] != course.course_id
                ):
                    raise ValueError("completed exam differs from committed course/model")
                for row in report["records"]:
                    path = output / row["artifact"]["path"]
                    if (
                        path.parent.resolve() != output.resolve()
                        or path.is_symlink()
                        or digest(path) != row["artifact"]["hash"]
                    ):
                        raise ValueError("exam raw artifact changed")
                    if row["arm"] == "role-foot-clearance" and not all(
                        row["preceding_clearance_identity"].values()
                    ):
                        raise ValueError("baseline identity failed; successor prohibited")
                    if (
                        residual is not None
                        and row["arm"] == "role-command-residual"
                        and not all(row["zero_residual_parent_identity"].values())
                    ):
                        raise ValueError("zero residual identity failed; next course prohibited")
                result = dict(
                    course=course.course_id,
                    source_hash=digest(complete),
                    native_steps=report["native_steps"],
                    records=report["records"],
                )
                write_once(a.output / f"stage-{course.course_id}.json", result)
                results.append(result)
            clean(a.expected_head)
            if digest(a.plan) != a.plan_hash or digest(training / "complete.json") != training_hash:
                raise ValueError("family commitment changed during execution")
            write_once(
                a.output / "complete.json",
                dict(
                    status="NATIVE_COMMAND_FAMILY_PHYSICALLY_MEASURED_NOT_PROMOTED",
                    source_head=a.expected_head,
                    plan_hash=a.plan_hash,
                    seed=a.seed,
                    training_hash=training_hash,
                    results=results,
                    native_steps=sum(row["native_steps"] for row in results),
                    optimizer_updates=0,
                    imported_BC_updates=plan["training_expected_updates"],
                    hardware_authorized=False,
                    promotion_authorized=False,
                    goal_complete=False,
                ),
            )
        except BaseException as error:
            write_once(
                a.output / "failed.json",
                dict(
                    error=repr(error),
                    seed=a.seed,
                    hardware_authorized=False,
                    goal_complete=False,
                    blind_retry=False,
                ),
            )
            raise


if __name__ == "__main__":
    main()
