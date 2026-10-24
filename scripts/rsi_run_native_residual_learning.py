"""One real bounded collection->TD3->all5 native physical exam generation.

Waits for its already launched collection; never restarts data generation or
failed/partial stages. Numerical improvement is not a physical promotion.
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
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean

    clean(a.expected_head)
    if a.plan.is_symlink() or digest(a.plan) != a.plan_hash or a.output.exists():
        raise ValueError("new learning output and exact declared plan required")
    plan = json.loads(a.plan.read_text())
    path = Path(plan["identity_plan"])
    if path.is_symlink() or digest(path) != plan["identity_plan_hash"]:
        raise ValueError("exact qualified physical parent plan required")
    base = json.loads(path.read_text())
    courses = native_team_course_bank()
    if (
        plan["schema"] != "soccer.native_role_residual_learning.v1"
        or plan["source_head"] != a.expected_head
        or preflight_receiving_courses(courses) != base["course_bank_hash"]
        or [c.course_id for c in courses] != base["courses"]
        or plan["hardware_authorized"] is not False
    ):
        raise ValueError("complete bounded five-course learning generation required")
    init = base["residual_initializations"][str(a.seed)]
    identity = plan["identities"][str(a.seed)]
    identity_family_root = Path(identity["root"])
    identity_path = identity_family_root / "complete.json"
    if identity_path.is_symlink() or digest(identity_path) != identity["hash"]:
        raise ValueError("exact completed parent identity family required")
    identity_family = json.loads(identity_path.read_text())
    a.output.mkdir(parents=True)
    with (a.output / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_once(
            a.output / "declaration.json",
            dict(
                schema=plan["schema"],
                source_head=a.expected_head,
                seed=a.seed,
                plan_hash=a.plan_hash,
                maximum_fits=1,
                maximum_exam_worlds=20,
                maximum_exam_native_steps=120000,
                blind_retry=False,
                collection_is_already_running_not_relaunched=True,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
        try:
            collection = Path(plan["collection_root_template"].format(seed=a.seed))
            unit = plan["collection_unit_template"].format(seed=a.seed)
            deadline = time.monotonic() + 3600
            while not (collection / "complete.json").exists():
                result = subprocess.run(
                    [
                        "systemctl",
                        "--user",
                        "show",
                        unit,
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
                    state.get("LoadState") != "loaded"
                    or state.get("ActiveState") not in ("active", "activating")
                    or state.get("Result") != "success"
                    or (collection / "failed.json").exists()
                    or time.monotonic() > deadline
                ):
                    raise ValueError(
                        "collection absent/failed/expired; retained, never auto-restarted"
                    )
                time.sleep(2)
            collected = json.loads((collection / "complete.json").read_text())
            if (
                collected["source_head"] != plan["collection_source_head"]
                or collected["plan_hash"] != plan["collection_plan_hash"]
                or collected["seed"] != a.seed
            ):
                raise ValueError("completed collection differs from learning commitment")
            collection_hash = digest(collection / "complete.json")
            fit = a.output / "fit"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.rsi_fit_native_role_residual_td3",
                    "--collection",
                    str(collection),
                    "--collection-hash",
                    collection_hash,
                    "--initialization",
                    init["root"],
                    "--initialization-hash",
                    init["hash"],
                    "--core-source-root",
                    plan["core_root"],
                    "--core-expected-head",
                    plan["core_head"],
                    "--expected-head",
                    a.expected_head,
                    "--seed",
                    str(a.seed),
                    "--output",
                    str(fit),
                ],
                check=True,
            )
            fitted = json.loads((fit / "complete.json").read_text())
            if (
                fitted["status"] != "NATIVE_ROLE_RESIDUAL_TD3_TRAINED_NOT_PHYSICALLY_EVALUATED"
                or fitted["collection_hash"] != collection_hash
            ):
                raise ValueError("fit differs from actual measured experience")
            write_once(
                a.output / "fit-bound.json",
                dict(
                    root=str(fit),
                    hash=digest(fit / "complete.json"),
                    numeric_report=fitted["numeric_report"],
                ),
            )
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
                    base["training_root_template"].format(seed=a.seed),
                    "--command-training-hash",
                    fitted["training_hash"],
                    "--command-residual",
                    str(fit / fitted["policy"]["path"]),
                    "--command-residual-hash",
                    fitted["policy"]["hash"],
                    "--expected-head",
                    base["source_head"],
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
                    command += [
                        "--" + option,
                        plan[key] if key in ("core_root", "core_head") else base[key],
                    ]
                foot = base["foot_courses"][course.course_id]
                command += ["--foot-reference", foot["root"], "--foot-reference-hash", foot["hash"]]
                subprocess.run(command, cwd=Path(plan["motor_source_root"]), check=True)
                report = json.loads((output / "complete.json").read_text())
                if not report["fresh_reexecution_exact"] or len(report["records"]) != 4:
                    raise ValueError("all paired source/fresh native executions required")
                check_pins(report["input_file_hashes"])
                for row in report["records"]:
                    path = output / row["artifact"]["path"]
                    if (
                        path.is_symlink()
                        or path.parent.resolve() != output.resolve()
                        or digest(path) != row["artifact"]["hash"]
                    ):
                        raise ValueError("physical examination arrays changed")
                    if row["arm"] == "role-foot-clearance" and not all(
                        row["preceding_clearance_identity"].values()
                    ):
                        raise ValueError("retained native teacher baseline changed")
                # Task outcome comparison uses the exact earlier zero-head
                # parent, not command-MSE or a survivor-only reward average.
                from rosclaw_soccer.rsi.native_role_residual_data import build_role_residual_episode

                identity_root = identity_family_root / course.course_id
                identity_case = next(
                    c for c in identity_family["results"] if c["course"] == course.course_id
                )
                if digest(identity_root / "complete.json") != identity_case["source_hash"]:
                    raise ValueError("parent task comparison report changed")
                identity_report = json.loads((identity_root / "complete.json").read_text())
                check_pins(identity_report["input_file_hashes"])
                for original in identity_report["records"]:
                    path = identity_root / original["artifact"]["path"]
                    if (
                        path.is_symlink()
                        or path.parent.resolve() != identity_root.resolve()
                        or digest(path) != original["artifact"]["hash"]
                    ):
                        raise ValueError("parent task comparison arrays changed")
                parent_row = next(
                    r
                    for r in identity_report["records"]
                    if r["arm"] == "role-command-residual" and r["repeat"] == 0
                )
                _, parent_metrics = build_role_residual_episode(
                    identity_root, parent_row, identity_report["input_file_hashes"]
                )
                row = next(
                    r
                    for r in report["records"]
                    if r["arm"] == "role-command-residual" and r["repeat"] == 0
                )
                _, candidate_metrics = build_role_residual_episode(
                    output, row, report["input_file_hashes"]
                )
                result = dict(
                    course=course.course_id,
                    source_hash=digest(output / "complete.json"),
                    parent_metrics=parent_metrics,
                    candidate_metrics=candidate_metrics,
                    candidate_healthy=row["healthy_full_horizon"],
                    native_steps=report["native_steps"],
                    candidate_residual_update_index=row["residual_update_index"],
                )
                write_once(a.output / f"stage-{course.course_id}.json", result)
                results.append(result)
            clean(a.expected_head)
            write_once(
                a.output / "complete.json",
                dict(
                    status="NATIVE_ROLE_TD3_ONE_GENERATION_PHYSICALLY_MEASURED_NOT_PROMOTED",
                    source_head=a.expected_head,
                    plan_hash=a.plan_hash,
                    seed=a.seed,
                    fit_hash=digest(fit / "complete.json"),
                    collection_hash=collection_hash,
                    numeric_report=fitted["numeric_report"],
                    results=results,
                    native_steps=sum(r["native_steps"] for r in results),
                    all5_cases_measured=True,
                    independent_Fresh_proven=False,
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
                    partial_outputs_retained=True,
                    blind_retry=False,
                    hardware_authorized=False,
                    goal_complete=False,
                ),
            )
            raise


if __name__ == "__main__":
    main()
