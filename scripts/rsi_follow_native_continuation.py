"""Finite SIM chain: await exact learner, examine fixed132, audit both arms.

Never restarts training, chooses weights, promotes policies or wakes an Agent.
Missing/failed workers stop the chain. The plan commits all inputs before wait.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--plan-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or digest(a.plan) != a.plan_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused external output, exact plan and frozen source required")
    plan = json.loads(a.plan.read_text())
    if (
        plan.get("schema") != "soccer.rsi.continuation132_followup_plan.v1"
        or plan.get("promotion_authorized") is not False
        or set(plan["tasks"]) != {"train64", "dev64"}
        or not re.fullmatch(r"[A-Za-z0-9_.@-]+\.service", plan["worker_unit"])
    ):
        raise ValueError("bounded simulation-only fixed132 follow-up plan required")
    pins = {str(a.plan): a.plan_hash}
    for task, inputs in plan["tasks"].items():
        for key in ("starts", "baseline") if task == "train64" else ("starts",):
            path = Path(inputs[key + "_root"]) / "complete.json"
            if digest(path) != inputs[key + "_hash"]:
                raise ValueError("exact task prerequisites required")
            pins[str(path)] = digest(path)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            maximum_wait_s=10800,
            fixed_checkpoint=132,
            automatic_promotion=False,
            hardware_authorized=False,
        ),
    )
    try:
        learner_root = Path(plan["learner_root"])
        deadline = time.monotonic() + 10800
        while True:
            unit = dict(
                line.split("=", 1)
                for line in subprocess.check_output(
                    [
                        "systemctl",
                        "--user",
                        "show",
                        plan["worker_unit"],
                        "-p",
                        "MainPID",
                        "-p",
                        "SubState",
                        "-p",
                        "ActiveState",
                        "-p",
                        "ExecMainStatus",
                        "-p",
                        "LoadState",
                    ],
                    text=True,
                ).splitlines()
            )
            if (
                unit["LoadState"] != "loaded"
                or unit["ActiveState"] == "failed"
                or (unit["SubState"] in ("exited", "dead") and unit["ExecMainStatus"] != "0")
            ):
                raise ValueError(
                    "declared learner missing/failed; no automatic restart: " + str(unit)
                )
            if unit["SubState"] in ("exited", "dead") and unit["MainPID"] == "0":
                if not (learner_root / "complete.json").is_file():
                    raise ValueError("terminal worker without complete evidence")
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("bounded follow-up wait expired; learner not restarted")
            print(json.dumps(dict(status="WAIT_DECLARED_LIVE_LEARNER", unit=unit)), flush=True)
            time.sleep(20)
        receipt = learner_root / "complete.json"
        r = json.loads(receipt.read_text())
        declaration = json.loads((learner_root / "declaration.json").read_text())
        if (
            r["status"] != "NATIVE_OPTION_CONTINUATION132_REQUIRES_FULL_EXAM"
            or r["end_episode"] != 132
            or declaration["source_head"] != plan["learner_source_head"]
            or r["resume_checkpoint_hash"] != plan["resume_checkpoint_hash"]
            or r["resume_source_hash"] != plan["resume_source_hash"]
            or r["resume_proof_hash"] != plan["resume_proof_hash"]
            or [x["episode"] for x in r["results"]] != list(range(20, 132))
        ):
            raise ValueError("learner is not the fixed declared continuation")
        pins.update(r["input_file_hashes"])
        learner_hash = digest(receipt)
        pins[str(receipt)] = learner_hash
        pins[str(learner_root / "declaration.json")] = digest(learner_root / "declaration.json")
        check_pins(pins)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            environment[name] = "1"

        def examine(task):
            inputs = plan["tasks"][task]
            output = a.output / task
            command = [
                sys.executable,
                "-m",
                "scripts.rsi_exam_native_contact_policy",
                "--task-set",
                task,
                "--checkpoint-intent",
                "bounded-recovery-continuation",
                "--learner-root",
                str(learner_root),
                "--learner-hash",
                learner_hash,
                "--starts-root",
                inputs["starts_root"],
                "--starts-hash",
                inputs["starts_hash"],
                "--expected-head",
                a.expected_head,
                "--output",
                str(output),
            ]
            if task == "train64":
                command += [
                    "--baseline-episodes-root",
                    inputs["baseline_root"],
                    "--baseline-episodes-hash",
                    inputs["baseline_hash"],
                ]
            with (a.output / (task + ".log")).open("x") as log:
                subprocess.run(
                    command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            exam_hash = digest(output / "complete.json")
            audit = a.output / (task + "-audit.json")
            with (a.output / (task + "-audit.log")).open("x") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "scripts.rsi_audit_finite_option_exam",
                        "--exam-root",
                        str(output),
                        "--exam-hash",
                        exam_hash,
                        "--output",
                        str(audit),
                    ],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            return dict(
                task=task,
                exam_hash=exam_hash,
                audit_hash=digest(audit),
                summary=json.loads((output / "complete.json").read_text())["summary"],
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(examine, ("train64", "dev64")))
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="CONTINUATION132_EXAMS_AUDITED_NOT_PROMOTED",
                input_file_hashes=pins,
                learner_hash=learner_hash,
                results=results,
                observer_is_not_agent_wakeup=True,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
        print(json.dumps(dict(status="FIXED_EXAM_CHAIN_CLOSED", results=results)), flush=True)
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                training_restart_attempted=False,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
