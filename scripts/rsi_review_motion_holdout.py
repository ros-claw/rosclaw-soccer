"""Authenticate all frozen paired near-holdout receipts; no NN or physics step."""

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from rosclaw_soccer.rsi.paired_motion_summary import summarize_pairs
from rosclaw_soccer.training.motion_portfolio_holdout import motion_portfolio_holdout_courses
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import check_pins, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--launch", type=Path, required=True)
    p.add_argument("--expected-launch-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != a.expected_head or a.output.exists() or digest(a.launch) != a.expected_launch_hash:
        raise ValueError("frozen review source, launch pin and unused output required")
    launch = json.loads(a.launch.read_text())
    pins = {str(a.launch): a.expected_launch_hash}
    for folder in ("src", "scripts"):
        pins.update({str(f.resolve()): digest(f) for f in Path(folder).rglob("*.py")})
    if digest(a.plan / "complete.json") != launch["plan_complete_hash"]:
        raise ValueError("sealed plan differs")
    plan = json.loads((a.plan / "complete.json").read_text())
    if digest(a.plan / "courses.json") != plan["courses_hash"]:
        raise ValueError("sealed course bank differs")
    courses = [asdict(c) for c in motion_portfolio_holdout_courses()]
    # JSON turns the course tuple into a list.
    courses = json.loads(json.dumps(courses))
    if json.loads((a.plan / "courses.json").read_text())["courses"] != courses:
        raise ValueError("whole192 generated declaration differs")
    pins.update({str(f): digest(f) for f in a.plan.iterdir() if f.is_file()})
    rows, seen, terminals = [], set(), {}
    for job in launch["jobs"]:
        unit = job["unit"] + ".service"
        terminal = dict(
            line.split("=", 1)
            for line in subprocess.check_output(
                [
                    "systemctl",
                    "--user",
                    "show",
                    unit,
                    "-p",
                    "MainPID",
                    "-p",
                    "Result",
                    "-p",
                    "ExecMainStatus",
                    "-p",
                    "SubState",
                ],
                text=True,
            ).splitlines()
        )
        if terminal != dict(MainPID="0", Result="success", ExecMainStatus="0", SubState="exited"):
            raise ValueError("worker not closed successfully: " + unit)
        terminals[unit] = terminal
        root = Path(job["output"])
        declaration = json.loads((root / "declaration.json").read_text())
        complete = json.loads((root / "complete.json").read_text())
        if (
            declaration["source_head"] != launch["source_head"]
            or declaration["plan_hash"] != launch["plan_complete_hash"]
            or declaration["indices"] != job["indices"]
            or complete["source_head"] != launch["source_head"]
            or complete["status"] != "NEAR_HOLDOUT_PAIRED_NATIVE_VERIFIED"
            or complete["optimizer_steps"] != 0
            or complete["new_native_steps"] != 12000 * len(job["indices"])
            or len(complete["results"]) != len(job["indices"])
        ):
            raise ValueError("closed worker declaration/accounting differs")
        check_pins(declaration["input_file_hashes"])
        pins.update(declaration["input_file_hashes"])
        for name in ("declaration.json", "complete.json"):
            pins[str(root / name)] = digest(root / name)
        for index, row in zip(job["indices"], complete["results"], strict=True):
            folder = root / f"course{index:03d}"
            if index in seen or row["course"] != courses[index] or row["new_native_steps"] != 12000:
                raise ValueError("duplicate, replaced or truncated course")
            seen.add(index)
            for path in (folder / "complete.json", root / f"progress{index:03d}.json"):
                if json.loads(path.read_text()) != row:
                    raise ValueError("course progress/terminal receipt mismatch")
                pins[str(path)] = digest(path)
            for variant in row["variants"]:
                part = folder / variant["variant"]
                receipt = part / "complete.json"
                if json.loads(receipt.read_text()) != variant:
                    raise ValueError("variant receipt mismatch")
                pins[str(receipt)] = digest(receipt)
                for name, key in (("native.npz", "native_hash"), ("sampling.npz", "sampling_hash")):
                    if digest(part / name) != variant[key]:
                        raise ValueError("committed raw trajectory differs")
                    pins[str(part / name)] = variant[key]
                replay = variant["replay"]
                if (
                    variant["native_steps"] != 6000
                    or replay["additional_native_steps"] != 3000
                    or replay["student_neural_and_muscle_native_bit_exact"] is not True
                    or replay["teacher_encoder_calls"] != 0
                    or replay["teacher_planner_calls"] != 0
                    or replay["muscle_decoder_calls"] != 280
                    or replay["intention_network_calls"]
                    != (300 if variant["variant"] == "baseline" else 380)
                    or replay["hardware_authorized"] is not False
                    or replay["promotion_authorized"] is not False
                ):
                    raise ValueError("independent neural/muscle/native proof differs")
            rows.append(row)
    if seen != set(range(192)) or len(rows) != 192:
        raise ValueError("exact whole192 paired coverage required")
    result = summarize_pairs(rows)
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            status="NEAR_HOLDOUT192_RECEIPTS_AUTHENTICATED",
            source_head=head,
            launch_hash=a.expected_launch_hash,
            input_file_hashes=pins,
            producer_terminals=terminals,
            summary=result,
            reviewed_native_steps=2304000,
            new_native_steps=0,
            optimizer_steps=0,
            neural_inference_calls=0,
            independent_replay_reexecuted_by_this_review=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
