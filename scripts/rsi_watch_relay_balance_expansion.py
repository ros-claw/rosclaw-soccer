"""Run expanded physics only after the exact complete pilot review passes."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("frontier-root", "pilot-review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (
        "frontier-head",
        "frontier-hash",
        "pilot-unit",
        "pilot-review-head",
        "expected-head",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--new-context-block", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--balance-teacher-pair", nargs=4, action="append", required=True)
    parser.add_argument("--timeout-sec", type=int, default=43200)
    args = parser.parse_args()
    clean(args.expected_head)
    if (
        len(args.balance_teacher_pair) != 3
        or args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or not 1 <= args.timeout_sec <= 43200
        or not args.pilot_unit.startswith("rosclaw-team-")
        or not args.pilot_unit.endswith(".service")
        or "/" in args.pilot_unit
    ):
        raise ValueError("three teacher identities, bounded pilot and unused output required")
    deadline = time.monotonic() + args.timeout_sec
    while True:
        query = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                args.pilot_unit,
                "-p",
                "ActiveState",
                "-p",
                "Result",
                "-p",
                "ExecMainStatus",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        state = dict(line.split("=", 1) for line in query.stdout.splitlines() if "=" in line)
        if query.returncode or not {"ActiveState", "Result", "ExecMainStatus"} <= state.keys():
            raise ValueError("cannot verify paired pilot process")
        if state["ActiveState"] != "active":
            if (
                state["Result"] != "success"
                or state["ExecMainStatus"] != "0"
                or not args.pilot_review.is_file()
                or args.pilot_review.is_symlink()
            ):
                raise ValueError("failed or missing pilot, no physics")
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded pilot wait expired")
        time.sleep(15)
    pilot_hash = digest(args.pilot_review)
    pilot = json.loads(args.pilot_review.read_text())
    if (
        pilot.get("status") != "RELAY_BALANCE_PAIRED_DEVELOPMENT_NOT_PROMOTED"
        or pilot.get("source_head") != args.pilot_review_head
        or pilot.get("paired_physical_and_private_handoff_prefix_exact") is not True
        or pilot.get("original_baselines_raw_byte_identical") is not True
        or len(pilot.get("records", [])) != 72
        or not any(
            row.get("body_failure_rescues", 0) > 0 for row in pilot.get("summaries", {}).values()
        )
    ):
        raise ValueError("pilot has no admitted physical rescue; expanded physics blocked")
    args.output.mkdir(parents=True)
    source = args.output / "source"
    command = [
        sys.executable,
        "scripts/rsi_collect_neural_relay_chain.py",
        "--frontier-root",
        str(args.frontier_root),
        "--frontier-head",
        args.frontier_head,
        "--frontier-hash",
        args.frontier_hash,
        "--new-context-block",
        str(args.new_context_block),
        "--balance-expanded",
        "--balance-pilot-review",
        str(args.pilot_review),
        "--balance-pilot-review-hash",
        pilot_hash,
        "--balance-pilot-review-head",
        args.pilot_review_head,
        "--output",
        str(source),
        "--expected-head",
        args.expected_head,
    ]
    for pair in args.balance_teacher_pair:
        command.extend(("--balance-teacher-pair", *pair))
    subprocess.run(command, check=True)
    source_hash = digest(source / "complete.json")
    subprocess.run(
        [
            sys.executable,
            "scripts/rsi_audit_neural_relay_chain.py",
            "--source-root",
            str(source),
            "--source-head",
            args.expected_head,
            "--source-hash",
            source_hash,
            "--output",
            str(args.output / "audit"),
            "--expected-head",
            args.expected_head,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
