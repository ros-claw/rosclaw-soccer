"""Boundedly chain a completed native pass experiment to independent replay."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("unit", "source-head", "expected-head"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--timeout-sec", type=int, default=43200)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--priority", action="store_true")
    modes.add_argument("--bilateral", action="store_true")
    modes.add_argument("--long-horizon", action="store_true")
    modes.add_argument("--relay-development", action="store_true")
    args = parser.parse_args()
    clean(args.expected_head)
    if (
        not 1 <= args.timeout_sec <= 43200
        or not args.unit.startswith("rosclaw-team-")
        or not args.unit.endswith(".service")
        or "/" in args.unit
    ):
        raise ValueError("bounded SIM-only unit and timeout required")
    if args.output.exists() or args.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external physics replay output required")
    deadline = time.monotonic() + args.timeout_sec
    source = args.source / "complete.json"
    while True:
        query = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                args.unit,
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
            raise ValueError("cannot verify native experiment unit; replay blocked")
        if source.is_file() and not source.is_symlink() and state.get("ActiveState") != "active":
            if state["Result"] != "success" or state["ExecMainStatus"] != "0":
                raise ValueError("native experiment failed; replay blocked")
            source_hash = digest(source)
            report = json.loads(source.read_text())
            if (
                report.get("status")
                != (
                    "NEURAL_RELAY_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
                    if args.relay_development
                    else "CAPTURE_LONG_HORIZON_SWEEP_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
                    if args.long_horizon
                    else "CAPTURE_BILATERAL_SWEEP_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
                    if args.bilateral
                    else "CAPTURE_PRIORITY_SWEEP_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
                    if args.priority
                    else "CAPTURE_GEOMETRY_SWEEP_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
                )
                or report.get("source_head") != args.source_head
                or len(report.get("records", [])) != 32
                or report.get("promotion_authorized") is not False
            ):
                raise ValueError("complete unpromoted native pass sweep required")
            break
        if state.get("ActiveState") in ("failed", "inactive") and not source.exists():
            raise ValueError("native experiment exited without completion; no retry")
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded watcher expired without complete native experiment")
        time.sleep(15)
    subprocess.run(
        [
            sys.executable,
            "scripts/rsi_audit_neural_relay_chain.py"
            if args.relay_development
            else "scripts/rsi_audit_native_receiving_roles.py",
            "--source-root",
            str(args.source),
            "--source-head",
            args.source_head,
            "--source-hash",
            source_hash,
            "--output",
            str(args.output),
            "--expected-head",
            args.expected_head,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
