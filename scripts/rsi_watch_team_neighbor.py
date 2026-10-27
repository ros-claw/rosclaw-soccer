"""Run preregistered SIM neighbor development only after an admitted gate.

This bounded deterministic watcher cannot train, promote, authorize hardware,
recover a failed unit, or independently decide a new experiment.
"""

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
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--timeout-sec", type=int, default=43200)
    args, runner_args = parser.parse_known_args()
    if runner_args and runner_args[0] == "--":
        runner_args = runner_args[1:]
    clean(args.expected_head)
    if (
        not 1 <= args.timeout_sec <= 43200
        or not args.unit.startswith("rosclaw-team-")
        or not args.unit.endswith(".service")
        or "/" in args.unit
    ):
        raise ValueError("bounded SIM-only unit and timeout required")
    if (
        "--admission-hash" in runner_args
        or "--expected-head" in runner_args
        or runner_args.count("--admission-proof") != 1
        or runner_args.count("--neighbor-bank") != 1
    ):
        raise ValueError("exactly one preregistered neighbor and admission proof required")
    proof_index = runner_args.index("--admission-proof")
    if proof_index + 1 >= len(runner_args) or Path(runner_args[proof_index + 1]) != args.assessment:
        raise ValueError("runner admission must bind watched assessment")
    deadline = time.monotonic() + args.timeout_sec
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
        if (
            args.assessment.is_file()
            and not args.assessment.is_symlink()
            and state.get("ActiveState") != "active"
        ):
            if (
                state.get("Result", "success") != "success"
                or state.get("ExecMainStatus", "0") != "0"
            ):
                raise ValueError("assessment failed; neighbor development blocked")
            proof_hash = digest(args.assessment)
            proof = json.loads(args.assessment.read_text())
            if (
                proof.get("status") != "TEAM_COVERAGE_RETENTION_ASSESSED_NOT_PROMOTED"
                or proof.get("hardware_authorized") is not False
            ):
                raise ValueError("unpromoted assessment required")
            if proof.get("result", {}).get("development_generalization_admitted") is not True:
                print("Retention gate rejected; no neighbor simulations started.", flush=True)
                return
            break
        if state.get("ActiveState") in ("failed", "inactive") and not args.assessment.exists():
            raise ValueError("assessment exited without completion; no retry or promotion")
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded watcher expired without complete assessment")
        time.sleep(15)
    subprocess.run(
        [
            sys.executable,
            "scripts/rsi_team_selector_smoke.py",
            *runner_args,
            "--admission-hash",
            proof_hash,
            "--expected-head",
            args.expected_head,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
