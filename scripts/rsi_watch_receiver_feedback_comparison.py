"""Wait for both independently replayed SIM reviews, then compare all seeds."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dense-campaign", "motion-campaign", "dense-review", "motion-review", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("dense-unit", "motion-unit", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--timeout-sec", type=int, default=14400)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        a.output.exists()
        or not 1 <= a.timeout_sec <= 14400
        or a.dense_unit != "rosclaw-receiver-dense-campaign-review-20261028.service"
        or a.motion_unit != "rosclaw-receiver-motion-campaign-review-20261028.service"
    ):
        raise ValueError("exact bounded review units and unused output required")
    deadline = time.monotonic() + a.timeout_sec
    for unit, receipt in ((a.dense_unit, a.dense_review), (a.motion_unit, a.motion_review)):
        while True:
            q = subprocess.run(
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
                    "-p",
                    "ExecMainStatus",
                ],
                text=True,
                capture_output=True,
                check=True,
            )
            state = dict(line.split("=", 1) for line in q.stdout.splitlines() if "=" in line)
            if state.get("ActiveState") in ("active", "activating"):
                if time.monotonic() >= deadline:
                    raise TimeoutError("reviews still active; do not restart")
                time.sleep(15)
                continue
            if (
                state.get("LoadState") not in ("loaded", "not-found")
                or state.get("ActiveState") != "inactive"
                or state.get("Result") != "success"
                or state.get("ExecMainStatus") != "0"
                or not receipt.is_file()
            ):
                raise ValueError("review did not finish with evidence; preserve failures")
            break
    command = [
        sys.executable,
        "-m",
        "scripts.rsi_compare_receiver_feedback_campaigns",
        "--expected-head",
        a.expected_head,
        "--output",
        str(a.output),
    ]
    for name in ("dense_campaign", "motion_campaign", "dense_review", "motion_review"):
        path = getattr(a, name)
        command.extend(
            (
                "--" + name.replace("_", "-"),
                str(path),
                "--" + name.replace("_", "-") + "-hash",
                digest(path),
            )
        )
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
