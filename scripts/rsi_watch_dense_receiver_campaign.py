"""Review the whole fixed SIM campaign only after its terminal receipt."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--unit", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timeout-sec", type=int, default=14400)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        not a.unit.startswith(
            ("rosclaw-receiver-dense-campaign-", "rosclaw-receiver-motion-campaign-")
        )
        or not a.unit.endswith(".service")
        or "/" in a.unit
        or a.output.exists()
        or not 1 <= a.timeout_sec <= 14400
    ):
        raise ValueError("exact bounded campaign and unused output required")
    deadline = time.monotonic() + a.timeout_sec
    while True:
        q = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                a.unit,
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
                raise TimeoutError("campaign still running; do not restart it")
            time.sleep(15)
            continue
        receipt = a.campaign / "complete.json"
        if (
            state.get("LoadState") not in ("loaded", "not-found")
            or state.get("ActiveState") != "inactive"
            or state.get("Result") != "success"
            or state.get("ExecMainStatus") != "0"
            or not receipt.is_file()
            or (a.campaign / "failed.json").exists()
        ):
            raise ValueError("campaign did not complete; preserve failures")
        break
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_review_dense_receiver_campaign",
            "--campaign",
            str(a.campaign),
            "--campaign-hash",
            digest(receipt),
            "--expected-head",
            a.expected_head,
            "--output",
            str(a.output),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
