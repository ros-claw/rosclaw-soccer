"""Wait for two exact SIM pilots and independently compare all first-block cases."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_native_fresh_common import clean
from scripts.rsi_watch_passer_residual_pair import worker_ready


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dense", "motion", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("dense-head", "motion-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--timeout-sec", type=int, default=7200)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or not 1 <= a.timeout_sec <= 7200:
        raise ValueError("unused output and bounded wait required")
    deadline = time.monotonic() + a.timeout_sec
    for root, unit in (
        (a.dense, "rosclaw-receiver-dense-pair-s1-b0-20261028.service"),
        (a.motion, "rosclaw-receiver-motion-pair-s1-b0-20261028.service"),
    ):
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
            if worker_ready(root, state):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("pilot still active; do not restart")
            time.sleep(15)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_compare_receiver_feedback_pilots",
            "--dense",
            str(a.dense),
            "--motion",
            str(a.motion),
            "--dense-head",
            a.dense_head,
            "--motion-head",
            a.motion_head,
            "--expected-head",
            a.expected_head,
            "--output",
            str(a.output),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
