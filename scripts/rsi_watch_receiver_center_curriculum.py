"""Wait for three named center-course workers, then review their replays."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_native_fresh_common import clean
from scripts.rsi_watch_passer_residual_pair import worker_ready


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", nargs=2, action="append", required=True, metavar=("ROOT", "UNIT"))
    p.add_argument("--producer-head", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--timeout-sec", default=43200, type=int)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        len(a.group) != 3
        or len({u for _, u in a.group}) != 3
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or not 1 <= a.timeout_sec <= 43200
        or any(
            not u.startswith("rosclaw-receiver-center-") or not u.endswith(".service") or "/" in u
            for _, u in a.group
        )
    ):
        raise ValueError("three bounded center workers and unused external output required")
    deadline = time.monotonic() + a.timeout_sec
    while True:
        ready = []
        for root, unit in a.group:
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
                    "-p",
                    "ExecMainStatus",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            state = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
            ready.append(worker_ready(Path(root), state))
        if all(ready):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("observation timeout; do not restart live physics")
        time.sleep(15)
    command = [
        sys.executable,
        "-m",
        "scripts.rsi_review_receiver_center_curriculum",
        "--producer-head",
        a.producer_head,
        "--expected-head",
        a.expected_head,
        "--output",
        str(a.output),
    ]
    for root, _ in a.group:
        command.extend(
            ("--group", str(Path(root) / "source"), str(Path(root) / "audit/complete.json"))
        )
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
