"""Wait for three actual precontact physics workers and review completed replays."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_native_fresh_common import clean
from scripts.rsi_watch_passer_residual_pair import worker_ready


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--group",
        nargs=4,
        action="append",
        required=True,
        metavar=("ROOT", "UNIT", "OLD", "OLD_AUDIT"),
    )
    for name in ("source-head", "old-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timeout-sec", type=int, default=43200)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        len(a.group) != 3
        or len({g[1] for g in a.group}) != 3
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or not 1 <= a.timeout_sec <= 43200
        or any(
            not g[1].startswith("rosclaw-precontact-fullbody-")
            or not g[1].endswith(".service")
            or "/" in g[1]
            for g in a.group
        )
    ):
        raise ValueError("three explicit bounded physical workers required")
    deadline = time.monotonic() + a.timeout_sec
    while True:
        ready = []
        for root, unit, _, _ in a.group:
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
                capture_output=True,
                text=True,
                check=True,
            )
            state = dict(line.split("=", 1) for line in q.stdout.splitlines() if "=" in line)
            ready.append(worker_ready(Path(root), state))
        if all(ready):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("observation expired; do not restart live physical workers")
        time.sleep(15)
    command = [
        sys.executable,
        "-m",
        "scripts.rsi_review_precontact_fullbody",
        "--source-head",
        a.source_head,
        "--old-head",
        a.old_head,
        "--expected-head",
        a.expected_head,
        "--output",
        str(a.output),
    ]
    for root, _, old, old_audit in a.group:
        command.extend(
            (
                "--group",
                str(Path(root) / "source"),
                str(Path(root) / "audit/complete.json"),
                old,
                old_audit,
            )
        )
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
