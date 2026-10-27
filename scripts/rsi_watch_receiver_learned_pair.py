"""Wait for all fixed paired workers and launch review only after own replay."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--group",
        nargs=4,
        action="append",
        required=True,
        metavar=("ROOT", "UNIT", "OLD", "OLD_AUDIT"),
    )
    for key in ("source-head", "old-head", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timeout-sec", type=int, default=43200)
    a = p.parse_args()
    clean(a.expected_head)
    units = [group[1] for group in a.group]
    if (
        len(units) != 3
        or len(set(units)) != 3
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or not 1 <= a.timeout_sec <= 43200
        or any(
            not u.startswith("rosclaw-receiver-learned-pair-recovery-")
            or not u.endswith(".service")
            or "/" in u
            for u in units
        )
    ):
        raise ValueError("three specific bounded workers and unused output required")
    deadline = time.monotonic() + a.timeout_sec
    while True:
        pending = False
        for unit in units:
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
                check=False,
            )
            state = dict(line.split("=", 1) for line in q.stdout.splitlines() if "=" in line)
            if (
                q.returncode
                or state.get("LoadState") != "loaded"
                or not {"ActiveState", "Result", "ExecMainStatus"} <= state.keys()
            ):
                raise ValueError("cannot verify exact physical worker state")
            if state["ActiveState"] in ("active", "activating"):
                pending = True
            elif (
                state["ActiveState"] not in ("inactive",)
                or state["Result"] != "success"
                or state["ExecMainStatus"] != "0"
            ):
                raise ValueError("paired physics or independent replay failed; no review")
        if not pending:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded observation expired; no retry or worker restart")
        time.sleep(15)
    command = [
        sys.executable,
        "-m",
        "scripts.rsi_review_receiver_learned_pair",
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
                str(Path(root) / "audit" / "complete.json"),
                old,
                old_audit,
            )
        )
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
