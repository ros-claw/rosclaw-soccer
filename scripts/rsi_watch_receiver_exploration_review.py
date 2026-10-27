"""Wait for complete stochastic physics, own replay and density audits."""

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
        metavar=("ROOT", "UNIT", "IDENTITY", "IDENTITY_AUDIT"),
    )
    p.add_argument("--identity-review", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    for name in (
        "identity-review-hash",
        "producer-head",
        "identity-producer-head",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--timeout-sec", type=int, default=43200)
    a = p.parse_args()
    clean(a.expected_head)
    units = [g[1] for g in a.group]
    if (
        len(a.group) != 3
        or len(set(units)) != 3
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or not 1 <= a.timeout_sec <= 43200
        or any(
            not u.startswith("rosclaw-team-") or not u.endswith(".service") or "/" in u
            for u in units
        )
    ):
        raise ValueError("three bounded SIM workers and unused output required")
    deadline = time.monotonic() + a.timeout_sec
    while True:
        pending = False
        for _, unit, _, _ in a.group:
            q = subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "show",
                    unit,
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
            if q.returncode or not {"ActiveState", "Result", "ExecMainStatus"} <= state.keys():
                raise ValueError("cannot verify physical and density worker")
            if state["ActiveState"] == "active":
                pending = True
            elif state["Result"] != "success" or state["ExecMainStatus"] != "0":
                raise ValueError("source, own replay or sampling-law audit failed; no review")
        if not pending:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded exploration wait expired; no retry")
        time.sleep(15)
    command = [
        sys.executable,
        "scripts/rsi_review_receiver_exploration.py",
        "--identity-review",
        str(a.identity_review),
        "--identity-review-hash",
        a.identity_review_hash,
        "--producer-head",
        a.producer_head,
        "--identity-producer-head",
        a.identity_producer_head,
        "--output",
        str(a.output),
        "--expected-head",
        a.expected_head,
    ]
    for root, _, prior, prior_audit in a.group:
        command.extend(
            (
                "--group",
                str(Path(root) / "source"),
                str(Path(root) / "audit" / "complete.json"),
                prior,
                prior_audit,
            )
        )
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
