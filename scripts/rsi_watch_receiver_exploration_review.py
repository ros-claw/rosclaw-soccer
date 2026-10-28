"""Wait for complete stochastic physics, own replay and density audits."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_watch_passer_residual_pair import worker_ready


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
    p.add_argument("--receiver-center-curriculum", action="store_true")
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
            not u.startswith(
                "rosclaw-receiver-center-" if a.receiver_center_curriculum else "rosclaw-team-"
            )
            or not u.endswith(".service")
            or "/" in u
            for u in units
        )
    ):
        raise ValueError("three bounded SIM workers and unused output required")
    deadline = time.monotonic() + a.timeout_sec
    while True:
        pending = False
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
                check=False,
            )
            state = dict(line.split("=", 1) for line in q.stdout.splitlines() if "=" in line)
            if q.returncode or not {"ActiveState", "Result", "ExecMainStatus"} <= state.keys():
                raise ValueError("cannot verify physical and density worker")
            pending = not worker_ready(Path(root), state) or pending
        if not pending:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded exploration wait expired; no retry")
        time.sleep(15)
    identity_hash = a.identity_review_hash
    if identity_hash == "auto":
        if not a.receiver_center_curriculum or a.identity_review.is_symlink():
            raise ValueError("automatic identity pin only for the explicit center course")
        identity_hash = digest(a.identity_review)
    command = [
        sys.executable,
        "-m",
        "scripts.rsi_review_receiver_exploration",
        "--identity-review",
        str(a.identity_review),
        "--identity-review-hash",
        identity_hash,
        "--producer-head",
        a.producer_head,
        "--identity-producer-head",
        a.identity_producer_head,
        "--output",
        str(a.output),
        "--expected-head",
        a.expected_head,
    ]
    if a.receiver_center_curriculum:
        command.append("--receiver-center-curriculum")
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
