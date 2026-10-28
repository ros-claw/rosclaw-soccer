"""Wait for all real passer probes and independent replays, then review once."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_native_fresh_common import clean


def worker_ready(root: Path, state: dict[str, str]) -> bool:
    if state.get("LoadState") == "not-found":
        if (
            state.get("ActiveState") != "inactive"
            or not (root / "source/complete.json").is_file()
            or not (root / "audit/complete.json").is_file()
            or (root / "source/failed.json").exists()
            or (root / "audit/failed.json").exists()
        ):
            raise ValueError("collected unit requires both completed physical evidence receipts")
        # Receipt existence is not a pass: the reviewer rechecks every replay.
        return True
    if (
        state.get("LoadState") != "loaded"
        or not {"ActiveState", "Result", "ExecMainStatus"} <= state.keys()
    ):
        raise ValueError("cannot verify exact worker state")
    if state["ActiveState"] in ("active", "activating"):
        return False
    if (
        state["ActiveState"] != "inactive"
        or state["Result"] != "success"
        or state["ExecMainStatus"] != "0"
    ):
        raise ValueError("physical source or independent replay failed")
    return True


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
    p.add_argument("--passer-contact-exit", action="store_true")
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
            not u.startswith("rosclaw-passer-residual-pair-")
            or not u.endswith(".service")
            or "/" in u
            for u in units
        )
    ):
        raise ValueError("three explicit bounded worker identities required")
    deadline = time.monotonic() + a.timeout_sec
    while True:
        done = []
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
            if q.returncode:
                raise ValueError("cannot inspect declared physical worker")
            state = dict(line.split("=", 1) for line in q.stdout.splitlines() if "=" in line)
            done.append(worker_ready(Path(root), state))
        if all(done):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("observation expired; never restart live physical worker")
        time.sleep(15)
    command = [
        sys.executable,
        "-m",
        "scripts.rsi_review_passer_residual_pair",
        "--source-head",
        a.source_head,
        "--old-head",
        a.old_head,
        "--expected-head",
        a.expected_head,
        "--output",
        str(a.output),
    ]
    if a.passer_contact_exit:
        command.append("--passer-contact-exit")
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
