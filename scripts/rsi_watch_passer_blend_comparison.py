"""Compare the exact paired reviews after the repaired independent replay finishes."""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("old", "new", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("unit", "old-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--timeout-sec", type=int, default=7200)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        re.fullmatch(r"rosclaw-passer-blend-replay-repair-b0-[0-9]{8}\.service", a.unit) is None
        or not 1 <= a.timeout_sec <= 7200
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.old.is_symlink()
        or digest(a.old) != a.old_hash
    ):
        raise ValueError("exact old review, bounded worker, and unused external output required")
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
            capture_output=True,
            text=True,
            check=True,
        )
        state = dict(line.split("=", 1) for line in q.stdout.splitlines() if "=" in line)
        if state.get("LoadState") != "loaded":
            raise ValueError("cannot inspect exact replay worker")
        if state.get("ActiveState") not in ("active", "activating"):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("replay observation expired; do not restart")
        time.sleep(15)
    if (
        state.get("ActiveState") != "inactive"
        or state.get("Result") != "success"
        or state.get("ExecMainStatus") != "0"
        or not a.new.is_file()
        or a.new.is_symlink()
        or digest(a.old) != a.old_hash
    ):
        raise ValueError("successful exact independent replay and complete reviews required")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_compare_passer_recovery_reviews",
            "--old",
            str(a.old),
            "--old-hash",
            a.old_hash,
            "--new",
            str(a.new),
            "--new-hash",
            digest(a.new),
            "--expected-head",
            a.expected_head,
            "--output",
            str(a.output),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
