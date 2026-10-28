"""Compare only after one explicitly named physical-review job completes."""

import argparse
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
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or not a.unit.startswith(
            (
                "rosclaw-receiver-foot-teacher-axis-review-",
                "rosclaw-receiver-foot-teacher-early-review-",
            )
        )
        or not a.unit.endswith(".service")
        or "/" in a.unit
        or not 1 <= a.timeout_sec <= 7200
        or a.old.is_symlink()
        or digest(a.old) != a.old_hash
    ):
        raise ValueError("exact bounded review unit and pinned old evidence required")
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
        if (
            state.get("ActiveState") in ("active", "activating")
            and state.get("LoadState") == "loaded"
        ):
            pass
        elif state.get("ActiveState") == "inactive" and (
            state.get("LoadState") == "not-found"
            or (
                state.get("LoadState") == "loaded"
                and state.get("Result") == "success"
                and state.get("ExecMainStatus") == "0"
            )
        ):
            if not a.new.is_file() or a.new.is_symlink():
                raise ValueError("completed own physical review receipt required")
            break
        else:
            raise ValueError("exact physical review failed or its state cannot be verified")
        if time.monotonic() >= deadline:
            raise TimeoutError("review observation expired; do not restart it")
        time.sleep(15)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_compare_foot_teacher_reviews",
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
