"""Observe two named physical-review jobs before comparing action memory."""

import argparse
import json
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
    for key in ("old-unit", "new-unit", "old-review-head", "new-review-head", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--timeout-sec", type=int, default=7200)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.old_unit == a.new_unit
        or not 1 <= a.timeout_sec <= 7200
        or any(
            not u.startswith("rosclaw-receiver-leg-limit-")
            or not u.endswith(".service")
            or "/" in u
            for u in (a.old_unit, a.new_unit)
        )
    ):
        raise ValueError("two exact bounded physical review jobs required")
    deadline = time.monotonic() + a.timeout_sec
    while True:
        done = []
        for unit, path in ((a.old_unit, a.old), (a.new_unit, a.new)):
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
            if state.get("LoadState") == "loaded" and state.get("ActiveState") in (
                "active",
                "activating",
            ):
                done.append(False)
            elif state.get("ActiveState") == "inactive" and (
                state.get("LoadState") == "not-found"
                or (
                    state.get("LoadState") == "loaded"
                    and state.get("Result") == "success"
                    and state.get("ExecMainStatus") == "0"
                )
            ):
                if not path.is_file() or path.is_symlink():
                    raise ValueError("completed physical review receipt required")
                done.append(True)
            else:
                raise ValueError("physical review failed or its state is unverifiable")
        if all(done):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("review observation expired; do not restart workers")
        time.sleep(15)
    for path, head in ((a.old, a.old_review_head), (a.new, a.new_review_head)):
        if json.loads(path.read_text()).get("source_head") != head:
            raise ValueError("exact frozen review producer required")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_compare_leg_projection_memory",
            "--old",
            str(a.old),
            "--old-hash",
            digest(a.old),
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
