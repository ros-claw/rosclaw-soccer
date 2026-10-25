"""Declared bounded successor: authenticate closed producer cohorts, then audit.

After= orders startup, NOT completion of Type=simple physics producers. Read
their actual state to wait for completion; never start/restart physical units.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from rosclaw_soccer.providers.g1.velocity_policy import file_hash


def producer_finished(state):
    """Return false for a live/queued job even if an output marker exists."""
    if state.get("ActiveState") == "failed" or state.get("Result") not in (None, "success"):
        raise ValueError("failed producer requires review, not automatic physics retry")
    if state.get("MainPID") not in (None, "0") or (
        state.get("Job", "").isdigit() and int(state["Job"]) > 0
    ):
        return False
    if state.get("ActiveState") in ("active", "activating", "deactivating"):
        return False
    if state.get("ActiveState") != "inactive":
        raise ValueError("unrecognized producer observation; cannot infer completion")
    return True  # May be garbage-collected; exact completed artifacts are checked next.


def wait_for_producers(units, *, timeout_seconds=1200):
    """Bounded read-only completion wait, not an agent wakeup or science loop."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        finished = []
        for unit in units:
            try:
                result = subprocess.run(
                    [
                        "systemctl",
                        "--user",
                        "show",
                        unit,
                        "-p",
                        "ActiveState",
                        "-p",
                        "MainPID",
                        "-p",
                        "Job",
                        "-p",
                        "Result",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                finished.append(False)  # Observation timeout is not physical termination.
                continue
            state = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
            if not state:
                finished.append(False)
                continue
            finished.append(producer_finished(state))
        if all(finished):
            return
        time.sleep(5)
    raise TimeoutError("producer completion not observed; preserve physics and request review")


def closed_shard_arguments(roots, producer_head):
    if len(roots) != 4 or len({p.resolve() for p in roots}) != 4:
        raise ValueError("four unique declared roots required")
    args = []
    fast_mode = None
    expanded_mode = None
    for i, root in enumerate(roots):
        path = root / "complete.json"
        if root.is_symlink() or path.is_symlink() or not path.is_file():
            raise ValueError("missing closed shard; no physics restart authorized")
        report = json.loads(path.read_text())
        fast = report.get("experimental_fast_approach", False)
        expanded = report.get("expanded_public_bank", False)
        count = 40 if expanded else 16
        if (
            report["source_head"] != producer_head
            or report["status"]
            != (
                "PUBLIC40_TEACHER_CEM_SHARD_MEASURED_NOT_TRAINED"
                if expanded
                else "PUBLIC16_TEACHER_CEM_SHARD_MEASURED_NOT_TRAINED"
            )
            or report["worlds"] != (count // 4) * 19
            or report["promotion_authorized"] is not False
            or [r["course_index"] for r in report["results"]] != list(range(i, count, 4))
            or type(expanded) is not bool
            or (expanded_mode is not None and expanded is not expanded_mode)
            or (
                expanded and (not fast or report["baseline_uses_original_strict_codec"] is not True)
            )
            or type(fast) is not bool
            or (fast_mode is not None and fast != fast_mode)
        ):
            raise ValueError("entire immutable producer cohort required before audit")
        fast_mode = fast
        expanded_mode = expanded
        args.extend(["--shard", str(root), file_hash(path)])
    return args


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard-root", type=Path, nargs=4, required=True)
    p.add_argument("--expected-producer-head", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--producer-unit", nargs=4)
    a = p.parse_args()
    from scripts.rsi_native_fresh_common import clean

    clean(a.expected_head)
    if a.producer_unit is not None:
        wait_for_producers(a.producer_unit)
    args = closed_shard_arguments(a.shard_root, a.expected_producer_head)
    subprocess.run(
        [
            sys.executable,
            "scripts/rsi_audit_walk_kick_teacher.py",
            *args,
            "--expected-head",
            a.expected_head,
            "--output",
            str(a.output),
        ],
        check=True,
    )
    clean(a.expected_head)


if __name__ == "__main__":
    main()
