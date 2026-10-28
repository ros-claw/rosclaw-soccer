"""Compare the exact paired reviews after the repaired independent replay finishes."""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import report


def collected_replay_verified(state: dict[str, str], review_path: Path) -> bool:
    """Accept a collected transient unit only with its bound completed replay."""
    if state.get("LoadState") != "not-found" or state.get("ActiveState") != "inactive":
        return False
    if review_path.is_symlink() or not review_path.is_file():
        raise ValueError("collected worker requires its exact completed review")
    review = json.loads(review_path.read_text())
    root = review_path.parent.parent
    _, audit, source_hash = report(
        root / "source",
        root / "audit/complete.json",
        review.get("audit_hash"),
        "NEURAL_RELAY_PASSER_VELOCITY_BLEND16_PENDING_OWN_REPLAY",
        worlds=16,
    )
    if (
        review.get("status") != "PASSER_VELOCITY_RECOVERY_REPLAY_REVIEWED"
        or review.get("source_hash") != source_hash
        or review.get("source_head") != audit.get("source_head")
        or any(review.get(k) is not False for k in ("promotion_authorized", "hardware_authorized"))
    ):
        raise ValueError("collected worker's review must bind exact successful source and replay")
    return True


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
        collected = collected_replay_verified(state, a.new)
        if collected:
            break
        if state.get("LoadState") != "loaded":
            raise ValueError("cannot inspect exact replay worker")
        if state.get("ActiveState") not in ("active", "activating"):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("replay observation expired; do not restart")
        time.sleep(15)
    if (
        not collected
        and (
            state.get("ActiveState") != "inactive"
            or state.get("Result") != "success"
            or state.get("ExecMainStatus") != "0"
        )
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
