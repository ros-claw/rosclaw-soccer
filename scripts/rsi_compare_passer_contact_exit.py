"""Compare replayed passer studies after a specific review job completes."""

import argparse
import json
import subprocess
import time
from pathlib import Path

from rosclaw_soccer.rsi.passer_contact_exit_compare import compare_passer_reviews
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("first-review", "exit-review", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("first-hash", "first-head", "exit-head", "review-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--review-unit", required=True)
    p.add_argument("--timeout-sec", type=int, default=43200)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.first_review.is_symlink()
        or digest(a.first_review) != a.first_hash
        or not a.review_unit.startswith("rosclaw-passer-contact-exit-review-")
        or not a.review_unit.endswith(".service")
        or "/" in a.review_unit
        or not 1 <= a.timeout_sec <= 43200
    ):
        raise ValueError("fixed first review, exact job and unused external output required")
    deadline = time.monotonic() + a.timeout_sec
    while True:
        q = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                a.review_unit,
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
        if state.get("ActiveState") in ("active", "activating"):
            if time.monotonic() >= deadline:
                raise TimeoutError("observation timeout; do not restart live review")
            time.sleep(15)
            continue
        if state.get("LoadState") == "not-found" and state.get("ActiveState") == "inactive":
            break  # Receipt integrity is still checked below.
        if not (
            state.get("LoadState") == "loaded"
            and state.get("ActiveState") == "inactive"
            and state.get("Result") == "success"
            and state.get("ExecMainStatus") == "0"
        ):
            raise ValueError("review job did not complete successfully")
        break
    if a.exit_review.is_symlink():
        raise ValueError("owned review receipt required")
    current_hash = digest(a.exit_review)
    first, current = json.loads(a.first_review.read_text()), json.loads(a.exit_review.read_text())
    if (
        first.get("producer_head") != a.first_head
        or current.get("producer_head") != a.exit_head
        or current.get("source_head") != a.review_head
    ):
        raise ValueError("exact physical producer and reviewer source histories required")
    comparison = compare_passer_reviews(first, current)
    if digest(a.first_review) != a.first_hash or digest(a.exit_review) != current_hash:
        raise ValueError("review changed during comparison")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="PASSER_CONTACT_EXIT_MATCHED_COMPARISON",
            first_review_hash=a.first_hash,
            exit_review_hash=current_hash,
            source_head=a.expected_head,
            **comparison,
        ),
    )


if __name__ == "__main__":
    main()
