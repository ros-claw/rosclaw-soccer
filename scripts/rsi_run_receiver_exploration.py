"""Wait for strict whole receiver identity, then sample and independently replay."""

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
    for name in ("frontier-root", "identity-review", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "frontier-head",
        "frontier-hash",
        "identity-unit",
        "identity-review-head",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--new-context-block", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--timeout-sec", type=int, default=43200)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or not 1 <= a.timeout_sec <= 43200
        or not a.identity_unit.startswith("rosclaw-team-")
        or not a.identity_unit.endswith(".service")
        or "/" in a.identity_unit
    ):
        raise ValueError("bounded identity gate and unused external output required")
    deadline = time.monotonic() + a.timeout_sec
    while True:
        q = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                a.identity_unit,
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
            raise ValueError("cannot verify identity review unit")
        if state["ActiveState"] != "active":
            if state["Result"] != "success" or state["ExecMainStatus"] != "0":
                raise ValueError("identity review failed; no exploration physics")
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded identity gate wait expired; no retry")
        time.sleep(15)
    if not a.identity_review.is_file() or a.identity_review.is_symlink():
        raise ValueError("complete identity review required")
    review_hash = digest(a.identity_review)
    review = json.loads(a.identity_review.read_text())
    if (
        review.get("status") != "RELAY_WHOLE_RECEIVER_ZERO_IDENTITY_QUALIFIED"
        or review.get("source_head") != a.identity_review_head
        or review.get("conditions") != 24
        or len(review.get("records", [])) != 24
        or review.get("identity_passed") is not True
        or review.get("own_physics_and_neural_replay_complete") is not True
        or any(
            review.get(k) is not False
            for k in ("trained", "promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("strict full zero identity required before exploration")
    a.output.mkdir(parents=True)
    source = a.output / "source"
    subprocess.run(
        [
            sys.executable,
            "scripts/rsi_collect_neural_relay_chain.py",
            "--frontier-root",
            str(a.frontier_root),
            "--frontier-head",
            a.frontier_head,
            "--frontier-hash",
            a.frontier_hash,
            "--new-context-block",
            str(a.new_context_block),
            "--receiver-exploration",
            "--identity-review",
            str(a.identity_review),
            "--identity-review-hash",
            review_hash,
            "--identity-review-head",
            a.identity_review_head,
            "--output",
            str(source),
            "--expected-head",
            a.expected_head,
        ],
        check=True,
    )
    source_hash = digest(source / "complete.json")
    subprocess.run(
        [
            sys.executable,
            "scripts/rsi_audit_neural_relay_chain.py",
            "--source-root",
            str(source),
            "--source-head",
            a.expected_head,
            "--source-hash",
            source_hash,
            "--output",
            str(a.output / "audit"),
            "--expected-head",
            a.expected_head,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
