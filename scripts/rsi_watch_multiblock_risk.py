"""Wait for two real paired reviews, then run one grouped SIM research fit."""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--block",
        nargs=5,
        action="append",
        required=True,
        metavar=("REVIEW", "REVIEW_HASH", "FEATURES", "FEATURE_HASH", "UNIT"),
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=int, default=3600)
    args = parser.parse_args()
    clean(args.expected_head)
    if (
        len(args.block) != 3
        or not 1 <= args.timeout_sec <= 7200
        or args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
    ):
        raise ValueError("three explicit blocks and a new external result required")
    units = [block[4] for block in args.block if block[4] != "complete"]
    if sorted(units) != [
        "rosclaw-context-b1-long-pair-b0-20261029.service",
        "rosclaw-context-b2-long-pair-b0-20261029.service",
    ] or any(
        re.fullmatch(r"rosclaw-context-b[12]-long-pair-b0-20261029\.service", unit) is None
        for unit in units
    ):
        raise ValueError("exact declared long-pair units required")
    for review, review_hash, feature, feature_hash, unit in args.block:
        if (
            Path(review).is_symlink()
            or Path(feature).is_symlink()
            or digest(Path(feature)) != feature_hash
        ):
            raise ValueError("exact pinned features and non-symlink reviews required")
        if (
            unit == "complete"
            and (not Path(review).is_file() or digest(Path(review)) != review_hash)
        ) or (unit != "complete" and review_hash != "pending"):
            raise ValueError("completed review hash or explicitly pending review required")
    deadline = time.monotonic() + args.timeout_sec
    while True:
        pending = [
            (Path(review), unit)
            for review, _, _, _, unit in args.block
            if not Path(review).is_file()
        ]
        if not pending:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("paired physics observation expired; never restart")
        for _, unit in pending:
            if unit == "complete":
                raise ValueError("declared prior completed review is missing")
            result = subprocess.run(
                ["systemctl", "--user", "show", unit, "-p", "ActiveState"],
                capture_output=True,
                text=True,
                check=True,
            )
            if result.stdout.strip() not in ("ActiveState=active", "ActiveState=activating"):
                raise ValueError("physical pair ended without a review; preserve failure")
        time.sleep(15)
    command = [sys.executable, "-m", "scripts.rsi_train_multiblock_receiver_risk"]
    for review, review_hash, feature, feature_hash, unit in args.block:
        path = Path(review)
        if path.is_symlink() or not path.is_file():
            raise ValueError("complete exact paired review required")
        actual_hash = digest(path)
        if unit == "complete" and actual_hash != review_hash:
            raise ValueError("prior completed review changed")
        command.extend(["--block", str(path), actual_hash, feature, feature_hash])
    subprocess.run(
        [*command, "--expected-head", args.expected_head, "--output", str(args.output)], check=True
    )


if __name__ == "__main__":
    main()
