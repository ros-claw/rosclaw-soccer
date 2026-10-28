"""Wait for all sampled physics review before exporting learning data."""

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
    p.add_argument("--pair", nargs=2, action="append", required=True, metavar=("SOURCE", "AUDIT"))
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    for name in ("review-unit", "review-head", "producer-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--timeout-sec", type=int, default=43200)
    p.add_argument("--receiver-center-strict", action="store_true")
    a = p.parse_args()
    clean(a.expected_head)
    if (
        len(a.pair) != 3
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or not 1 <= a.timeout_sec <= 43200
        or not a.review_unit.startswith(
            "rosclaw-receiver-center-" if a.receiver_center_strict else "rosclaw-team-"
        )
        or not a.review_unit.endswith(".service")
        or "/" in a.review_unit
    ):
        raise ValueError("bounded full review and unused external output required")
    deadline = time.monotonic() + a.timeout_sec
    while True:
        q = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                a.review_unit,
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
            raise ValueError("cannot verify complete physical review")
        if state["ActiveState"] != "active":
            if state["Result"] != "success" or state["ExecMainStatus"] != "0":
                raise ValueError("physical review failed; no sampled dataset")
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded review wait expired; no retry")
        time.sleep(15)
    if not a.review.is_file() or a.review.is_symlink():
        raise ValueError("complete immutable review required")
    rh = digest(a.review)
    r = json.loads(a.review.read_text())
    if (
        r.get("status") != "RELAY_RECEIVER_EXPLORATION_PAIRED_NOT_TRAINED"
        or r.get("source_head") != a.review_head
        or r.get("conditions") != 24
        or len(r.get("records", [])) != 72
        or len(r.get("source_pins", [])) != 3
        or r.get("own_physics_neural_and_density_audited") is not True
        or r.get("zero_parent_byte_identical") is not True
        or any(
            r.get(k) is not False
            for k in ("optimizer_authorized", "promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("full own-physics and density review required")
    command = [
        sys.executable,
        "-m",
        "scripts.rsi_export_receiver_onpolicy",
        "--review",
        str(a.review),
        "--review-hash",
        rh,
        "--review-head",
        a.review_head,
        "--producer-head",
        a.producer_head,
        "--expected-head",
        a.expected_head,
        "--output",
        str(a.output),
    ]
    if a.receiver_center_strict:
        command.append("--receiver-center-strict")
    for pair in a.pair:
        command.extend(("--pair", *pair))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
