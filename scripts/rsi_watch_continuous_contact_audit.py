"""Wait for complete continuous-data export, then run independent readback."""

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
    p.add_argument("--pair", nargs=2, action="append", required=True, metavar=("ROOT", "AUDIT"))
    for name in ("dataset", "features", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("dataset-hash", "unit", "producer-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--timeout-sec", type=int, default=43200)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        len(a.pair) != 6
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or not 1 <= a.timeout_sec <= 43200
        or not a.unit.startswith("rosclaw-team-")
        or not a.unit.endswith(".service")
        or "/" in a.unit
    ):
        raise ValueError("bounded data producer and unused external output required")
    deadline = time.monotonic() + a.timeout_sec
    while True:
        q = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                a.unit,
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
            raise ValueError("cannot verify data producer")
        if state["ActiveState"] != "active":
            if state["Result"] != "success" or state["ExecMainStatus"] != "0":
                raise ValueError("data producer failed; no audit")
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded producer wait expired; no retry")
        time.sleep(15)
    receipt = a.features / "complete.json"
    if not receipt.is_file() or receipt.is_symlink():
        raise ValueError("complete feature export required")
    feature_hash = digest(receipt)
    r = json.loads(receipt.read_text())
    if r.get("source_head") != a.producer_head or r.get("dataset_hash") != a.dataset_hash:
        raise ValueError("exact producer and parent dataset required")
    command = [
        sys.executable,
        "scripts/rsi_audit_continuous_contact_forecast.py",
        "--dataset",
        str(a.dataset),
        "--dataset-hash",
        a.dataset_hash,
        "--features",
        str(a.features),
        "--features-hash",
        feature_hash,
        "--producer-head",
        a.producer_head,
        "--expected-head",
        a.expected_head,
        "--output",
        str(a.output),
    ]
    for pair in a.pair:
        command.extend(("--pair", *pair))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
