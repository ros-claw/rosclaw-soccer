"""Wait for bounded offline training, then independently replay all three shards."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("unit", "training-head", "expected-head"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--timeout-sec", type=int, default=43200)
    args = parser.parse_args()
    clean(args.expected_head)
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or not 1 <= args.timeout_sec <= 43200
        or not args.unit.startswith("rosclaw-team-")
        or not args.unit.endswith(".service")
        or "/" in args.unit
    ):
        raise ValueError("one bounded SIM unit and unused external output required")
    deadline = time.monotonic() + args.timeout_sec
    while True:
        query = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                args.unit,
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
        state = dict(line.split("=", 1) for line in query.stdout.splitlines() if "=" in line)
        if query.returncode or not {"ActiveState", "Result", "ExecMainStatus"} <= state.keys():
            raise ValueError("cannot verify training unit")
        if state["ActiveState"] != "active":
            if state["Result"] != "success" or state["ExecMainStatus"] != "0":
                raise ValueError("failed training; no audit retry")
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded training wait expired")
        time.sleep(15)
    dp = args.root / "data" / "complete.json"
    if not dp.is_file() or dp.is_symlink():
        raise ValueError("complete dataset required")
    dataset_hash = digest(dp)
    models = []
    for shard in (0, 1, 2):
        report = args.root / f"train-{shard}" / "complete.json"
        if not report.is_file() or report.is_symlink():
            raise ValueError("all three complete model shards required")
        model_hash = digest(report)
        model = json.loads(report.read_text())
        if (
            model.get("status") != "RELAY_FALL_SUPERVISED_CONTEXT_HOLDOUT_NOT_CONTROL"
            or model.get("source_head") != args.training_head
            or model.get("dataset_hash") != dataset_hash
            or model.get("shard") != shard
            or model.get("optimizer_updates") != 2400
        ):
            raise ValueError("exact complete model shard identity required")
        models.append((report.parent, model_hash))
    args.output.mkdir(parents=True)
    for shard, (source, source_hash) in enumerate(models):
        subprocess.run(
            [
                sys.executable,
                "scripts/rsi_audit_relay_fall_predictor.py",
                "--source",
                str(source),
                "--source-hash",
                source_hash,
                "--dataset",
                str(args.root / "data"),
                "--dataset-hash",
                dataset_hash,
                "--output",
                str(args.output / f"shard-{shard}" / "complete.json"),
                "--expected-head",
                args.expected_head,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
