"""Read back paired decision data; shared extractor, no new physics."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-hash", required=True)
    parser.add_argument("--pair", nargs=3, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    receipt, data = args.dataset / "complete.json", args.dataset / "dataset.npz"
    if (
        len(args.pair) != 6
        or args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or receipt.is_symlink()
        or data.is_symlink()
        or digest(receipt) != args.dataset_hash
    ):
        raise ValueError("exact source data, six audited inputs and unused output required")
    source = json.loads(receipt.read_text())
    if (
        source.get("status") != "PAIRED_RELAY_BALANCE_DECISIONS_EXPORTED_NOT_TRAINED"
        or digest(data) != source["dataset_hash"]
    ):
        raise ValueError("exact paired decision data required")
    args.output.mkdir(parents=True)
    command = [
        sys.executable,
        "scripts/rsi_export_relay_balance_decisions.py",
        "--output",
        str(args.output / "readback"),
        "--expected-head",
        args.expected_head,
    ]
    for pair in args.pair:
        command.extend(("--pair", *pair))
    subprocess.run(command, check=True)
    readback = json.loads((args.output / "readback" / "complete.json").read_text())
    for key in (
        "source_pins",
        "teacher_pairs",
        "conditions",
        "handoff_decisions",
        "no_handoff_conditions",
        "groups",
        "rows",
    ):
        if source[key] != readback[key]:
            raise ValueError("complete original source and readback metadata differ")
    with (
        np.load(data, allow_pickle=False) as a,
        np.load(args.output / "readback" / "dataset.npz", allow_pickle=False) as b,
    ):
        if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
            raise ValueError("all actual pre-intervention features and outcomes must match")
    if digest(receipt) != args.dataset_hash or digest(data) != source["dataset_hash"]:
        raise ValueError("source dataset changed during readback")
    clean(args.expected_head)
    write_once(
        args.output / "complete.json",
        dict(
            status="PAIRED_RELAY_BALANCE_DECISIONS_READBACK_NOT_REINTEGRATED",
            source_head=args.expected_head,
            dataset_hash=args.dataset_hash,
            source_pairs=source["source_pins"],
            features_reuse_export_code=True,
            own_physics_reintegrated=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
