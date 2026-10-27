"""Run a fixed shared-world learned/zero receiver block and its own replay."""

import argparse
import subprocess
import sys
from pathlib import Path

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in (
        "frontier-root",
        "receiver-checkpoint",
        "receiver-candidate-audit",
        "receiver-dataset",
        "output",
    ):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "frontier-head",
        "frontier-hash",
        "receiver-checkpoint-hash",
        "receiver-candidate-audit-hash",
        "receiver-dataset-hash",
        "expected-head",
    ):
        p.add_argument("--" + key, required=True)
    p.add_argument("--new-context-block", type=int, choices=(0, 1, 2), required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external evidence root required")
    a.output.mkdir(parents=True)
    source = a.output / "source"
    command = [
        sys.executable,
        "-m",
        "scripts.rsi_collect_neural_relay_chain",
        "--receiver-learned-pair",
        "--new-context-block",
        str(a.new_context_block),
        "--output",
        str(source),
    ]
    for key in (
        "frontier_root",
        "frontier_head",
        "frontier_hash",
        "receiver_checkpoint",
        "receiver_checkpoint_hash",
        "receiver_candidate_audit",
        "receiver_candidate_audit_hash",
        "receiver_dataset",
        "receiver_dataset_hash",
        "expected_head",
    ):
        command.extend(("--" + key.replace("_", "-"), str(getattr(a, key))))
    subprocess.run(command, check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_audit_neural_relay_chain",
            "--source-root",
            str(source),
            "--source-head",
            a.expected_head,
            "--source-hash",
            digest(source / "complete.json"),
            "--output",
            str(a.output / "audit"),
            "--expected-head",
            a.expected_head,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
