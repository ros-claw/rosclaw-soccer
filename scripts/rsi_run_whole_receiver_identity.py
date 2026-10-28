"""Run one whole-receiver SIM identity block and its own native replay."""

import argparse
import subprocess
import sys
from pathlib import Path

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("frontier-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("frontier-head", "frontier-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--new-context-block", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--receiver-center-curriculum", action="store_true")
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external evidence root required")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import rosclaw.continual; import rosclaw_soccer.training.native_pass_courses; "
            "import scripts.rsi_collect_neural_relay_chain",
        ],
        check=True,
    )
    a.output.mkdir(parents=True)
    source = a.output / "source"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_collect_neural_relay_chain",
            "--frontier-root",
            str(a.frontier_root),
            "--frontier-head",
            a.frontier_head,
            "--frontier-hash",
            a.frontier_hash,
            "--new-context-block",
            str(a.new_context_block),
            "--receiver-whole-identity",
            *(["--receiver-center-curriculum"] if a.receiver_center_curriculum else []),
            "--output",
            str(source),
            "--expected-head",
            a.expected_head,
        ],
        check=True,
    )
    sh = digest(source / "complete.json")
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
            sh,
            "--output",
            str(a.output / "audit"),
            "--expected-head",
            a.expected_head,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
