"""Run one fixed foot-teacher/zero-authority pair and own native replay."""

import argparse
import subprocess
import sys
from pathlib import Path

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("frontier-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("frontier-head", "frontier-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--new-context-block", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--predictive-foot-shin", action="store_true")
    p.add_argument("--axis-local-fallback", action="store_true")
    p.add_argument("--early-intercept", action="store_true")
    a = p.parse_args()
    if a.axis_local_fallback and not a.predictive_foot_shin:
        p.error("axis-local fallback requires predictive-foot-shin")
    if a.early_intercept and not (a.axis_local_fallback and a.predictive_foot_shin):
        p.error("early intercept requires predictive-foot-shin and axis-local fallback")
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external evidence root required")
    a.output.mkdir(parents=True)
    source = a.output / "source"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_collect_neural_relay_chain",
            "--receiver-foot-teacher-pair",
            *(["--receiver-predictive-foot-shin"] if a.predictive_foot_shin else []),
            *(["--receiver-axis-local-fallback"] if a.axis_local_fallback else []),
            *(["--receiver-early-foot-intercept"] if a.early_intercept else []),
            "--new-context-block",
            str(a.new_context_block),
            "--frontier-root",
            str(a.frontier_root),
            "--frontier-head",
            a.frontier_head,
            "--frontier-hash",
            a.frontier_hash,
            "--expected-head",
            a.expected_head,
            "--output",
            str(source),
        ],
        check=True,
    )
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
            "--expected-head",
            a.expected_head,
            "--output",
            str(a.output / "audit"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
