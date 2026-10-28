"""Bounded observer: compare only a completed exact candidate SIM probe."""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "snapshots", "scene", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("baseline-hash", "snapshot-hash", "candidate-head", "expected-head", "unit"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--timeout-sec", type=int, default=7200)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        re.fullmatch(r"rosclaw-relay-leg-probe-candidate-[a-z]-b0-[0-9]{8}\.service", a.unit)
        is None
        or not 1 <= a.timeout_sec <= 7200
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.baseline.is_symlink()
        or digest(a.baseline) != a.baseline_hash
        or a.snapshots.is_symlink()
        or digest(a.snapshots) != a.snapshot_hash
    ):
        raise ValueError("exact bounded worker and pinned evidence required")
    deadline = time.monotonic() + a.timeout_sec
    while True:
        q = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                a.unit,
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
                raise TimeoutError("candidate observation expired; never restart")
            time.sleep(15)
            continue
        if not (
            (
                state.get("LoadState") == "loaded"
                and state.get("ActiveState") == "inactive"
                and state.get("Result") == "success"
                and state.get("ExecMainStatus") == "0"
            )
            or (state.get("LoadState") == "not-found" and state.get("ActiveState") == "inactive")
        ):
            raise ValueError("candidate worker failed or cannot be inspected")
        break
    if a.candidate.is_symlink() or not a.candidate.is_file():
        raise ValueError("completed candidate receipt required")
    candidate_hash = digest(a.candidate)
    candidate = json.loads(a.candidate.read_text())
    if (
        candidate.get("source_head") != a.candidate_head
        or candidate.get("status") != "RELAY_LEG_RESIDUAL_PHYSICAL_PROBE_NOT_FOOTBALL_SUCCESS"
        or candidate.get("snapshot_hash") != a.snapshot_hash
        or len(candidate.get("rows", [])) != 8
        or any(
            candidate.get(k) is not False
            for k in ("goal_complete", "promotion_authorized", "hardware_authorized")
        )
        or digest(a.candidate) != candidate_hash
    ):
        raise ValueError("exact unpromoted eight-case candidate receipt required")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_compare_restored_receiver_probe",
            "--baseline",
            str(a.baseline),
            "--baseline-hash",
            a.baseline_hash,
            "--candidate",
            str(a.candidate),
            "--candidate-hash",
            candidate_hash,
            "--snapshots",
            str(a.snapshots),
            "--snapshot-hash",
            a.snapshot_hash,
            "--scene",
            str(a.scene),
            "--output",
            str(a.output),
            "--expected-head",
            a.expected_head,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
