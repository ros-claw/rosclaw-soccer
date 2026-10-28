"""Review a single SIM pilot after the exact physics/replay unit terminates."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_native_fresh_common import clean
from scripts.rsi_watch_passer_residual_pair import worker_ready


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "prior", "prior-audit", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("unit", "source-head", "prior-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--timeout-sec", type=int, default=7200)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        not a.unit.startswith("rosclaw-precontact-smooth-")
        or not a.unit.endswith(".service")
        or "/" in a.unit
        or a.output.exists()
        or not 1 <= a.timeout_sec <= 7200
    ):
        raise ValueError("exact bounded experiment and unused output required")
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
            text=True,
            capture_output=True,
            check=True,
        )
        state = dict(line.split("=", 1) for line in q.stdout.splitlines() if "=" in line)
        if worker_ready(a.root, state):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("pilot still unverified; do not restart it")
        time.sleep(15)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_review_smooth_pilot",
            "--source",
            str(a.root / "source"),
            "--audit",
            str(a.root / "audit/complete.json"),
            "--prior",
            str(a.prior),
            "--prior-audit",
            str(a.prior_audit),
            "--output",
            str(a.output),
            "--source-head",
            a.source_head,
            "--prior-head",
            a.prior_head,
            "--expected-head",
            a.expected_head,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
