"""Wait for one SIM producer and start its own replay without shell expansion.

This is a bounded dependency, not an autonomous research decision. It neither
retries a failed producer nor restarts an interrupted or failed replay.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from scripts.rsi_native_fresh_common import clean


def producer_status(unit: str) -> tuple[str, str, int]:
    if not re.fullmatch(r"rosclaw-[a-z0-9-]+\.service", unit):
        raise ValueError("explicit project-owned systemd unit required")
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "-p",
            "LoadState",
            "-p",
            "ActiveState",
            "-p",
            "Result",
            "-p",
            "ExecMainStatus",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if fields.get("LoadState") != "loaded":
        raise ValueError("producer unit is not loaded")
    return fields["ActiveState"], fields["Result"], int(fields["ExecMainStatus"])


def receipt_hash(source: Path, source_head: str, status: str) -> str:
    path = source / "complete.json"
    if (
        source.is_symlink()
        or path.is_symlink()
        or not path.is_file()
        or (source / "failed.json").exists()
    ):
        raise ValueError("closed non-symlink source receipt required")
    raw = path.read_bytes()
    receipt = json.loads(raw)
    if (
        receipt.get("source_head") != source_head
        or receipt.get("status") != status
        or any(
            receipt.get(k) is not False
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError("exact unpromoted simulation source contract required")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--unit", required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-head", required=True)
    p.add_argument("--source-status", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--wait-seconds", type=int, default=1200)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        not re.fullmatch(r"[0-9a-f]{40}", a.source_head)
        or not 1 <= a.wait_seconds <= 7200
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
    ):
        raise ValueError("full source head, bounded wait and unused external output required")
    deadline = time.monotonic() + a.wait_seconds
    while True:
        state, result, exit_code = producer_status(a.unit)
        if state in ("inactive", "failed"):
            if state != "inactive" or result != "success" or exit_code != 0:
                raise RuntimeError("producer failed; replay not launched")
            break
        if state not in ("active", "activating") or time.monotonic() >= deadline:
            raise RuntimeError("producer not safely terminal before bounded deadline")
        time.sleep(min(10.0, max(0.0, deadline - time.monotonic())))
    digest = receipt_hash(a.source, a.source_head, a.source_status)
    clean(a.expected_head)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_audit_neural_relay_chain",
            "--source-root",
            str(a.source),
            "--source-head",
            a.source_head,
            "--source-hash",
            digest,
            "--expected-head",
            a.expected_head,
            "--output",
            str(a.output),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
