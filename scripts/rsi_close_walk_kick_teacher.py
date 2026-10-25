"""Declared bounded successor: authenticate closed producer cohorts, then audit.

Use systemd After= for ordering only. Never Requires=/start/restart physical
units to recover missing artifacts. Missing/failed cohorts require agent review.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rosclaw_soccer.providers.g1.velocity_policy import file_hash


def closed_shard_arguments(roots, producer_head):
    if len(roots) != 4 or len({p.resolve() for p in roots}) != 4:
        raise ValueError("four unique declared roots required")
    args = []
    fast_mode = None
    for i, root in enumerate(roots):
        path = root / "complete.json"
        if root.is_symlink() or path.is_symlink() or not path.is_file():
            raise ValueError("missing closed shard; no physics restart authorized")
        report = json.loads(path.read_text())
        fast = report.get("experimental_fast_approach", False)
        if (
            report["source_head"] != producer_head
            or report["status"] != "PUBLIC16_TEACHER_CEM_SHARD_MEASURED_NOT_TRAINED"
            or report["worlds"] != 76
            or report["promotion_authorized"] is not False
            or [r["course_index"] for r in report["results"]] != list(range(i, 16, 4))
            or type(fast) is not bool
            or (fast_mode is not None and fast != fast_mode)
        ):
            raise ValueError("entire immutable producer cohort required before audit")
        fast_mode = fast
        args.extend(["--shard", str(root), file_hash(path)])
    return args


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard-root", type=Path, nargs=4, required=True)
    p.add_argument("--expected-producer-head", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_native_fresh_common import clean

    clean(a.expected_head)
    args = closed_shard_arguments(a.shard_root, a.expected_producer_head)
    subprocess.run(
        [
            sys.executable,
            "scripts/rsi_audit_walk_kick_teacher.py",
            *args,
            "--expected-head",
            a.expected_head,
            "--output",
            str(a.output),
        ],
        check=True,
    )
    clean(a.expected_head)


if __name__ == "__main__":
    main()
