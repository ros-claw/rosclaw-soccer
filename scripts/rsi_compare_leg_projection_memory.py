"""Compare pinned independently replayed projection-memory variants."""

import argparse
import json
from pathlib import Path

from rosclaw_soccer.rsi.leg_projection_memory_comparison import compare_projection_memory
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("old", "new", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("old-hash", "new-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.old.is_symlink()
        or a.new.is_symlink()
        or digest(a.old) != a.old_hash
        or digest(a.new) != a.new_hash
    ):
        raise ValueError("two pinned reviews and unused external output required")
    result = compare_projection_memory(json.loads(a.old.read_text()), json.loads(a.new.read_text()))
    if digest(a.old) != a.old_hash or digest(a.new) != a.new_hash:
        raise ValueError("review evidence changed during comparison")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="LEG_PROJECTION_MEMORY_PAIRED_COMPARISON",
            source_head=a.expected_head,
            old_hash=a.old_hash,
            new_hash=a.new_hash,
            **result,
        ),
    )
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
