"""Compare two hashed, independently replayed developmental passer studies."""

import argparse
import json
from pathlib import Path

from rosclaw_soccer.rsi.passer_recovery_pair_comparison import compare_passer_recovery_reviews
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("old", "new", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("old-hash", "new-hash", "expected-head"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or args.old.is_symlink()
        or args.new.is_symlink()
        or digest(args.old) != args.old_hash
        or digest(args.new) != args.new_hash
    ):
        raise ValueError("two exact reviews and unused external evidence output required")
    old, new = (json.loads(path.read_text()) for path in (args.old, args.new))
    result = compare_passer_recovery_reviews(old, new)
    if digest(args.old) != args.old_hash or digest(args.new) != args.new_hash:
        raise ValueError("review evidence changed during comparison")
    clean(args.expected_head)
    args.output.mkdir(parents=True)
    write_once(
        args.output / "complete.json",
        dict(
            status="PAIRED_PASSER_RECOVERY_REVIEW_COMPARISON",
            source_head=args.expected_head,
            old_hash=args.old_hash,
            new_hash=args.new_hash,
            **result,
        ),
    )
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
