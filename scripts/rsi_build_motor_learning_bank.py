"""Export a read-only offline learning bank, never a runtime expert selector."""

import argparse
import json
from pathlib import Path

from rosclaw_soccer.rsi.motor_learning_bank import build_bank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training-root", "validation-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new bank output required")
    bank = build_bank(args.training_root, args.validation_root)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(bank, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                k: bank[k]
                for k in (
                    "sample_count",
                    "distinct_consumed_course_count",
                    "successful_teacher_course_count",
                    "report_hash",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
