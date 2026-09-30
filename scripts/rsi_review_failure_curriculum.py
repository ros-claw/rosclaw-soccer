"""Read-only review; does not run simulations or alter training artifacts."""

import argparse
import json
from pathlib import Path

from rosclaw_soccer.rsi.failure_curriculum_evidence import review_curriculum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new review output required; previous reviews are immutable")
    result = review_curriculum(args.training_root, args.validation_root)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "training_complete": result["training_complete"],
                "complete_candidates": result["complete_candidate_count"],
                "best": result["best"]["score"],
                "report_hash": result["report_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
