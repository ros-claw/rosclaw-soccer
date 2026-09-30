"""Read-only physical/statistical/usage ledger review of the neural fresh exam."""

import argparse
import json
from pathlib import Path

from rosclaw_soccer.rsi.neural_fresh_evidence import review_fresh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "quarantine", "pool-ledger"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            review_fresh(args.root, args.quarantine, args.pool_ledger),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
