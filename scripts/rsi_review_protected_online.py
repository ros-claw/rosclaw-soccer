"""Read-only physical replay/weight/retention review; never runs simulations."""

import argparse
import json
from pathlib import Path

from rosclaw_soccer.rsi.protected_online_evidence import review_online


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "bank", "preview-root", "validation-root"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            review_online(args.root, args.bank, args.preview_root, args.validation_root),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
