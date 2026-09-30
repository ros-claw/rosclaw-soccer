"""Independently review raw online physics before exporting a pending Core manifest."""

import argparse
import json
from pathlib import Path

from rosclaw_soccer.rsi.core_online_growth import online_growth_payload
from rosclaw_soccer.rsi.protected_online_evidence import review_online


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "bank", "preview-root", "validation-root", "model"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    review = review_online(args.root, args.bank, args.preview_root, args.validation_root)
    model = json.loads(args.model.read_text())
    print(
        json.dumps(online_growth_payload(review, model), indent=2, sort_keys=True, allow_nan=False)
    )


if __name__ == "__main__":
    main()
