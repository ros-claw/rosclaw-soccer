"""Read-only stable actor recalibration review and optional pending Core export."""

import argparse
import json
from pathlib import Path

from rosclaw_soccer.rsi.core_online_growth import online_growth_payload
from rosclaw_soccer.rsi.stable_motor_evidence import review_stable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "online-root", "bank", "preview-root", "validation-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--core-export", action="store_true")
    args = parser.parse_args()
    report = review_stable(
        args.root, args.online_root, args.bank, args.preview_root, args.validation_root
    )
    if args.core_export:
        model = json.loads((args.root / "stable_model.json").read_text())
        report = online_growth_payload(report, model)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
