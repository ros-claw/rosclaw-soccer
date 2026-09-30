"""Write a pending/rejected Core Growth manifest from audited motor evidence."""

import argparse
import json
from pathlib import Path

from rosclaw_soccer.rsi.core_motor_growth import export_motor_growth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("validation-root", "motor-policy", "zero-parent-policy", "protocol", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output required")
    report = export_motor_growth(
        validation_root=args.validation_root,
        policy_path=args.motor_policy,
        protocol_path=args.protocol,
        zero_parent_policy_path=args.zero_parent_policy,
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "decision": report["consolidation_manifest"]["decision"],
                "report_hash": report["report_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
