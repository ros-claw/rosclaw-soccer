"""Independently replay CPU physics and reconstruct causal neural commands."""

import argparse
import json
from pathlib import Path

from rosclaw_soccer.rsi.cpu_motor_transfer_evidence import audit_cpu_transfer
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_cpu_transfer(args.root, args.source)
    write_once(args.output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
