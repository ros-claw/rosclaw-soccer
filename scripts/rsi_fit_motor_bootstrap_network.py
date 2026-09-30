"""CPU-only offline neural warm start, no simulation or policy activation."""

import argparse
import json
from pathlib import Path

from rosclaw_soccer.rsi.motor_bootstrap_network import fit_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2500)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new numerical model output required")
    model = fit_bootstrap(json.loads(args.bank.read_text(encoding="utf-8")), epochs=args.epochs)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(model, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "model_hash": model["model_hash"],
                "diagnostics": model["diagnostics"],
                "runtime_execution_authorized": False,
            }
        )
    )


if __name__ == "__main__":
    main()
