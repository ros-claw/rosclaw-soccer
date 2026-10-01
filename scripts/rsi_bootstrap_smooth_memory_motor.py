"""Create a zero-residual AR-capable child without changing the memory parent."""

import argparse
import json
from pathlib import Path

from rosclaw_soccer.rsi.smooth_memory_motor import initial_model
from scripts.rsi_atomic_artifacts import write_once


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    model = initial_model(json.loads(args.parent_model.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=False)
    write_once(args.output, model)
    print(json.dumps(dict(model_hash=model["model_hash"], parent_hash=model["parent_model_hash"])))


if __name__ == "__main__":
    main()
