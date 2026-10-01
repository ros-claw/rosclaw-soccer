"""Build a separately sealed narrow-memory candidate without new training."""

import argparse
import json
from pathlib import Path

from rosclaw_soccer.rsi.selective_phase_memory import make_model
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = make_model(json.loads(args.transfer_model.read_text()))
    write_once(args.output, model)
    print(json.dumps(dict(model_hash=model["model_hash"], new_optimizer_steps=0)), flush=True)


if __name__ == "__main__":
    main()
