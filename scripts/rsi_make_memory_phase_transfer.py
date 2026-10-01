"""Build an explicit trained-head transfer; this command performs no training."""

import argparse
import json
from pathlib import Path

from rosclaw_soccer.rsi.memory_guarded_phase_transfer import make_model
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("phase-model", "kernel-model", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    model = make_model(
        json.loads(args.phase_model.read_text()), json.loads(args.kernel_model.read_text())
    )
    write_once(args.output, model)
    print(json.dumps(dict(model_hash=model["model_hash"], new_optimizer_steps=0)), flush=True)


if __name__ == "__main__":
    main()
