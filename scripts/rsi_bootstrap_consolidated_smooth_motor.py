"""Seal an explicit memory-consolidation proposal; no new optimizer or physics."""

import argparse
import json
from pathlib import Path

from rosclaw_soccer.rsi.consolidated_smooth_motor import make_model
from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from scripts.rsi_atomic_artifacts import write_once


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-model", "memory-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.base_model.read_text())
    manifest = _sealed(args.memory_root / "manifest.json")
    memory = json.loads((args.memory_root / "memory.json").read_text())
    model = make_model(base, memory, manifest)
    write_once(args.output, model)
    print(
        dict(
            model_hash=model["model_hash"],
            base_model_hash=base["model_hash"],
            current_parent_hash=base["parent_model_hash"],
            new_optimizer_steps=0,
            physical_executions_added=0,
            qualification="UNQUALIFIED_CONSOLIDATION_ABLATION",
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
