"""Seal an untrained residual MLP on the complete later-parent output memory."""

import argparse
import json
from pathlib import Path

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.output_memory_step_motor import initial_model
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-model", "memory-root", "output-root"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    parent = json.loads(args.parent_model.read_text())
    manifest = _sealed(args.memory_root / "manifest.json")
    memory = json.loads((args.memory_root / "memory.json").read_text())
    evidence_keys = (
        "schema",
        "predecessor_manifest_hash",
        "predecessor_memory_hash",
        "parent_model_hash",
        "bank_summary_hash",
        "bank_review_hash",
        "records",
        "source_hash",
        "promotion_authorized",
        "hardware_authorized",
    )
    if (
        manifest["schema"] != "soccer.rsi.full_parent_memory_extension.v1"
        or manifest["memory_hash"] != memory["memory_hash"]
        or manifest["parent_model_hash"] != parent["model_hash"]
        or memory["evidence_hash"] != hash_json({k: manifest[k] for k in evidence_keys})
        or manifest["recorded_frames"] != len(memory["observations"])
        or manifest["inherited_frames"] + manifest["added_frames"] != manifest["recorded_frames"]
        or any(
            manifest.get(k) is not False
            for k in (
                "physical_policy_execution_qualified",
                "promotion_authorized",
                "hardware_authorized",
            )
        )
    ):
        raise ValueError("complete sealed later-parent training memory required")
    model = initial_model(parent, memory)
    args.output_root.mkdir(parents=True, exist_ok=False)
    write_once(args.output_root / "model.json", model)
    print(
        json.dumps(
            dict(
                model_hash=model["model_hash"],
                memory_hash=memory["memory_hash"],
                recorded_frames=manifest["recorded_frames"],
                independent_contexts=manifest["independent_contexts"],
                new_optimizer_steps=0,
                physical_policy_execution_qualified=False,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
