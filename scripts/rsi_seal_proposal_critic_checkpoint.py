"""Explicit metadata-only migration of authenticated frozen-actor MC critics.

The original fitting source proves that actions are raw actor proposals. Early
pilot code omitted the checkpoint semantic field. Do not rerun optimization or
silently interpret those critics as applied-torque critics. Retain originals.
"""

import argparse
import io
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-hash", required=True)
    p.add_argument("--producer-head", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    path = a.source / "complete.json"
    if a.output.exists() or path.is_symlink() or digest(path) != a.source_hash:
        raise ValueError("unused migration output and exact original fitting report required")
    report = json.loads(path.read_text())
    if (
        report["status"] != "PROPOSAL_CRITIC_CALIBRATION_EXPERIMENT_ACTOR_FROZEN"
        or report["source_head"] != a.producer_head
        or report["actual_critic_Adam_steps"] != 128
        or report["actor_optimizer_updates"] != 0
        or report["SAC_update_index"] != 0
        or not report["actor_binary_unchanged"]
        or report["declaration"]["protocol"] != "paired_frozen_proposal_critic_scale.v1"
    ):
        raise ValueError("authenticated frozen actor/raw-proposal MC calibration required")
    checkpoint_path = a.source / report["checkpoint"]["path"]
    if (
        checkpoint_path.is_symlink()
        or checkpoint_path.parent.resolve() != a.source.resolve()
        or digest(checkpoint_path) != report["checkpoint"]["hash"]
    ):
        raise ValueError("exact original critic checkpoint required")
    pins = dict(report["input_file_hashes"])
    pins[str(path)], pins[str(checkpoint_path)] = a.source_hash, report["checkpoint"]["hash"]
    source_pins(pins)
    check_pins(pins)
    import torch

    original = torch.load(
        io.BytesIO(checkpoint_path.read_bytes()), map_location="cpu", weights_only=True
    )
    if (
        original["update_index"] != 0
        or original.get("replay_action_semantics") is not None
        or not original["critic_optimizer"]["state"]
    ):
        raise ValueError("only the explicitly unbound initial MC pilot may be migrated")
    steps = {int(value["step"]) for value in original["critic_optimizer"]["state"].values()}
    if steps != {128}:
        raise ValueError("actual critic Adam state differs from authenticated128steps")
    sealed = dict(original)
    sealed["replay_action_semantics"] = "ACTOR_PROPOSAL_THROUGH_GUARD"
    buffer = io.BytesIO()
    torch.save(sealed, buffer)
    restored = torch.load(io.BytesIO(buffer.getvalue()), map_location="cpu", weights_only=True)

    def identical(left, right):
        if type(left) is not type(right):
            return False
        if isinstance(left, torch.Tensor):
            return left.dtype == right.dtype and torch.equal(left, right)
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(
                identical(left[key], right[key]) for key in left
            )
        if isinstance(left, (list, tuple)):
            return len(left) == len(right) and all(
                identical(x, y) for x, y in zip(left, right, strict=True)
            )
        if isinstance(left, (str, int, float, bool)) or left is None:
            return left == right
        raise ValueError("unexpected checkpoint value type")

    old_numeric = {
        key: value for key, value in original.items() if key != "replay_action_semantics"
    }
    new_numeric = {
        key: value for key, value in restored.items() if key != "replay_action_semantics"
    }
    if not identical(old_numeric, new_numeric):
        raise ValueError("metadata migration changed a model/Adam/normalizer/RNG value")
    check_pins(pins)
    a.output.mkdir(parents=True, exist_ok=False)
    target = a.output / "sealed-critic-checkpoint.pt"
    with target.open("xb") as stream:
        stream.write(buffer.getvalue())
    write_once(
        a.output / "complete.json",
        dict(
            status="RAW_PROPOSAL_CRITIC_SEMANTICS_EXPLICITLY_SEALED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            original_fitting_hash=a.source_hash,
            replay_action_semantics=restored["replay_action_semantics"],
            numeric_checkpoint_identity=True,
            original_retained=True,
            checkpoint=dict(path=target.name, hash=digest(target)),
            new_optimizer_updates=0,
            new_physics_steps=0,
            actor_binary_unchanged=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
