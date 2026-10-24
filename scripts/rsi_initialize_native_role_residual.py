"""Initialize a separate zero role residual over a retained1024-step encoder."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "core-source-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("training-hash", "core-expected-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from scripts.rsi_active_core_source import pin_active_core
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    path = a.training / "complete.json"
    if a.output.exists() or path.is_symlink() or digest(path) != a.training_hash:
        raise ValueError("unused output and exact completed encoder training required")
    fitted = json.loads(path.read_text())
    if (
        fitted["status"] != "NATIVE_ROLE_COMMAND_DAGGER_TRAINED_NOT_PHYSICALLY_EVALUATED"
        or fitted["optimizer_updates"] != 1024
        or fitted["actual_critic_optimizer_updates"] != 0
        or fitted["original_Adam_and_RNG_and_normalization_retained"] is not True
    ):
        raise ValueError("retained1024-step role encoder required; not model promotion")
    pins = dict(fitted["input_file_hashes"])
    pins[str(path)] = a.training_hash
    encoder = a.training / fitted["policy"]["path"]
    if (
        encoder.is_symlink()
        or encoder.parent.resolve() != a.training.resolve()
        or digest(encoder) != fitted["policy"]["hash"]
    ):
        raise ValueError("exact confined retained encoder required")
    pins[str(encoder)] = fitted["policy"]["hash"]
    queries = [
        Path(k)
        for k, v in pins.items()
        if v == fitted["query_hash"] and k.endswith("/complete.json")
    ]
    if (
        len(queries) != 1
        or json.loads(queries[0].read_text())["original_teacher_labels_exact"] != 22000
    ):
        raise ValueError("all original parent labels must already be independently reconstructed")
    pin_active_core(a.core_source_root, a.core_expected_head, pins)
    source_pins(pins)
    check_pins(pins)
    from rosclaw_soccer.rsi import native_role_teacher, native_team_intent
    from rosclaw_soccer.rsi.native_role_command import NativeRoleCommandPolicy
    from rosclaw_soccer.rsi.native_role_residual import (
        NativeRoleResidualPolicy,
        qualified_role_parent_hash,
        serialize_role_residual,
    )
    from rosclaw_soccer.sim.contracts import hash_json
    from rosclaw_soccer.skills import planar_interception
    from rosclaw_soccer.world import player_clearance

    # Before the present source was added, these exact implementation bytes
    # already had22k actual-command labels validated in each query bank.
    ancestors = set(fitted["input_file_hashes"].values())
    for module in (native_role_teacher, native_team_intent, planar_interception, player_clearance):
        if digest(Path(module.__file__)) not in ancestors:
            raise ValueError("current coached function differs from validated parent")
    policy = NativeRoleCommandPolicy(
        encoder, expected_hash=fitted["policy"]["hash"], expected_binding=fitted["binding_hash"]
    )
    parent_hash = qualified_role_parent_hash()
    dataset_hash = hash_json(
        dict(
            encoder_training=a.training_hash,
            verified_parent_queries=fitted["query_hash"],
            initialization="zero_parent_role_residual121",
            new_training_samples=0,
        )
    )
    value = serialize_role_residual(
        encoder_hash=policy.artifact_hash,
        binding_hash=fitted["binding_hash"],
        parent_hash=parent_hash,
        dataset_hash=dataset_hash,
    )
    a.output.mkdir(parents=True)
    copied = a.output / "encoder.json"
    with copied.open("xb") as f:
        f.write(encoder.read_bytes())
    write_once(a.output / "residual-head.json", value)
    artifact_hash = digest(a.output / "residual-head.json")
    NativeRoleResidualPolicy(
        a.output / "residual-head.json",
        expected_hash=artifact_hash,
        encoder=policy,
        expected_encoder_hash=policy.artifact_hash,
        expected_binding=fitted["binding_hash"],
        expected_parent_hash=parent_hash,
    )
    if digest(copied) != policy.artifact_hash:
        raise ValueError("encoder bytes changed during transfer")
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="NATIVE_ROLE_PARENT_RESIDUAL_INITIALIZED_NOT_TRAINED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            training_hash=a.training_hash,
            seed=fitted["seed"],
            binding_hash=fitted["binding_hash"],
            parent_hash=parent_hash,
            dataset_hash=dataset_hash,
            encoder=dict(path=copied.name, hash=policy.artifact_hash),
            policy=dict(path="residual-head.json", hash=artifact_hash),
            historical_encoder_optimizer_updates=1024,
            optimizer_updates=0,
            new_residual_head_optimizer_updates=0,
            new_critic_optimizer_updates=0,
            observation_dim=121,
            actor_feature_dim=131,
            action_dim=3,
            maximum_ratio=0.05,
            old_absolute_head_unused=True,
            physical_identity_not_yet_proven=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
