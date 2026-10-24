"""Bind a zero residual head to a real encoder and verified fast parent; no training."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "parent", "own", "core-source-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("training-hash", "parent-hash", "own-hash", "core-expected-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from scripts.rsi_active_core_source import pin_active_core
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused zero-initialization output required")
    reports, pins = {}, {}
    for name in ("training", "parent", "own"):
        path = getattr(a, name) / "complete.json"
        expected = getattr(a, name + "_hash")
        if path.is_symlink() or digest(path) != expected:
            raise ValueError("exact completed ancestor required")
        report = json.loads(path.read_text())
        for key, value in report["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting source ancestry")
            pins[key] = value
        pins[str(path)] = expected
        reports[name] = report
    training, parent, own = (reports[k] for k in ("training", "parent", "own"))
    declaration_path = a.parent / "declaration.json"
    if own["input_file_hashes"].get(str(declaration_path)) != digest(declaration_path):
        raise ValueError("parent declaration must be authenticated by own replay")
    declaration = json.loads(declaration_path.read_text())
    if (
        not own["all_exact"]
        or own["source_hash"] != a.parent_hash
        or len(parent["records"]) != 16
        or not all(r["healthy_full_horizon"] for r in parent["records"])
        or declaration["protocol"] != "robonaldo.reference_clock_authority.v1"
        or declaration["reference_rate"] != 1.18
        or training["actual_BC_actor_optimizer_updates"] != 390
        or training["SAC_update_index"] != 0
    ):
        raise ValueError("actual390-step encoder and exact complete fast16 parent required")
    original = a.training / training["policy"]["path"]
    if (
        original.parent.resolve() != a.training.resolve()
        or original.is_symlink()
        or digest(original) != training["policy"]["hash"]
    ):
        raise ValueError("confined original trained encoder binary required")
    pins[str(original)] = training["policy"]["hash"]
    pin_active_core(a.core_source_root, a.core_expected_head, pins)
    source_pins(pins)
    check_pins(pins)
    import numpy as np
    from rosclaw.simforge.g1_neural_torque import load_g1_neural_torque_artifact
    from rosclaw.simforge.g1_parent_residual_torque import serialize_parent_residual_head

    from rosclaw_soccer.sim.contracts import hash_json

    encoder = load_g1_neural_torque_artifact(
        original, expected_hash=training["policy"]["hash"], expected_body_hash=training["body_hash"]
    )
    parent_hash = declaration["reference_controller_hash"]
    dataset_hash = hash_json(
        dict(
            encoder_hash=encoder.artifact_hash,
            physical_parent_hash=a.parent_hash,
            new_training_samples=0,
            initialization="zero-residual-context-head.v1",
        )
    )
    payload = serialize_parent_residual_head(
        encoder=encoder,
        parent_policy_hash=parent_hash,
        dataset_hash=dataset_hash,
        residual_limit_ratio=0.01,
    )
    decoded = json.loads(payload)
    if np.any(decoded["head_weight"]) or np.any(decoded["head_bias"]):
        raise ValueError("zero head initialization changed")
    a.output.mkdir(parents=True)

    def artifact(name, content):
        with (a.output / name).open("xb") as f:
            f.write(content)
        return dict(path=name, hash=digest(a.output / name))

    copied = artifact("encoder.bin", original.read_bytes())
    head = artifact("zero-residual-head.json", payload)
    if copied["hash"] != encoder.artifact_hash:
        raise ValueError("encoder transfer changed weights or metadata")
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="CORE_PARENT_RESIDUAL_ZERO_INITIALIZED_NOT_TRAINED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            policy=head,
            encoder=copied,
            body_hash=encoder.body_hash,
            parent_policy_hash=parent_hash,
            dataset_hash=dataset_hash,
            training_hash=a.training_hash,
            parent_hash=a.parent_hash,
            own_hash=a.own_hash,
            encoder_historical_BC_updates=390,
            actual_BC_actor_optimizer_updates=0,
            actual_SAC_actor_optimizer_updates=0,
            new_residual_head_optimizer_updates=0,
            optimizer_updates=0,
            residual_limit_ratio=0.01,
            effective_observation_dim=131,
            encoder_observation_dim=102,
            encoder_weights_unchanged=True,
            actual_task_gain_not_claimed=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
