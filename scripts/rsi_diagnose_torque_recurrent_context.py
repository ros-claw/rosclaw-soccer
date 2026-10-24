"""Compare trained16-frame windows with persistent runtime GRU on exact zero runs.

Pure inference on authenticated observations; no physics, optimizer or policy
promotion. Includes the runtime's OOD reset schedule, not an unbounded ideal GRU.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("pair", "training", "core-source-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("zero-hash", "own-hash", "training-hash", "core-expected-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from scripts.rsi_active_core_source import pin_active_core
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused context diagnostic output required")
    pins, reports = {}, {}
    for name, root, expected in (
        ("zero", a.pair / "zero", a.zero_hash),
        ("own", a.pair / "zero-manual", a.own_hash),
        ("training", a.training, a.training_hash),
    ):
        path = root / "complete.json"
        if path.is_symlink() or digest(path) != expected:
            raise ValueError("exact completed diagnostic ancestry required")
        report = json.loads(path.read_text())
        for key, value in report["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting diagnostic ancestry")
            pins[key] = value
        pins[str(path)] = expected
        reports[name] = report
    source, own, training = (reports[k] for k in ("zero", "own", "training"))
    if (
        not own["all_exact"]
        or own["source_hash"] != a.zero_hash
        or len(source["records"]) != 16
        or len(own["records"]) != 16
    ):
        raise ValueError("entire16 independently replayed zero run required")

    def pin(root, entry):
        path = root / entry["path"]
        if (
            path.parent.resolve() != root.resolve()
            or path.is_symlink()
            or digest(path) != entry["hash"]
        ):
            raise ValueError("confined exact artifact required")
        pins[str(path)] = entry["hash"]
        return path

    files = []
    for row in source["records"]:
        entry = row["torque_execution_artifacts"]["a/"]
        if (
            row["torque_intervention_fraction"] != 0
            or not row["healthy_full_horizon"]
            or entry["policy_receipt"]["state_guard_fallback_count"] != 0
            or entry["policy_receipt"]["nonfinite_fallback_count"] != 0
        ):
            raise ValueError(
                "finite full healthy zero context with no state-recovery branch required"
            )
        files.append(pin(a.pair / "zero", entry))
    checkpoint = pin(a.training, training["checkpoint"])
    binary = pin(a.training, training["policy"])
    pin_active_core(a.core_source_root, a.core_expected_head, pins)
    source_pins(pins)
    check_pins(pins)
    import numpy as np
    import torch
    from rosclaw.simforge.g1_neural_torque import (
        G1NeuralTorquePolicy,
        G1TorqueSafetyConfig,
        load_g1_neural_torque_artifact,
    )
    from rosclaw.simforge.g1_neural_torque_learning import (
        G1ContinualTorqueActorCritic,
        G1NeuralTorqueLearnerConfig,
    )

    torch.set_num_threads(1)
    learner = G1ContinualTorqueActorCritic(
        G1NeuralTorqueLearnerConfig(**training["config"]),
        safety=G1TorqueSafetyConfig(**training["safety"]),
    )
    learner.restore_checkpoint(checkpoint.read_bytes())
    artifact = load_g1_neural_torque_artifact(
        binary, expected_hash=training["policy"]["hash"], expected_body_hash=training["body_hash"]
    )
    results = []
    for row, path in zip(source["records"], files, strict=True):
        policy = G1NeuralTorquePolicy(
            artifact,
            expected_body_hash=training["body_hash"],
            expected_parent_policy_hash=training["parent_policy_hash"],
        )
        with np.load(path) as ep:
            obs, targets = ep["observations"].copy(), ep["parent_actions"].copy()
        normalized = np.stack([policy._normalize(x) for x in obs])
        warmup = artifact.safety.warmup_steps
        reset = np.max(np.abs(normalized), axis=1) > artifact.safety.maximum_observation_z
        reset[:warmup] = False
        count = ~reset
        count[:warmup] = False
        if (
            int(reset.sum())
            != row["torque_execution_artifacts"]["a/"]["policy_receipt"][
                "out_of_distribution_fallback_count"
            ]
            or int(count.sum())
            != row["torque_execution_artifacts"]["a/"]["actual_NN_call_counts"]["head_calls"]
        ):
            raise ValueError("reconstructed actual inference/reset schedule differs from source")
        full = np.zeros((len(obs), 29), np.float32)
        with torch.no_grad():
            start = 0
            while start < len(obs):
                if reset[start]:
                    start += 1
                    continue
                stop = start + 1
                while stop < len(obs) and not reset[stop]:
                    stop += 1
                encoded, _ = learner.actor.gru(torch.from_numpy(normalized[start:stop])[None])
                mean, _ = learner.actor.head(encoded[0]).chunk(2, dim=-1)
                full[start:stop] = (mean.tanh() * learner.actor.action_limits).numpy()
                start = stop
            indices = np.arange(15, len(obs), 10)
            indices = indices[count[indices]]
            short = []
            for start in range(0, len(indices), 256):
                batch = indices[start : start + 256]
                windows = np.stack([normalized[i - 15 : i + 1] for i in batch])
                short.append(learner.actor.deterministic(torch.from_numpy(windows)).numpy())
        short = np.concatenate(short)
        native = []
        for i, value in enumerate(normalized):
            if reset[i]:
                policy._hidden.fill(0)
            elif i < warmup:
                policy._advance_hidden(value)
            else:
                predicted = policy._infer_normalized(value)
                if i in indices:
                    native.append(predicted)
        native = np.asarray(native)
        limits = artifact.tensors["action_limits"]
        targets = np.clip(targets[indices], -limits, limits)
        error = float(np.max(np.abs((native - full[indices]) / limits)))
        if error > 1e-4:
            raise ValueError("Torch persistent and actual Core NumPy inference disagree")
        result = dict(
            index=row["index"],
            head_calls=int(count.sum()),
            OOD_resets=int(reset.sum()),
            sampled_rows=len(indices),
            native_torch_max_normalized_difference=error,
            short16_mse=float(np.mean(((short - targets) / limits) ** 2)),
            persistent_runtime_mse=float(np.mean(((native - targets) / limits) ** 2)),
            short_vs_persistent_mse=float(np.mean(((native - short) / limits) ** 2)),
        )
        results.append(result)
        print(json.dumps(result), flush=True)
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="TORQUE_RECURRENT_CONTEXT_OFFLINE_DIAGNOSED_NOT_PROMOTED",
            records=results,
            input_file_hashes=pins,
            source_head=a.expected_head,
            training_hash=a.training_hash,
            zero_hash=a.zero_hash,
            zero_own_hash=a.own_hash,
            new_physics_steps=0,
            optimizer_updates=0,
            counterfactual_policy_effect_not_proven=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
