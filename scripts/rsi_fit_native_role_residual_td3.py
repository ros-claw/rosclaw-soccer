"""One actual TD3 update stage on newly measured native residual experience.

No body/gait/encoder updates. Old absolute-policy critics are never loaded.
The produced proposal must still pass complete fresh physical evaluation.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("collection", "initialization", "core-source-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("collection-hash", "initialization-hash", "core-expected-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, choices=(926061, 926062, 926063), required=True)
    a = p.parse_args()
    from scripts.rsi_active_core_source import pin_active_core
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused TD3 fit output required")
    loaded, pins = {}, {}
    for key in ("collection", "initialization"):
        path = getattr(a, key) / "complete.json"
        expected = getattr(a, key + "_hash")
        if path.is_symlink() or digest(path) != expected:
            raise ValueError("exact complete collection and initialization required")
        report = json.loads(path.read_text())
        if report["seed"] != a.seed:
            raise ValueError("own seed's retained encoder and experience required")
        for k, v in report["input_file_hashes"].items():
            if k in pins and pins[k] != v:
                raise ValueError("conflicting learning ancestry")
            pins[k] = v
        pins[str(path)] = expected
        loaded[key] = report
    source, init = loaded["collection"], loaded["initialization"]
    if (
        source["status"] != "NATIVE_ROLE_RESIDUAL_EXPLORATION_MEASURED_NOT_TRAINED"
        or source["initialization_hash"] != a.initialization_hash
        or len(source["results"]) != 5
        or source["optimizer_updates"] != 0
        or init["optimizer_updates"] != 0
    ):
        raise ValueError("entire5-course new interaction bank and zero initial head required")
    datasets = []
    for case in source["results"]:
        path = Path(case["path"])
        if (
            path.parent.parent.resolve() != a.collection.resolve()
            or path.is_symlink()
            or digest(path) != case["hash"]
        ):
            raise ValueError("exact confined course collection required")
        report = json.loads(path.read_text())
        if len(report["records"]) != 4:
            raise ValueError("all4 exploration trials required")
        for k, v in report["input_file_hashes"].items():
            if k in pins and pins[k] != v:
                raise ValueError("conflicting physical source hashes")
            pins[k] = v
        pins[str(path)] = case["hash"]
        data_path = path.parent / report["dataset"]["path"]
        if (
            data_path.parent.resolve() != path.parent.resolve()
            or data_path.is_symlink()
            or digest(data_path) != report["dataset"]["hash"]
        ):
            raise ValueError("exact ordinary course transition arrays required")
        pins[str(data_path)] = report["dataset"]["hash"]
        datasets.append((case["course"], data_path))
    for item in (init["encoder"], init["policy"]):
        path = a.initialization / item["path"]
        if (
            path.parent.resolve() != a.initialization.resolve()
            or path.is_symlink()
            or digest(path) != item["hash"]
        ):
            raise ValueError("exact retained encoder/head required")
        pins[str(path)] = item["hash"]
    pin_active_core(a.core_source_root, a.core_expected_head, pins)
    source_pins(pins)
    check_pins(pins)
    import numpy as np
    import torch
    from rosclaw.growth.parent_residual_td3 import ParentResidualTD3Config, fit_parent_residual_td3

    from rosclaw_soccer.providers.g1 import velocity_policy
    from rosclaw_soccer.rsi import (
        native_role_command,
        native_role_residual,
        native_role_residual_data,
    )
    from rosclaw_soccer.rsi.native_role_command import NativeRoleCommandPolicy
    from rosclaw_soccer.rsi.native_role_residual import (
        NativeRoleResidualPolicy,
        serialize_role_residual,
    )
    from rosclaw_soccer.rsi.native_role_residual_task import residual_reward_hash
    from rosclaw_soccer.sim.contracts import hash_json

    torch.set_num_threads(1)
    if source["reward_contract_hash"] != residual_reward_hash():
        raise ValueError("reward contract changed after exploration")
    encoder = NativeRoleCommandPolicy(
        a.initialization / init["encoder"]["path"],
        expected_hash=init["encoder"]["hash"],
        expected_binding=init["binding_hash"],
    )
    initial = NativeRoleResidualPolicy(
        a.initialization / init["policy"]["path"],
        expected_hash=init["policy"]["hash"],
        encoder=encoder,
        expected_encoder_hash=encoder.artifact_hash,
        expected_binding=init["binding_hash"],
        expected_parent_hash=init["parent_hash"],
    )
    arrays = {}
    for course, path in datasets:
        with np.load(path) as data:
            for key in data.files:
                arrays.setdefault(key, []).append(data[key])
            if not np.all(
                data["training"]
                == (course in ("outbound-north", "incoming-north", "incoming-east"))
            ):
                raise ValueError("fixed public train/mirror-validation partition changed")
    bank = {k: np.concatenate(v) for k, v in arrays.items()}
    mask = bank["training"]
    if (
        mask.dtype != np.bool_
        or not mask.any()
        or mask.all()
        or bank["context"].shape[1] != 152
        or bank["features"].shape[1] != 131
    ):
        raise ValueError("complete distinct training and mirror-validation rows required")
    # Retain the encoder's train-only normalization; privileged gait memory and
    # normalized parent entries use unit scale. No validation-derived statistics.
    mean = np.concatenate((encoder.arrays["mean"], np.zeros(34, np.float32)))
    std = np.concatenate((encoder.arrays["std"], np.ones(34, np.float32)))
    context = np.clip((bank["context"] - mean) / std, -10, 10).astype(np.float32)
    next_context = np.clip((bank["next_context"] - mean) / std, -10, 10).astype(np.float32)
    context_hash = hash_json(
        dict(
            schema="soccer.native_role_residual_td3_context.v1",
            binding_hash=init["binding_hash"],
            encoder_hash=encoder.artifact_hash,
            parent_hash=init["parent_hash"],
            maximum_ratio=init["maximum_ratio"],
            reward_contract_hash=residual_reward_hash(),
            control_hz=50,
            actor_context_dim=121,
            actor_feature_dim=131,
            critic_context_dim=152,
            action_dim=3,
            critic_memory="own_prior_neural_action29_unmasked_phase2_no_current_command",
            action_semantics="NORMALIZED_RESIDUAL_BEFORE_ENVIRONMENT_GUARD",
            sampler_truncations_bootstrap_actual_next_state=True,
            guard_implementation_hash=digest(Path(native_role_residual.__file__)),
            gait_codec_hash=digest(Path(velocity_policy.__file__)),
            encoder_implementation_hash=digest(Path(native_role_command.__file__)),
            transition_extractor_hash=digest(Path(native_role_residual_data.__file__)),
            critic_normalization="retained_encoder118_unit_scale34_clip10_v1",
        )
    )
    result = fit_parent_residual_td3(
        context=context[mask],
        features=bank["features"][mask],
        actions=bank["actions"][mask],
        rewards=bank["rewards"][mask],
        next_context=next_context[mask],
        next_features=bank["next_features"][mask],
        terminated=bank["terminated"][mask],
        actor_weight=initial.tensors["head_weight"],
        actor_bias=initial.tensors["head_bias"],
        context_hash=context_hash,
        config=ParentResidualTD3Config(seed=a.seed),
    )
    a.output.mkdir(parents=True)
    data_path = a.output / "measured-replay-bank.npz"
    np.savez_compressed(data_path, **bank)
    policy = serialize_role_residual(
        encoder_hash=encoder.artifact_hash,
        binding_hash=init["binding_hash"],
        parent_hash=init["parent_hash"],
        dataset_hash=digest(data_path),
        update_index=result["report"]["total_actor_optimizer_updates"],
        maximum_ratio=init["maximum_ratio"],
        head_weight=result["actor_weight"],
        head_bias=result["actor_bias"],
    )
    write_once(a.output / "residual-head.json", policy)
    checkpoint_path = a.output / "td3-checkpoint.pt"
    torch.save(result["checkpoint"], checkpoint_path)
    # Preserve a strict portable-inference comparison after actual gradients.
    fitted = NativeRoleResidualPolicy(
        a.output / "residual-head.json",
        expected_hash=digest(a.output / "residual-head.json"),
        encoder=encoder,
        expected_encoder_hash=encoder.artifact_hash,
        expected_binding=init["binding_hash"],
        expected_parent_hash=init["parent_hash"],
    )
    with torch.no_grad():
        expected = torch.tanh(
            torch.from_numpy(bank["features"][:64]) @ torch.from_numpy(result["actor_weight"]).T
            + torch.from_numpy(result["actor_bias"])
        ).numpy()
    actual = np.stack(
        [
            fitted.propose(state[:118], parent).normalized_action
            for state, parent in zip(
                bank["context"][:64], bank["parent_commands"][:64], strict=True
            )
        ]
    )
    maximum_error = float(np.max(abs(actual - expected)))
    if maximum_error > 2e-6:
        raise ValueError("portable residual mean differs from fitted Torch actor")
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="NATIVE_ROLE_RESIDUAL_TD3_TRAINED_NOT_PHYSICALLY_EVALUATED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            seed=a.seed,
            training_hash=init["training_hash"],
            initialization_hash=a.initialization_hash,
            collection_hash=a.collection_hash,
            binding_hash=init["binding_hash"],
            parent_hash=init["parent_hash"],
            encoder_hash=encoder.artifact_hash,
            context_hash=context_hash,
            policy=dict(path="residual-head.json", hash=digest(a.output / "residual-head.json")),
            checkpoint=dict(path=checkpoint_path.name, hash=digest(checkpoint_path)),
            dataset=dict(path=data_path.name, hash=digest(data_path)),
            training_rows=int(mask.sum()),
            mirror_validation_rows=int((~mask).sum()),
            unique_initial_geometries=5,
            mirror_validation_is_not_independent_generalization=True,
            optimizer_updates=result["report"]["total_actor_optimizer_updates"],
            numeric_report=result["report"],
            maximum_torch_numpy_error_64_examples=maximum_error,
            raw_action_semantics="NORMALIZED_RESIDUAL_BEFORE_ENVIRONMENT_GUARD",
            original_gait_and_encoder_unchanged=True,
            new_joint_torque_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
