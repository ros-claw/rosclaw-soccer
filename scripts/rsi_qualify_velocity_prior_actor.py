"""Recorded native-state Torch initialization qualification, not new physics/RL."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probe-root", "policy", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("probe-unit", "probe-hash", "expected-head"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        args.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != args.expected_head
    ):
        raise ValueError("unused output and frozen clean source required")
    wait_for_terminal_collection(args.probe_unit)
    receipt = args.probe_root / "complete.json"
    if digest(receipt) != args.probe_hash:
        raise ValueError("immutable actual native mobility receipt required")
    probe = json.loads(receipt.read_text())
    if (
        probe["status"] != "VELOCITY_PRIOR_NEW_BODY_MOBILITY_PROBE_NOT_FOOTBALL_QUALIFIED"
        or not probe["all_four_independent_full_repeats_exact"]
        or not all(row["command_response_research_gate"] for row in probe["results"])
    ):
        raise ValueError("all original mobility research gates required")
    model_hash = probe["input_file_hashes"][str(args.policy)]
    pins = {str(receipt): args.probe_hash, str(args.policy): model_hash}
    for case in range(4):
        for repeat in range(2):
            folder = args.probe_root / f"case{case}-repeat{repeat}"
            complete = json.loads((folder / "complete.json").read_text())
            pins[str(folder / "complete.json")] = digest(folder / "complete.json")
            pins[str(folder / "native300.npz")] = complete["raw_hash"]
    for directory in (Path.cwd() / "src", Path.cwd() / "scripts"):
        for source in directory.rglob("*.py"):
            pins[str(source)] = digest(source)
    check_pins(pins)
    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            source_head=args.expected_head,
            input_file_hashes=pins,
            recorded_native_observations=1200,
            torch_rows_including_checkpoint_reload=6000,
            reference_suffix_features=70,
            reference_suffixes="zero and deterministic sine, synthetic influence test only",
            maximum_action_error=1e-5,
            independent_actor_instances=2,
            new_physics_steps=0,
            optimizer_steps=0,
            not_reference_conditioned_skill=True,
            hardware_authorized=False,
        ),
    )
    calls = 0
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        recorded = []
        actions = []
        for case in range(4):
            with np.load(
                args.probe_root / f"case{case}-repeat0/native300.npz", allow_pickle=False
            ) as raw:
                recorded.append(raw["observation"].copy())
                actions.append(raw["action"].copy())
        obs = np.concatenate(recorded)
        gold = np.concatenate(actions)
        if obs.shape != (1200, 98) or gold.shape != (1200, 29):
            raise ValueError(
                "all four complete actual native observation/action sequences required"
            )
        suffix = np.sin(np.arange(1200 * 70, dtype=np.float32).reshape(1200, 70) * np.float32(0.01))
        first = None
        maximum = 0.0
        for _repeat in range(2):
            actor = VelocityPriorActor.from_onnx(args.policy, expected_hash=model_hash)
            actor.eval()
            outputs = []
            with torch.no_grad():
                for extra in (np.zeros_like(suffix), suffix):
                    value = actor(torch.from_numpy(np.concatenate((obs, extra), axis=1)))
                    calls += len(obs)
                    outputs.append(value.numpy().copy())
            if not np.array_equal(outputs[0], outputs[1]):
                raise ValueError("zero-initialized reference inputs changed prior output")
            error = float(np.max(np.abs(outputs[0] - gold)))
            maximum = max(maximum, error)
            if not np.isfinite(outputs[0]).all() or error > 1e-5:
                raise ValueError("Torch initialization differs from real native ONNX outputs")
            if first is None:
                first = outputs[0]
                torch.save(
                    dict(
                        actor_state_dict=actor.state_dict(),
                        reference_features=70,
                        velocity_features=98,
                        optimizer_steps=0,
                        original_onnx_hash=model_hash,
                    ),
                    args.output / "prior-actor-initialization.pt",
                )
                restored = torch.load(
                    args.output / "prior-actor-initialization.pt", weights_only=True
                )
                replica = VelocityPriorActor.from_onnx(args.policy, expected_hash=model_hash)
                replica.load_state_dict(restored["actor_state_dict"], strict=True)
                with torch.no_grad():
                    check = replica(torch.from_numpy(np.concatenate((obs, suffix), axis=1))).numpy()
                    calls += len(obs)
                if not np.array_equal(check, first):
                    raise ValueError("safe checkpoint reload changed initialized policy")
            elif not np.array_equal(outputs[0], first):
                raise ValueError("independent actor initialization differs")
        np.savez_compressed(
            args.output / "recorded-state-torch-output.npz", output=first, onnx_action=gold
        )
        check_pins(pins)
        write_once(
            args.output / "complete.json",
            dict(
                status="TORCH_VELOCITY_PRIOR_INITIALIZATION_QUALIFIED_NOT_LEARNED_SKILL",
                source_head=args.expected_head,
                input_file_hashes=pins,
                maximum_action_error=maximum,
                torch_rows_including_checkpoint_reload=calls,
                reference_influence_initially_exact_zero=True,
                independent_actor_repeat_exact=True,
                safe_checkpoint_reload_exact=True,
                checkpoint_hash=digest(args.output / "prior-actor-initialization.pt"),
                output_hash=digest(args.output / "recorded-state-torch-output.npz"),
                optimizer_steps=0,
                new_physics_steps=0,
                hardware_authorized=False,
                promotion_authorized=False,
                football_skill_proven=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            args.output / "failure.json",
            dict(error=repr(error), torch_rows=calls, automatic_retry_authorized=False),
        )
        raise


if __name__ == "__main__":
    main()
