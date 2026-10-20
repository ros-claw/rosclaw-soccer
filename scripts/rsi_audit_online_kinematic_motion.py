"""Recompute all216 outcomes and re-execute all12 PPO updates for ONE pipeline.

No new physics or candidate generation. Training re-execution uses the SAME
implementation in a new process, not an independent derivation of PPO math.
"""

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "core-source", "pipeline", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "expected-head",
        "expected-inputs-hash",
        "expected-pipeline-source",
        "pipeline-unit",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, choices=(17, 29, 43), required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_motion_portfolio_common import admit
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    spec, bd, pins, _ = admit(a)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            expected_pipeline_source=a.expected_pipeline_source,
            pipeline_unit=a.pipeline_unit,
            pipeline=str(a.pipeline),
            seed=a.seed,
            new_native_steps=0,
            candidate_optimizer_updates=0,
            planned_reproduced_optimizer_steps=12,
            planned_reviewed_episodes=216,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    try:
        wait_for_terminal_collection(a.pipeline_unit, wait=True)
        d = json.loads((a.pipeline / "declaration.json").read_text())
        c = json.loads((a.pipeline / "complete.json").read_text())
        if (
            d["source_head"] != a.expected_pipeline_source
            or c["source_head"] != a.expected_pipeline_source
            or d["mode"] != "pipeline"
            or d["seed"] != a.seed
            or d["rounds"] != 3
            or c["status"] != "ONLINE_KINEMATIC_PIPELINE_CLOSED"
            or c["optimizer_steps"] != 12
            or c["new_native_steps"] != 1296000
            or len(c["result"]["rounds"]) != 3
            or c["hardware_authorized"] is not False
            or c["promotion_authorized"] is not False
        ):
            raise ValueError("whole three-round producer must close unchanged")
        sources = [
            Path(k).parent.parent
            for k in d["input_file_hashes"]
            if k.endswith("/scripts/rsi_online_kinematic_motion.py")
        ]
        if (
            len(sources) != 1
            or subprocess.check_output(
                ["git", "-C", str(sources[0]), "rev-parse", "HEAD"], text=True
            ).strip()
            != a.expected_pipeline_source
        ):
            raise ValueError("one exact original actor/update implementation required")
        for folder in ("src", "scripts"):
            for old in (sources[0] / folder).rglob("*.py"):
                if digest(old) != digest(Path.cwd() / old.relative_to(sources[0])):
                    raise ValueError("review cannot alter original actor, PPO or physics helpers")
        for k, h in d["input_file_hashes"].items():
            if k in pins and pins[k] != h:
                raise ValueError("conflicting pinned provenance")
            pins[k] = h
        initial_paths = [
            Path(k)
            for k, h in d["input_file_hashes"].items()
            if h == d["behavior_checkpoint_hash"] and k.endswith("/candidate.pt")
        ]
        if len(initial_paths) != 1:
            raise ValueError("one committed initial supervised policy required")
        initial = initial_paths[0]
        check_pins(pins)
        summaries = audit(a, spec, bd, pins, initial, d["behavior_checkpoint_hash"])
        if summaries != c["result"]["rounds"]:
            raise ValueError("root summary differs from reconstructed whole216 evidence")
        for name in ("declaration.json", "complete.json"):
            pins[str(a.pipeline / name)] = digest(a.pipeline / name)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="ONLINE_THREE_ROUNDS_TENSOR_AND_OUTCOME_REPRODUCED",
                source_head=a.expected_head,
                producer_source=a.expected_pipeline_source,
                input_file_hashes=pins,
                seed=a.seed,
                rounds=summaries,
                reviewed_native_steps=1296000,
                new_native_steps=0,
                candidate_optimizer_updates=0,
                reproduced_optimizer_steps=12,
                training_reexecution_uses_same_implementation=True,
                independently_reintegrated_by_this_audit=False,
                raw_outcomes_recomputed=True,
                adam_state_and_rng_tensor_exact=True,
                checkpoint_file_hash_equality_claimed=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        print("ONLINE_FULL_PIPELINE_AUDITED", a.seed, flush=True)
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                no_new_physics=True,
                no_candidate_promotion=True,
                automatic_retry_authorized=False,
            ),
        )
        raise


def audit(a, spec, bd, pins, initial, behavior_hash):
    import mujoco
    import torch

    from rosclaw_soccer.rsi.optimizer_lineage_audit import exact_tensor_tree
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import digest
    from scripts.rsi_online_kinematic_motion import load_model, update_policy

    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True)
    # Keep RNG consumption identical to the recorded parent process.
    policy = load_model(spec, initial)
    policy.parent.requires_grad_(False)
    optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=1e-4)
    world = mujoco.MjModel.from_binary_path(spec["world"])
    args = SimpleNamespace(inputs=a.inputs, policy=initial, expected_head=a.expected_head)
    summaries = []
    for iteration in range(3):
        original = a.pipeline / f"round{iteration}"
        output = a.output / f"round{iteration}"
        output.mkdir()
        collections = read_phase(a, bd, pins, world, original, "collect", iteration, behavior_hash)
        learning = json.loads((original / "learning-complete.json").read_text())
        if learning["behavior_checkpoint_hash"] != behavior_hash:
            raise ValueError("learner consumed stale predecessor")
        checkpoint = original / "candidate.pt"
        if digest(checkpoint) != learning["checkpoint_hash"]:
            raise ValueError("recorded candidate hash differs")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            payload["source_head"] != a.expected_pipeline_source
            or payload["behavior_checkpoint_hash"] != behavior_hash
            or payload["round_index"] != iteration
            or payload["optimizer_steps"] != 4 * (iteration + 1)
        ):
            raise ValueError("checkpoint temporal lineage differs")
        update = update_policy(
            args, policy, optimizer, collections, output, behavior_hash, iteration
        )
        replay = torch.load(output / "candidate.pt", map_location="cpu", weights_only=True)
        for key in ("state_dict", "optimizer", "cpu_rng_state"):
            exact_tensor_tree(replay[key], payload[key], path=key)
        for key in (
            "updates",
            "parameter_changes",
            "batched_vs_step_log_probability_max_error",
            "persistent_adam_update_count",
            "input_file_hashes",
        ):
            exact_tensor_tree(update[key], learning[key], path="learning." + key)
        behavior_before, behavior_hash = behavior_hash, learning["checkpoint_hash"]
        exams = read_phase(a, bd, pins, world, original, "evaluate", iteration, behavior_hash)
        rows = [r for _, c in exams for r in c["result"]["results"]]
        passed = sorted(
            r["course_index"]
            for r in rows
            if r["score"]["qualified"]
            and r["prefix_diagnostics"]["prefix_physical_safe"]
            and r["prefix_diagnostics"]["post_boundary_control_can_cause_first_touch"]
        )
        summary = dict(
            round_index=iteration,
            parent_behavior_checkpoint_hash=behavior_before,
            checkpoint_hash=behavior_hash,
            actual_new_optimizer_steps=4,
            cumulative_optimizer_steps=4 * (iteration + 1),
            success_indices=passed,
            regressions=sorted({0, 10} - set(passed)),
            validation_success=sum(bd["rows"][i]["split"] == "validation" for i in passed),
            body_balance=sum(
                r["score"]["minimum_pelvis_z_m"] >= 0.65 and r["score"]["minimum_upright"] >= 0.9
                for r in rows
            ),
            nonfoot_cases=sum(r["score"]["maximum_nonfoot_force_n"] > 1 for r in rows),
            new_native_steps=432000,
            promotion_authorized=False,
        )
        if json.loads((original / "complete.json").read_text()) != summary:
            raise ValueError("round summary contradicts recomputed physical outcomes")
        for name in ("learning-complete.json", "candidate.pt", "complete.json"):
            pins[str(original / name)] = digest(original / name)
        write_once(
            output / "verification.json",
            dict(
                actual_checkpoint_hash=behavior_hash,
                recomputed_checkpoint_hash=update["checkpoint_hash"],
                numerical_state_adam_rng_bit_exact=True,
                physical_outcomes_recomputed=True,
                reproduced_optimizer_steps=4,
                candidate_optimizer_updates=0,
                new_native_steps=0,
                summary=summary,
            ),
        )
        write_once(a.output / f"progress{iteration}.json", summary)
        summaries.append(summary)
        print("ONLINE_ROUND_NUMERICALLY_REPRODUCED", a.seed, iteration, passed, flush=True)
    return summaries


def read_phase(a, bd, pins, world, original, mode, iteration, weight_hash):
    import numpy as np

    from rosclaw_soccer.rsi.entry_prefix_diagnostics import diagnose_entry_prefix
    from rosclaw_soccer.rsi.full_entry_forecast import score_full_entry
    from scripts.rsi_motion_student_rollout import validate_motion_recording
    from scripts.rsi_native_context_recovery_common import digest
    from scripts.rsi_online_kinematic_motion import episode_noise_seed

    indices = [i for i, r in enumerate(bd["rows"]) if mode == "evaluate" or r["split"] == "train"]
    result = []
    for part in range(4):
        root = original / f"{mode}{part}"
        d = json.loads((root / "declaration.json").read_text())
        c = json.loads((root / "complete.json").read_text())
        if (
            d["source_head"] != a.expected_pipeline_source
            or d["mode"] != mode
            or d["indices"] != indices[part::4]
            or d["behavior_checkpoint_hash"] != weight_hash
            or d["episode_noise_seed_base"] != episode_noise_seed(a.seed, iteration)
            or c["status"] != "CURRENT_KINEMATIC_EPISODES_REPLAY_VERIFIED"
            or c["new_native_steps"] != 6000 * len(d["indices"])
            or c["optimizer_steps"] != 0
            or c["source_head"] != a.expected_pipeline_source
            or [r["course_index"] for r in c["result"]["results"]] != d["indices"]
        ):
            raise ValueError("whole exact current-policy phase required")
        for key, h in d["input_file_hashes"].items():
            if key in pins and pins[key] != h:
                raise ValueError("phase input commitment conflict")
            pins[key] = h
        for name in ("declaration.json", "complete.json"):
            pins[str(root / name)] = digest(root / name)
        for row in c["result"]["results"]:
            i = row["course_index"]
            folder = root / f"course{i:02d}"
            if (
                row["split"] != bd["rows"][i]["split"]
                or row["seed"] != episode_noise_seed(a.seed, iteration) + i
                or json.loads((folder / "complete.json").read_text()) != row
            ):
                raise ValueError("course identity/split/noise differs")
            for name, h in (
                ("native.npz", row["native_hash"]),
                ("sampling.npz", row["sampling_hash"]),
            ):
                if digest(folder / name) != h:
                    raise ValueError("native/sampling bytes differ")
                pins[str(folder / name)] = h
            with np.load(folder / "native.npz", allow_pickle=False) as z:
                trace = {k: z[k].copy() for k in z.files}
            validate_motion_recording(trace, nbody=world.nbody)
            score = score_full_entry(world, {k: v[20:] for k, v in trace.items()})
            prefix = diagnose_entry_prefix(world, {k: v[:20] for k, v in trace.items()})
            if score != row["score"] or prefix != row["prefix_diagnostics"]:
                raise ValueError("raw physical primitives contradict recorded gate")
            proof = row["replay"]
            if (
                row["native_steps"] != 6000
                or proof["additional_native_steps"] != 3000
                or proof["student_neural_and_muscle_native_bit_exact"] is not True
                or row["regenerated_noise_and_prequant_likelihood_bit_exact"] is not True
                or proof["intention_network_calls"] != 600
                or proof["muscle_decoder_calls"] != 280
                or proof["teacher_encoder_calls"] != 0
                or proof["teacher_planner_calls"] != 0
                or proof["hardware_authorized"] is not False
                or proof["promotion_authorized"] is not False
            ):
                raise ValueError("independent per-step neural/noise/muscle/native replay required")
            pins[str(folder / "complete.json")] = digest(folder / "complete.json")
        result.append((root, c))
    return result


if __name__ == "__main__":
    main()
