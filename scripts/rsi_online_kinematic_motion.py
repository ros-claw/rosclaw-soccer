"""Three bounded ACTUAL on-policy rounds with persistent actor/critic/Adam.

SIM research only. Every round recollects ALL32 training contexts with current
weights and new explicit noise, independently replays physics, performs at most
four PPO updates, then executes ALL40 development contexts deterministically.
No private Fresh, automatic promotion, threshold adaptation or silent retry.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def episode_noise_seed(seed, round_index):
    if (
        type(seed) is not int
        or seed not in (17, 29, 43)
        or type(round_index) is not int
        or round_index not in range(3)
    ):
        raise ValueError("explicit declared experiment seed and round required")
    return 20261020 + seed * 1000 + round_index * 1000000


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("pipeline", "collect", "evaluate"), required=True)
    for name in ("inputs", "core-source", "policy", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-head", "expected-inputs-hash", "expected-policy-hash"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, choices=(17, 29, 43), required=True)
    p.add_argument("--round-index", type=int, choices=(0, 1, 2), default=0)
    p.add_argument("--indices", nargs="+", type=int)
    p.add_argument("--initial-review", type=Path)
    p.add_argument("--expected-review-hash")
    return p.parse_args()


def main():
    a = arguments()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_motion_portfolio_common import admit
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    spec, bd, pins, _ = admit(a)
    if digest(a.policy) != a.expected_policy_hash:
        raise ValueError("exact current behavior checkpoint required")
    pins[str(a.policy)] = a.expected_policy_hash
    if a.mode == "pipeline":
        if a.indices or a.initial_review is None or not a.expected_review_hash:
            raise ValueError("pipeline requires authenticated complete developmental exam")
        if digest(a.initial_review / "complete.json") != a.expected_review_hash:
            raise ValueError("initial full40 review differs")
        review = json.loads((a.initial_review / "complete.json").read_text())
        if (
            review["status"] != "KINEMATIC_COLUMN_ALL40_AUTHENTICATED"
            or review["candidate_hash"] != a.expected_policy_hash
            or review["courses"] != 40
        ):
            raise ValueError("actual initial candidate identity differs")
        pins.update(review["input_file_hashes"])
        pins[str(a.initial_review / "complete.json")] = a.expected_review_hash
    else:
        if (
            not a.indices
            or len(a.indices) != len(set(a.indices))
            or any(i not in range(40) for i in a.indices)
            or (a.mode == "collect" and any(bd["rows"][i]["split"] != "train" for i in a.indices))
        ):
            raise ValueError("unique declared indices; validation forbidden in collection")
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            schema="soccer.rsi.online_kinematic_motion.v1",
            source_head=a.expected_head,
            mode=a.mode,
            input_file_hashes=pins,
            behavior_checkpoint_hash=a.expected_policy_hash,
            seed=a.seed,
            episode_noise_seed_base=episode_noise_seed(a.seed, a.round_index),
            indices=a.indices,
            rounds=3 if a.mode == "pipeline" else None,
            maximum_ppo_updates_per_round=4,
            learning_rate=1e-4,
            critic_new_column_trainable=True,
            parent_entirely_frozen=True,
            original_32_contexts_repeated_with_new_current_policy_noise=True,
            persistent_adam_between_rounds=True,
            maximum_native_steps=1296000 if a.mode == "pipeline" else 6000 * len(a.indices),
            original_far_or_private_fresh=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    try:
        if a.mode == "pipeline":
            result = pipeline(a, spec, bd, pins)
        else:
            import torch

            from scripts.rsi_motion_ppo_pilot import episodes

            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            model = load_model(spec, a.policy)
            a.world, a.model_root = Path(spec["world"]), Path(spec["model_root"])
            a.seed = episode_noise_seed(a.seed, a.round_index)
            result = episodes(a, model.eval().requires_grad_(False), bd, pins)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="ONLINE_KINEMATIC_PIPELINE_CLOSED"
                if a.mode == "pipeline"
                else "CURRENT_KINEMATIC_EPISODES_REPLAY_VERIFIED",
                source_head=a.expected_head,
                result=result,
                new_native_steps=result["new_native_steps"],
                optimizer_steps=result.get("optimizer_steps", 0),
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                exact_partial_physical_charge_requires_preserved_child_receipt_review=True,
                silent_retry_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


def load_model(spec, path):
    import mujoco
    import torch

    from rosclaw_soccer.rsi.kinematic_motion_residual import create_kinematic_motion_residual
    from rosclaw_soccer.rsi.measured_foot_kinematics import MeasuredFootKinematics
    from scripts.rsi_motion_portfolio_common import load_frozen_experts
    from scripts.rsi_native_context_recovery_common import digest

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload["model_kind"]
        not in ("kinematic_motion_residual64", "online_kinematic_motion_residual64")
        or payload["parent_checkpoint_hash"] != spec["experts"][0]["checkpoint_hash"]
        or payload["world_hash"] != digest(Path(spec["world"]))
    ):
        raise ValueError("bound frozen-parent/geometry model required")
    parent = load_frozen_experts(spec)[0]
    before = {k: v.clone() for k, v in parent.state_dict().items()}
    extractor = MeasuredFootKinematics(
        mujoco.MjModel.from_binary_path(spec["world"]),
        expected_world_hash=digest(Path(spec["world"])),
    )
    model = create_kinematic_motion_residual(parent, extractor)
    model.load_state_dict(payload["state_dict"], strict=True)
    if any(not torch.equal(v, before[k]) for k, v in parent.state_dict().items()):
        raise ValueError("immutable physical knowledge parent changed")
    return model


def pipeline(a, spec, bd, pins):
    import torch

    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins

    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True)
    policy = load_model(spec, a.policy)
    policy.parent.requires_grad_(False)
    optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=1e-4)
    train = [i for i, r in enumerate(bd["rows"]) if r["split"] == "train"]
    if len(train) != 32:
        raise ValueError("ALL32 original train contexts required")
    path, weight_hash, summaries = a.policy, a.expected_policy_hash, []
    for round_index in range(3):
        folder = a.output / f"round{round_index}"
        folder.mkdir()
        collection = run_shards(a, folder, "collect", path, weight_hash, train, round_index)
        update = update_policy(a, policy, optimizer, collection, folder, weight_hash, round_index)
        check_pins(pins)
        path = folder / "candidate.pt"
        weight_hash = update["checkpoint_hash"]
        exam = run_shards(a, folder, "evaluate", path, weight_hash, list(range(40)), round_index)
        passed = sorted(
            r["course_index"]
            for _, c in exam
            for r in c["result"]["results"]
            if r["score"]["qualified"]
            and r["prefix_diagnostics"]["prefix_physical_safe"]
            and r["prefix_diagnostics"]["post_boundary_control_can_cause_first_touch"]
        )
        summary = dict(
            round_index=round_index,
            parent_behavior_checkpoint_hash=update["behavior_checkpoint_hash"],
            checkpoint_hash=weight_hash,
            actual_new_optimizer_steps=4,
            cumulative_optimizer_steps=4 * (round_index + 1),
            success_indices=passed,
            regressions=sorted({0, 10} - set(passed)),
            validation_success=sum(bd["rows"][i]["split"] == "validation" for i in passed),
            body_balance=sum(
                r["score"]["minimum_pelvis_z_m"] >= 0.65 and r["score"]["minimum_upright"] >= 0.9
                for _, c in exam
                for r in c["result"]["results"]
            ),
            nonfoot_cases=sum(
                r["score"]["maximum_nonfoot_force_n"] > 1
                for _, c in exam
                for r in c["result"]["results"]
            ),
            new_native_steps=432000,
            promotion_authorized=False,
        )
        write_once(folder / "complete.json", summary)
        write_once(a.output / f"progress-round{round_index}.json", summary)
        summaries.append(summary)
        print("ONLINE_ROUND_CLOSED", a.seed, summary, flush=True)
    return dict(
        rounds=summaries,
        new_native_steps=1296000,
        optimizer_steps=12,
        final_checkpoint_hash=weight_hash,
        fresh_exam_pass=False,
    )


def run_shards(a, folder, mode, policy, weight_hash, indices, round_index):
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    jobs = []
    try:
        for part in range(4):
            output = folder / f"{mode}{part}"
            command = [
                sys.executable,
                "-u",
                "scripts/rsi_online_kinematic_motion.py",
                "--mode",
                mode,
                "--inputs",
                str(a.inputs),
                "--expected-inputs-hash",
                a.expected_inputs_hash,
                "--core-source",
                str(a.core_source),
                "--expected-head",
                a.expected_head,
                "--policy",
                str(policy),
                "--expected-policy-hash",
                weight_hash,
                "--seed",
                str(a.seed),
                "--round-index",
                str(round_index),
                "--indices",
                *[str(i) for i in indices[part::4]],
                "--output",
                str(output),
            ]
            # Independent exec processes, never multiprocessing/spawned Python state.
            log = (folder / f"{mode}{part}.log").open("xb")
            try:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            finally:
                log.close()
            jobs.append((process, output))
        outcomes = []
        for process, output in jobs:
            if process.wait(timeout=3600) != 0:
                raise ValueError("declared child failed; preserve all siblings and do not retry")
            d = json.loads((output / "declaration.json").read_text())
            c = json.loads((output / "complete.json").read_text())
            if (
                d["source_head"] != a.expected_head
                or d["behavior_checkpoint_hash"] != weight_hash
                or d["mode"] != mode
                or c["status"] != "CURRENT_KINEMATIC_EPISODES_REPLAY_VERIFIED"
                or c["new_native_steps"] != 6000 * len(d["indices"])
            ):
                raise ValueError("actual current-policy child identity/accounting differs")
            check_pins(d["input_file_hashes"])
            for r in c["result"]["results"]:
                i = r["course_index"]
                part = output / f"course{i:02d}"
                if (
                    digest(part / "native.npz") != r["native_hash"]
                    or digest(part / "sampling.npz") != r["sampling_hash"]
                    or r["native_steps"] != 6000
                    or not r["regenerated_noise_and_prequant_likelihood_bit_exact"]
                    or not r["replay"]["student_neural_and_muscle_native_bit_exact"]
                ):
                    raise ValueError("entire real current-policy episode/replay required")
            outcomes.append((output, c))
        actual = [r["course_index"] for _, c in outcomes for r in c["result"]["results"]]
        if sorted(actual) != sorted(indices):
            raise ValueError(
                "whole declared context set required; no duplicate or dropped failures"
            )
        return outcomes
    finally:
        # Do not leave unaccounted child physics running after a sibling failure.
        # Waiting does not retry, kill, replace or promote any experiment.
        for process, _ in jobs:
            if process.poll() is None:
                process.wait(timeout=3600)


def update_policy(a, policy, optimizer, collection, folder, behavior_hash, round_index):
    import numpy as np
    import torch

    from rosclaw_soccer.rsi.motion_ppo_math import clipped_ppo_loss, full_horizon_gae
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import digest

    fields, experience, seen = [], {}, set()
    for root, complete in collection:
        for r in complete["result"]["results"]:
            if r["split"] != "train":
                raise ValueError("validation cannot enter online gradients")
            if r["course_index"] in seen:
                raise ValueError("duplicate context in online update")
            seen.add(r["course_index"])
            part = root / f"course{r['course_index']:02d}"
            if (
                digest(part / "native.npz") != r["native_hash"]
                or digest(part / "sampling.npz") != r["sampling_hash"]
            ):
                raise ValueError("on-policy input changed before optimizer admission")
            with np.load(part / "native.npz", allow_pickle=False) as z:
                x, value = z["student_features"].copy(), z["value_estimate"].copy()
            with np.load(part / "sampling.npz", allow_pickle=False) as z:
                raw, lp, reward, mean, std = [
                    z[k].copy() for k in ("raw", "log_prob", "rewards", "mean", "std")
                ]
            fields.append((x, raw, lp, value, reward, mean, std))
            experience[str(part / "native.npz")] = digest(part / "native.npz")
            experience[str(part / "sampling.npz")] = digest(part / "sampling.npz")
    if len(seen) != 32:
        raise ValueError("all32 actual training contexts required before gradients")
    x, raw, lp, old_value, reward, mean, std = [
        torch.from_numpy(np.stack([r[k] for r in fields])) for k in range(7)
    ]
    policy.eval()
    with torch.no_grad():
        recomputed, _, _, old_mean, old_std = policy.evaluate_raw(x, raw)
    # Batched GRU/matmul rounding differs from single-step inference. The latter
    # already has exact independently replayed NN/physics and checkpoint hashes.
    torch.testing.assert_close(old_mean, mean, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(old_std, std, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(recomputed, lp, atol=0.005, rtol=0)
    likelihood_error = float((recomputed - lp).abs().max())
    lp = recomputed.detach()  # same batched recurrence convention; initial ratio=1
    adv, returns = full_horizon_gae(reward.numpy(), old_value.numpy())
    # Reconstruct a frozen reference without deep-copying MuJoCo C++ scratch.
    reference = load_model(json.loads(a.inputs.read_text()), a.policy)
    reference.load_state_dict(policy.state_dict(), strict=True)
    reference.eval().requires_grad_(False)
    before = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    history = []
    policy.train()
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss, stats = clipped_ppo_loss(
            policy,
            reference,
            x,
            raw,
            lp,
            old_value,
            torch.from_numpy(adv),
            torch.from_numpy(returns),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        history.append(
            dict(
                step=step + 1,
                loss=float(loss.detach()),
                **{k: float(v.detach()) for k, v in stats.items()},
            )
        )
    changes = {
        k: float((v.detach() - before[k]).double().square().sum())
        for k, v in policy.state_dict().items()
    }
    if (
        any(v != 0 for k, v in changes.items() if k.startswith(("parent.", "geometry_")))
        or not any(v > 0 for k, v in changes.items() if k.startswith("adapter."))
        or not any(v > 0 for k, v in changes.items() if k.startswith("value_adapter."))
    ):
        raise ValueError("actual new actor AND critic update, exact frozen knowledge required")
    policy.eval()
    from scripts.rsi_native_context_recovery_common import check_pins

    check_pins(experience)
    torch.save(
        dict(
            state_dict=policy.state_dict(),
            optimizer=optimizer.state_dict(),
            source_head=a.expected_head,
            model_kind="online_kinematic_motion_residual64",
            parent_checkpoint_hash=json.loads(a.inputs.read_text())["experts"][0][
                "checkpoint_hash"
            ],
            world_hash=digest(Path(json.loads(a.inputs.read_text())["world"])),
            behavior_checkpoint_hash=behavior_hash,
            round_index=round_index,
            optimizer_steps=4 * (round_index + 1),
            cpu_rng_state=torch.get_rng_state(),
            hardware_authorized=False,
            promotion_authorized=False,
        ),
        folder / "candidate.pt",
    )
    result = dict(
        behavior_checkpoint_hash=behavior_hash,
        checkpoint_hash=digest(folder / "candidate.pt"),
        input_file_hashes=experience,
        training_episodes=32,
        unique_training_contexts=32,
        validation_gradient_episodes=0,
        updates=history,
        parameter_changes=changes,
        batched_vs_step_log_probability_max_error=likelihood_error,
        persistent_adam_update_count=4 * (round_index + 1),
        new_native_steps=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    write_once(folder / "learning-complete.json", result)
    return result


if __name__ == "__main__":
    main()
