"""Pinned SIM-only on-policy collection, recurrent PPO update and deterministic exam.

This is a bounded research stage, never an automatic policy promotion. All
native episodes are independently re-executed before their data are admitted.
"""

import argparse
import json
import subprocess
from pathlib import Path


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("collect", "learn", "evaluate"))
    for key in ("bank", "train-root", "world", "model-root", "core-source", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    for key in ("expected-head", "expected-bank-hash", "expected-training-hash"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--indices", type=int, nargs="+")
    p.add_argument("--seed", type=int, default=92029)
    p.add_argument("--collection", action="append", type=Path, default=[])
    p.add_argument("--collection-unit", action="append", default=[])
    p.add_argument("--candidate", type=Path)
    p.add_argument("--expected-candidate-hash")
    p.add_argument("--candidate-unit")
    p.add_argument("--expected-collection-source")
    p.add_argument(
        "--credit-assignment", choices=("per_tick", "contact_return"), default="per_tick"
    )
    p.add_argument("--quantized-likelihood", action="store_true")
    return p.parse_args()


def main():
    a = arguments()
    if a.mode != "learn" and (
        a.credit_assignment != "per_tick"
        or a.expected_collection_source is not None
        or a.quantized_likelihood
    ):
        raise ValueError("credit relabeling and collection-source override are learner-only")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if (
        head != a.expected_head
        or a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
    ):
        raise ValueError("clean frozen source and unused output required before allocation")
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        digest(a.bank / "complete.json") != a.expected_bank_hash
        or digest(a.train_root / "complete.json") != a.expected_training_hash
    ):
        raise ValueError("declared bank and base training identity required")
    bd = json.loads((a.bank / "declaration.json").read_text())
    td = json.loads((a.train_root / "declaration.json").read_text())
    tc = json.loads((a.train_root / "complete.json").read_text())
    if (
        tc["model_kind"] != "motion_intention_fsq64"
        or tc["optimizer_steps"] != 3000
        or td["seed"] != 29
        or len(bd["rows"]) != 40
    ):
        raise ValueError("fixed seed29 supervised developmental baseline required")
    pins = {**bd["input_file_hashes"], **td["input_file_hashes"]}
    for root in (a.bank, a.train_root):
        pins.update({str(f): digest(f) for f in root.iterdir() if f.is_file()})
    for root in (Path.cwd() / "src", Path.cwd() / "scripts", a.core_source):
        pins.update({str(f): digest(f) for f in root.rglob("*.py")})
    if pins.get(str(a.world)) != digest(a.world):
        raise ValueError("original pinned native world required")
    for name in (
        "planner_sonic.onnx",
        "low_latency/model_encoder.onnx",
        "low_latency/model_decoder.onnx",
        "low_latency/observation_config.yaml",
    ):
        if pins.get(str(a.model_root / name)) != digest(a.model_root / name):
            raise ValueError("original muscle assets required")
    check_pins(pins)
    if a.candidate_unit:
        from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection

        if a.mode != "evaluate" or a.candidate is None:
            raise ValueError("closed learner dependency only supported for candidate exams")
        wait_for_terminal_collection(a.candidate_unit, wait=True)
        root = a.candidate.parent
        receipt = json.loads((root / "complete.json").read_text())
        learner = json.loads((root / "declaration.json").read_text())
        admitted = {
            "NATIVE_EXPERIENCE_PPO_UPDATED_REQUIRES_EXAM": ("learn", 4),
            "MOTION_MEMORY_CONSOLIDATED_REQUIRES_EXAM": ("consolidate", 3000),
        }.get(receipt.get("status"))
        if (
            admitted is None
            or (learner["mode"], receipt["optimizer_steps"]) != admitted
            or receipt["source_head"] != head
            or learner["base_weights_hash"] != tc["checkpoint_hash"]
            or learner["bank_hash"] != a.expected_bank_hash
        ):
            raise ValueError("closed same-source learner receipt differs")
        if a.expected_candidate_hash not in (None, receipt["checkpoint_hash"]):
            raise ValueError("explicit candidate commitment differs from learner")
        a.expected_candidate_hash = receipt["checkpoint_hash"]
        for name in ("complete.json", "declaration.json", "experience-commitment.json"):
            pins[str(root / name)] = digest(root / name)
    indices = a.indices or []
    if a.mode != "learn" and (
        not indices
        or len(indices) != len(set(indices))
        or any(i not in range(40) for i in indices)
        or (a.mode == "collect" and any(bd["rows"][i]["split"] != "train" for i in indices))
    ):
        raise ValueError("declared distinct admitted courses; no validation gradients")
    base_path = a.train_root / tc["checkpoint"]
    if digest(base_path) != tc["checkpoint_hash"]:
        raise ValueError("base model hash differs")
    if a.candidate is not None:
        if a.mode != "evaluate" or digest(a.candidate) != a.expected_candidate_hash:
            raise ValueError("explicit frozen candidate allowed only for deterministic exam")
        pins[str(a.candidate)] = a.expected_candidate_hash
    check_pins(pins)
    a.output.mkdir(parents=True, exist_ok=False)
    declaration = dict(
        schema="soccer.rsi.motion_ppo_pilot.v1",
        mode=a.mode,
        source_head=head,
        input_file_hashes=pins,
        bank_hash=a.expected_bank_hash,
        base_training_hash=a.expected_training_hash,
        base_weights_hash=tc["checkpoint_hash"],
        indices=indices,
        seed=a.seed,
        candidate_hash=a.expected_candidate_hash,
        candidate_unit=a.candidate_unit,
        expected_collection_source=a.expected_collection_source or head,
        credit_assignment=a.credit_assignment,
        quantized_likelihood=a.quantized_likelihood,
        collection=[str(r) for r in a.collection],
        collection_units=a.collection_unit,
        prefix_frames=20,
        horizon_frames=300,
        post_contact_required_frames=60,
        frozen_muscle=True,
        frozen_encoder_and_memory=True,
        foundation_prefix_neural_math_replayed=False,
        seed29_selected_posthoc_on_development_results=True,
        reward_is_narrow_foot_only_protocol_not_general_football_legality=True,
        training_not_publicity_or_promotion=True,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    write_once(a.output / "declaration.json", declaration)
    if a.mode == "learn":
        from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection

        if not a.collection or len(a.collection) != len(a.collection_unit):
            raise ValueError("explicit paired producer units required")
        # Wait before importing Torch/allocating models, not just before gradients.
        for unit in a.collection_unit:
            wait_for_terminal_collection(unit, wait=True)
    import torch

    from rosclaw_soccer.rsi.motion_actor_critic import create_motion_actor_critic

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(a.seed)
    policy = create_motion_actor_critic().eval()
    payload = torch.load(base_path, map_location="cpu", weights_only=True)
    policy.base.load_state_dict(payload["state_dict"], strict=True)
    if a.candidate is not None:
        cp = torch.load(a.candidate, map_location="cpu", weights_only=True)
        if (
            cp["base_weights_hash"] != tc["checkpoint_hash"]
            or cp["source_head"] != head
            or cp["model_kind"]
            not in ("motion_intention_ppo64", "motion_intention_consolidation64")
        ):
            raise ValueError("candidate provenance differs")
        policy.load_state_dict(cp["state_dict"], strict=True)
    result = learn(a, policy, declaration) if a.mode == "learn" else episodes(a, policy, bd, pins)
    check_pins(pins)
    result.update(source_head=head, hardware_authorized=False, promotion_authorized=False)
    write_once(a.output / "complete.json", result)
    print(
        "PPO_STAGE_COMPLETE",
        a.mode,
        json.dumps({k: result[k] for k in ("status", "new_native_steps")}),
        flush=True,
    )


def episodes(a, policy, bd, pins):
    import mujoco
    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.providers.g1.sonic_navigation import (
        G1SonicNavigation,
        SonicNavigationConfig,
    )
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupConfig, G1SonicRunupController
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.entry_prefix_diagnostics import diagnose_entry_prefix
    from rosclaw_soccer.rsi.foundation_motor_codec import FoundationMotorCodec
    from rosclaw_soccer.rsi.full_entry_forecast import nominal_prefix20, score_full_entry
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_motion_ppo_rollout import RewardedBody, SamplingPolicy, checked_private_gap
    from scripts.rsi_motion_student_rollout import replay_motion_student, rollout_motion_student
    from scripts.rsi_motor_student_rollout import exact_array
    from scripts.rsi_native_context_recovery_common import digest

    world = mujoco.MjModel.from_binary_path(str(a.world))
    backend = G1SonicRunupController(
        a.model_root, G1SonicRunupConfig(model_variant="low_latency"), inference_threads=1
    )
    priors = [Path(k) for k in pins if Path(k).name == "entry-affordance.json"]
    if len(priors) != 1:
        raise ValueError("one original prefix prior required")
    stored = json.loads(priors[0].read_text())
    prior = SkillEntryAffordance(
        tuple(stored["ball_offset_xy_m"]),
        stored["historical_frame"],
        stored["historical_forward_speed_mps"],
        stored["historical_artifact_hash"],
    )
    if prior.contract_hash != stored["contract_hash"]:
        raise ValueError("prefix prior changed")
    results = []
    for index in a.indices:
        record = bd["rows"][index]
        folder = a.output / f"course{index:02d}"
        folder.mkdir()

        def body(record=record):
            return RewardedBody(
                G1TargetEpisode(
                    world,
                    world_hash=digest(a.world),
                    kp=backend.kp.copy(),
                    kd=backend.kd.copy(),
                    course=tuple(record["course"]["course"]),
                    defaults=G1SonicRunupController.default_angles.copy(),
                    record_all_ball_contacts=True,
                )
            )

        actual = body()
        checked_private_gap(actual.observer, actual.body)
        nav = G1SonicNavigation(
            a.model_root,
            record["course"]["agent_id"],
            SonicNavigationConfig(
                maximum_frames=300,
                model_variant="low_latency",
                inference_threads=1,
                record_default_planner_events=True,
            ),
        )
        prefix = nominal_prefix20(nav, actual.body, prior)
        del nav
        raw = Path(record["source_artifact"])
        if digest(raw) != record["source_artifact_hash"]:
            raise ValueError("original prefix recording changed")
        with np.load(raw, allow_pickle=False) as old:
            for k, v in prefix.items():
                exact_array(v, old[k][:20])
        seed = a.seed + index
        sampling = SamplingPolicy(policy, seed=seed, stochastic=a.mode == "collect")
        progress = {}
        try:
            trace = rollout_motion_student(
                sampling, FoundationMotorCodec(backend), actual, prefix, progress=progress
            )
        except Exception:
            rows = progress.get("rows", [])
            if rows:
                np.savez_compressed(
                    folder / "partial.npz",
                    **{k: np.concatenate([r[k] for r in rows]) for k in rows[0]},
                )
            write_once(
                folder / "failure.json",
                dict(native_steps=actual.integrated_steps, attempted_steps=actual.attempted_steps),
            )
            raise
        diagnostics = diagnose_entry_prefix(world, {k: v[:20] for k, v in trace.items()})
        score = score_full_entry(world, {k: v[20:] for k, v in trace.items()})
        if not diagnostics["prefix_physical_safe"]:
            raise ValueError("unsafe fixed prefix invalidates pilot data")
        rewards = actual.finish(score, diagnostics)
        sampled = sampling.recording()
        np.savez_compressed(folder / "native.npz", **trace)
        np.savez_compressed(folder / "sampling.npz", **sampled, rewards=rewards)
        # New body/history/RNG; no recorded post-prefix motion action is consumed.
        independent = body()
        again = SamplingPolicy(policy, seed=seed, stochastic=a.mode == "collect")
        proof = replay_motion_student(again, FoundationMotorCodec(backend), independent, trace)
        for k, v in again.recording().items():
            exact_array(v, sampled[k])
        exact_array(independent.finish(score, diagnostics), rewards)
        if independent.terms != actual.terms or actual.integrated_steps != 3000:
            raise ValueError("causal reward/native horizon differs")
        row = dict(
            course_index=index,
            split=record["split"],
            seed=seed,
            score=score,
            prefix_diagnostics=diagnostics,
            sum_reward=float(rewards.sum(dtype=np.float64)),
            reward_terms=actual.terms,
            native_hash=digest(folder / "native.npz"),
            sampling_hash=digest(folder / "sampling.npz"),
            native_steps=actual.integrated_steps + independent.integrated_steps,
            regenerated_noise_and_prequant_likelihood_bit_exact=True,
            replay=proof,
        )
        write_once(folder / "complete.json", row)
        results.append(row)
        print(
            "PPO_NATIVE_REPLAY_VERIFIED",
            a.mode,
            index,
            score["qualified"],
            row["sum_reward"],
            flush=True,
        )
    return dict(
        status="ALL_DECLARED_EPISODES_AND_NOISE_REPLAY_VERIFIED",
        results=results,
        optimizer_steps=0,
        new_native_steps=sum(r["native_steps"] for r in results),
        qualified=sum(r["score"]["qualified"] for r in results),
    )


def learn(a, policy, declaration):
    import copy

    import numpy as np
    import torch

    from rosclaw_soccer.rsi.motion_ppo_math import clipped_ppo_loss, full_horizon_gae
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if not a.collection or len(a.collection) != len(a.collection_unit):
        raise ValueError("explicit paired producer units required")
    bank = json.loads((a.bank / "declaration.json").read_text())
    sequences, all_indices, data_pins = [], [], {}
    for root, unit in zip(a.collection, a.collection_unit, strict=True):
        wait_for_terminal_collection(unit, wait=True)
        d = json.loads((root / "declaration.json").read_text())
        c = json.loads((root / "complete.json").read_text())
        if (
            d["source_head"] != (a.expected_collection_source or a.expected_head)
            or d["mode"] != "collect"
            or d["bank_hash"] != a.expected_bank_hash
            or d["base_weights_hash"] != declaration["base_weights_hash"]
            or c["status"] != "ALL_DECLARED_EPISODES_AND_NOISE_REPLAY_VERIFIED"
            or [r["course_index"] for r in c["results"]] != d["indices"]
        ):
            raise ValueError("same policy/source and fully replayed collection required")
        check_pins(d["input_file_hashes"])
        # New learner/relabeling code may consume old on-policy experience only
        # if every pre-existing actor/physics/helper stays byte-identical.
        sources = [
            Path(k).parent.parent
            for k in d["input_file_hashes"]
            if k.endswith("/scripts/rsi_motion_ppo_pilot.py")
        ]
        if (
            len(sources) != 1
            or subprocess.check_output(
                ["git", "-C", str(sources[0]), "rev-parse", "HEAD"], text=True
            ).strip()
            != d["source_head"]
        ):
            raise ValueError("one exact frozen on-policy producer source required")
        for folder in ("src", "scripts"):
            for original in (sources[0] / folder).rglob("*.py"):
                relative = original.relative_to(sources[0])
                if str(relative) not in (
                    "scripts/rsi_motion_ppo_pilot.py",
                    "src/rosclaw_soccer/rsi/motion_ppo_math.py",
                ) and digest(Path.cwd() / relative) != digest(original):
                    raise ValueError("on-policy actor/physics/helper changed: " + str(relative))
        data_pins.update({str(f): digest(f) for f in root.rglob("*") if f.is_file()})
        for row in c["results"]:
            i = row["course_index"]
            if bank["rows"][i]["split"] != "train" or row["split"] != "train":
                raise ValueError("validation experience cannot enter optimizer")
            folder = root / f"course{i:02d}"
            if (
                digest(folder / "native.npz") != row["native_hash"]
                or digest(folder / "sampling.npz") != row["sampling_hash"]
                or row["native_steps"] != 6000
                or not row["regenerated_noise_and_prequant_likelihood_bit_exact"]
            ):
                raise ValueError("closed trajectory commitment differs")
            with np.load(folder / "native.npz", allow_pickle=False) as z:
                x, values = z["student_features"].copy(), z["value_estimate"].copy()
                actual_token = z["student_motion_token"].copy()
            with np.load(folder / "sampling.npz", allow_pickle=False) as z:
                raw, lp, rewards = (z[k].copy() for k in ("raw", "log_prob", "rewards"))
            # NumPy tanh can differ from Torch at a last-bit rounding boundary;
            # compare using the sampler's exact Torch arithmetic instead.
            reconstructed = (
                torch.round(
                    15.515501022338867 * torch.tanh(torch.from_numpy(raw) + 0.03223699703812599)
                    - 0.5
                )
                / 16
            ).numpy()
            if not np.array_equal(reconstructed, actual_token):
                raise ValueError("recorded raw sample does not produce consumed FSQ action")
            if a.credit_assignment == "contact_return":
                from rosclaw_soccer.rsi.contact_credit import contact_return_credit, credit_audit

                before_rewards = rewards.copy()
                frame = row["score"]["first_foot_contact_global_frame"]
                rewards = contact_return_credit(
                    rewards,
                    row["reward_terms"],
                    first_contact_frame=frame,
                    qualified=row["score"]["qualified"],
                    causal=row["prefix_diagnostics"]["post_boundary_control_can_cause_first_touch"],
                )
                write_once(
                    a.output / f"credit-course{i:02d}.json",
                    credit_audit(before_rewards, rewards, frame),
                )
            sequences.append((x, raw, lp, values, rewards))
            all_indices.append(i)
    if len(all_indices) != len(set(all_indices)):
        raise ValueError("pilot must not duplicate courses across declared shards")
    write_once(
        a.output / "experience-commitment.json",
        dict(
            input_file_hashes=data_pins,
            training_indices=all_indices,
            replayed_physical_training_episodes=len(all_indices),
            validation_episodes_in_gradients=0,
        ),
    )
    fields = [torch.from_numpy(np.stack([s[k] for s in sequences])) for k in range(5)]
    x, raw, lp, old_values, rewards = fields
    adv, returns = full_horizon_gae(rewards.numpy(), old_values.numpy())
    reference = copy.deepcopy(policy).eval().requires_grad_(False)
    before = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=1e-4)
    history = []
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss, stats = clipped_ppo_loss(
            policy,
            reference,
            x,
            raw,
            lp,
            old_values,
            torch.from_numpy(adv),
            torch.from_numpy(returns),
            quantized_likelihood=a.quantized_likelihood,
        )
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        row = dict(
            step=step + 1,
            loss=float(loss.detach()),
            gradient_norm=float(norm),
            **{k: float(v.detach()) for k, v in stats.items()},
        )
        history.append(row)
        write_once(a.output / f"update{step + 1}.json", row)
    for name, value in policy.state_dict().items():
        if name.startswith(
            ("base.encoder.", "base.memory.", "base.observation_")
        ) and not torch.equal(value, before[name]):
            raise ValueError("frozen perceptual/memory/normalizer prior changed")
    changes = {
        k: float((v - before[k]).double().square().sum())
        for k, v in policy.state_dict().items()
        if not torch.equal(v, before[k])
    }
    if not any(k.startswith("base.actor.") for k in changes) or not any(
        k.startswith("base.critic.") for k in changes
    ):
        raise ValueError("actor and critic both must actually update")
    check_pins(data_pins)
    torch.save(
        dict(
            state_dict=policy.state_dict(),
            optimizer=optimizer.state_dict(),
            rng_state=torch.get_rng_state(),
            optimizer_steps=4,
            base_weights_hash=declaration["base_weights_hash"],
            source_head=a.expected_head,
            model_kind="motion_intention_ppo64",
            hardware_authorized=False,
            promotion_authorized=False,
        ),
        a.output / "candidate.pt",
    )
    return dict(
        status="NATIVE_EXPERIENCE_PPO_UPDATED_REQUIRES_EXAM",
        optimizer_steps=4,
        checkpoint_hash=digest(a.output / "candidate.pt"),
        updates=history,
        parameter_change_l2_squared=changes,
        training_episodes=len(all_indices),
        new_native_steps=0,
        closed_loop_skill_verified=False,
    )


if __name__ == "__main__":
    main()
