"""Four autonomous causal-PPO cycles per declared seed with physical rollback."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("audit", "review", "memory", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("audit-hash", "review-hash", "memory-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--shard-root", type=Path, nargs=4, required=True)
    p.add_argument("--seed", type=int, choices=(926211, 926212, 926213), required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused cohort output required, never restart a charged campaign")
    reports, pins = [], {}
    for path, h, status in (
        (a.audit, a.audit_hash, "ALL80_KICK_HEAD_PPO_CANDIDATE_TRIALS_NATIVE_VERIFIED"),
        (
            a.review / "complete.json",
            a.review_hash,
            "ALL88_CAUSAL_EVENT_REWARDS_PREFIX_VERIFIED_AND_NONFOOT_GEOMETRY_INSPECTED",
        ),
        (
            a.memory / "complete.json",
            a.memory_hash,
            "ALL_ORIGINAL40_FULL_NEURAL_HISTORIES_EXACTLY_RECONSTRUCTED_NOT_TRAINED",
        ),
    ):
        if path.is_symlink() or digest(path) != h:
            raise ValueError("exact closed scientific prerequisites required")
        report = json.loads(path.read_text())
        if report["status"] != status or report["promotion_authorized"] is not False:
            raise ValueError("completed unpromoted source reports required")
        for name, value in {**report["input_file_hashes"], str(path): h}.items():
            if name in pins and pins[name] != value:
                raise ValueError("conflicting source identity")
            pins[name] = value
        reports.append(report)
    audit, review, memory = reports
    originals = {}
    for shard, root in enumerate(a.shard_root):
        cp = root / "complete.json"
        if cp.is_symlink() or pins.get(str(cp)) != digest(cp):
            raise ValueError("exact original native producer required")
        report = json.loads(cp.read_text())
        if [r["course_index"] for r in report["results"]] != list(range(shard, 40, 4)):
            raise ValueError("all forty fixed source programs required")
        for item in report["results"]:
            originals[item["course_index"]] = item["rows"]["onnx"]
    if sorted(originals) != list(range(40)):
        raise ValueError("complete40 original states required")
    for row in memory["results"]:
        if row.get("memory"):
            path = a.memory / row["memory"]["path"]
            if path.is_symlink() or path.parent.resolve() != a.memory.resolve():
                raise ValueError("confined actual full-history memories required")
            pins[str(path)] = row["memory"]["hash"]
    source_pins(pins)
    check_pins(pins)

    def one(suffix):
        paths = [Path(k) for k in pins if k.endswith(suffix)]
        if len(paths) != 1:
            raise ValueError("unique bound research asset required: " + suffix)
        return paths[0]

    import mujoco
    import numpy as np
    import torch
    from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.experimental_velocity_evidence import admit_heading_evidence
    from rosclaw_soccer.rsi.kick_causal_learning import (
        CausalKickLearner,
        collect_episode,
        copy_actor,
        snapshot_factory,
    )
    from rosclaw_soccer.rsi.kick_event_reward import event_reward_contract
    from rosclaw_soccer.rsi.kick_prior_actor import (
        KICK_PRIOR_HASH,
        KickPriorActor,
        torch_kick_policy_type,
    )
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.rsi.walk_kick_teacher_episode import evaluate_episode
    from rosclaw_soccer.rsi.walk_kick_teacher_plan import search_course_bank
    from rosclaw_soccer.sim.contracts import hash_json
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

    if review["reward_contract"] != event_reward_contract():
        raise ValueError("actual reviewed causal reward semantics required")
    courses = search_course_bank(expanded_public=True)
    preflight_receiving_courses(courses)
    configs = []
    for name in pins:
        path = Path(name)
        if path.suffix == ".json" and path.stat().st_size < 2048:
            cfg = json.loads(path.read_text())
            if cfg.get("schema") == "soccer.native_single_finisher_experimental_approach.v1":
                configs.append(path)
    if len(configs) != 1:
        raise ValueError("unique verified approach codec required")
    fast, _ = admit_heading_evidence(configs[0], pins[str(configs[0])])
    vm, vc = one("/velocity/v0/exported/policy.onnx"), one("/velocity/v0/params/deploy.yaml")
    assets = dict(
        velocity_model=vm,
        velocity_config=vc,
        velocity_model_hash=pins[str(vm)],
        velocity_config_hash=pins[str(vc)],
        **fast,
    )
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(
            one("/cached-standard/declared-scene.xml").read_text(), prefixes=("a/",)
        )
    )
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    state, output, policy, mapping = _load_robonaldo(
        one("/policy/robonaldo/FreeKick.py").parents[2]
    )
    original_factory = torch_kick_policy_type(
        policy, one("/policy/robonaldo/model/policy-obs-aic.onnx")
    )
    anchors, retained = [], []
    for item in memory["results"]:
        index = item["course_index"]
        required = (
            originals[index]["outcome"]["task_pass"] and originals[index]["healthy_full_horizon"]
        )
        if item.get("original_success_to_retain", False) != required:
            raise ValueError("full-history success memory set differs from actual source")
        if required:
            with np.load(a.memory / item["memory"]["path"]) as saved:
                anchors.append(saved["observations"].copy())
            retained.append(index)
    if len(retained) != 30 or sum(len(x) for x in anchors) != 7131:
        raise ValueError("all30 full successful histories/7131anchors required")
    anchor_tensor = torch.from_numpy(np.concatenate(anchors))
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.g1.causal_head_ppo_physical_retention.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            seed=a.seed,
            all_cohort_seeds=[926211, 926212, 926213],
            cycles=4,
            new_own_worlds_per_cycle=8,
            maximum_retention_worlds_per_cycle=40,
            final_worlds=40,
            world_ceiling=232,
            native_step_ceiling=835200,
            causal_reward=event_reward_contract(),
            actor_proposals_per_cycle=1,
            critic_steps_per_cycle=4,
            raw_std=0.015,
            ar_rho=0.9,
            local_behavior_KL_limit=0.005,
            full_history_prior_KL_soft_penalty=0.1,
            no_global_prior_KL_safety_certificate=True,
            successful_memory_rows=7131,
            physical_gate=(
                "all original body health, all original safe outcomes, all original task successes"
            ),
            retention_stops_at_first_failure=True,
            failed_context_added_to_next_curriculum=True,
            motor_and_Adam_rollback_on_regression=True,
            critic_keeps_current_policy_training=True,
            only3741motor_readout_parameters_trainable=True,
            original_approach_is_privileged_per_course=True,
            public_consumed_development_not_Fresh=True,
            no_best_seed_or_checkpoint_selection=True,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    actor = KickPriorActor.from_onnx(
        one("/policy/robonaldo/model/policy-obs-aic.onnx"), expected_hash=KICK_PRIOR_HASH
    )
    learner = CausalKickLearner(actor, anchor_tensor)
    initial = copy_actor(actor)
    worlds, native, motor_samples, numeric_accepts = 0, 0, 0, 0
    hard = set(audit["lost_success_indices"]) | set(audit["safety_regression_indices"])
    cycles = []
    physical_records = {}

    def checkpoint(name):
        path = a.output / (name + ".pt")
        torch.save(
            dict(
                schema="soccer.g1.causal_kick_cohort_checkpoint.v1",
                actor=actor.state_dict(),
                critic=learner.critic.state_dict(),
                actor_optimizer=learner.actor_optimizer.state_dict(),
                value_optimizer=learner.value_optimizer.state_dict(),
                pending=learner.pending,
                torch_rng=torch.get_rng_state(),
                numpy_rng=rng.bit_generator.state,
                value_steps=learner.value_steps,
                actor_proposals=learner.actor_proposals,
                physical_motor_commits=learner.motor_commits,
                seed=a.seed,
                source_head=a.expected_head,
                body_hash=hash_json(layout.profile()),
                prior_hash=KICK_PRIOR_HASH,
                reward_contract=event_reward_contract(),
                input_file_hashes=pins,
                simulation_only=True,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
            path,
        )
        restored = torch.load(path, map_location="cpu", weights_only=True)
        if any(not torch.equal(v, restored["actor"][k]) for k, v in actor.state_dict().items()):
            raise ValueError("safe-loaded current actor differs")
        return dict(path=path.name, hash=digest(path))

    def persist(folder, name, index, row, arrays, version):
        nonlocal worlds, native
        worlds += 1
        native += row["native_steps"]
        if worlds > 232 or native > 835200:
            raise ValueError("declared scientific execution ceiling exceeded")
        if hash_json(
            relabel_arrays(
                arrays, controls=row["controls"], ball_address=layout.bq, root_address=layout.rq
            )
        ) != hash_json(row["outcome"]):
            raise ValueError("new actual physics label failed readback")
        if np.any(
            abs(arrays["torque"][:, layout.ai]) > 0.85 * np.array(G1_HARD_TORQUE_LIMITS) + 1e-12
        ):
            raise ValueError("applied hard torque bound changed")
        target = folder / (name + ".npz")
        np.savez_compressed(target, **arrays)
        row.update(
            course_index=index,
            policy_checkpoint=version,
            artifact=dict(path=target.name, hash=digest(target)),
        )
        attempt = folder / (name + "-attempt.json")
        row["attempt_record_hash"] = digest(attempt)
        write_once(folder / (name + ".json"), row)
        physical_records[str((folder / (name + ".json")).relative_to(a.output))] = digest(
            folder / (name + ".json")
        )

    def deterministic(index, snapshot):
        return evaluate_episode(
            model=model,
            assets=dict(assets, experimental_fast=originals[index]["experimental_fast_approach"]),
            policy_types=(state, output, snapshot_factory(original_factory, snapshot), mapping),
            course=courses[index].course,
            parameters=originals[index]["parameters"],
            hard_limits=np.array(G1_HARD_TORQUE_LIMITS),
        )

    active = checkpoint("initial")
    try:
        for cycle in range(4):
            folder = a.output / f"cycle-{cycle:02d}"
            folder.mkdir()
            chosen = []
            for bucket in range(4):
                priority = sorted(i for i in hard if i // 10 == bucket)
                normal = [bucket * 10 + (a.seed + cycle * 2 + j) % 10 for j in range(10)]
                pool = (
                    priority[cycle % len(priority) :] + priority[: cycle % len(priority)]
                    if priority
                    else []
                ) + normal
                picks = list(dict.fromkeys(pool))[:2]
                chosen.extend(picks)
            write_once(
                folder / "declaration.json",
                dict(
                    cycle=cycle,
                    courses=chosen,
                    active_checkpoint=active,
                    known_failures=sorted(hard),
                ),
            )
            experiences = []
            for index in chosen:
                seed = int(rng.integers(0, 2**31))
                name = f"own-{index:02d}"
                write_once(
                    folder / (name + "-attempt.json"),
                    dict(course_index=index, seed=seed, policy_checkpoint=active),
                )
                row, arrays, experience, generator_state = collect_episode(
                    actor,
                    original_factory,
                    seed=seed,
                    state_type=state,
                    output_type=output,
                    mapping=mapping,
                    model=model,
                    assets=dict(
                        assets, experimental_fast=originals[index]["experimental_fast_approach"]
                    ),
                    course=courses[index].course,
                    parameters=originals[index]["parameters"],
                    hard_limits=np.array(G1_HARD_TORQUE_LIMITS),
                )
                if generator_state is not None:
                    arrays["own_generator_state"] = generator_state.numpy()
                row["exploration_seed"] = seed
                persist(folder, name, index, row, arrays, active)
                motor_samples += row["own_motor_samples"]
                if experience is not None:
                    experiences.append(experience)
            if not experiences:
                raise ValueError("no own motor experience, cannot manufacture gradients")
            proposal = learner.propose(experiences)
            candidate = checkpoint(f"candidate-{cycle:02d}")
            accepted = proposal["actor_decision"]["accepted"]
            numeric_accepts += int(accepted)
            gate = []
            failed = None
            if accepted:
                snapshot = copy_actor(actor)
                order = sorted(hard) + [
                    i for j in range(10) for i in range(j, 40, 10) if i not in hard
                ]
                for index in order:
                    name = f"retention-{index:02d}"
                    write_once(
                        folder / (name + "-attempt.json"),
                        dict(course_index=index, policy_checkpoint=candidate),
                    )
                    row, arrays = deterministic(index, snapshot)
                    persist(folder, name, index, row, arrays, candidate)
                    old = originals[index]
                    loss = (
                        (old["healthy_full_horizon"] and not row["healthy_full_horizon"])
                        or (old["outcome"]["safe"] and not row["outcome"]["safe"])
                        or (old["outcome"]["task_pass"] and not row["outcome"]["task_pass"])
                    )
                    gate.append(dict(course_index=index, regression=bool(loss)))
                    if loss:
                        failed = index
                        hard.add(index)
                        break
                learner.physical_decision(failed is None and len(gate) == 40)
            active = checkpoint(f"active-{cycle:02d}")
            record = dict(
                cycle=cycle,
                chosen_courses=chosen,
                proposal=proposal,
                candidate=candidate,
                retention=gate,
                failed_context=failed,
                active_checkpoint=active,
                physically_committed=bool(accepted and failed is None and len(gate) == 40),
                total_worlds=worlds,
                total_native_steps=native,
                total_own_motor_samples=motor_samples,
                physical_motor_commits=learner.motor_commits,
            )
            cycles.append(record)
            write_once(folder / "complete.json", record)
        final = []
        snapshot = copy_actor(actor)
        folder = a.output / "final40"
        folder.mkdir()
        for index in range(40):
            write_once(
                folder / f"course-{index:02d}-attempt.json",
                dict(course_index=index, policy_checkpoint=active),
            )
            row, arrays = deterministic(index, snapshot)
            persist(folder, f"course-{index:02d}", index, row, arrays, active)
            final.append(
                dict(
                    course_index=index, outcome=row["outcome"], healthy=row["healthy_full_horizon"]
                )
            )
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="FOUR_CAUSAL_PPO_CYCLES_PHYSICALLY_GATED_AND_FINAL40_MEASURED_NOT_PROMOTED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                seed=a.seed,
                cycles=cycles,
                final=final,
                worlds=worlds,
                native_steps=native,
                own_motor_samples=motor_samples,
                numerical_actor_accepts=numeric_accepts,
                physical_motor_commits=learner.motor_commits,
                value_optimizer_updates=learner.value_steps,
                final_checkpoint=active,
                physical_record_hashes=physical_records,
                changed_active_actor_parameters=sum(
                    int((v - initial[k]).count_nonzero()) for k, v in actor.state_dict().items()
                ),
                passes=sum(x["healthy"] and x["outcome"]["task_pass"] for x in final),
                independent_initial_courses=40,
                public_consumed_not_Fresh=True,
                promotion_authorized=False,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                completed_cycles=len(cycles),
                worlds=worlds,
                native_steps=native,
                no_automatic_retry=True,
            ),
        )
        raise


if __name__ == "__main__":
    main()
