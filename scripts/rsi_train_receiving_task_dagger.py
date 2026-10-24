"""Task-conditioned BC400->800 with verified OWN-state correction/retention.

No physics, online RL, task relabeling or promotion. All 96 teacher trajectories
stay intact. Three learner cohorts contribute only their six unique source
worlds, never deterministic repeats or own replays as extra observations.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

SEEDS = (732023, 732024, 732025)
CASES = (0, 6, 24, 32, 60, 64)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("bc-root", "teacher-source-root", "scene", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("bc-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--continuation-root", type=Path)
    p.add_argument("--continuation-hash")
    p.add_argument("--paired-review-root", type=Path)
    p.add_argument("--paired-review-hash")
    p.add_argument(
        "--student-pair",
        nargs=3,
        action="append",
        required=True,
        metavar=("SOURCE", "OWN", "OWN_HASH"),
    )
    a = p.parse_args()
    if len(a.student_pair) != 3:
        p.error("all three predeclared learner source/own pairs required")
    continuation = any(
        v is not None
        for v in (
            a.continuation_root,
            a.continuation_hash,
            a.paired_review_root,
            a.paired_review_hash,
        )
    )
    if continuation and any(
        v is None
        for v in (
            a.continuation_root,
            a.continuation_hash,
            a.paired_review_root,
            a.paired_review_hash,
        )
    ):
        p.error(
            "second round needs exact first-round training and complete three-seed paired review"
        )
    import mujoco
    import numpy as np
    import torch

    from rosclaw_soccer.rsi.measured_receiving_intent import receiving_intent
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.receiving_dagger_labels import receiving_dagger_label
    from rosclaw_soccer.rsi.receiving_task_command_actor import (
        ReceivingTaskCommandActor,
        receiving_task_context6,
    )
    from rosclaw_soccer.rsi.receiving_task_command_learning import continue_task_command
    from rosclaw_soccer.rsi.refinement_onpolicy_checkpoint import file_hash
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused once-only task DAgger output required")
    pins = {}

    def read(root, expected):
        rp = root / "complete.json"
        if (
            root.is_symlink()
            or rp.is_symlink()
            or (root / "failed.json").exists()
            or file_hash(rp) != expected
        ):
            raise ValueError("exact closed ordinary source evidence required")
        r = json.loads(rp.read_text())
        for key, value in r["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting DAgger ancestry")
            pins[key] = value
        pins[str(rp)] = expected
        return r

    bc = read(a.bc_root, a.bc_hash)
    if (
        bc["status"] != "RECEIVING_COMMAND_BC_THREE_SEEDS_TRAINED_PENDING_OWN_NATIVE"
        or bc["optimizer_updates"] != 1200
        or [r["seed"] for r in bc["candidates"]] != list(SEEDS)
    ):
        raise ValueError("exact BC400 three-seed parent required")
    tp = a.teacher_source_root / "complete.json"
    if str(tp) not in pins:
        raise ValueError("teacher source must already bind the BC bank")
    teacher = read(a.teacher_source_root, pins[str(tp)])
    if (
        len(teacher["records"]) != 96
        or teacher["config"].get("receiver_intent_mode") != "projected-capture-v3"
    ):
        raise ValueError("original complete96 projected teacher required")
    bank_path = a.bc_root / "teacher-command-bank.npz"
    if bank_path.is_symlink() or file_hash(bank_path) != bc["bank_hash"]:
        raise ValueError("exact parent BC bank required")
    pins[str(bank_path)] = bc["bank_hash"]
    for candidate in bc["candidates"]:
        cp = Path(candidate["checkpoint_path"])
        if (
            cp.is_symlink()
            or cp.resolve().parent != a.bc_root.resolve()
            or file_hash(cp) != candidate["checkpoint_hash"]
        ):
            raise ValueError("all original parent command weights required")
        pins[str(cp)] = candidate["checkpoint_hash"]
    cohorts = {}
    for sr, orr, oh in a.student_pair:
        root, own_root = Path(sr), Path(orr)
        own = read(own_root, oh)
        source = read(root, own["source_hash"])
        seed = source["command_seed"]
        if (
            seed not in SEEDS
            or (0, seed) in cohorts
            or source["status"] != "RECEIVING_NEURAL_COMMAND_BC_PILOT_PENDING_OWN"
            or own["status"] != "RECEIVING_NEURAL_COMMAND_BC_PILOT_ALL_OWN_MATCH"
            or source["command_training_hash"] != a.bc_hash
            or own["command_training_hash"] != a.bc_hash
            or source.get("command_reference_untrained", False) is not False
            or source["source_case_indices"] != list(CASES)
            or len(source["records"]) != 12
            or own["independent_whole_world_replay"] is not True
            or own["records"]
            != [{k: v for k, v in r.items() if k != "artifacts"} for r in source["records"]]
        ):
            raise ValueError("all exact BC400 own-history student worlds required")
        for i, r in enumerate(source["records"]):
            if (
                r["source_case_index"] != CASES[i // 2]
                or r["course"] != teacher["records"][CASES[i // 2]]["course"]
                or r["sampling_seed"] != teacher["records"][CASES[i // 2]]["sampling_seed"]
            ):
                raise ValueError("same predeclared original initial/goal/private A seed required")
        cohorts[(0, seed)] = (root, source)
    parents = bc
    if continuation:
        from rosclaw_soccer.rsi.receiving_command_comparison import admit_pair
        from scripts.rsi_native_fresh_common import receipt

        parents = read(a.continuation_root, a.continuation_hash)
        review = read(a.paired_review_root, a.paired_review_hash)
        if (
            parents["status"] != "RECEIVING_TASK_COMMAND_DAGGER_THREE_SEEDS_PENDING_NATIVE"
            or parents["parent_bc_hash"] != a.bc_hash
            or parents["optimizer_updates"] != 1200
            or [r["seed"] for r in parents["candidates"]] != list(SEEDS)
            or review["status"] != "THREE_SEED_TASK_DAGGER_NATIVE_REVIEWED_NOT_PROMOTED"
            or review["parent_training_hash"] != a.bc_hash
            or review["child_training_hash"] != a.continuation_hash
            or [r["seed"] for r in review["cohorts"]] != list(SEEDS)
            or review["goal_complete"] is not False
            or review["promotion_authorized"] is not False
        ):
            raise ValueError(
                "complete paired first-round physics and all three actual parents required"
            )
        for group in review["cohorts"]:
            sp, source = receipt(pins, group["child_source_hash"])
            _, own = receipt(pins, group["child_own_hash"])
            admit_pair(
                source,
                own,
                source_hash=group["child_source_hash"],
                training_hash=a.continuation_hash,
                seed=group["seed"],
                task=True,
            )
            cohorts[(1, group["seed"])] = (sp.parent, source)
        for candidate in parents["candidates"]:
            cp = Path(candidate["checkpoint_path"])
            if (
                cp.is_symlink()
                or cp.resolve().parent != a.continuation_root.resolve()
                or file_hash(cp) != candidate["checkpoint_hash"]
            ):
                raise ValueError("exact first-round weights required")
            pins[str(cp)] = candidate["checkpoint_hash"]
    if pins.get(str(a.scene)) != file_hash(a.scene):
        raise ValueError("original body layout required")
    source_pins(pins)
    check_pins(pins)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = mujoco.MjModel.from_xml_path(str(a.scene))
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    f = (
        np.array(teacher["records"][64]["course"]["ball_planar_offset_world"])
        - np.array(teacher["records"][32]["course"]["ball_planar_offset_world"])
    ) / 0.01
    if abs(np.linalg.norm(f) - 1) > 1e-10:
        raise ValueError("unchanged declared task direction required")
    lateral = np.array([-f[1], f[0]])
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            parent_bc_hash=a.bc_hash,
            **(
                dict(
                    parent_training_hash=a.continuation_hash,
                    paired_review_hash=a.paired_review_hash,
                    continuation_round=2,
                )
                if continuation
                else {}
            ),
            seeds=list(SEEDS),
            updates_per_seed=400,
            **(
                dict(prior_adam800_task400_continues=True, normalization_unchanged=True)
                if continuation
                else dict(old_adam400_continues=True)
            ),
            private_batch_rng_carried=True,
            minibatch_old=128,
            minibatch_own_critical=64,
            minibatch_own_all=64,
            own_worlds=36 if continuation else 18,
            repeats_not_training_rows=True,
            new_physical_steps=0,
            advice_not_executed_in_student_world=True,
            hardware_authorized=False,
        ),
    )
    total_steps, candidates = 0, []
    try:
        with np.load(bank_path, allow_pickle=False) as z:
            old_x = z["observations"].copy()
            old_y = z["commands"].copy()
            old_tr = z["training"].copy()
            old_va = z["validation"].copy()
            old_world = z["world"].copy()

        def primitives(root, row, keys):
            result = {}
            for key in keys:
                art = row["artifacts"][key]
                path = root / art["path"]
                if (
                    path.is_symlink()
                    or path.resolve().parent != (root / f"trial{row['index']}").resolve()
                    or pins.get(str(path)) != art["hash"]
                ):
                    raise ValueError(
                        "every consumed primitive must be confined and independently pinned"
                    )
                with np.load(path, allow_pickle=False) as z:
                    result[key] = {k: z[k].copy() for k in z.files}
            return result

        def poses(raw, n):
            q = np.concatenate((raw["initial"]["qpos"][None], raw["physical"]["qpos"][:-1, -1]))
            if q.shape != (n, model.nq):
                raise ValueError("complete original PRE-action state history required")
            return q

        old_tasks = []
        for i, row in enumerate(teacher["records"]):
            raw = primitives(a.teacher_source_root, row, ("initial", "physical"))
            q = poses(raw, row["controls"])
            goal = np.array(row["course"]["target"], dtype=np.float64)
            ids = np.flatnonzero(old_world == i)
            if len(ids) != len(q) or not np.array_equal(
                ids, np.arange(len(old_tasks), len(old_tasks) + len(q))
            ):
                raise ValueError("exact old bank/world rows required")
            old_tasks.extend(
                receiving_task_context6(q[t, layout.rq : layout.rq + 7], goal, f)
                for t in range(len(q))
            )
        old_x118 = np.c_[old_x, np.stack(old_tasks)].astype(np.float32)
        own_x, own_y, critical, kinds, provenance = [], [], [], [], []
        for (generation, seed), (root, source) in sorted(cohorts.items()):
            for row in source["records"][::2]:
                raw = primitives(root, row, ("initial", "physical", "B"))
                n = row["controls"]
                q = poses(raw, n)
                v = np.concatenate((raw["initial"]["qvel"][None], raw["physical"]["qvel"][:-1, -1]))
                b = raw["B"]
                goal = np.array(row["course"]["target"], dtype=np.float64)
                anchor = goal[:2] + 0.25 * f + 0.12 * lateral
                if b["command_inputs112"].shape != (n, 112) or not np.array_equal(
                    b["command_inputs112"][:, 110:112].astype(bool), b["contact_flags_before"]
                ):
                    raise ValueError("actual own causal inputs/contact history required")
                o = row["outcome"]
                success = bool(
                    o["B_legal_functional_receive_before_deadline"]
                    and o["whole_body_healthy"]
                    and not o["B_excluded_contact"]
                )
                for t in range(n):
                    pose = q[t, layout.rq : layout.rq + 7]
                    w, x, y, z = pose[3:7]
                    yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
                    advice = receiving_intent(
                        root_xy=pose[:2],
                        yaw=yaw,
                        ball_xy=q[t, layout.bq : layout.bq + 2],
                        ball_velocity_xy=v[t, layout.bv : layout.bv + 2],
                        home_xy=raw["initial"]["qpos"][layout.rq : layout.rq + 2],
                        pass_direction=f,
                        launch_seen=bool(b["command_launch_seen_before"][t]),
                        ready_anchor_xy=anchor,
                        capture_target_xy=goal[:2],
                        mode="projected-capture-v3",
                    )
                    important = advice.role_state == "TRACK_RECEIVING_PLANE" and not bool(
                        b["contact_flags_before"][t].any()
                    )
                    label, kind = receiving_dagger_label(
                        b["command_policy_action3"][t],
                        np.array(advice.command_body, np.float32),
                        observed_safe_functional=success,
                        observed_strict=bool(o["combined_precision_and_functional_receive"]),
                        plane_before_contact=important,
                    )
                    own_x.append(
                        np.r_[b["command_inputs112"][t], receiving_task_context6(pose, goal, f)]
                    )
                    if generation == 1 and not np.array_equal(own_x[-1], b["command_inputs118"][t]):
                        raise ValueError(
                            "new student task context must equal actual executed inputs"
                        )
                    own_y.append(label)
                    critical.append(important)
                    kinds.append(kind)
                    provenance.append(
                        (generation, seed, row["source_case_index"], t)
                        if continuation
                        else (seed, row["source_case_index"], t)
                    )
        own_x = np.stack(own_x).astype(np.float32)
        own_y = np.stack(own_y).astype(np.float32)
        critical = np.array(critical, dtype=bool)
        data_path = a.output / "task-dagger-bank.npz"
        np.savez_compressed(
            data_path,
            old_x=old_x118,
            old_y=old_y,
            old_training=old_tr,
            old_validation=old_va,
            own_x=own_x,
            own_y=own_y,
            own_critical=critical,
            own_label_kind=np.array(kinds),
            own_provenance=np.array(provenance, dtype=np.int64),
        )
        data_hash = file_hash(data_path)
        tensors = [
            torch.from_numpy(x) for x in (old_x118, old_y, old_tr, old_va, own_x, own_y, critical)
        ]
        for parent in parents["candidates"]:
            seed = parent["seed"]
            state = torch.load(parent["checkpoint_path"], map_location="cpu", weights_only=True)
            if (
                state["bank_hash"] != parents["bank_hash"]
                or state["seed"] != seed
                or state["optimizer_updates"] != 400
            ):
                raise ValueError("exact actual parent state/RNG/Adam required")
            if continuation:
                from rosclaw_soccer.rsi.receiving_task_command_checkpoint import load_task_command

                original = next(c for c in bc["candidates"] if c["seed"] == seed)
                original_state = torch.load(
                    original["checkpoint_path"], map_location="cpu", weights_only=True
                )
                load_task_command(state, parents, parent, original_state)

            def observe(step, loss, norm, current_seed=seed):
                nonlocal total_steps
                total_steps += 1
                if step % 50 == 0:
                    print(
                        json.dumps(
                            dict(seed=current_seed, new_steps=step, loss=loss, gradient_norm=norm)
                        ),
                        flush=True,
                    )

            fitted = continue_task_command(
                state,
                *tensors,
                steps=400,
                continuation_round=2 if continuation else 1,
                on_step=observe,
            )
            actor = fitted["actor"]
            cp = a.output / f"task-command-seed{seed}.pt"
            torch.save(
                dict(
                    schema="soccer.receiving_task_command_dagger.v3"
                    if continuation
                    else "soccer.receiving_task_command_dagger.v2",
                    actor=actor.state_dict(),
                    optimizer=fitted["optimizer"],
                    batch_rng=fitted["batch_rng"],
                    seed=seed,
                    optimizer_updates=400,
                    cumulative_prior_steps=1200 if continuation else 800,
                    parent_checkpoint_hash=parent["checkpoint_hash"],
                    parent_bc_hash=a.bc_hash,
                    **(dict(parent_training_hash=a.continuation_hash) if continuation else {}),
                    bank_hash=data_hash,
                    policy_domain=actor.policy_domain,
                ),
                cp,
            )
            restored = torch.load(cp, map_location="cpu", weights_only=True)
            from rosclaw_soccer.rsi.receiving_command_actor import ReceivingCommandActor

            pw = (
                {
                    k.removeprefix("prior."): v
                    for k, v in state["actor"].items()
                    if k.startswith("prior.")
                }
                if continuation
                else state["actor"]
            )
            pr = ReceivingCommandActor(pw["observation_mean"], pw["observation_scale"])
            pr.load_state_dict(pw, strict=True)
            replica = ReceivingTaskCommandActor(
                pr, restored["actor"]["task_mean"], restored["actor"]["task_scale"]
            )
            replica.load_state_dict(restored["actor"], strict=True)
            replica.eval().requires_grad_(False)
            with torch.no_grad():
                if not torch.equal(actor(tensors[4][:32]), replica(tensors[4][:32])):
                    raise ValueError("task checkpoint reload changed policy")
            result = {
                k: fitted[k]
                for k in (
                    "optimizer_updates",
                    "cumulative_prior_steps",
                    "optimizer_parameter_steps",
                    "changed_parameter_tensors",
                    "exact_parent_function_before_updates",
                    "before",
                    "after",
                    "own_critical_rows",
                )
            }
            result.update(
                seed=seed,
                checkpoint_path=str(cp),
                checkpoint_hash=file_hash(cp),
                parent_checkpoint_hash=parent["checkpoint_hash"],
            )
            candidates.append(result)
            write_once(a.output / f"seed{seed}.json", result)
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="RECEIVING_TASK_COMMAND_DAGGER_SECOND_ROUND_PENDING_NATIVE"
                if continuation
                else "RECEIVING_TASK_COMMAND_DAGGER_THREE_SEEDS_PENDING_NATIVE",
                source_head=a.expected_head,
                input_file_hashes=pins,
                parent_bc_hash=a.bc_hash,
                **(
                    dict(
                        parent_training_hash=a.continuation_hash,
                        paired_review_hash=a.paired_review_hash,
                        continuation_round=2,
                    )
                    if continuation
                    else {}
                ),
                bank_hash=data_hash,
                candidates=candidates,
                optimizer_updates=total_steps,
                own_unique_worlds=36 if continuation else 18,
                old_rows=len(old_x118),
                own_rows=len(own_x),
                own_critical_rows=int(critical.sum()),
                own_label_counts=dict(Counter(kinds)),
                teacher_advice_not_executed=True,
                old_validation_unchanged_not_Fresh=True,
                new_physical_steps=0,
                critic_updates=0,
                online_RL_proven=False,
                neural_physical_growth_proven=False,
                promotion_authorized=False,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                actual_optimizer_steps=total_steps,
                completed_candidates=candidates,
                new_physical_steps=0,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
