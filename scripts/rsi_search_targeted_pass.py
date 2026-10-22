"""Bounded one-body target-conditioned pass teacher CEM, NOT student training.

One frozen process per context,2 goals,72 maximum actual worlds. Qualified
two-body transfers and own-history replay are still required before IL.
No output or model allocation at import, no automatic retries or resets of
an active episode. Every CEM failure remains an explicit physical record.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("frontier-root", "chain-root", "chain-audit-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "frontier-head",
        "frontier-hash",
        "chain-hash",
        "chain-audit-hash",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--context", type=int, choices=(1, 4, 20, 21), required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import admit_frontier, clean, receipt, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused target-pass search root required")
    pins, base_hash, candidates, contexts = admit_frontier(
        a.frontier_root, a.frontier_head, a.frontier_hash
    )
    if [c["seed"] for c in candidates] != [65023]:
        raise ValueError("same independently retained frozen motor required")
    reports = []
    for root, h in ((a.chain_root, a.chain_hash), (a.chain_audit_root, a.chain_audit_hash)):
        path = root / "complete.json"
        if path.is_symlink() or (root / "failed.json").exists() or digest(path) != h:
            raise ValueError("closed independently replayed team-entry study required")
        reports.append(json.loads(path.read_text()))
        pins[str(path)] = h
    chain, own = reports
    if (
        chain["status"] != "NEURAL_RELAY_TEAM_ENTRY_COLLECTED_PENDING_OWN_REPLAY"
        or own["status"] != "ALL_NEURAL_RELAY_OWN_HISTORY_NATIVE_REPLAY_MATCH"
        or own["producer_hash"] != a.chain_hash
        or own["counts"] != chain["counts"]
        or own["worlds"] != 128
        or len(own["records"]) != 128
        or any(
            r[k] is not False
            for r in reports
            for k in ("goal_complete", "promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("whole128 causal source closure required before new search")
    source_declaration = a.chain_root / "declaration.json"
    if source_declaration.is_symlink() or digest(source_declaration) != chain["declaration_hash"]:
        raise ValueError("exact closed chain declaration required")
    declared = json.loads(source_declaration.read_text())
    if declared["candidate"] != candidates[0]:
        raise ValueError("same exact frozen motor used by actual source study required")
    pins[str(source_declaration)] = digest(source_declaration)
    source_pins(pins)
    check_pins(pins)
    goals = [dict(distance_m=3.75, lateral_m=y) for y in (0.0, 0.2)]
    seed = 66023 + a.context * 101
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.targeted_pass_joint_teacher_CEM72.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            candidate=candidates[0],
            baseline_training_hash=base_hash,
            context=a.context,
            goals=goals,
            seed=seed,
            generations=2,
            candidates_per_generation=16,
            selected_repetitions=2,
            maximum_worlds=72,
            maximum_native_steps=288000,
            maximum_actor_calls=28800,
            zero_teacher_mean_identity_and_whole_raw_repeat_required=True,
            original_action_authority_rad=0.2,
            all_search_failures_retained=True,
            one_body_reachability_ONLY=True,
            teammate_receiving_verified=False,
            optimizer_updates=0,
            old_exam_contract_changed=False,
            hardware_authorized=False,
            promotion_authorized=False,
            arrival_error_m=0.1,
            arrival_speed_cap_mps=3.0,
            rank_rule="healthy100NNrecovery,cleanfoot,actualfoot,-error-.03*max(0,speed-3)^2",
            formal_arrival_horizon_controls=300,
            separate_neural_recovery_controls=100,
        ),
    )
    records, generations, selections = [], [], []
    counts = dict(native_steps=0, actor_calls=0)
    try:
        from dataclasses import asdict

        import numpy as np

        from rosclaw_soccer.physics.native_ball_dimensions import inspect_native_ball_dimensions
        from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
        from rosclaw_soccer.rsi.stroke_search import sample_generation, update_elites
        from rosclaw_soccer.rsi.targeted_pass_precision import (
            precision_rank,
            targeted_pass_precision,
        )
        from scripts.rsi_stroke_search_common import collect, open_motor

        bench, base = open_motor(pins, base_hash, {a.context: contexts[a.context]})
        dimensions = inspect_native_ball_dimensions(
            bench.template.model, geom_id=int(bench.template.model.geom("soccer_ball_geom").id)
        )
        if not dimensions.size_and_mass_in_ifab_range:
            raise ValueError("same compiled standard ball dimensions required")
        candidate = candidates[0]
        _, training = receipt(pins, candidate["training_hash"])
        actor, _ = load_dual_candidate_for_sim_audit(
            base.parent,
            parent_lineage_hash=training["parent_lineage_hash"],
            root=Path(candidate["training_root"]),
            expected_receipt_hash=candidate["training_hash"],
            checkpoint_kind=candidate["kind"],
        )
        anchor_rows = [
            r
            for r in chain["records"]
            if r["course"]["context"] == a.context
            and r["course"]["lateral_m"] == -0.55
            and r["course"]["arm"]["start"] == 0
            and r["course"]["arm"]["memory"] == "new_skill_zero"
            and r["course"]["arm"]["assigned_receiver_pass"]
            and not r["course"]["arm"]["align_successor"]
        ]
        if len(anchor_rows) != 1:
            raise ValueError("unique closed own-replayed TRAIN contact anchor required")
        first = anchor_rows[0]["result"]["incoming_delivery"]["A_first_foot_strike_substep"]
        if type(first) is not int or not 98 <= first // 10 <= 180:
            raise ValueError("declared bounded stroke requires observed source contact98..180")
        center = first // 10
        original = bench.make(contexts[a.context]["case"])
        source_name = contexts[a.context]["case"]["source"]
        if ("left" in source_name) == ("right" in source_name):
            raise ValueError("explicit named contact leg required; no foot guessing")
        foot = "left" if "left" in source_name else "right"
        q, v = original.layout.canonical(original.env.data)
        forward = original.goal[:2] - q[36:38]
        forward /= np.linalg.norm(forward)
        lateral = np.array([-forward[1], forward[0]])
        tasks = []
        for goal in goals:
            point = q[36:38] + goal["distance_m"] * forward + goal["lateral_m"] * lateral
            target = np.array([point[0], point[1], 0.11])
            case = {
                k: val
                for k, val in contexts[a.context]["case"].items()
                if k not in ("ball_local", "velocity_local")
            }
            case.update(
                ball_world_initial=q[36:39].tolist(),
                velocity_world_initial=v[35:37].tolist(),
                goal_world=target.tolist(),
            )
            ctx = dict(contexts[a.context], case=case, context_id=a.context)
            ep = bench.make(case)
            q2, v2 = ep.layout.canonical(ep.env.data)
            if not np.array_equal(q, q2) or not np.array_equal(v, v2):
                raise ValueError(
                    "new task goal must not change any original physical initialization"
                )
            tasks.append((ctx, target))
        write_once(
            a.output / "preflight.json",
            dict(
                center=center,
                tasks=[dict(context=c, goal=g.tolist()) for c, g in tasks],
                native_steps=bench.native_steps,
                ball_dimensions=asdict(dimensions),
                stroke_foot=foot,
            ),
        )
        if bench.native_steps:
            raise ValueError("all tasks must preflight before any physics")

        def world(task, latent, tag):
            if len(records) >= 72:
                raise ValueError("bounded72-world teacher search exhausted")
            ctx, target = tasks[task]
            row, blobs = collect(
                bench, actor, ctx, center, latent, a.output / tag, stroke_foot=foot
            )
            if not latent.any() and any(
                not np.array_equal(z["NN_action"], z["NN_nominal_action"]) for z in blobs.values()
            ):
                raise ValueError("zero teacher changed actual neural action")
            qn = np.concatenate([z["qpos"] for z in blobs.values()]).reshape(-1, 43)
            vn = np.concatenate([z["qvel"] for z in blobs.values()]).reshape(-1, 41)
            fn = np.concatenate([z["forces"] for z in blobs.values()]).reshape(-1, 3)
            stand = row["stand"]
            measure = targeted_pass_precision(
                qn,
                vn,
                fn,
                target,
                body_healthy=not row["result"]["body_failed"],
                recovery_observed=bool(stand and stand["healthy"] and stand["stability_observed"]),
            )
            row.update(
                task=task,
                target=target.tolist(),
                passing_result=measure,
                precision_rank=list(precision_rank(measure)),
                stroke_foot=foot,
            )
            records.append(row)
            counts["native_steps"] += row["native_steps"]
            counts["actor_calls"] += row["actor_decisions"]
            if counts["native_steps"] > 288000 or counts["actor_calls"] > 28800:
                raise ValueError("declared native/actor budget exceeded")
            write_once(a.output / tag / "complete.json", row)
            print(
                json.dumps(dict(world=len(records), tag=tag, result=measure, counts=counts)),
                flush=True,
            )
            return row

        for task in range(2):
            rng = np.random.default_rng(seed + task)
            mean, std = np.zeros((2, 6)), np.full((2, 6), 0.4)
            incumbent = world(task, np.zeros((2, 6)), f"task{task}-baseline")
            repeated = world(task, np.zeros((2, 6)), f"task{task}-baseline-repeat")
            if repeated["passing_result"] != incumbent["passing_result"] or {
                k: v["hash"] for k, v in repeated["artifacts"].items()
            } != {k: v["hash"] for k, v in incumbent["artifacts"].items()}:
                raise ValueError(
                    "zero teacher whole raw NN/physics baseline failed bit-exact repeat"
                )
            for generation in range(2):
                samples, draws = sample_generation(rng, mean, std)
                rows = [incumbent] + [
                    world(task, z, f"task{task}-g{generation}-c{k}") for k, z in enumerate(samples)
                ]
                points = [np.array(r["latent"]) for r in rows]
                ranks = [precision_rank(r["passing_result"]) for r in rows]
                updated, spread, elite = update_elites(points, ranks)
                best = max(range(17), key=lambda k: (ranks[k], -k))
                generations.append(
                    dict(
                        task=task,
                        generation=generation,
                        draws=draws,
                        mean_before=mean.tolist(),
                        std_before=std.tolist(),
                        tags=[r["tag"] for r in rows],
                        ranks=[list(r) for r in ranks],
                        elite=elite,
                        mean_after=updated.tolist(),
                        std_after=spread.tolist(),
                        selected=rows[best]["tag"],
                    )
                )
                incumbent = rows[best]
                mean, std = updated, spread
            repeats = [
                world(task, np.array(incumbent["latent"]), f"task{task}-selected-repeat{k}")
                for k in range(2)
            ]
            if any(r["passing_result"] != incumbent["passing_result"] for r in repeats):
                raise ValueError(
                    "same fixed selected teacher did not repeat actual physical result"
                )
            selections.append(
                dict(
                    task=task,
                    selected_tag=incumbent["tag"],
                    latent=incumbent["latent"],
                    passing_result=incumbent["passing_result"],
                )
            )
        if len(records) != 72:
            raise ValueError("complete72-world declared search required")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="TARGETED_PASS_CEM72_COLLECTED_PENDING_OWN_REPLAY_AND_TEAM_TRANSFER",
                source_head=a.expected_head,
                input_file_hashes=pins,
                context=a.context,
                counts=counts,
                records=records,
                generations=generations,
                selections=selections,
                optimizer_updates=0,
                teammate_receiving_verified=False,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="TARGETED_PASS_SEARCH_FAILED_PRESERVE_NO_RETRY",
                error_type=type(error).__name__,
                error=str(error),
                counts=counts,
                records=records,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
