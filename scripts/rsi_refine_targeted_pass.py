"""Learn local pass response from own-audited failures, validate6 bounded steps.

One-body TRAIN teacher only. Fits3 ridge models,6 proposals per goal, maximum
18 actual worlds per context. No student update or transferred team claim.
"""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "audit-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-hash", "audit-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins
    from scripts.rsi_pass_response_common import admit, plan_task, response_from_blobs

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused learned response refinement root required")
    s, d, pf, pins = admit(a.root, a.source_hash, a.audit_root, a.audit_hash)
    source_pins(pins)
    check_pins(pins)
    plans = [plan_task(a.root, s, t) for t in range(2)]
    a.output.mkdir(parents=True)
    write_once(
        a.output / "plan.json",
        dict(
            source_hash=a.source_hash,
            audit_hash=a.audit_hash,
            tasks=plans,
            planning_physics_steps=0,
            student_optimizer_updates=0,
        ),
    )
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.learned_local_pass_response_refinement.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            source_hash=a.source_hash,
            audit_hash=a.audit_hash,
            source_context=s["context"],
            plan_hash=digest(a.output / "plan.json"),
            maximum_worlds=18,
            maximum_native_steps=72000,
            maximum_actor_calls=7200,
            feasible_success_before_soft_cost=True,
            precision_predicate_unchanged=True,
            all_source_failures_retained=True,
            fit_only_safe_unique_contact_trajectories=True,
            control_projection_is_declared_latent_only=True,
            no_physical_state_projection=True,
            one_body_teacher_ONLY=True,
            neural_actor_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    records = []
    selections = []
    counts = dict(native_steps=0, actor_calls=0)
    try:
        import json
        from dataclasses import asdict

        import numpy as np

        from rosclaw_soccer.physics.native_ball_dimensions import inspect_native_ball_dimensions
        from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
        from rosclaw_soccer.rsi.pass_response_model import feasible_first_rank
        from rosclaw_soccer.rsi.targeted_pass_precision import targeted_pass_precision
        from scripts.rsi_stroke_search_common import collect, make_episode, open_motor

        context = s["context"]
        foot = pf["stroke_foot"]
        center = pf["center"]
        bench, base = open_motor(
            pins, d["baseline_training_hash"], {context: pf["tasks"][0]["context"]}
        )
        dimensions = inspect_native_ball_dimensions(
            bench.template.model, geom_id=int(bench.template.model.geom("soccer_ball_geom").id)
        )
        if not dimensions.size_and_mass_in_ifab_range:
            raise ValueError("same actual compiled standard ball required")
        c = d["candidate"]
        _, training = receipt(pins, c["training_hash"])
        actor, _ = load_dual_candidate_for_sim_audit(
            base.parent,
            parent_lineage_hash=training["parent_lineage_hash"],
            root=Path(c["training_root"]),
            expected_receipt_hash=c["training_hash"],
            checkpoint_kind=c["kind"],
        )
        for task in pf["tasks"]:
            ep = make_episode(bench, task["context"])
            if ep.native_steps:
                raise ValueError("all new worlds must preflight before any physics")
        write_once(
            a.output / "preflight.json",
            dict(
                tasks=pf["tasks"],
                center=center,
                stroke_foot=foot,
                ball_dimensions=asdict(dimensions),
                native_steps=bench.native_steps,
            ),
        )
        if bench.native_steps:
            raise ValueError("whole task admission required before physics")

        def world(task, latent, tag, prediction=None):
            if len(records) >= 18:
                raise ValueError("bounded18-world refinement exhausted")
            ctx = pf["tasks"][task]["context"]
            goal = np.array(pf["tasks"][task]["goal"])
            row, blobs = collect(
                bench,
                actor,
                ctx,
                center,
                np.array(latent, np.float64),
                a.output / tag,
                stroke_foot=foot,
            )
            stand = row["stand"]
            measure = targeted_pass_precision(
                np.concatenate([z["qpos"] for z in blobs.values()]).reshape(-1, 43),
                np.concatenate([z["qvel"] for z in blobs.values()]).reshape(-1, 41),
                np.concatenate([z["forces"] for z in blobs.values()]).reshape(-1, 3),
                goal,
                body_healthy=not row["result"]["body_failed"],
                recovery_observed=bool(stand and stand["healthy"] and stand["stability_observed"]),
            )
            actual = response_from_blobs(blobs, measure, goal)
            row.update(
                task=task,
                target=goal.tolist(),
                stroke_foot=foot,
                passing_result=measure,
                feasible_first_rank=list(feasible_first_rank(measure)),
                model_prediction=prediction,
                actual_response=None if actual is None else actual.tolist(),
            )
            records.append(row)
            counts["native_steps"] += row["native_steps"]
            counts["actor_calls"] += row["actor_decisions"]
            if counts["native_steps"] > 72000 or counts["actor_calls"] > 7200:
                raise ValueError("refinement budget exceeded")
            write_once(a.output / tag / "complete.json", row)
            print(json.dumps(dict(tag=tag, result=measure, counts=counts)), flush=True)
            return row

        for plan in plans:
            task = plan["task"]
            baseline = world(task, plan["start_latent"], f"task{task}-baseline")
            old = next(r for r in s["records"] if r["tag"] == plan["start_tag"])
            if baseline["passing_result"] != old["passing_result"] or {
                k: v["hash"] for k, v in baseline["artifacts"].items()
            } != {k: v["hash"] for k, v in old["artifacts"].items()}:
                raise ValueError("new own baseline differs from source whole raw NN/physics")
            values = [baseline] + [
                world(task, proposal["latent"], f"task{task}-proposal{k}", proposal)
                for k, proposal in enumerate(plan["fit"]["candidates"])
            ]
            best = max(
                range(len(values)),
                key=lambda k: (feasible_first_rank(values[k]["passing_result"]), -k),
            )
            selected = values[best]
            for repeat in range(2):
                actual = world(task, selected["latent"], f"task{task}-selected-repeat{repeat}")
                if actual["passing_result"] != selected["passing_result"] or {
                    k: v["hash"] for k, v in actual["artifacts"].items()
                } != {k: v["hash"] for k, v in selected["artifacts"].items()}:
                    raise ValueError("selected learned response teacher did not repeat whole raw")
            selections.append(
                dict(
                    task=task,
                    selected_tag=selected["tag"],
                    latent=selected["latent"],
                    passing_result=selected["passing_result"],
                    fit_status=plan["fit"]["status"],
                )
            )
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="LEARNED_PASS_RESPONSE_REFINEMENT_COLLECTED_PENDING_OWN_AND_TRANSFER",
                source_head=a.expected_head,
                input_file_hashes=pins,
                context=context,
                declaration_hash=digest(a.output / "declaration.json"),
                plan_hash=digest(a.output / "plan.json"),
                preflight_hash=digest(a.output / "preflight.json"),
                source_hash=a.source_hash,
                audit_hash=a.audit_hash,
                records=records,
                selections=selections,
                counts=counts,
                neural_actor_optimizer_updates=0,
                goal_complete=False,
                teammate_receiving_verified=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="PASS_RESPONSE_REFINEMENT_FAILED_PRESERVE_NO_RETRY",
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
