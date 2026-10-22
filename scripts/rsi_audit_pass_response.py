"""Recompute learned teacher fits, then whole own-history/native refinement.

Shared codecs/response fitting/stroke rules are evaluated on admitted data
and own actual physics. Saved joint actions or hidden states are NOT executed.
One-body proof remains insufficient for team transfer or student promotion.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-hash", "producer-head", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins
    from scripts.rsi_pass_response_common import admit, plan_task, response_from_blobs

    clean(a.expected_head)
    paths = [
        a.root / n for n in ("complete.json", "declaration.json", "plan.json", "preflight.json")
    ]
    if (
        a.output.exists()
        or any(x.is_symlink() for x in paths)
        or (a.root / "failed.json").exists()
        or digest(paths[0]) != a.source_hash
    ):
        raise ValueError("exact completed refinement and unused audit root required")
    s, d, plan, pf = [json.loads(x.read_text()) for x in paths]
    if (
        s["status"] != "LEARNED_PASS_RESPONSE_REFINEMENT_COLLECTED_PENDING_OWN_AND_TRANSFER"
        or d["protocol"] != "soccer.learned_local_pass_response_refinement.v1"
        or any(r["source_head"] != a.producer_head for r in (s, d))
        or s["declaration_hash"] != digest(paths[1])
        or s["plan_hash"] != digest(paths[2])
        or d["plan_hash"] != digest(paths[2])
        or s["preflight_hash"] != digest(paths[3])
        or any(
            r[k] is not False
            for r in (s, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
        or len(plan["tasks"]) != 2
        or d["maximum_worlds"] != 18
        or d["maximum_native_steps"] != 72000
        or d["maximum_actor_calls"] != 7200
    ):
        raise ValueError("complete finite one-body refinement declaration required")
    pins = dict(s["input_file_hashes"])
    cp, _ = receipt(pins, s["source_hash"])
    op, _ = receipt(pins, s["audit_hash"])
    pool, pool_d, pool_pf, _ = admit(cp.parent, s["source_hash"], op.parent, s["audit_hash"])
    recomputed = [plan_task(cp.parent, pool, t) for t in range(2)]
    if (
        recomputed != plan["tasks"]
        or pf["tasks"] != pool_pf["tasks"]
        or pf["center"] != pool_pf["center"]
        or pf["stroke_foot"] != pool_pf["stroke_foot"]
    ):
        raise ValueError("own data selection/unique rows/regression/proposals/initial tasks differ")
    for path in paths:
        pins[str(path)] = digest(path)
    for claim in s["records"]:
        for art in claim["artifacts"].values():
            path = a.root / art["path"]
            if (
                path.is_symlink()
                or path.resolve().parent != (a.root / claim["tag"]).resolve()
                or digest(path) != art["hash"]
            ):
                raise ValueError("all exact confined new physical trajectories required")
            pins[str(path)] = art["hash"]
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            producer_hash=a.source_hash,
            input_file_hashes=pins,
            all_local_response_models_recomputed=True,
            all_source_data_own_audited=True,
            source_actions_executed=False,
            teacher_hidden_state_borrowed=False,
            shared_codecs_and_teacher_rule_on_own_states=True,
            producer_integrator_used=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    counts = dict(native=0, neural=0, kinematic_intent=0)
    records = []
    try:
        import numpy as np

        from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
        from rosclaw_soccer.rsi.pass_response_model import feasible_first_rank
        from rosclaw_soccer.rsi.targeted_pass_precision import targeted_pass_precision
        from scripts.rsi_stroke_search_common import open_motor
        from scripts.rsi_stroke_world_audit import replay_stroke_world

        bench, base = open_motor(
            pins, pool_d["baseline_training_hash"], {s["context"]: pf["tasks"][0]["context"]}
        )
        c = pool_d["candidate"]
        _, training = receipt(pins, c["training_hash"])
        actor, _ = load_dual_candidate_for_sim_audit(
            base.parent,
            parent_lineage_hash=training["parent_lineage_hash"],
            root=Path(c["training_root"]),
            expected_receipt_hash=c["training_hash"],
            checkpoint_kind=c["kind"],
        )
        cursor = 0
        selections = []

        def replay(task, tag, latent, prediction=None):
            nonlocal cursor
            claim = s["records"][cursor]
            cursor += 1
            if (
                claim["task"] != task
                or claim["tag"] != tag
                or claim["latent"] != latent
                or claim["model_prediction"] != prediction
                or claim["target"] != pf["tasks"][task]["goal"]
                or claim["stroke_foot"] != pf["stroke_foot"]
            ):
                raise ValueError("each planned bounded teacher input must match declared order")
            raw = {}
            for key, art in claim["artifacts"].items():
                with np.load(a.root / art["path"], allow_pickle=False) as z:
                    raw[key] = {k: z[k].copy() for k in z.files}
            result, stand, good, _, goal = replay_stroke_world(
                bench,
                actor,
                pf["tasks"][task]["context"],
                pf["center"],
                np.array(latent),
                raw,
                counts,
                stroke_foot=pf["stroke_foot"],
            )
            if (
                result != claim["result"]
                or stand != claim["stand"]
                or good != claim["healthy_quality"]
            ):
                raise ValueError("own original contact/recovery diagnostics differ")
            measure = targeted_pass_precision(
                np.concatenate([r["qpos"] for r in raw.values()]).reshape(-1, 43),
                np.concatenate([r["qvel"] for r in raw.values()]).reshape(-1, 41),
                np.concatenate([r["forces"] for r in raw.values()]).reshape(-1, 3),
                goal,
                body_healthy=not result["body_failed"],
                recovery_observed=bool(stand and stand["healthy"] and stand["stability_observed"]),
            )
            response = response_from_blobs(raw, measure, goal)
            if (
                measure != claim["passing_result"]
                or (None if response is None else response.tolist()) != claim["actual_response"]
                or list(feasible_first_rank(measure)) != claim["feasible_first_rank"]
            ):
                raise ValueError("own physical model-response and success outcome differ")
            row = dict(
                tag=tag, task=task, passing_result=measure, actual_response=claim["actual_response"]
            )
            records.append(row)
            write_once(a.output / (tag + ".json"), row)
            print(json.dumps(dict(tag=tag, passing_result=measure, counts=counts)), flush=True)
            return claim

        for task_plan in recomputed:
            task = task_plan["task"]
            baseline = replay(task, f"task{task}-baseline", task_plan["start_latent"])
            old = next(r for r in pool["records"] if r["tag"] == task_plan["start_tag"])
            if baseline["passing_result"] != old["passing_result"] or {
                k: v["hash"] for k, v in baseline["artifacts"].items()
            } != {k: v["hash"] for k, v in old["artifacts"].items()}:
                raise ValueError("whole raw baseline changed from source archived teacher")
            values = [baseline] + [
                replay(task, f"task{task}-proposal{k}", v["latent"], v)
                for k, v in enumerate(task_plan["fit"]["candidates"])
            ]
            best = max(
                range(len(values)),
                key=lambda k: (feasible_first_rank(values[k]["passing_result"]), -k),
            )
            chosen = values[best]
            for repeat in range(2):
                row = replay(task, f"task{task}-selected-repeat{repeat}", chosen["latent"])
                if row["passing_result"] != chosen["passing_result"] or {
                    k: v["hash"] for k, v in row["artifacts"].items()
                } != {k: v["hash"] for k, v in chosen["artifacts"].items()}:
                    raise ValueError("selected model-based teacher whole raw repeat differs")
            selections.append(
                dict(
                    task=task,
                    selected_tag=chosen["tag"],
                    latent=chosen["latent"],
                    passing_result=chosen["passing_result"],
                    fit_status=task_plan["fit"]["status"],
                )
            )
        if (
            cursor != len(s["records"])
            or selections != s["selections"]
            or counts["native"] != s["counts"]["native_steps"]
            or counts["neural"] != s["counts"]["actor_calls"]
            or counts["kinematic_intent"] != 0
        ):
            raise ValueError("whole refinement selection and native accounting differ")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="LEARNED_PASS_RESPONSE_ALL_MODELS_AND_WHOLE_NATIVE_REPLAY_MATCH",
                source_head=a.expected_head,
                producer_hash=a.source_hash,
                input_file_hashes=pins,
                context=s["context"],
                counts=counts,
                records=records,
                selections=selections,
                neural_actor_optimizer_updates=0,
                goal_complete=False,
                teammate_receiving_verified=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="PASS_RESPONSE_AUDIT_FAILED_PRESERVE_NO_RETRY",
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
