"""Compare ALL48 model predictions with independently reproduced native outcomes."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("closure-root", "model-root", "shards-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("closure-head", "model-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    from rosclaw_soccer.rsi.pass_response_validation import response_validation
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    cp, mp = a.closure_root / "complete.json", a.model_root / "complete.json"
    if a.output.exists() or any(p.is_symlink() for p in (a.closure_root, cp, a.model_root, mp)):
        raise ValueError("unused analysis output and ordinary whole completed artifacts required")
    if (a.closure_root / "failed.json").exists() or digest(mp) != a.model_hash:
        raise ValueError("exact closed study and model required")
    c, models = [json.loads(p.read_text()) for p in (cp, mp)]
    if (
        c["status"] != "TWO_BODY_RESPONSE_FOUR_STATE_REFINEMENT_CLOSED_NOT_NEURAL_TRAINING"
        or c["source_head"] != a.closure_head
        or models["fit_protocol"] != "soccer.two_body_response_fit.v2"
        or c["new_worlds_source"] != 64
        or c["new_worlds_own"] != 64
        or c["input_file_hashes"].get(str(mp)) != a.model_hash
    ):
        raise ValueError("whole independently authenticated local response study required")
    pins = {str(cp): digest(cp), str(mp): a.model_hash}
    tasks = []
    for state in (20, 23, 26, 29):
        proof = next(x for x in c["proofs"] if x["state_id"] == state)
        sr, own = (
            a.shards_root / f"state{state}" / "source",
            a.shards_root / f"state{state}" / "own",
        )
        paths = [sr / "complete.json", own / "complete.json"]
        if any(p.is_symlink() for p in (sr, own, *paths)) or any(
            (r / "failed.json").exists() for r in (sr, own)
        ):
            raise ValueError("whole source and own evidence required")
        if any(
            digest(path) != proof[k]
            for path, k in zip(paths, ("source_hash", "own_hash"), strict=True)
        ):
            raise ValueError("exact closed-study shard hashes required")
        s, o = [json.loads(p.read_text()) for p in paths]
        if (
            o["producer_hash"] != digest(paths[0])
            or o["status"] != "TWO_BODY_RESPONSE_REFINEMENT_ALL16_OWN_NATIVE_MATCH"
            or o["records"]
            != [{k: v for k, v in r.items() if k != "artifacts"} for r in s["records"]]
        ):
            raise ValueError("actual whole independently reproduced proposals required")
        pins.update({str(p): digest(p) for p in paths})
        for task in (0, 1):
            plan = next(x for x in models["plans"] if x["state_id"] == state and x["task"] == task)
            proposals = [r for r in o["records"] if r["task"] == task and r["role"] == "proposal"]
            if len(proposals) != 6 or any(
                r["course"]["latent"] != z["latent"]
                or r["predicted_response"] != z["predicted_response"]
                for r, z in zip(proposals, plan["new_distinct_candidates"], strict=True)
            ):
                raise ValueError("exact all six fixed model predictions required")
            tasks.append(
                dict(
                    state_id=state,
                    task=task,
                    **response_validation(
                        anchor=plan["actual_start_response"], proposals=proposals
                    ),
                )
            )
    source_pins(pins)
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    report = dict(
        status="ALL48_TWO_BODY_PREDICTIONS_COMPARED_WITH_WHOLE_OWN_PHYSICS",
        source_head=a.expected_head,
        input_file_hashes=pins,
        tasks=tasks,
        predicted_improvements=sum(t["predicted_improvements"] for t in tasks),
        actual_improvements=sum(t["actual_improvements"] for t in tasks),
        missing_responses=sum(t["missing"] for t in tasks),
        predicted_improvement_but_actual_not=sum(
            t["predicted_improvement_but_actual_not"] for t in tasks
        ),
        diagnostics_are_not_new_physics=True,
        actual_improvements_are_delivery_cost_only=True,
        distinct_initial_states=4,
        goal_tasks=8,
        model_proposals=48,
        physical_steps_this_analysis=0,
        student_optimizer_updates=0,
        Fresh_cases_read=0,
        student_improvement_verified=False,
        correction_bank_ready=False,
        hardware_authorized=False,
        promotion_authorized=False,
        goal_complete=False,
    )
    write_once(a.output / "complete.json", report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "predicted_improvements",
                    "actual_improvements",
                    "missing_responses",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
