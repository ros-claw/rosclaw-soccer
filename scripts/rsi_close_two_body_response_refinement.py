"""Close all four learned-response shards, without confusing predictions with physics."""

import argparse
import json
from pathlib import Path


def summarize(selected):
    from rosclaw_soccer.rsi.pass_transfer_learning import controlled_pass_label

    controlled = [controlled_pass_label(r["outcome"]) for r in selected]
    precise = [r["outcome"]["combined_precision_and_functional_receive"] for r in selected]
    clean = all(
        o["whole_body_healthy"]
        and o["passer_full100NNrecovery"]
        and o["passer_clean_foot"]
        and not o["B_excluded_contact"]
        for o in (r["outcome"] for r in selected)
    )
    return dict(
        controlled=sum(controlled),
        strict_composite=sum(precise),
        actual_contact_and_strict=sum(c and p for c, p in zip(controlled, precise, strict=True)),
        all_healthy_clean=clean,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("shards-root", "plan", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("plan-hash", "producer-head", "auditor-head", "expected-head"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_two_body_response_common import anchor

    if a.output.exists() or a.plan.is_symlink() or digest(a.plan) != a.plan_hash:
        raise ValueError("unused closure and exact preregistration required")
    plan = json.loads(a.plan.read_text())
    if (
        plan["schema"] != "soccer.two_body_response_refinement_plan.v1"
        or plan["source_head"] != a.producer_head
        or plan["states"] != [20, 23, 26, 29]
        or plan["source_worlds"] != 64
        or plan["independent_own_worlds"] != 64
    ):
        raise ValueError("whole four-state preregistered refinement required")
    pins = {str(a.plan): a.plan_hash}
    selected, anchors, proofs = [], [], []
    counts = dict(native_steps=0, actor_calls=0)
    for state in plan["states"]:
        root = a.shards_root / f"state{state}"
        paths = [
            root / r / n
            for r, n in (
                ("source", "complete.json"),
                ("own", "complete.json"),
                ("source", "declaration.json"),
                ("source", "scene.xml"),
            )
        ]
        if any(x.is_symlink() for x in (root, *paths)) or any(
            (root / k / "failed.json").exists() for k in ("source", "own")
        ):
            raise ValueError("all four source/own shards must close without failure")
        s, o, d = [json.loads(x.read_text()) for x in paths[:3]]
        if (
            s["status"] != "TWO_BODY_RESPONSE_REFINEMENT16_COLLECTED_PENDING_ALL_OWN"
            or o["status"] != "TWO_BODY_RESPONSE_REFINEMENT_ALL16_OWN_NATIVE_MATCH"
            or s["source_head"] != a.producer_head
            or d["source_head"] != a.producer_head
            or o["source_head"] != a.auditor_head
            or o["producer_hash"] != digest(paths[0])
            or s["declaration_hash"] != digest(paths[2])
            or s["scene_hash"] != digest(paths[3])
            or any(x["state_id"] != state for x in (s, o, d))
            or any(x["model_hash"] != plan["model_hash"] for x in (s, o, d))
            or o["counts"] != s["counts"]
            or o["selections"] != s["selections"]
            or len(s["records"]) != 16
            or o["records"]
            != [{k: v for k, v in r.items() if k != "artifacts"} for r in s["records"]]
            or o["manual_native_clock"] is not True
            or o["source_actions_executed"] is not False
            or o["borrowed_hidden_state"] is not False
            or o["actual_responses_rederived"] is not True
            or o["joint_selection_reconstructed"] is not True
            or any(
                x[k] != 0 for x in (s, o) for k in ("student_optimizer_updates", "Fresh_cases_read")
            )
            or any(
                x[k] is not False
                for x in (s, o, d)
                for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
            )
        ):
            raise ValueError("exact all16 independent native proof required per shard")
        # Own auditor already re-admitted full raw ancestry and reconstructed all choices.
        for x in (s, o):
            for key, h in x["input_file_hashes"].items():
                if key in pins and pins[key] != h:
                    raise ValueError("conflicting whole physical ancestry")
                pins[key] = h
        for path in paths:
            pins[str(path)] = digest(path)
        model_path = Path(d["model_root"]) / "complete.json"
        if digest(model_path) != plan["model_hash"]:
            raise ValueError("unchanged fitted model plan required")
        models = json.loads(model_path.read_text())
        old_root = Path(next(x["source_root"] for x in models["plans"] if x["state_id"] == state))
        old_paths = [old_root / n for n in ("complete.json", "declaration.json")]
        old, old_decl = [json.loads(x.read_text()) for x in old_paths]
        cal_root = Path(old_decl["calibration_root"])
        cal_path = cal_root / "complete.json"
        for x in (*old_paths, cal_path):
            if str(x) not in pins or digest(x) != pins[str(x)]:
                raise ValueError("exact already audited anchor ancestry required")
        cal = json.loads(cal_path.read_text())
        for item in models["plans"]:
            if item["state_id"] == state:
                measured = anchor(item, old, cal, cal_root, old_root)
                anchors.append(
                    dict(state_id=state, task=item["task"], outcome=measured["row"]["outcome"])
                )
        for key in counts:
            counts[key] += s["counts"][key]
        selected.extend(dict(state_id=state, **x) for x in s["selections"])
        proofs.append(dict(state_id=state, source_hash=digest(paths[0]), own_hash=digest(paths[1])))
    if len(selected) != 8 or len(anchors) != 8:
        raise ValueError("all eight unchanged goal tasks required")
    before, after = summarize(anchors), summarize(selected)
    source_pins(pins)
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    report = dict(
        status="TWO_BODY_RESPONSE_FOUR_STATE_REFINEMENT_CLOSED_NOT_NEURAL_TRAINING",
        source_head=a.expected_head,
        input_file_hashes=pins,
        proofs=proofs,
        anchors=anchors,
        selected=selected,
        before=before,
        after=after,
        precision_teacher_demonstrated=after["actual_contact_and_strict"] > 0
        and after["all_healthy_clean"],
        next_agent_action="EXPAND_TEACHER_COVERAGE_BEFORE_FITTING"
        if after["actual_contact_and_strict"] > 0
        else "DIAGNOSE_PHYSICAL_RESPONSE_ERROR_NOT_REPEAT_SAME_SWEEP",
        distinct_initial_states=4,
        goal_tasks=8,
        new_worlds_source=64,
        new_worlds_own=64,
        new_native_steps_source_plus_own=2 * counts["native_steps"],
        physical_steps_this_aggregation=0,
        student_optimizer_updates=0,
        Fresh_cases_read=0,
        student_improvement_verified=False,
        correction_bank_ready=False,
        hardware_authorized=False,
        promotion_authorized=False,
        goal_complete=False,
    )
    write_once(a.output / "complete.json", report)
    print(json.dumps(dict(status=report["status"], before=before, after=after)), flush=True)


if __name__ == "__main__":
    main()
