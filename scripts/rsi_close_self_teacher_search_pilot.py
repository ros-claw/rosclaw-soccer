"""Aggregate ALL4 source/whole-own CEM shards against their registered plan.

Receipt reanalysis only, not another physical replay or learned-policy proof.
The4 initial states x2 goals are8 tasks, NOT272 independent physical states.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("shards-root", "plan", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("plan-hash", "producer-head", "auditor-head", "expected-head"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    from rosclaw_soccer.rsi.pass_transfer_learning import controlled_pass_label
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_self_teacher_search_common import verify_search

    if a.output.exists() or a.plan.is_symlink() or digest(a.plan) != a.plan_hash:
        raise ValueError("unused review and sealed pilot plan required")
    plan = json.loads(a.plan.read_text())
    if (
        plan["schema"] != "soccer.whole_learned_self_teacher_CEM_pilot.v1"
        or plan["source_head"] != a.producer_head
        or plan["design"]["states"] != [20, 23, 26, 29]
    ):
        raise ValueError("unchanged predeclared pilot required")
    pins = {str(a.plan): a.plan_hash}
    proofs = []
    selected = []
    counts = dict(native_steps=0, actor_calls=0)
    cal_path = Path(plan["public_calibration"]["root"]) / "complete.json"
    if digest(cal_path) != plan["public_calibration"]["hash"]:
        raise ValueError("same original whole calibration required")
    cal = json.loads(cal_path.read_text())
    pins[str(cal_path)] = digest(cal_path)
    for state in (20, 23, 26, 29):
        root = a.shards_root / f"state{state}"
        sp, op, dp = [
            root / r / n
            for r, n in (
                ("source", "complete.json"),
                ("own", "complete.json"),
                ("source", "declaration.json"),
            )
        ]
        if any(x.is_symlink() for x in (root, sp, op, dp)) or any(
            (root / r / "failed.json").exists() for r in ("source", "own")
        ):
            raise ValueError("ALL4 source/own shards must close")
        s, o, d = [json.loads(f.read_text()) for f in (sp, op, dp)]
        if (
            s["status"] != "WHOLE_LEARNED_SELF_TEACHER_SEARCH_COLLECTED_PENDING_ALL68_OWN"
            or o["status"] != "WHOLE_LEARNED_SELF_TEACHER_ALL68_OWN_NATIVE_MATCH"
            or s["source_head"] != a.producer_head
            or o["source_head"] != a.auditor_head
            or o["producer_hash"] != digest(sp)
            or s["declaration_hash"] != digest(dp)
            or d["preflight_only"] is not False
            or d["state_id"] != state
            or s["state_id"] != state
            or o["state_id"] != state
            or d["calibration_hash"] != plan["public_calibration"]["hash"]
            or d["calibration_own_hash"] != plan["public_calibration"]["own_hash"]
            or d["review_hash"] != plan["public_calibration"]["review_hash"]
            or o["counts"] != s["counts"]
            or o["selections"] != s["selections"]
            or len(s["records"]) != 68
            or o["records"]
            != [{k: v for k, v in r.items() if k != "artifacts"} for r in s["records"]]
            or o["manual_native_clock"] is not True
            or o["source_actions_executed"] is not False
            or o["borrowed_hidden_state"] is not False
            or o["all_CEM_decisions_reconstructed"] is not True
            or any(
                v[k] is not False
                for v in (s, o, d)
                for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
            )
        ):
            raise ValueError("complete exact independently reproduced searches required")
        verify_search(s | dict(records=o["records"]), cal)
        for key in counts:
            counts[key] += o["counts"][key]
        for x in (sp, op, dp):
            pins[str(x)] = digest(x)
        proofs.append(dict(state_id=state, source_hash=digest(sp), own_hash=digest(op)))
        selected.extend(
            dict(state_id=state, task=r["task"], outcome=r["outcome"], latent=r["latent"])
            for r in o["selections"]
        )
    controlled = sum(controlled_pass_label(r["outcome"]) for r in selected)
    healthy_clean = all(
        o["whole_body_healthy"]
        and o["passer_full100NNrecovery"]
        and o["passer_clean_foot"]
        and not o["B_excluded_contact"]
        for o in (r["outcome"] for r in selected)
    )
    precise = sum(r["outcome"]["combined_precision_and_functional_receive"] for r in selected)
    source_pins(pins)
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    report = dict(
        status="SELF_TEACHER_CEM_FOUR_STATE_PILOT_CLOSED_NOT_NEURAL_TRAINING",
        source_head=a.expected_head,
        input_file_hashes=pins,
        proofs=proofs,
        selected=selected,
        selected_controlled=controlled,
        all_selected_healthy_clean=healthy_clean,
        selected_strict_composite=precise,
        functional_teacher_feasible=controlled >= 6 and healthy_clean,
        precision_teacher_demonstrated=precise >= 1,
        distinct_initial_states=4,
        goal_tasks=8,
        new_worlds_source=272,
        new_worlds_own=272,
        new_native_steps_source_plus_own=2 * counts["native_steps"],
        physical_steps_this_aggregation=0,
        optimizer_updates=0,
        Fresh_cases_read=0,
        expanded_teacher_coverage_verified=False,
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
                    "selected_controlled",
                    "all_selected_healthy_clean",
                    "selected_strict_composite",
                    "functional_teacher_feasible",
                    "precision_teacher_demonstrated",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
