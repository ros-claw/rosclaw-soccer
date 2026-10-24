"""Whole-pilot admission and fixed nearby TRAIN transfer decisions."""

import json

from scripts.rsi_native_context_recovery_common import check_pins, digest
from scripts.rsi_native_fresh_common import receipt
from scripts.rsi_self_teacher_search_common import calibrations
from scripts.rsi_two_body_response_common import admit_models


def admit_pilot(root, expected_hash, expected_head, model_root, model_hash, model_head):
    path = root / "complete.json"
    if (
        root.is_symlink()
        or path.is_symlink()
        or (root / "failed.json").exists()
        or digest(path) != expected_hash
    ):
        raise ValueError("exact whole independent response pilot closure required")
    c = json.loads(path.read_text())
    if (
        c["status"] != "TWO_BODY_RESPONSE_FOUR_STATE_REFINEMENT_CLOSED_NOT_NEURAL_TRAINING"
        or c["source_head"] != expected_head
        or c["distinct_initial_states"] != 4
        or c["goal_tasks"] != 8
        or len(c["proofs"]) != 4
        or c["after"]["all_healthy_clean"] is not True
        or c["after"]["controlled"] < 6
        or c["student_optimizer_updates"] != 0
        or c["Fresh_cases_read"] != 0
        or any(
            c[k] is not False
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError("audited safe functional pilot, not a precision promotion, required")
    _, _, _, cal, scope, pf, pins, bank, cr, _ = admit_models(
        model_root, model_hash, model_head, 20
    )
    already = dict(pins)
    for key, h in c["input_file_hashes"].items():
        if key in pins and pins[key] != h:
            raise ValueError("conflicting complete pilot ancestry")
        pins[key] = h
    pins[str(path)] = expected_hash
    donors = []
    from rosclaw_soccer.rsi.pass_teacher_transfer_plan import transfer_plans
    from rosclaw_soccer.rsi.pass_transfer_learning import controlled_pass_label

    for state in (20, 23, 26, 29):
        proof = next(x for x in c["proofs"] if x["state_id"] == state)
        sp, s = receipt(pins, proof["source_hash"])
        op, o = receipt(pins, proof["own_hash"])
        if (
            s["status"] != "TWO_BODY_RESPONSE_REFINEMENT16_COLLECTED_PENDING_ALL_OWN"
            or o["status"] != "TWO_BODY_RESPONSE_REFINEMENT_ALL16_OWN_NATIVE_MATCH"
            or o["producer_hash"] != digest(sp)
            or len(o["records"]) != 16
            or o["records"]
            != [{k: v for k, v in r.items() if k != "artifacts"} for r in s["records"]]
            or o["selections"] != s["selections"]
            or o["counts"] != s["counts"]
            or s["state_id"] != state
            or o["state_id"] != state
            or s["model_hash"] != model_hash
        ):
            raise ValueError("all four complete source and independent replay matches required")
        for task in (0, 1):
            repeats = [
                r for r in o["records"] if r["task"] == task and r["role"] == "selected_repeat"
            ]
            if (
                len(repeats) != 2
                or repeats[0]["outcome"] != repeats[1]["outcome"]
                or repeats[0]["course"] != repeats[1]["course"]
            ):
                raise ValueError("reproducible selected whole teacher required")
            row = repeats[0]
            outcome = row["outcome"]
            safe = bool(
                outcome["whole_body_healthy"]
                and outcome["passer_full100NNrecovery"]
                and outcome["passer_clean_foot"]
                and not outcome["B_excluded_contact"]
            )
            donors.append(
                dict(
                    course=row["course"],
                    controlled=controlled_pass_label(outcome),
                    healthy_clean=safe,
                    evidence_hash=digest(op),
                )
            )
    courses = [
        calibrations(cal, state, task)[0]["course"]
        for state in (21, 22, 24, 25, 27, 28)
        for task in (0, 1)
    ]
    plans = transfer_plans(courses, donors)
    # Inputs already hashed by admit_models need not be redundantly rehashed
    # within this same preallocation phase; new receipt/raw inputs still are.
    check_pins({k: h for k, h in pins.items() if already.get(k) != h})
    return c, plans, cal, scope, pf, pins, bank, cr


def verify_expansion(report, plans, calibration, state):
    from rosclaw_soccer.rsi.two_body_pass_response import refinement_rank

    local = [p for p in plans if p["state_id"] == state]
    if (
        [p["task"] for p in local] != [0, 1]
        or len(report["records"]) != 6
        or len(report["selections"]) != 2
    ):
        raise ValueError("both goals with one transfer plus two whole repeats required")
    cursor = 0
    for plan in local:
        task = plan["task"]

        def take(course, tag, role, task=task, plan=plan):
            nonlocal cursor
            row = report["records"][cursor]
            if (
                row["index"] != cursor
                or row["course"] != course
                or row["tag"] != tag
                or row["task"] != task
                or row["role"] != role
                or row["proposal_donor_state"] != plan["donor_state"]
                or row["proposal_donor_controlled"] != plan["donor_controlled"]
                or type(row["prefix_identity_fields"]) is not int
                or row["prefix_identity_fields"] <= 0
                or row["neural_actor_optimizer_updates"] != 0
            ):
                raise ValueError("exact fixed proposal and selected repeats required")
            cursor += 1
            return dict(row=row, tag=tag)

        values = [
            dict(row=r, tag=f"cal-t{task}-p{r['student_prefix_controls']}")
            for r in calibrations(calibration, state, task)
        ]
        values.append(take(plan["course"], f"t{task}-transfer", "transfer"))
        best = max(
            enumerate(values), key=lambda kv: (refinement_rank(kv[1]["row"]["outcome"]), -kv[0])
        )[1]
        for repeat in range(2):
            r = take(best["row"]["course"], f"t{task}-repeat{repeat}", "selected_repeat")
            if r["row"]["outcome"] != best["row"]["outcome"]:
                raise ValueError("selected physical outcome failed exact repetition")
        expected = dict(
            task=task,
            selected=best["tag"],
            latent=best["row"]["course"]["latent"],
            outcome=best["row"]["outcome"],
            transfer_selected=best["tag"] == f"t{task}-transfer",
        )
        if report["selections"][task] != expected:
            raise ValueError("fallback must not masquerade as transfer success")
