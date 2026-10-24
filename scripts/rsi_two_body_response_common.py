"""Bounded-model admission and actual ball-response extraction for TRAIN only."""

import json
from pathlib import Path

from scripts.rsi_native_context_recovery_common import check_pins, digest
from scripts.rsi_native_fresh_common import source_pins
from scripts.rsi_self_teacher_search_common import admit, calibrations


def admit_models(root, expected_hash, expected_head, state):
    path = root / "complete.json"
    if (
        root.is_symlink()
        or path.is_symlink()
        or digest(path) != expected_hash
        or (root / "failed.json").exists()
    ):
        raise ValueError("exact closed fitted response models required")
    m = json.loads(path.read_text())
    if (
        m["status"] != "TWO_BODY_SELF_TEACHER_RESPONSE_MODELS_FIT_NOT_PHYSICALLY_VALIDATED"
        or m["source_head"] != expected_head
        or m["fit_protocol"] != "soccer.two_body_response_fit.v2"
        or m["anchor_rule"] != "joint-normalized"
        or len(m["plans"]) != 8
        or {(p["state_id"], p["task"]) for p in m["plans"]}
        != {(i, t) for i in (20, 23, 26, 29) for t in (0, 1)}
        or any(
            m[k] != 0 for k in ("physical_steps", "student_optimizer_updates", "Fresh_cases_read")
        )
        or any(
            m[k] is not False
            for k in (
                "predictions_are_physical_evidence",
                "hardware_authorized",
                "promotion_authorized",
                "goal_complete",
            )
        )
    ):
        raise ValueError("whole explicit versioned unvalidated TRAIN response fit required")
    plans = [p for p in m["plans"] if p["state_id"] == state]
    if [p["task"] for p in plans] != [0, 1] or any(
        p["fit"]["status"] != "LOCAL_RESPONSE_MODELS_FIT_NOT_PHYSICALLY_VALIDATED"
        or len(p["new_distinct_candidates"]) != 6
        for p in plans
    ):
        raise ValueError("both predeclared six-proposal task plans required")
    sr = Path(plans[0]["source_root"])
    sp, dp, op = sr / "complete.json", sr / "declaration.json", sr.parent / "own" / "complete.json"
    if (
        any(m["input_file_hashes"].get(str(p)) != digest(p) for p in (sp, dp, op))
        or digest(sp) != plans[0]["source_hash"]
        or digest(op) != plans[0]["source_own_hash"]
        or any(
            p["source_root"] != str(sr)
            or p["source_hash"] != plans[0]["source_hash"]
            or p["source_own_hash"] != plans[0]["source_own_hash"]
            for p in plans
        )
    ):
        raise ValueError("same complete source/own evidence for both goals required")
    source, scope, own = [json.loads(p.read_text()) for p in (sp, dp, op)]
    if (
        own["producer_hash"] != digest(sp)
        or own["status"] != "WHOLE_LEARNED_SELF_TEACHER_ALL68_OWN_NATIVE_MATCH"
        or source["declaration_hash"] != digest(dp)
    ):
        raise ValueError("whole independent native search qualification required")
    cr, co, rr = [
        Path(scope[k]) for k in ("calibration_root", "calibration_own_root", "review_root")
    ]
    cal, ctx, pf, pins, bank = admit(cr, co, rr, scope["calibration_head"])
    for key, h in m["input_file_hashes"].items():
        if key in pins and pins[key] != h:
            raise ValueError("conflicting physical model ancestry")
        pins[key] = h
    pins[str(path)] = expected_hash
    source_pins(pins)
    check_pins(pins)
    import numpy as np

    from rosclaw_soccer.rsi.stroke_search import checked_latent

    for plan in plans:
        start = checked_latent(plan["start_latent"])
        if [p["candidate_index"] for p in plan["new_distinct_candidates"]] != list(range(6)):
            raise ValueError("exact complete distinct model proposals required")
        for p in plan["new_distinct_candidates"]:
            z = checked_latent(p["latent"])
            prediction = np.asarray(p["predicted_response"])
            if (
                (abs(z - start) > 0.2 + 1e-12).any()
                or prediction.shape != (4,)
                or not np.isfinite(prediction).all()
            ):
                raise ValueError("bounded finite empirical response proposal required")
    return m, plans, source, cal, ctx, pf, pins, bank, cr, sr


def anchor(plan, source, cal, cal_root, source_root):
    task, state = plan["task"], plan["state_id"]
    rows = [
        (cal_root, f"cal-t{task}-p{r['student_prefix_controls']}", r)
        for r in calibrations(cal, state, task)
    ] + [(source_root, r["tag"], r) for r in source["records"] if r["search_task"] == task]
    found = [(root, row) for root, tag, row in rows if tag == plan["start_tag"]]
    if len(found) != 1 or found[0][1]["course"]["latent"] != plan["start_latent"]:
        raise ValueError("exact already measured model anchor required")
    root, row = found[0]
    return dict(root=root, row=row, tag=plan["start_tag"], latent=plan["start_latent"])


def actual_response(root, row, layout):
    import numpy as np

    from rosclaw_soccer.rsi.two_body_pass_response import measured_response

    paths = {}
    for name in ("initial", "physical"):
        art = row["artifacts"][name]
        p = root / art["path"]
        if (
            p.is_symlink()
            or not p.resolve().is_relative_to(root.resolve())
            or digest(p) != art["hash"]
        ):
            raise ValueError("exact confined native ball arrays required")
        paths[name] = p
    qi, vi, nq, nv = layout
    with np.load(paths["initial"], allow_pickle=False) as z:
        initial = z["qpos"][qi : qi + 3].copy()
    with np.load(paths["physical"], allow_pickle=False) as z:
        if z["qpos"].shape[1:] != (10, nq) or z["qvel"].shape[1:] != (10, nv):
            raise ValueError("same compiled native layout required")
        pos = z["qpos"].reshape(-1, nq)[:, qi : qi + 3].copy()
        vel = z["qvel"].reshape(-1, nv)[:, vi : vi + 3].copy()
    response = measured_response(
        pos, vel, initial, np.array(row["course"]["target"]), row["outcome"]
    )
    return None if response is None else response.tolist()


def verify_refinement(report, plans, anchors):
    from rosclaw_soccer.rsi.two_body_pass_response import refinement_rank

    if len(report["records"]) != 16 or len(report["selections"]) != 2:
        raise ValueError("all16 declared refinement worlds required")
    cursor = 0

    def take(plan, z, tag, role, index, prediction):
        nonlocal cursor
        row = report["records"][cursor]
        course = anchors[plan["task"]]["row"]["course"] | dict(latent=z)
        if (
            row["index"] != cursor
            or row["course"] != course
            or row["tag"] != tag
            or row["task"] != plan["task"]
            or row["role"] != role
            or row["model_candidate_index"] != index
            or row["predicted_response"] != prediction
            or type(row["prefix_identity_fields"]) is not int
            or row["prefix_identity_fields"] <= 0
            or row["neural_actor_optimizer_updates"] != 0
        ):
            raise ValueError("exact ordered model proposals and whole repetitions required")
        cursor += 1
        return dict(row=row, tag=tag, latent=z)

    for plan in plans:
        task = plan["task"]
        values = [anchors[task]]
        for p in plan["new_distinct_candidates"]:
            values.append(
                take(
                    plan,
                    p["latent"],
                    f"t{task}-model{p['candidate_index']}",
                    "proposal",
                    p["candidate_index"],
                    p["predicted_response"],
                )
            )
        best = max(
            enumerate(values), key=lambda kv: (refinement_rank(kv[1]["row"]["outcome"]), -kv[0])
        )[1]
        for repeat in range(2):
            row = take(
                plan, best["latent"], f"t{task}-repeat{repeat}", "selected_repeat", None, None
            )
            if row["row"]["outcome"] != best["row"]["outcome"]:
                raise ValueError("selected actual outcome did not repeat")
        expected = dict(
            task=task,
            selected=best["tag"],
            latent=best["latent"],
            rank=list(refinement_rank(best["row"]["outcome"])),
            outcome=best["row"]["outcome"],
        )
        if report["selections"][task] != expected:
            raise ValueError("actual joint-cost selection differs")
    if cursor != 16:
        raise ValueError("all16 new worlds required")
