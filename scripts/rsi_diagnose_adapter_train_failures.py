"""Reanalyse public consumed TRAIN failures; never open sealed Fresh cases.

No rollouts, fitting, reward changes or promotion. Distinguishes omitted safe
anchors from actual closed-loop failure on tasks that already had a selected
physically qualified teacher. Counts are observations, not causal attribution.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def diagnose(records, candidate, plan):
    from rosclaw_soccer.rsi.pass_transfer_learning import controlled_pass_label

    old = {
        (r["source_course"]["state_id"], r["source_course"]["task"]): r
        for r in records
        if r["role"] == "new_unassisted"
    }
    keys = {(i, t) for i in range(40) for t in (0, 1)}
    if len(candidate) != 80 or set(old) != keys:
        raise ValueError("all80 matched consumed TRAIN tasks required")
    selected = {
        (records[i]["source_course"]["state_id"], records[i]["source_course"]["task"])
        for i in plan["teacher_indices"] + plan["baseline_pass_indices"]
    }
    result = {}
    seen = set()
    for row in candidate:
        c, y = row["course"], row["outcome"]
        key = (c["state_id"], c["task"])
        if key not in old or key in seen or c != old[key]["source_course"]:
            raise ValueError("exact unique public TRAIN task inputs required")
        seen.add(key)
        ctx = str(c["context"])
        count = result.setdefault(ctx, Counter())
        x = old[key]["measured_outcome"]
        for name, o in (("baseline", x), ("candidate", y)):
            count[name + "_controlled"] += controlled_pass_label(o)
            count[name + "_clean_foot"] += o["passer_clean_foot"]
            count[name + "_actual_foot_strike"] += (
                o["incoming"]["A_first_foot_strike_substep"] is not None
            )
            count[name + "_B_contact"] += o["incoming"]["B_first_ball_contact_substep"] is not None
            count[name + "_B_control"] += o["B_legal_functional_receive_before_deadline"]
            count[name + "_precise_moderate_delivery"] += o["precise_moderate_delivery"]
        count["baseline_clean_without_selected_plan"] += (
            x["passer_clean_foot"] and key not in selected
        )
        unsafe = x["passer_clean_foot"] and not y["passer_clean_foot"]
        count["new_nonfoot"] += unsafe
        count["new_nonfoot_with_selected_plan"] += unsafe and key in selected
        count["new_nonfoot_without_selected_plan"] += unsafe and key not in selected
        count["regressions"] += controlled_pass_label(x) and not controlled_pass_label(y)
        count["rescues"] += not controlled_pass_label(x) and controlled_pass_label(y)
        count["tasks"] += 1
    if set(result) != {"1", "4", "20", "21"} or any(v["tasks"] != 20 for v in result.values()):
        raise ValueError("all4 original balanced TRAIN contexts required")
    return {k: dict(v) for k, v in result.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("bank-root", "manifest", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("bank-hash", "manifest-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    bp = a.bank_root / "complete.json"
    if a.output.exists() or digest(bp) != a.bank_hash or digest(a.manifest) != a.manifest_hash:
        raise ValueError("unused analysis output and exact public TRAIN commitments required")
    bank, manifest = [json.loads(f.read_text()) for f in (bp, a.manifest)]
    if (
        manifest["schema"] != "soccer.pass_fresh_frontier_inputs.v1"
        or [c["seed"] for c in manifest["candidates"]] != [105023, 105024, 105025]
        or len(bank["records"]) != 274
        or bank["Fresh_fit_rows"] != 0
    ):
        raise ValueError("whole public274 bank and all3 candidates required")
    pins = {str(bp): a.bank_hash, str(a.manifest): a.manifest_hash}
    # Pin before importing learning code. This analysis reads receipts, not the
    # physical arrays again; it does not claim a new independent native replay.
    source_pins(pins)
    check_pins(pins)
    from rosclaw_soccer.rsi.consolidated_adapter_learning import adapter_fit_plan

    plan = adapter_fit_plan(bank["records"])
    reports = []
    for model in manifest["candidates"]:
        spec = model["pass_own"]
        op = Path(spec["root"]) / "complete.json"
        if op.is_symlink() or digest(op) != spec["hash"]:
            raise ValueError("exact public whole-own replay witness required")
        own = json.loads(op.read_text())
        rp, source = receipt(own["input_file_hashes"], own["producer_hash"])
        if (
            own["status"] != "TEAM_PASS_CANDIDATE80_WHOLE_OWN_NATIVE_REPLAY_MATCH"
            or own["summary"] != source["summary"]
            or own["training_hash"] != source["training_hash"]
            or own["records"]
            != [{k: v for k, v in r.items() if k != "artifacts"} for r in source["records"]]
            or own["input_file_hashes"].get(str(bp)) != a.bank_hash
            or source["checkpoint_kind"] != "consolidated-adapter-v1"
            or any(
                r[k] is not False
                for r in (source, own)
                for k in (
                    "Fresh_verified",
                    "hardware_authorized",
                    "promotion_authorized",
                    "goal_complete",
                )
            )
        ):
            raise ValueError(
                "exact independently matched unpromoted public TRAIN candidate required"
            )
        _, trained = receipt(own["input_file_hashes"], own["training_hash"])
        if trained["config"]["seed"] != model["seed"]:
            raise ValueError("exact model identity required")
        pins.update({str(op): spec["hash"], str(rp): own["producer_hash"]})
        reports.append(
            dict(seed=model["seed"], contexts=diagnose(bank["records"], source["records"], plan))
        )
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="PUBLIC_TRAIN_ADAPTER_FAILURE_COMPONENTS_REANALYSED_NOT_NEW_PHYSICS",
            source_head=a.expected_head,
            input_file_hashes=pins,
            reports=reports,
            fit_plan_hash=plan["hash"],
            selected_teacher_histories=len(plan["teacher_indices"]),
            selected_baseline_success_histories=len(plan["baseline_pass_indices"]),
            original_memory_anchors=len(plan["old_anchor_indices"]),
            physical_steps=0,
            optimizer_updates=0,
            Fresh_cases_read=0,
            causal_failure_reason_proven=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    print(json.dumps(reports), flush=True)


if __name__ == "__main__":
    main()
