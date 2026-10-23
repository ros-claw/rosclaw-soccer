"""Apply the pre-registered TRAIN pilot screen after whole native replay.

Receipt reanalysis only: no sampling, model fitting, physics or Fresh access.
Passing means qualified correction data exists, not policy promotion.
"""

import argparse
import json
from pathlib import Path


def screen_rows(rows, bank_records, limits):
    from rosclaw_soccer.rsi.pass_transfer_learning import controlled_pass_label

    baseline = {
        (r["source_course"]["state_id"], r["source_course"]["task"]): r
        for r in bank_records
        if r["role"] == "new_unassisted" and r["source_course"]["context"] == 20
    }
    expected = {(i, t, p) for i in range(20, 30) for t in (0, 1) for p in (0, 80)}
    seen = set()
    counts = {p: dict(controlled=0, clean=0, new_nonfoot=0) for p in (0, 80)}
    eligible = []
    for row in rows:
        c, o, prefix = row["course"], row["outcome"], row["student_prefix_controls"]
        key = (c["state_id"], c["task"], prefix)
        if key not in expected or key in seen or c["context"] != 20 or c["assisted"] is not True:
            raise ValueError("ALL40 distinct declared public TRAIN prefix tasks required")
        seen.add(key)
        old = baseline[key[:2]]
        for k in (
            "context",
            "state_id",
            "task",
            "side",
            "foot",
            "center",
            "target",
            "ball_planar_offset_world",
        ):
            if c[k] != old["source_course"][k]:
                raise ValueError("same original public TRAIN physical state and task required")
        positive = controlled_pass_label(o)
        if row["actual_continuation_controlled_pass"] is not positive:
            raise ValueError("unchanged physical qualification labels required")
        counts[prefix]["controlled"] += positive
        counts[prefix]["clean"] += o["passer_clean_foot"]
        counts[prefix]["new_nonfoot"] += (
            old["measured_outcome"]["passer_clean_foot"] and not o["passer_clean_foot"]
        )
        if prefix == 80 and positive:
            eligible.append(row["index"])
    if seen != expected or len(baseline) != 20:
        raise ValueError("whole paired20 tasks required, not only successful continuations")
    after = counts[80]
    return dict(
        prefix0=counts[0],
        prefix80=after,
        fit_eligible_source_indices=eligible,
        physically_qualified_corrections=len(eligible),
        negative_prefix80_trajectories_preserved=20 - len(eligible),
        pilot_research_screen_passed=(
            after["controlled"] >= limits["prefix80_controlled_pass_at_least"]
            and after["new_nonfoot"] <= limits["prefix80_new_nonfoot_vs_original_baseline_at_most"]
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "own-root", "plan", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("plan-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    paths = [
        a.plan,
        a.root / "complete.json",
        a.own_root / "complete.json",
        a.root / "declaration.json",
    ]
    if (
        a.output.exists()
        or any(f.is_symlink() for f in paths)
        or digest(a.plan) != a.plan_hash
        or any((r / "failed.json").exists() for r in (a.root, a.own_root))
    ):
        raise ValueError(
            "unused review and complete exact pre-registered source/own witnesses required"
        )
    plan, s, own, d = [json.loads(f.read_text()) for f in paths]
    if (
        plan["schema"] != "soccer.adapter_actual_state_distribution_pilot.v1"
        or s["status"] != "STUDENT_PASS_CONTINUATION_SHARD_COLLECTED_PENDING_WHOLE_OWN_REPLAY"
        or own["status"] != "STUDENT_PASS_CONTINUATION_SHARD_WHOLE_OWN_NATIVE_REPLAY_MATCH"
        or s["source_head"] != plan["source"]["head"]
        or own["source_head"] != plan["auditor"]["head"]
        or s["declaration_hash"] != digest(paths[3])
        or own["producer_hash"] != digest(paths[1])
        or own["counts"] != s["counts"]
        or len(s["records"]) != 40
        or own["records"]
        != [{k: v for k, v in r.items() if k != "artifacts"} for r in s["records"]]
        or d["student_kind"] != "consolidated-adapter-v1"
        or d["protocol"] != "soccer.adapter_actual_prefix_expert_suffix_TRAIN80.v1"
        or d["student_seed"] != plan["selection_before_Fresh_results"]["seed"]
        or d["preflight_only"] is not False
        or d["prefixes"] != [0, 80]
        or d["context"] != plan["design"]["context"]
        or any(
            r[k] is not False
            for r in (s, own, d)
            for k in (
                "Fresh_verified",
                "hardware_authorized",
                "promotion_authorized",
                "goal_complete",
            )
        )
    ):
        raise ValueError("ALL40 matched independently reproduced original pilot required")
    bp, bank = receipt(own["input_file_hashes"], d["bank_hash"])
    pins = {str(f): digest(f) for f in paths}
    pins[str(bp)] = d["bank_hash"]
    source_pins(pins)
    check_pins(pins)
    summary = screen_rows(own["records"], bank["records"], plan["predeclared_research_screen"])
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="ADAPTER_ACTUAL_STATE_PILOT_REVIEWED_NOT_TRAINED_OR_PROMOTED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            **summary,
            physical_steps=0,
            optimizer_updates=0,
            Fresh_cases_read=0,
            whole40_independent_replay_matched=True,
            correction_bank_ready=False,
            accepted_policy_improvement=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
