"""Close all ten TRAIN states without confusing local fallback with transfer."""

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
    from scripts.rsi_native_fresh_common import clean, receipt

    clean(a.expected_head)
    from rosclaw_soccer.rsi.pass_teacher_coverage import teacher_coverage
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_pass_teacher_expansion_common import admit_pilot, verify_expansion
    from scripts.rsi_self_teacher_search_common import calibrations

    if a.output.exists() or a.plan.is_symlink() or digest(a.plan) != a.plan_hash:
        raise ValueError("unused closure output and exact expansion preregistration required")
    plan = json.loads(a.plan.read_text())
    if (
        plan["schema"] != "soccer.safe_pass_teacher_TRAIN20_expansion.v1"
        or plan["source_head"] != a.producer_head
        or plan["pilot_states"] != [20, 23, 26, 29]
        or plan["expanded_states"] != [21, 22, 24, 25, 27, 28]
        or plan["minimum_full20_controlled"] != 8
        or plan["minimum_net_rescues"] != 2
        or plan["maximum_old_controlled_regressions"] != 0
        or plan["require_all_selected_healthy_clean_full100NNrecovery"] is not True
    ):
        raise ValueError("unchanged functional research screen, not an easier final exam, required")
    pilot, plans, cal, _, _, pins, _, cr = admit_pilot(
        Path(plan["pilot_root"]),
        plan["pilot_hash"],
        plan["pilot_head"],
        Path(plan["model_root"]),
        plan["model_hash"],
        plan["model_head"],
    )
    verified = dict(pins)
    pins[str(a.plan)] = a.plan_hash
    selected, baseline, transfers, proofs = [], [], [], []
    counts = dict(native_steps=0, actor_calls=0)

    def selected_row(root, source, state, task, source_hash, own_hash, transfer_selected):
        rows = [
            r for r in source["records"] if r["task"] == task and r["role"] == "selected_repeat"
        ]
        if len(rows) != 2 or rows[0]["outcome"] != rows[1]["outcome"]:
            raise ValueError("whole selected repetitions required for every bank pointer")
        row = rows[0]
        return dict(
            state_id=state,
            task=task,
            source_root=str(root),
            source_hash=source_hash,
            own_hash=own_hash,
            trial_index=row["index"],
            course=row["course"],
            outcome=row["outcome"],
            transfer_selected=transfer_selected,
        )

    for state in plan["pilot_states"]:
        proof = next(x for x in pilot["proofs"] if x["state_id"] == state)
        sp, source = receipt(pins, proof["source_hash"])
        for task in (0, 1):
            selected.append(
                selected_row(
                    sp.parent, source, state, task, proof["source_hash"], proof["own_hash"], False
                )
            )
    for state in plan["expanded_states"]:
        root = a.shards_root / f"state{state}"
        paths = [
            root / r / n
            for r, n in (
                ("source", "complete.json"),
                ("own", "complete.json"),
                ("source", "declaration.json"),
            )
        ]
        if any(p.is_symlink() for p in (root, *paths)) or any(
            (root / k / "failed.json").exists() for k in ("source", "own")
        ):
            raise ValueError("all six source/own shards must close without failure")
        s, o, d = [json.loads(p.read_text()) for p in paths]
        if (
            s["status"] != "PASS_PROPOSAL_TRANSFER6_COLLECTED_PENDING_WHOLE_OWN"
            or o["status"] != "PASS_PROPOSAL_TRANSFER_ALL6_OWN_NATIVE_MATCH"
            or s["source_head"] != a.producer_head
            or d["source_head"] != a.producer_head
            or o["source_head"] != a.auditor_head
            or o["producer_hash"] != digest(paths[0])
            or s["declaration_hash"] != digest(paths[2])
            or s["pilot_hash"] != plan["pilot_hash"]
            or d["pilot_hash"] != plan["pilot_hash"]
            or d["plans"] != plans
            or d["preflight_only"] is not False
            or any(v["state_id"] != state for v in (s, o, d))
            or o["records"]
            != [{k: v for k, v in r.items() if k != "artifacts"} for r in s["records"]]
            or o["selections"] != s["selections"]
            or o["counts"] != s["counts"]
            or o["manual_native_clock"] is not True
            or o["fixed_proposal_and_selection_reconstructed"] is not True
            or o["source_actions_executed"] is not False
            or o["borrowed_hidden_state"] is not False
            or any(
                v[k] != 0 for v in (s, o) for k in ("student_optimizer_updates", "Fresh_cases_read")
            )
            or any(
                v[k] is not False
                for v in (s, o, d)
                for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
            )
        ):
            raise ValueError("exact complete independently reproduced expansion required")
        verify_expansion(s | dict(records=o["records"]), plans, cal, state)
        for v in (s, o):
            for key, h in v["input_file_hashes"].items():
                if key in pins and pins[key] != h:
                    raise ValueError("conflicting whole expansion ancestry")
                pins[key] = h
        pins.update({str(p): digest(p) for p in paths})
        for task in (0, 1):
            choice = s["selections"][task]
            selected.append(
                selected_row(
                    root / "source",
                    s,
                    state,
                    task,
                    digest(paths[0]),
                    digest(paths[1]),
                    choice["transfer_selected"],
                )
            )
        transfers.extend(
            dict(
                state_id=state,
                task=r["task"],
                outcome=r["outcome"],
                donor_state=r["proposal_donor_state"],
                donor_controlled=r["proposal_donor_controlled"],
            )
            for r in s["records"]
            if r["role"] == "transfer"
        )
        for key in counts:
            counts[key] += s["counts"][key]
        proofs.append(dict(state_id=state, source_hash=digest(paths[0]), own_hash=digest(paths[1])))
    for state in range(20, 30):
        for task in (0, 1):
            row = calibrations(cal, state, task)[0]
            baseline.append(
                dict(
                    state_id=state,
                    task=task,
                    outcome=row["outcome"],
                    source_root=str(cr),
                    trial_index=row["index"],
                )
            )
    coverage = teacher_coverage(baseline, selected, transfers)
    check_pins({k: h for k, h in pins.items() if verified.get(k) != h})
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    report = dict(
        status="PASS_TEACHER_TRAIN20_COVERAGE_CLOSED_NOT_STUDENT_LEARNING",
        source_head=a.expected_head,
        input_file_hashes=pins,
        coverage=coverage,
        selected_sources=selected,
        baseline=baseline,
        transfers=transfers,
        proofs=proofs,
        new_worlds_source=36,
        new_worlds_own=36,
        new_native_steps_source_plus_own=2 * counts["native_steps"],
        physical_steps_this_aggregation=0,
        student_optimizer_updates=0,
        Fresh_cases_read=0,
        correction_bank_ready=False,
        learned_policy_improvement_verified=False,
        hardware_authorized=False,
        promotion_authorized=False,
        goal_complete=False,
    )
    write_once(a.output / "complete.json", report)
    print(json.dumps(dict(status=report["status"], coverage=coverage)), flush=True)


if __name__ == "__main__":
    main()
