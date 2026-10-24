"""One fixed safe proposal on each remaining TRAIN task, plus selected repeats."""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("pilot-root", "model-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("pilot-hash", "pilot-head", "model-hash", "model-head", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--state-id", type=int, choices=(21, 22, 24, 25, 27, 28), required=True)
    p.add_argument("--preflight-only", action="store_true")
    a = p.parse_args()
    from scripts.rsi_native_fresh_common import clean

    clean(a.expected_head)
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_output_capacity import require_output_capacity
    from scripts.rsi_pass_teacher_expansion_common import admit_pilot, verify_expansion
    from scripts.rsi_self_teacher_search_common import calibrations, compare_common, open_setup

    if a.output.exists():
        raise ValueError("unused expansion root required")
    require_output_capacity(a.output, 2 * 1024**3)
    _, plans, cal, scope, pf, pins, bank, cr = admit_pilot(
        a.pilot_root, a.pilot_hash, a.pilot_head, a.model_root, a.model_hash, a.model_head
    )
    local = [v for v in plans if v["state_id"] == a.state_id]
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.safe_pass_proposal_transfer6.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            pilot_root=str(a.pilot_root),
            pilot_hash=a.pilot_hash,
            pilot_head=a.pilot_head,
            model_root=str(a.model_root),
            model_hash=a.model_hash,
            model_head=a.model_head,
            state_id=a.state_id,
            plans=plans,
            preflight_only=a.preflight_only,
            maximum_worlds=6,
            maximum_native_steps=24000,
            maximum_actor_calls=4800,
            optimizer_updates=0,
            Fresh_cases_read=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    counts = dict(native_steps=0, actor_calls=0)
    records, selections = [], []
    try:
        from rosclaw_soccer.rsi.two_body_pass_response import refinement_rank
        from scripts.rsi_pass_transfer_trial import collect_trial

        setup, actor = open_setup(scope, pf, pins, bank, cr / "scene.xml")
        (a.output / "scene.xml").write_text(setup.xml)
        if a.preflight_only:
            check_pins(pins)
            clean(a.expected_head)
            write_once(
                a.output / "complete.json",
                dict(
                    status="PASS_TRANSFER_ALL40_PREFLIGHT_NOT_PHYSICS",
                    source_head=a.expected_head,
                    input_file_hashes=pins,
                    all40_compiled_states_match=True,
                    physical_steps=0,
                    optimizer_updates=0,
                    hardware_authorized=False,
                    promotion_authorized=False,
                    goal_complete=False,
                ),
            )
            return

        def run(plan, course, tag, role):
            if len(records) >= 6:
                raise ValueError("six-world expansion budget exhausted")
            require_output_capacity(a.output, 256 * 1024**2)
            row = collect_trial(
                setup,
                actor,
                course,
                output=a.output,
                index=len(records),
                counts=counts,
                emit_course_rows=False,
            )
            row.update(
                task=plan["task"],
                tag=tag,
                role=role,
                proposal_donor_state=plan["donor_state"],
                proposal_donor_controlled=plan["donor_controlled"],
            )
            records.append(row)
            zero = calibrations(cal, a.state_id, plan["task"])[0]
            row["prefix_identity_fields"] = compare_common(a.output, row, cr, zero, prefix=80)
            write_once(a.output / f"trial{row['index']}.json", row)
            return dict(root=a.output, row=row, tag=tag)

        for plan in local:
            task = plan["task"]
            values = [
                dict(root=cr, row=r, tag=f"cal-t{task}-p{r['student_prefix_controls']}")
                for r in calibrations(cal, a.state_id, task)
            ]
            values.append(run(plan, plan["course"], f"t{task}-transfer", "transfer"))
            best = max(
                enumerate(values), key=lambda kv: (refinement_rank(kv[1]["row"]["outcome"]), -kv[0])
            )[1]
            for repeat in range(2):
                again = run(
                    plan, best["row"]["course"], f"t{task}-repeat{repeat}", "selected_repeat"
                )
                if again["row"]["outcome"] != best["row"]["outcome"]:
                    raise ValueError("selected exact outcome did not repeat")
                compare_common(a.output, again["row"], best["root"], best["row"])
            selections.append(
                dict(
                    task=task,
                    selected=best["tag"],
                    latent=best["row"]["course"]["latent"],
                    outcome=best["row"]["outcome"],
                    transfer_selected=best["tag"] == f"t{task}-transfer",
                )
            )
        verify_expansion(dict(records=records, selections=selections), plans, cal, a.state_id)
        if counts["native_steps"] > 24000 or counts["actor_calls"] > 4800:
            raise ValueError("bounded six-world accounting required")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="PASS_PROPOSAL_TRANSFER6_COLLECTED_PENDING_WHOLE_OWN",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                scene_hash=digest(a.output / "scene.xml"),
                pilot_hash=a.pilot_hash,
                state_id=a.state_id,
                counts=counts,
                records=records,
                selections=selections,
                student_optimizer_updates=0,
                Fresh_cases_read=0,
                correction_bank_ready=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as e:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(e).__name__,
                error=str(e),
                records=records,
                counts=counts,
                automatic_retry=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
