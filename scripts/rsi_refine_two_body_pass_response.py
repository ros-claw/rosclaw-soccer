"""Validate learned local response proposals on unchanged actual G1 physics.

Perstate2tasks x(6modelproposals+2selectedwhole repetitions)=16 newworlds.
No additional random search, neural optimizer update, task regrading or Fresh.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("model-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("model-hash", "model-head", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--state-id", type=int, choices=(20, 23, 26, 29), required=True)
    p.add_argument("--preflight-only", action="store_true")
    a = p.parse_args()
    from scripts.rsi_native_fresh_common import clean

    clean(a.expected_head)
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_self_teacher_search_common import calibrations, compare_common, open_setup
    from scripts.rsi_two_body_response_common import (
        actual_response,
        admit_models,
        anchor,
        verify_refinement,
    )

    if a.output.exists():
        raise ValueError("unused bounded refinement output required")
    models, plans, source, cal, scope, pf, pins, bank, cr, sr = admit_models(
        a.model_root, a.model_hash, a.model_head, a.state_id
    )
    anchors = [anchor(p, source, cal, cr, sr) for p in plans]
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.learned_two_body_response_refinement16.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            model_root=str(a.model_root),
            model_hash=a.model_hash,
            model_head=a.model_head,
            state_id=a.state_id,
            preflight_only=a.preflight_only,
            maximum_worlds=16,
            maximum_native_steps=64000,
            maximum_actor_calls=12800,
            optimizer_updates=0,
            Fresh_cases_read=0,
            rank="healthy+recovery+noBexcluded,cleanfoot,controlled,strictcomposite,-error/.1-max(speed-3,0)/.2",
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    counts = dict(native_steps=0, actor_calls=0)
    records = []
    selections = []
    try:
        import numpy as np

        from rosclaw_soccer.rsi.two_body_pass_response import refinement_rank
        from scripts.rsi_pass_transfer_trial import collect_trial

        setup, actor = open_setup(scope, pf, pins, bank, cr / "scene.xml")
        c = anchors[0]["row"]["course"]
        _, world, _, _, _ = setup.prepare(
            c["context"],
            c["side"],
            ball_planar_offset_world=np.array(c["ball_planar_offset_world"]),
        )
        body_layout = world.layouts["a/"]
        layout = (body_layout.bq, body_layout.bv, world.model.nq, world.model.nv)
        (a.output / "scene.xml").write_text(setup.xml)
        if a.preflight_only:
            check_pins(pins)
            clean(a.expected_head)
            write_once(
                a.output / "complete.json",
                dict(
                    status="TWO_BODY_RESPONSE_REFINEMENT_ALL40_PREFLIGHT_NOT_PHYSICS",
                    source_head=a.expected_head,
                    input_file_hashes=pins,
                    model_hash=a.model_hash,
                    all40_compiled_states_match=True,
                    physical_steps=0,
                    optimizer_updates=0,
                    hardware_authorized=False,
                    promotion_authorized=False,
                    goal_complete=False,
                ),
            )
            print(json.dumps(dict(stage="RESPONSE_PREFLIGHT", physical_steps=0)), flush=True)
            return

        def run(plan, z, tag, role, index=None, prediction=None):
            if len(records) >= 16:
                raise ValueError("declared16-world refinement exhausted")
            course = anchors[plan["task"]]["row"]["course"] | dict(latent=z)
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
                tag=tag,
                task=plan["task"],
                role=role,
                model_candidate_index=index,
                predicted_response=prediction,
            )
            records.append(row)
            row["actual_response"] = actual_response(a.output, row, layout)
            zero = calibrations(cal, a.state_id, plan["task"])[0]
            row["prefix_identity_fields"] = compare_common(a.output, row, cr, zero, prefix=80)
            write_once(a.output / f"trial{row['index']}.json", row)
            return dict(root=a.output, row=row, tag=tag, latent=z)

        for plan in plans:
            task = plan["task"]
            values = [anchors[task]]
            for point in plan["new_distinct_candidates"]:
                values.append(
                    run(
                        plan,
                        point["latent"],
                        f"t{task}-model{point['candidate_index']}",
                        "proposal",
                        point["candidate_index"],
                        point["predicted_response"],
                    )
                )
            best = max(
                enumerate(values), key=lambda kv: (refinement_rank(kv[1]["row"]["outcome"]), -kv[0])
            )[1]
            for repeat in range(2):
                again = run(plan, best["latent"], f"t{task}-repeat{repeat}", "selected_repeat")
                if again["row"]["outcome"] != best["row"]["outcome"]:
                    raise ValueError("selected physical outcome did not repeat")
                compare_common(a.output, again["row"], best["root"], best["row"])
            selections.append(
                dict(
                    task=task,
                    selected=best["tag"],
                    latent=best["latent"],
                    rank=list(refinement_rank(best["row"]["outcome"])),
                    outcome=best["row"]["outcome"],
                )
            )
        if len(records) != 16 or counts["native_steps"] > 64000 or counts["actor_calls"] > 12800:
            raise ValueError("whole bounded16-world study required")
        verify_refinement(dict(records=records, selections=selections), plans, anchors)
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="TWO_BODY_RESPONSE_REFINEMENT16_COLLECTED_PENDING_ALL_OWN",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                scene_hash=digest(a.output / "scene.xml"),
                state_id=a.state_id,
                model_hash=a.model_hash,
                counts=counts,
                records=records,
                selections=selections,
                all_model_proposals_physically_tested=True,
                predictions_accepted_without_physics=False,
                prediction_accuracy_proven=False,
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
                counts=counts,
                records=records,
                automatic_retry=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
