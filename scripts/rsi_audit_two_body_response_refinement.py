"""ALL16 manual-native-clock replay of empirical response refinement trials."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("producer-head", "expected-head"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_self_teacher_search_common import open_setup
    from scripts.rsi_two_body_response_common import (
        actual_response,
        admit_models,
        anchor,
        verify_refinement,
    )

    paths = [a.root / n for n in ("complete.json", "declaration.json", "scene.xml")]
    if a.output.exists() or any(p.is_symlink() for p in paths) or (a.root / "failed.json").exists():
        raise ValueError("completed source and unused own replay output required")
    s, d = [json.loads(p.read_text()) for p in paths[:2]]
    if (
        s["status"] != "TWO_BODY_RESPONSE_REFINEMENT16_COLLECTED_PENDING_ALL_OWN"
        or d["protocol"] != "soccer.learned_two_body_response_refinement16.v1"
        or any(r["source_head"] != a.producer_head for r in (s, d))
        or s["declaration_hash"] != digest(paths[1])
        or s["scene_hash"] != digest(paths[2])
        or d["preflight_only"] is not False
        or d["state_id"] != s["state_id"]
        or d["model_hash"] != s["model_hash"]
        or d["maximum_worlds"] != 16
        or d["maximum_native_steps"] != 64000
        or d["maximum_actor_calls"] != 12800
        or any(
            v[k] is not False
            for v in (s, d)
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError("exact complete bounded empirical refinement required")
    models, plans, source, cal, scope, pf, pins, bank, cr, sr = admit_models(
        Path(d["model_root"]), d["model_hash"], d["model_head"], s["state_id"]
    )
    anchors = [anchor(p, source, cal, cr, sr) for p in plans]
    verify_refinement(s, plans, anchors)
    for k, h in s["input_file_hashes"].items():
        if k in pins and pins[k] != h:
            raise ValueError("conflicting model/source ancestry")
        pins[k] = h
    for path in paths:
        pins[str(path)] = digest(path)
    for i, r in enumerate(s["records"]):
        if set(r["artifacts"]) != {"initial", "physical", "A", "B"}:
            raise ValueError("whole actual neural/physics primitives required")
        for art in r["artifacts"].values():
            f = a.root / art["path"]
            if f.is_symlink() or f.resolve().parent != (a.root / f"trial{i}").resolve():
                raise ValueError("confined exact source primitives required")
            pins[str(f)] = art["hash"]
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            producer_hash=digest(paths[0]),
            input_file_hashes=pins,
            manual_native_clock=True,
            source_actions_executed=False,
            borrowed_hidden_state=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    counts = dict(native_steps=0, actor_calls=0)
    records = []
    try:
        import numpy as np

        from scripts.rsi_pass_transfer_replay_trial import replay_trial

        setup, actor = open_setup(scope, pf, pins, bank, cr / "scene.xml")
        if setup.xml != paths[2].read_text():
            raise ValueError("same compiled actual scene required")
        c = anchors[0]["row"]["course"]
        _, world, _, _, _ = setup.prepare(
            c["context"],
            c["side"],
            ball_planar_offset_world=np.array(c["ball_planar_offset_world"]),
        )
        body_layout = world.layouts["a/"]
        layout = (body_layout.bq, body_layout.bv, world.model.nq, world.model.nv)
        prior = None
        for i, row in enumerate(s["records"]):
            extra = {
                k: row[k]
                for k in (
                    "tag",
                    "task",
                    "role",
                    "model_candidate_index",
                    "predicted_response",
                    "actual_response",
                    "prefix_identity_fields",
                )
            }
            claim = {k: v for k, v in row.items() if k not in extra}
            own, prior = replay_trial(
                setup,
                actor,
                row["course"],
                claim,
                raw_root=a.root,
                output=a.output,
                index=i,
                counts=counts,
                prior_initial=prior,
                emit_course_rows=False,
            )
            observed = actual_response(a.root, row, layout)
            if observed != extra["actual_response"]:
                raise ValueError("actual empirical response differs from raw physical proof")
            records.append(own | extra)
        if counts != s["counts"]:
            raise ValueError("whole actual native/NN accounting differs")
        verify_refinement(s | dict(records=records), plans, anchors)
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="TWO_BODY_RESPONSE_REFINEMENT_ALL16_OWN_NATIVE_MATCH",
                source_head=a.expected_head,
                producer_hash=digest(paths[0]),
                input_file_hashes=pins,
                state_id=s["state_id"],
                model_hash=s["model_hash"],
                counts=counts,
                records=records,
                selections=s["selections"],
                manual_native_clock=True,
                source_actions_executed=False,
                borrowed_hidden_state=False,
                actual_responses_rederived=True,
                joint_selection_reconstructed=True,
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
