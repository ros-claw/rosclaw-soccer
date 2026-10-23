"""Replay every student/expert continuation on an independent native clock.

Own observations, both private recurrent histories, B intent, PD/torques,
ball dynamics and physical outcomes must match every recorded row exactly.
No saved action or saved h/c is ever executed. No producer clock is used.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("producer-head", "expected-head"):
        p.add_argument("--" + key, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-hash")
    group.add_argument("--completed-predecessor", action="store_true")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    paths = [
        a.root / n for n in ("complete.json", "declaration.json", "preflight.json", "scene.xml")
    ]
    if a.completed_predecessor:
        a.source_hash = digest(paths[0])
    if (
        a.output.exists()
        or any(x.is_symlink() for x in paths)
        or (a.root / "failed.json").exists()
        or digest(paths[0]) != a.source_hash
    ):
        raise ValueError("unused audit and exact whole closed continuation required")
    s, d, pf = [json.loads(x.read_text()) for x in paths[:3]]
    if (
        s["status"] != "STUDENT_PASS_CONTINUATION_SHARD_COLLECTED_PENDING_WHOLE_OWN_REPLAY"
        or d["protocol"]
        not in (
            "soccer.student_actual_prefix_expert_suffix_TRAIN80.v1",
            "soccer.adapter_actual_prefix_expert_suffix_TRAIN80.v1",
        )
        or any(r["source_head"] != a.producer_head for r in (s, d))
        or s["declaration_hash"] != digest(paths[1])
        or s["preflight_hash"] != digest(paths[2])
        or s["scene_hash"] != digest(paths[3])
        or s["context"] != d["context"]
        or d["prefixes"] != [0, 80]
        or d["maximum_worlds"] != 40
        or len(s["records"]) != 40
        or len(d["global_courses"]) != 80
        or pf["global_initial_conditions"] != 40
        or any(
            r[k] is not False
            for r in (s, d)
            for k in (
                "goal_complete",
                "Fresh_verified",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
    ):
        raise ValueError("whole closed40-world unchanged continuation protocol required")
    pins = dict(s["input_file_hashes"])
    _, bank = receipt(pins, d["bank_hash"])
    tp, training = receipt(pins, d["student_training_hash"])
    wp, warm = receipt(pins, d["teacher_training_hash"])
    from rosclaw_soccer.rsi.adapter_continuation_identity import continuation_protocol

    kind = d.get("student_kind", "team-pass-consolidated-v1")
    seed = d.get("student_seed", 75024)
    child_hash = training["data_hash"] if kind == "consolidated-adapter-v1" else None
    child_bank = receipt(pins, child_hash)[1] if child_hash is not None else None
    protocol = continuation_protocol(
        kind=kind,
        seed=seed,
        training=training,
        bank=bank,
        bank_hash=d["bank_hash"],
        child_bank=child_bank,
        child_bank_hash=child_hash,
    )
    if protocol != d["protocol"] or d["teacher_training_hash"] != bank["warm_start_hash"]:
        raise ValueError("same frozen research student and retained teacher required")
    courses = d["global_courses"]
    selected = [
        (i, c, prefix)
        for i, c in enumerate(courses)
        if c["context"] == d["context"]
        for prefix in (0, 80)
    ]
    for index, (row, (i, c, prefix)) in enumerate(zip(s["records"], selected, strict=True)):
        if (
            row["index"] != index
            or row["original_task_index"] != i
            or row["course"] != c
            or row["student_prefix_controls"] != prefix
        ):
            raise ValueError("all original paired prefix commitments required")
        if set(row["artifacts"]) != {"initial", "physical", "A", "B"}:
            raise ValueError("all original primitives required")
        for art in row["artifacts"].values():
            path = a.root / art["path"]
            if (
                path.is_symlink()
                or path.resolve().parent != (a.root / f"trial{index}").resolve()
                or digest(path) != art["hash"]
            ):
                raise ValueError("all confined exact native and privateNN traces required")
            pins[str(path)] = art["hash"]
    pins.update({str(x): digest(x) for x in paths})
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir(parents=True)
    counts = dict(native_steps=0, actor_calls=0, shadow_actor_calls=0)
    records = []
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            producer_hash=a.source_hash,
            input_file_hashes=pins,
            own_neural_histories=True,
            manual_independent_native_clock=True,
            source_actions_executed=False,
            borrowed_hidden_state=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    try:
        import numpy as np

        from rosclaw_soccer.rsi.consolidated_adapter_checkpoint import (
            load_motor_candidate_for_sim_audit,
        )
        from rosclaw_soccer.rsi.pass_transfer_learning import controlled_pass_label
        from rosclaw_soccer.rsi.shared_relay_setup import SharedRelaySetup
        from rosclaw_soccer.sim.contracts import hash_json
        from rosclaw_soccer.training.native_pass_courses import NativePassCourse
        from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
        from scripts.rsi_pass_transfer_replay_trial import replay_trial
        from scripts.rsi_stroke_search_common import open_motor

        contexts = (
            {int(k): v for k, v in bank["contexts"].items()}
            if isinstance(bank["contexts"], dict)
            else dict(enumerate(bank["contexts"]))
        )
        bench, student = open_motor(
            pins,
            d["student_training_hash"],
            {i: contexts[i] for i in (1, 4, 20, 21)},
            checkpoint_kind=kind,
        )
        from rosclaw_soccer.rsi.same_state_motor_pair import paired_reference

        teacher, _ = load_motor_candidate_for_sim_audit(
            paired_reference(student),
            parent_lineage_hash=warm["parent_lineage_hash"],
            root=wp.parent,
            expected_receipt_hash=d["teacher_training_hash"],
            checkpoint_kind="contact-consolidated-v2",
        )
        setup = SharedRelaySetup(pins, contexts, bench=bench, actor=teacher)
        if setup.xml != paths[3].read_text():
            raise ValueError("same compiled physical scene required")
        typed = []
        for i in range(0, 80, 2):
            c = courses[i]
            _, w, forward, home, _ = setup.prepare(
                c["context"],
                c["side"],
                ball_planar_offset_world=np.array(c["ball_planar_offset_world"]),
            )
            q, _ = w.layouts["a/"].canonical(w.data)
            rel = home - q[36:38]
            typed.append(
                NativePassCourse(
                    c["state_id"],
                    float(rel @ forward),
                    float(rel @ np.array([-forward[1], forward[0]])),
                    hash_json(dict(qpos=w.data.qpos.tolist(), qvel=w.data.qvel.tolist())),
                )
            )
            if w.native_steps:
                raise ValueError("whole independent preflight before physics required")
        binding = preflight_receiving_courses(tuple(typed))
        if binding != pf["course_binding"]:
            raise ValueError("whole own preflight binding differs")
        prior = None
        for index, row in enumerate(s["records"]):
            extra = {
                k: row[k]
                for k in (
                    "original_task_index",
                    "identity_fields_compared",
                    "actual_continuation_controlled_pass",
                )
            }
            claim = {k: v for k, v in row.items() if k not in extra}
            own, prior = replay_trial(
                setup,
                teacher,
                row["course"],
                claim,
                raw_root=a.root,
                output=a.output,
                index=index,
                counts=counts,
                prior_initial=prior,
                student=student,
                student_prefix_controls=row["student_prefix_controls"],
            )
            if (
                controlled_pass_label(own["outcome"])
                != extra["actual_continuation_controlled_pass"]
            ):
                raise ValueError("own posthoc controlled-pass qualification differs")
            records.append(own | extra)
        if counts != s["counts"]:
            raise ValueError("whole actual and shadow budget accounting differs")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="STUDENT_PASS_CONTINUATION_SHARD_WHOLE_OWN_NATIVE_REPLAY_MATCH",
                source_head=a.expected_head,
                producer_hash=a.source_hash,
                input_file_hashes=pins,
                context=s["context"],
                counts=counts,
                records=records,
                prefix0_controlled=s["prefix0_controlled"],
                prefix80_controlled=s["prefix80_controlled"],
                optimizer_updates=0,
                PPO_actor_rows=0,
                correction_bank_ready=False,
                TRAIN_only=True,
                Fresh_verified=False,
                team_game_verified=False,
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="CONTINUATION_OWN_FAILED_PRESERVE_NO_RETRY",
                error_type=type(error).__name__,
                error=str(error),
                counts=counts,
                records=records,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
