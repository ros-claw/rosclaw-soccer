"""Own-history/manual native replay of ALL80 unassisted candidate tasks.

Reconstructs exact source TRAIN inputs and keeps old retention, blind Fresh
and full-game requirements separate. No producer step/saved action/h is used.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint-kind",
        choices=("team-pass-consolidated-v1", "team-pass-corrected-v2", "team-pass-selected-v3"),
        default="team-pass-consolidated-v1",
    )
    for key in ("root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-hash", "producer-head", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    paths = [
        a.root / n for n in ("complete.json", "declaration.json", "preflight.json", "scene.xml")
    ]
    if (
        a.output.exists()
        or any(x.is_symlink() for x in paths)
        or (a.root / "failed.json").exists()
        or digest(paths[0]) != a.source_hash
    ):
        raise ValueError("exact closed whole80 source and unused own audit required")
    s, d, pf = [json.loads(x.read_text()) for x in paths[:3]]
    if (
        s["status"] != "TEAM_PASS_CANDIDATE80_COLLECTED_PENDING_OWN_NATIVE_AND_RETENTION"
        or d["protocol"] != "dual256.team_pass_unassisted_TRAIN80.v1"
        or any(r["source_head"] != a.producer_head for r in (s, d))
        or s["declaration_hash"] != digest(paths[1])
        or s["preflight_hash"] != digest(paths[2])
        or s["scene_hash"] != digest(paths[3])
        or s["training_hash"] != d["training_hash"]
        or s["checkpoint_kind"] != a.checkpoint_kind
        or d.get("checkpoint_kind", "team-pass-consolidated-v1") != a.checkpoint_kind
        or d["teacher_enabled"] is not False
        or d["maximum_worlds"] != 80
        or d["maximum_native_steps"] != 320000
        or d["maximum_actor_calls"] != 64000
        or d["formal_pass_receive_controls"] != 300
        or d["required_A_neural_recovery_controls"] != 100
        or len(s["records"]) != 80
        or len(d["courses"]) != 80
        or any(
            r[k] is not False
            for r in (s, d)
            for k in (
                "goal_complete",
                "Fresh_verified",
                "old_retention_proven",
                "promotion_authorized",
                "hardware_authorized",
            )
        )
    ):
        raise ValueError("unchanged finite unassisted TRAIN80 protocol required")
    pins = dict(s["input_file_hashes"])
    tp, training = receipt(pins, d["training_hash"])
    _, cpu = receipt(pins, d["cpu_audit_hash"])
    _, bank = receipt(pins, d["data_hash"])
    if (
        cpu["status"] != "DUAL_RECURRENT_TRAINED_CHECKPOINT_CPU_OWN_HISTORY_AUDIT_PASS_NOT_PHYSICS"
        or cpu["source_hash"] != d["training_hash"]
        or cpu["checkpoint_hash"] != training["checkpoint_hash"]
        or bank["status"]
        != (
            "TEAM_PASS_WHOLE_HISTORY_BANK177_READY_NOT_TRAINED"
            if a.checkpoint_kind == "team-pass-consolidated-v1"
            else "TEAM_PASS_ACTUAL_STUDENT_CORRECTION_BANK274_READY_NOT_TRAINED"
        )
    ):
        raise ValueError("same whole-history independently CPU-checked trained motor required")
    task_bank = (
        bank
        if a.checkpoint_kind == "team-pass-consolidated-v1"
        else receipt(pins, bank["base_data_hash"])[1]
    )
    if (
        task_bank["status"] != "TEAM_PASS_WHOLE_HISTORY_BANK177_READY_NOT_TRAINED"
        or len(task_bank["records"]) != 177
    ):
        raise ValueError("original complete physical task witness required")
    baseline = sorted(
        (r for r in task_bank["records"] if r["role"] == "new_unassisted"),
        key=lambda r: (r["source_course"]["state_id"], r["source_course"]["task"]),
    )
    courses = [r["source_course"] for r in baseline]
    if (
        len(courses) != 80
        or courses != d["courses"]
        or any(c["assisted"] is not False for c in courses)
        or any(r["index"] != i or r["course"] != courses[i] for i, r in enumerate(s["records"]))
    ):
        raise ValueError("all original80 matched unassisted inputs required")
    for path in paths:
        pins[str(path)] = digest(path)
    for i, row in enumerate(s["records"]):
        if set(row["artifacts"]) != {"initial", "physical", "A", "B"}:
            raise ValueError("all initial/native/two-neural artifacts required")
        for art in row["artifacts"].values():
            path = a.root / art["path"]
            if (
                path.is_symlink()
                or path.resolve().parent != (a.root / f"trial{i}").resolve()
                or digest(path) != art["hash"]
            ):
                raise ValueError("all exact confined candidate primitives required")
            pins[str(path)] = art["hash"]
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            producer_hash=a.source_hash,
            input_file_hashes=pins,
            manual_independent_native_clock=True,
            own_neural_history=True,
            source_actions_executed=False,
            borrowed_hidden_state=False,
            shared_codecs_on_own_states=True,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    counts = dict(native_steps=0, actor_calls=0)
    records = []
    try:
        import numpy as np

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
            else {i: v for i, v in enumerate(bank["contexts"])}
        )
        bench, actor = open_motor(
            pins,
            d["training_hash"],
            {i: contexts[i] for i in (1, 4, 20, 21)},
            checkpoint_kind=a.checkpoint_kind,
        )
        setup = SharedRelaySetup(pins, contexts, bench=bench, actor=actor)
        if setup.xml != paths[3].read_text():
            raise ValueError("own compiled shared scene differs")
        typed = []
        for index in range(0, 80, 2):
            c = courses[index]
            _, w, forward, home, _ = setup.prepare(
                c["context"],
                c["side"],
                ball_planar_offset_world=np.array(c["ball_planar_offset_world"]),
            )
            old = baseline[index]
            path = Path(old["source_root"]) / f"trial{old['source_index']}" / "initial.npz"
            if str(path) not in pins or digest(path) != pins[str(path)]:
                raise ValueError("bound original baseline state required")
            with np.load(path, allow_pickle=False) as z:
                if not np.array_equal(z["qpos"], w.data.qpos) or not np.array_equal(
                    z["qvel"], w.data.qvel
                ):
                    raise ValueError("own matched original initial state differs")
            q, _ = w.layouts["a/"].canonical(w.data)
            relative = home - q[36:38]
            lateral = np.array([-forward[1], forward[0]])
            typed.append(
                NativePassCourse(
                    c["state_id"],
                    float(relative @ forward),
                    float(relative @ lateral),
                    hash_json(dict(qpos=w.data.qpos.tolist(), qvel=w.data.qvel.tolist())),
                )
            )
        expected_pf = dict(
            initial_conditions=40,
            worlds=80,
            course_binding=preflight_receiving_courses(tuple(typed)),
            ball_dimensions=setup.ball_dimensions,
            native_steps=0,
        )
        if json.loads(json.dumps(expected_pf, allow_nan=False)) != pf:
            raise ValueError("whole40 own initial course preflight differs")
        prior = None
        for i, c in enumerate(courses):
            row, prior = replay_trial(
                setup,
                actor,
                c,
                s["records"][i],
                raw_root=a.root,
                output=a.output,
                index=i,
                counts=counts,
                prior_initial=prior,
            )
            records.append(row)
        comparison = []
        for row, old in zip(records, baseline, strict=True):
            b, c = (
                controlled_pass_label(old["measured_outcome"]),
                controlled_pass_label(row["outcome"]),
            )
            comparison.append(
                dict(
                    state_id=row["course"]["state_id"],
                    task=row["course"]["task"],
                    baseline=b,
                    candidate=c,
                    rescue=c and not b,
                    regression=b and not c,
                )
            )
        summary = dict(
            baseline_controlled=sum(r["baseline"] for r in comparison),
            candidate_controlled=sum(r["candidate"] for r in comparison),
            rescues=sum(r["rescue"] for r in comparison),
            regressions=sum(r["regression"] for r in comparison),
            both_healthy=sum(r["outcome"]["whole_body_healthy"] for r in records),
            A_full100NNstable=sum(r["outcome"]["passer_full100NNrecovery"] for r in records),
            B_functional_receive=sum(
                r["outcome"]["B_legal_functional_receive_before_deadline"] for r in records
            ),
            strict_static_point_composite=sum(
                r["outcome"]["combined_precision_and_functional_receive"] for r in records
            ),
            distinct_physical_initial_conditions=40,
            goal_tasks=80,
            source_context_clusters=4,
            posthoc_training_label=True,
            Fresh_verified=False,
            old_retention_proven=False,
        )
        if comparison != s["comparison"] or summary != s["summary"] or counts != s["counts"]:
            raise ValueError("all own comparisons and native/NN counters differ")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="TEAM_PASS_CANDIDATE80_WHOLE_OWN_NATIVE_REPLAY_MATCH",
                source_head=a.expected_head,
                producer_hash=a.source_hash,
                input_file_hashes=pins,
                training_hash=d["training_hash"],
                counts=counts,
                records=records,
                comparison=comparison,
                summary=summary,
                TRAIN_only=True,
                Fresh_verified=False,
                old_retention_proven=False,
                team_game_verified=False,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="TEAM_PASS_CANDIDATE_AUDIT_FAILED_PRESERVE_NO_RETRY",
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
