"""Actual UNASSISTED neural pass candidate on all80 consumed TRAIN tasks.

The actor is the sole intervention against own-audited65023 baseline data.
B stays frozen; no teacher stroke, Cartesian correction, ball reset inside
an episode, source action execution or hidden-state borrowing is enabled.
This is training-domain evaluation, NOT Fresh, old retention or full game.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("training-root", "cpu-audit-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("training-hash", "cpu-audit-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    paths = [a.training_root / "complete.json", a.cpu_audit_root / "complete.json"]
    if (
        a.output.exists()
        or any(x.is_symlink() for x in paths)
        or any((r / "failed.json").exists() for r in (a.training_root, a.cpu_audit_root))
        or [digest(x) for x in paths] != [a.training_hash, a.cpu_audit_hash]
    ):
        raise ValueError(
            "exact closed trained model/independent CPU audit and unused exam required"
        )
    t, own = [json.loads(x.read_text()) for x in paths]
    if (
        own["status"] != "DUAL_RECURRENT_TRAINED_CHECKPOINT_CPU_OWN_HISTORY_AUDIT_PASS_NOT_PHYSICS"
        or own["checkpoint_kind"] != "team-pass-consolidated-v1"
        or own["source_hash"] != a.training_hash
        or own["checkpoint_hash"] != t["checkpoint_hash"]
        or own["data_hash"] != t["data_hash"]
        or own["own_actor_decisions"] != t["cpu_offline_decisions"]
        or own["physical_steps_executed"] != 0
        or any(
            r[k] is not False
            for r in (t, own)
            for k in ("goal_complete", "promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("whole independent own-history CPU model replay required")
    pins = dict(own["input_file_hashes"])
    pins.update({str(x): digest(x) for x in paths})
    bp, bank = receipt(pins, t["data_hash"])
    if (
        bank["status"] != "TEAM_PASS_WHOLE_HISTORY_BANK177_READY_NOT_TRAINED"
        or len(bank["records"]) != 177
    ):
        raise ValueError("same all177 whole-history TRAIN data bank required")
    baseline = [r for r in bank["records"] if r["role"] == "new_unassisted"]
    baseline.sort(key=lambda r: (r["source_course"]["state_id"], r["source_course"]["task"]))
    courses = [r["source_course"] for r in baseline]
    if (
        len(courses) != 80
        or {(c["state_id"], c["task"]) for c in courses}
        != {(i, j) for i in range(40) for j in (0, 1)}
        or any(c["assisted"] is not False for c in courses)
    ):
        raise ValueError("ALL80 matched tasks with teacher completely disabled required")
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="dual256.team_pass_unassisted_TRAIN80.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            training_hash=a.training_hash,
            cpu_audit_hash=a.cpu_audit_hash,
            data_hash=t["data_hash"],
            courses=courses,
            maximum_worlds=80,
            maximum_native_steps=320000,
            maximum_actor_calls=64000,
            formal_pass_receive_controls=300,
            required_A_neural_recovery_controls=100,
            teacher_enabled=False,
            B_policy_frozen=True,
            actor_only_intervention=True,
            TRAIN_only=True,
            Fresh_verified=False,
            old_retention_proven=False,
            goal_complete=False,
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
        from scripts.rsi_pass_transfer_trial import collect_trial
        from scripts.rsi_stroke_search_common import open_motor

        contexts = (
            {int(k): v for k, v in bank["contexts"].items()}
            if isinstance(bank["contexts"], dict)
            else {i: v for i, v in enumerate(bank["contexts"])}
        )
        bench, actor = open_motor(
            pins,
            a.training_hash,
            {i: contexts[i] for i in (1, 4, 20, 21)},
            checkpoint_kind="team-pass-consolidated-v1",
        )
        setup = SharedRelaySetup(pins, contexts, bench=bench, actor=actor)
        typed = []
        for index in range(0, 80, 2):
            course = courses[index]
            _, w, forward, home, _ = setup.prepare(
                course["context"],
                course["side"],
                ball_planar_offset_world=np.array(course["ball_planar_offset_world"]),
            )
            original = baseline[index]
            rp = Path(original["source_root"]) / f"trial{original['source_index']}" / "initial.npz"
            if str(rp) not in pins or digest(rp) != pins[str(rp)]:
                raise ValueError("bound original matched initial state required")
            with np.load(rp, allow_pickle=False) as z:
                if not np.array_equal(z["qpos"], w.data.qpos) or not np.array_equal(
                    z["qvel"], w.data.qvel
                ):
                    raise ValueError(
                        "candidate physical initialization changed from original baseline"
                    )
            q, _ = w.layouts["a/"].canonical(w.data)
            relative = home - q[36:38]
            lateral = np.array([-forward[1], forward[0]])
            typed.append(
                NativePassCourse(
                    course["state_id"],
                    float(relative @ forward),
                    float(relative @ lateral),
                    hash_json(dict(qpos=w.data.qpos.tolist(), qvel=w.data.qvel.tolist())),
                )
            )
            if w.native_steps:
                raise ValueError("entire matched state bank admission required before any physics")
        binding = preflight_receiving_courses(tuple(typed))
        write_once(
            a.output / "preflight.json",
            dict(
                initial_conditions=40,
                worlds=80,
                course_binding=binding,
                ball_dimensions=setup.ball_dimensions,
                native_steps=0,
            ),
        )
        (a.output / "scene.xml").write_text(setup.xml)
        for index, course in enumerate(courses):
            row = collect_trial(setup, actor, course, output=a.output, index=index, counts=counts)
            with np.load(a.output / row["artifacts"]["A"]["path"], allow_pickle=False) as z:
                if z["teacher_delta_HG"].any() or not np.array_equal(
                    z["NN_action"], z["executed_action"]
                ):
                    raise ValueError(
                        "unassisted candidate cannot receive any teacher motor correction"
                    )
            records.append(row)
        if len(records) != 80 or counts["native_steps"] > 320000 or counts["actor_calls"] > 64000:
            raise ValueError("whole bounded80-task candidate evaluation required")
        comparison = []
        for row, old in zip(records, baseline, strict=True):
            b, c = (
                controlled_pass_label(old["measured_outcome"]),
                controlled_pass_label(row["outcome"]),
            )
            if b != old["task_success"]:
                raise ValueError("original posthoc training label commitment differs")
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
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="TEAM_PASS_CANDIDATE80_COLLECTED_PENDING_OWN_NATIVE_AND_RETENTION",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                preflight_hash=digest(a.output / "preflight.json"),
                scene_hash=digest(a.output / "scene.xml"),
                training_hash=a.training_hash,
                checkpoint_kind="team-pass-consolidated-v1",
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
                status="TEAM_PASS_CANDIDATE_EXAM_FAILED_PRESERVE_NO_RETRY",
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
