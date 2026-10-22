"""Forty new nearby TRAIN ball states; paired physics teacher transfer.

Each of four explicitly selected context shards preflights the SAME entire
40-state declaration, runs two zero-perturbation whole-raw identity controls,
then its40 paired trials (10states x2goals x2arms). No state replacement,
search, model updates or automatic promotion. Whole-bank review requires all
four shards and their later own native audits, not a winning shard.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "audit-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-hash", "audit-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--context", type=int, choices=(1, 4, 20, 21), required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    paths = [a.root / "complete.json", a.audit_root / "complete.json", a.root / "declaration.json"]
    if (
        a.output.exists()
        or any(x.is_symlink() for x in paths)
        or any((r / "failed.json").exists() for r in (a.root, a.audit_root))
        or [digest(x) for x in paths[:2]] != [a.source_hash, a.audit_hash]
    ):
        raise ValueError("unused TRAIN output and whole closed own32 proof required")
    s, own, d = [json.loads(x.read_text()) for x in paths]
    if (
        s["status"] != "PASS_TEACHER_TWO_BODY32_COLLECTED_PENDING_OWN_REPLAY"
        or own["status"] != "PASS_TEACHER_TWO_BODY32_WHOLE_OWN_NATIVE_REPLAY_MATCH"
        or own["producer_hash"] != a.source_hash
        or own["counts"] != s["counts"]
        or s["declaration_hash"] != digest(paths[2])
        or d["protocol"] != "soccer.audited_pass_teacher_two_body_transfer32.v1"
        or len(s["records"]) != 32
        or len(own["records"]) != 32
        or any(
            r[k] is not False
            for r in (s, own)
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
        or any(
            {k: v for k, v in r.items() if k != "artifacts"} != q
            for r, q in zip(s["records"], own["records"], strict=True)
        )
    ):
        raise ValueError("all32 original paired native and ownNN outcomes required")
    pins = dict(own["input_file_hashes"])
    pins.update({str(path): digest(path) for path in paths})
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir(parents=True)
    counts = dict(native_steps=0, actor_calls=0)
    records, calibration = [], []
    try:
        import numpy as np

        from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
        from rosclaw_soccer.rsi.pass_transfer_curriculum import train_ball_states
        from rosclaw_soccer.rsi.shared_relay_setup import SharedRelaySetup
        from rosclaw_soccer.sim.contracts import hash_json
        from rosclaw_soccer.training.native_pass_courses import NativePassCourse
        from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
        from scripts.rsi_pass_transfer_trial import collect_trial
        from scripts.rsi_stroke_search_common import open_motor

        contexts = {int(k): v for k, v in d["contexts"].items()}
        bench, base = open_motor(pins, d["baseline_training_hash"], contexts)
        c = d["candidate"]
        if c["seed"] != 65023 or c["kind"] != "contact-consolidated-v2":
            raise ValueError("same retained frozen neural actor required")
        _, tr = receipt(pins, c["training_hash"])
        actor, _ = load_dual_candidate_for_sim_audit(
            base.parent,
            parent_lineage_hash=tr["parent_lineage_hash"],
            root=Path(c["training_root"]),
            expected_receipt_hash=c["training_hash"],
            checkpoint_kind=c["kind"],
        )
        setup = SharedRelaySetup(pins, contexts, bench=bench, actor=actor)
        if (
            setup.xml != (a.root / "scene.xml").read_text()
            or digest(a.root / "scene.xml") != s["scene_hash"]
        ):
            raise ValueError("unchanged actual scene and physical properties required")
        directions = {i: setup.prepare(i, -0.55)[2] for i in (1, 4, 20, 21)}
        states = train_ball_states(directions)
        typed = []
        bindings = []
        for state in states:
            _, w, forward, home, _ = setup.prepare(
                state["context"],
                state["side"],
                ball_planar_offset_world=np.array(state["ball_planar_offset_world"]),
            )
            q, _ = w.layouts["a/"].canonical(w.data)
            relative = home - q[36:38]
            lateral = np.array([-forward[1], forward[0]])
            if w.native_steps:
                raise ValueError("whole40-state admission must precede any native step")
            initial_hash = hash_json(dict(qpos=w.data.qpos.tolist(), qvel=w.data.qvel.tolist()))
            typed.append(
                NativePassCourse(
                    state["state_id"],
                    float(relative @ forward),
                    float(relative @ lateral),
                    initial_hash,
                )
            )
            bindings.append(dict(state=state, initial_world_state_hash=initial_hash))
        binding = preflight_receiving_courses(tuple(typed))
        courses = []
        for state in states:
            for task in (0, 1):
                for assisted in (False, True):
                    original = next(
                        r["course"]
                        for r in s["records"]
                        if r["course"]["context"] == state["context"]
                        and r["course"]["task"] == task
                        and r["course"]["assisted"] == assisted
                    )
                    courses.append(
                        original
                        | dict(
                            side=state["side"],
                            state_id=state["state_id"],
                            ball_planar_offset_world=state["ball_planar_offset_world"],
                        )
                    )
        selected = [c for c in courses if c["context"] == a.context]
        if len(states) != 40 or len(courses) != 160 or len(selected) != 40:
            raise ValueError("entire fixed40-state/160-trial bank required")
        write_once(
            a.output / "declaration.json",
            dict(
                protocol="soccer.pass_teacher_train_perturbation40.v1",
                source_head=a.expected_head,
                input_file_hashes=pins,
                source_hash=a.source_hash,
                audit_hash=a.audit_hash,
                context=a.context,
                global_states=states,
                global_courses=courses,
                selected_courses=selected,
                candidate=c,
                baseline_training_hash=d["baseline_training_hash"],
                contexts=d["contexts"],
                calibration_worlds=2,
                selected_worlds=40,
                maximum_worlds=42,
                maximum_native_steps=168000,
                maximum_actor_calls=33600,
                formal_pass_receive_controls=300,
                required_A_neural_recovery_controls=100,
                TRAIN_only=True,
                Fresh_verified=False,
                neural_actor_optimizer_updates=0,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
        write_once(
            a.output / "preflight.json",
            dict(
                global_initial_conditions=40,
                global_trials=160,
                local_initial_conditions=10,
                course_binding=binding,
                states=bindings,
                native_steps=0,
                ball_dimensions=setup.ball_dimensions,
            ),
        )
        (a.output / "scene.xml").write_text(setup.xml)
        cal_root = a.output / "calibration"
        cal_root.mkdir()
        for i, assisted in enumerate((False, True)):
            old = next(
                r
                for r in s["records"]
                if r["course"]["context"] == a.context
                and r["course"]["task"] == 0
                and r["course"]["side"] == -0.55
                and r["course"]["assisted"] == assisted
            )
            row = collect_trial(
                setup, actor, old["course"], output=cal_root, index=i, counts=counts
            )
            if {k: v["hash"] for k, v in row["artifacts"].items()} != {
                k: v["hash"] for k, v in old["artifacts"].items()
            } or {k: v for k, v in row.items() if k not in ("index", "artifacts")} != {
                k: v for k, v in old.items() if k not in ("index", "artifacts")
            }:
                raise ValueError(
                    "refactored zero-offset control differs from original whole raw NN/native"
                )
            calibration.append(row)
        for index, course in enumerate(selected):
            records.append(
                collect_trial(setup, actor, course, output=a.output, index=index, counts=counts)
            )
        if (
            len(records) != 40
            or len(calibration) != 2
            or counts["native_steps"] > 168000
            or counts["actor_calls"] > 33600
        ):
            raise ValueError("all local40 paired TRAIN trajectories and2 controls required")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="PASS_TRAIN_PERTURBATION40_SHARD_COLLECTED_PENDING_ALL_SHARDS_OWN",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                preflight_hash=digest(a.output / "preflight.json"),
                scene_hash=digest(a.output / "scene.xml"),
                context=a.context,
                counts=counts,
                records=records,
                calibration=calibration,
                TRAIN_only=True,
                Fresh_verified=False,
                team_game_verified=False,
                neural_actor_optimizer_updates=0,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="PASS_PERTURBATION_FAILED_PRESERVE_NO_RETRY",
                error_type=type(error).__name__,
                error=str(error),
                counts=counts,
                records=records,
                calibration=calibration,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
