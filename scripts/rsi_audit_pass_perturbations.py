"""Independently preflight the global40 TRAIN states and replay a whole shard.

Reconstructs all course inputs from the independently audited prior32 bank;
both zero-offset controls and all40 new trials use private NN state and the
manual audit clock. No producer integration, saved actions or borrowed h.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
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
        raise ValueError("unused audit and exact completed TRAIN shard required")
    s, d, pf = [json.loads(x.read_text()) for x in paths[:3]]
    if (
        s["status"] != "PASS_TRAIN_PERTURBATION40_SHARD_COLLECTED_PENDING_ALL_SHARDS_OWN"
        or d["protocol"] != "soccer.pass_teacher_train_perturbation40.v1"
        or any(r["source_head"] != a.producer_head for r in (s, d))
        or s["declaration_hash"] != digest(paths[1])
        or s["preflight_hash"] != digest(paths[2])
        or s["scene_hash"] != digest(paths[3])
        or s["context"] != d["context"]
        or d["context"] not in (1, 4, 20, 21)
        or d["maximum_worlds"] != 42
        or d["maximum_native_steps"] != 168000
        or d["maximum_actor_calls"] != 33600
        or d["formal_pass_receive_controls"] != 300
        or d["required_A_neural_recovery_controls"] != 100
        or len(s["calibration"]) != 2
        or len(s["records"]) != 40
        or any(
            r[k] is not False
            for r in (s, d)
            for k in (
                "goal_complete",
                "Fresh_verified",
                "promotion_authorized",
                "hardware_authorized",
            )
        )
        or any(r["TRAIN_only"] is not True for r in (s, d))
    ):
        raise ValueError("whole42 unpromoted TRAIN-only finite shard required")
    pins = dict(s["input_file_hashes"])
    parent_path, parent = receipt(pins, d["source_hash"])
    _, parent_own = receipt(pins, d["audit_hash"])
    parent_dp = parent_path.parent / "declaration.json"
    if (
        parent["status"] != "PASS_TEACHER_TWO_BODY32_COLLECTED_PENDING_OWN_REPLAY"
        or parent_own["status"] != "PASS_TEACHER_TWO_BODY32_WHOLE_OWN_NATIVE_REPLAY_MATCH"
        or parent_own["producer_hash"] != d["source_hash"]
        or parent_own["counts"] != parent["counts"]
        or len(parent["records"]) != 32
        or len(parent_own["records"]) != 32
        or digest(parent_dp) != parent["declaration_hash"]
        or any(
            {k: v for k, v in r.items() if k != "artifacts"} != o
            for r, o in zip(parent["records"], parent_own["records"], strict=True)
        )
    ):
        raise ValueError("whole independently audited parent32 task identity required")
    pd = json.loads(parent_dp.read_text())
    if any(d[k] != pd[k] for k in ("candidate", "baseline_training_hash", "contexts")):
        raise ValueError("unchanged models and original physical course sources required")
    for path in (*paths, parent_dp):
        pins[str(path)] = digest(path)
    for folder, rows in ((a.root, s["records"]), (a.root / "calibration", s["calibration"])):
        for i, row in enumerate(rows):
            if row["index"] != i or set(row["artifacts"]) != {"initial", "physical", "A", "B"}:
                raise ValueError("all ordered initial/native/NN rows required")
            for art in row["artifacts"].values():
                path = folder / art["path"]
                if (
                    path.is_symlink()
                    or path.resolve().parent != (folder / f"trial{i}").resolve()
                    or digest(path) != art["hash"]
                ):
                    raise ValueError("exact confined whole raw shard artifacts required")
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
            independent_manual_native_clock=True,
            own_neural_history=True,
            source_actions_executed=False,
            borrowed_memory=False,
            global40preflight_reconstructed=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    counts = dict(native_steps=0, actor_calls=0)
    calibration, records = [], []
    try:
        import numpy as np

        from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
        from rosclaw_soccer.rsi.pass_transfer_curriculum import train_ball_states
        from rosclaw_soccer.rsi.shared_relay_setup import SharedRelaySetup
        from rosclaw_soccer.sim.contracts import hash_json
        from rosclaw_soccer.training.native_pass_courses import NativePassCourse
        from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
        from scripts.rsi_pass_transfer_replay_trial import replay_trial
        from scripts.rsi_stroke_search_common import open_motor

        contexts = {int(k): v for k, v in pd["contexts"].items()}
        bench, base = open_motor(pins, pd["baseline_training_hash"], contexts)
        c = pd["candidate"]
        _, tr = receipt(pins, c["training_hash"])
        actor, _ = load_dual_candidate_for_sim_audit(
            base.parent,
            parent_lineage_hash=tr["parent_lineage_hash"],
            root=Path(c["training_root"]),
            expected_receipt_hash=c["training_hash"],
            checkpoint_kind=c["kind"],
        )
        setup = SharedRelaySetup(pins, contexts, bench=bench, actor=actor)
        if setup.xml != paths[3].read_text():
            raise ValueError("own compiled shared scene differs")
        states = train_ball_states({i: setup.prepare(i, -0.55)[2] for i in (1, 4, 20, 21)})
        bindings, typed = [], []
        for state in states:
            _, w, forward, home, _ = setup.prepare(
                state["context"],
                state["side"],
                ball_planar_offset_world=np.array(state["ball_planar_offset_world"]),
            )
            q, _ = w.layouts["a/"].canonical(w.data)
            relative = home - q[36:38]
            lateral = np.array([-forward[1], forward[0]])
            initial_hash = hash_json(dict(qpos=w.data.qpos.tolist(), qvel=w.data.qvel.tolist()))
            if w.native_steps:
                raise ValueError("all40 own initial states required before any integration")
            typed.append(
                NativePassCourse(
                    state["state_id"],
                    float(relative @ forward),
                    float(relative @ lateral),
                    initial_hash,
                )
            )
            bindings.append(dict(state=state, initial_world_state_hash=initial_hash))
        expected_pf = dict(
            global_initial_conditions=40,
            global_trials=160,
            local_initial_conditions=10,
            course_binding=preflight_receiving_courses(tuple(typed)),
            states=bindings,
            native_steps=0,
            ball_dimensions=setup.ball_dimensions,
        )
        if (
            json.loads(json.dumps(expected_pf, allow_nan=False)) != pf
            or states != d["global_states"]
        ):
            raise ValueError("whole40 own new physical initial states differ")
        courses = []
        for state in states:
            for task in (0, 1):
                for assisted in (False, True):
                    old = next(
                        r["course"]
                        for r in parent["records"]
                        if r["course"]["context"] == state["context"]
                        and r["course"]["task"] == task
                        and r["course"]["assisted"] == assisted
                    )
                    courses.append(
                        old
                        | dict(
                            side=state["side"],
                            state_id=state["state_id"],
                            ball_planar_offset_world=state["ball_planar_offset_world"],
                        )
                    )
        selected = [c for c in courses if c["context"] == d["context"]]
        if (
            courses != d["global_courses"]
            or selected != d["selected_courses"]
            or len(selected) != 40
        ):
            raise ValueError("all paired task inputs differ from original audited teacher")
        cal_out = a.output / "calibration"
        cal_out.mkdir()
        prior = None
        for i, assisted in enumerate((False, True)):
            old = next(
                r
                for r in parent["records"]
                if r["course"]["context"] == d["context"]
                and r["course"]["task"] == 0
                and r["course"]["side"] == -0.55
                and r["course"]["assisted"] == assisted
            )
            claim = s["calibration"][i]
            if claim["course"] != old["course"] or {
                k: v["hash"] for k, v in claim["artifacts"].items()
            } != {k: v["hash"] for k, v in old["artifacts"].items()}:
                raise ValueError("whole raw zero-offset controls differ from prior32")
            row, prior = replay_trial(
                setup,
                actor,
                old["course"],
                claim,
                raw_root=a.root / "calibration",
                output=cal_out,
                index=i,
                counts=counts,
                prior_initial=prior,
            )
            calibration.append(row)
        prior = None
        for i, course in enumerate(selected):
            claim = s["records"][i]
            if claim["course"] != course:
                raise ValueError("entire ordered paired perturbation trial required")
            row, prior = replay_trial(
                setup,
                actor,
                course,
                claim,
                raw_root=a.root,
                output=a.output,
                index=i,
                counts=counts,
                prior_initial=prior,
            )
            records.append(row)
        if counts != s["counts"] or len(records) != 40 or len(calibration) != 2:
            raise ValueError("whole42 own native and NN accounting differs")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="PASS_TRAIN_PERTURBATION40_SHARD_WHOLE_OWN_NATIVE_REPLAY_MATCH",
                source_head=a.expected_head,
                producer_hash=a.source_hash,
                input_file_hashes=pins,
                context=d["context"],
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
                status="PASS_PERTURBATION_AUDIT_FAILED_PRESERVE_NO_RETRY",
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
