"""SIM TRAIN: actual student-prefix/expert-suffix trajectories, not advice labels.

Every context shard admits ALL40 original physical states first. Prefix0 must
reproduce the complete old teacher trajectory; prefix80 must reproduce the
student's first80 actual controls before a continuous expert takeover. Both
private neural histories consume actual actions; no state or h/c restoration.
All outcomes, including failures, are retained. A later independent native
replay and outcome qualification are required before any correction-bank fit.
"""

import argparse
import json
from pathlib import Path


def compare_trajectory(new_root, row, old_root, old, *, prefix):
    """Exact primitive equality; added private-advice fields are not old evidence."""
    import numpy as np

    from scripts.rsi_native_context_recovery_common import digest

    compared = 0
    for name in ("initial", "physical", "A", "B"):
        x, y = new_root / row["artifacts"][name]["path"], old_root / old["artifacts"][name]["path"]
        if digest(y) != old["artifacts"][name]["hash"]:
            raise ValueError("exact original whole physical/neural primitives required")
        with np.load(x, allow_pickle=False) as a, np.load(y, allow_pickle=False) as b:
            if not set(b.files) <= set(a.files):
                raise ValueError("continuation dropped an original primitive")
            for key in b.files:
                left, right = a[key], b[key]
                if prefix is not None and name != "initial":
                    if len(left) < prefix or len(right) < prefix:
                        raise ValueError("whole declared student prefix required")
                    left, right = left[:prefix], right[:prefix]
                if left.dtype != right.dtype or not np.array_equal(left, right):
                    raise ValueError(f"actual trajectory identity mismatch: {name}.{key}")
                compared += 1
    return compared


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("bank-root", "native-own-root", "retention-own-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("bank-hash", "native-own-hash", "retention-own-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--context", type=int, choices=(1, 4, 20, 21), required=True)
    p.add_argument(
        "--student-kind",
        choices=("team-pass-consolidated-v1", "consolidated-adapter-v1"),
        default="team-pass-consolidated-v1",
    )
    p.add_argument("--student-seed", type=int, default=75024)
    p.add_argument("--preflight-only", action="store_true")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    roots = (a.bank_root, a.native_own_root, a.retention_own_root)
    paths = [r / "complete.json" for r in roots]
    if (
        a.output.exists()
        or any(x.is_symlink() for x in paths)
        or any((r / "failed.json").exists() for r in roots)
        or [digest(x) for x in paths] != [a.bank_hash, a.native_own_hash, a.retention_own_hash]
    ):
        raise ValueError("exact three closed witnesses and unused output required")
    bank, native_own, retained = [json.loads(x.read_text()) for x in paths]
    if (
        bank["status"] != "TEAM_PASS_WHOLE_HISTORY_BANK177_READY_NOT_TRAINED"
        or native_own["status"] != "TEAM_PASS_CANDIDATE80_WHOLE_OWN_NATIVE_REPLAY_MATCH"
        or retained["status"] != "DUAL_RECURRENT_ALL44_WHOLE_NATIVE_OWN_HISTORY_AUDIT_PASS"
        or retained["worlds"] != 44
        or retained["checkpoint_kind"] != a.student_kind
        or retained["training_hash"] != native_own["training_hash"]
        or len(native_own["records"]) != 80
        or any(
            r[k] is not False
            for r in (bank, native_own, retained)
            for k in ("goal_complete", "hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("fully replayed actual same unpromoted student required")
    pins = {}
    for r in (bank, native_own, retained):
        for key, value in r["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting source proof pin")
            pins[key] = value
    pins.update({str(x): digest(x) for x in paths})
    tp, training = receipt(pins, retained["training_hash"])
    _, warm = receipt(pins, bank["warm_start_hash"])
    npth, native = receipt(pins, native_own["producer_hash"])
    from rosclaw_soccer.rsi.adapter_continuation_identity import continuation_protocol

    child_hash = training["data_hash"] if a.student_kind == "consolidated-adapter-v1" else None
    child_bank = receipt(pins, child_hash)[1] if child_hash is not None else None
    protocol = continuation_protocol(
        kind=a.student_kind,
        seed=a.student_seed,
        training=training,
        bank=bank,
        bank_hash=a.bank_hash,
        child_bank=child_bank,
        child_bank_hash=child_hash,
    )
    if (
        training["checkpoint_hash"] != retained["checkpoint_hash"]
        or len(native["records"]) != 80
        or len(retained["results"]) != 44
    ):
        raise ValueError("research student75024 exact complete lineage required; not promotion")
    from rosclaw_soccer.rsi.acquired_motor_retention import native_successes

    required = {r["source_context"] for r in bank["records"] if r["role"] == "retained_neural17"}
    if len(required) != 17 or not required <= native_successes(retained["results"]):
        raise ValueError("ALL17 acquired old skills required, not original subset only")
    teachers = sorted(
        (r for r in bank["records"] if r["role"] == "new_teacher"),
        key=lambda r: (r["source_course"]["state_id"], r["source_course"]["task"]),
    )
    courses = [r["source_course"] for r in teachers]
    if len(courses) != 80 or {(c["state_id"], c["task"]) for c in courses} != {
        (i, j) for i in range(40) for j in (0, 1)
    }:
        raise ValueError("ALL80 original teacher tasks required")
    originals = []
    for row in teachers:
        path = Path(row["source_root"]) / "complete.json"
        if pins.get(str(path)) != digest(path):
            raise ValueError("bound whole source teacher receipt required")
        original = json.loads(path.read_text())["records"][row["source_index"]]
        if original["course"] != row["source_course"]:
            raise ValueError("teacher task commitment differs")
        originals.append(original)
    if any(
        old["course"] != (c | dict(assisted=False))
        for c, old in zip(courses, native["records"], strict=True)
    ):
        raise ValueError("same actual student and expert goal tasks required")
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir(parents=True)
    counts = dict(native_steps=0, actor_calls=0, shadow_actor_calls=0)
    records = []
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
        from scripts.rsi_pass_transfer_trial import collect_trial
        from scripts.rsi_stroke_search_common import open_motor

        contexts = (
            {int(k): v for k, v in bank["contexts"].items()}
            if isinstance(bank["contexts"], dict)
            else dict(enumerate(bank["contexts"]))
        )
        bench, student = open_motor(
            pins,
            retained["training_hash"],
            {i: contexts[i] for i in (1, 4, 20, 21)},
            checkpoint_kind=a.student_kind,
        )
        wp, _ = receipt(pins, bank["warm_start_hash"])
        from rosclaw_soccer.rsi.same_state_motor_pair import paired_reference

        teacher, _ = load_motor_candidate_for_sim_audit(
            paired_reference(student),
            parent_lineage_hash=warm["parent_lineage_hash"],
            root=wp.parent,
            expected_receipt_hash=bank["warm_start_hash"],
            checkpoint_kind="contact-consolidated-v2",
        )
        setup = SharedRelaySetup(pins, contexts, bench=bench, actor=teacher)
        typed = []
        for i in range(0, 80, 2):
            c = courses[i]
            _, w, forward, home, _ = setup.prepare(
                c["context"],
                c["side"],
                ball_planar_offset_world=np.array(c["ball_planar_offset_world"]),
            )
            old = originals[i]["artifacts"]["initial"]
            ip = Path(teachers[i]["source_root"]) / old["path"]
            if pins.get(str(ip)) != digest(ip) or digest(ip) != old["hash"]:
                raise ValueError("bound original initial physics required")
            with np.load(ip, allow_pickle=False) as z:
                if not np.array_equal(z["qpos"], w.data.qpos) or not np.array_equal(
                    z["qvel"], w.data.qvel
                ):
                    raise ValueError("same initial physical state required")
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
            if w.native_steps:
                raise ValueError("entire40-state admission before physics required")
        binding = preflight_receiving_courses(tuple(typed))
        write_once(
            a.output / "declaration.json",
            dict(
                protocol=protocol,
                student_kind=a.student_kind,
                student_seed=a.student_seed,
                preflight_only=a.preflight_only,
                source_head=a.expected_head,
                input_file_hashes=pins,
                bank_hash=a.bank_hash,
                native_own_hash=a.native_own_hash,
                retention_own_hash=a.retention_own_hash,
                context=a.context,
                student_training_hash=retained["training_hash"],
                teacher_training_hash=bank["warm_start_hash"],
                global_courses=courses,
                prefixes=[0, 80],
                maximum_worlds=40,
                maximum_native_steps=160000,
                maximum_actual_motor_decisions=32000,
                maximum_shadow_actor_calls=16000,
                formal_pass_receive_controls=300,
                required_A_neural_recovery_controls=100,
                student_h_borrowed=False,
                TRAIN_only=True,
                Fresh_verified=False,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
        write_once(
            a.output / "preflight.json",
            dict(
                global_initial_conditions=40,
                local_initial_conditions=10,
                local_goal_tasks=20,
                local_worlds=40,
                course_binding=binding,
                native_steps=0,
                ball_dimensions=setup.ball_dimensions,
            ),
        )
        (a.output / "scene.xml").write_text(setup.xml)
        if a.preflight_only:
            from rosclaw_soccer.rsi.shared_native_neural_player import SharedNativeNeuralPlayer
            from rosclaw_soccer.rsi.shared_teacher_pair import SharedTeacherPair

            c = next(c for c in courses if c["context"] == a.context)
            ep, world, _, _, _ = setup.prepare(
                c["context"],
                c["side"],
                ball_planar_offset_world=np.array(c["ball_planar_offset_world"]),
            )

            def player(actor):
                return SharedNativeNeuralPlayer(
                    world,
                    "a/",
                    actor,
                    clip=ep.clip,
                    torso_index=ep.torso_index,
                    build_observation=ep.build_observation,
                    goal=np.array(c["target"]),
                    offset=ep.offset,
                    scale=ep.scale,
                    order=ep.order,
                )

            pair = SharedTeacherPair(player(student), player(teacher))
            pair.propose()
            if world.native_steps or any(counts.values()):
                raise ValueError("preflight cannot execute physical steps")
            check_pins(pins)
            clean(a.expected_head)
            write_once(
                a.output / "complete.json",
                dict(
                    status="STUDENT_CONTINUATION_ALL40_PREFLIGHT_PRIVATE_PAIR_NOT_PHYSICS",
                    source_head=a.expected_head,
                    input_file_hashes=pins,
                    declaration_hash=digest(a.output / "declaration.json"),
                    student_kind=a.student_kind,
                    student_seed=a.student_seed,
                    global_initial_conditions=40,
                    physical_steps=0,
                    unexecuted_neural_proposals=2,
                    optimizer_updates=0,
                    same_actual_observation=True,
                    borrowed_hidden_state=False,
                    hardware_authorized=False,
                    promotion_authorized=False,
                    goal_complete=False,
                ),
            )
            print(
                json.dumps(dict(stage="STUDENT_TEACHER_PREFLIGHT", states=40, native_steps=0)),
                flush=True,
            )
            return
        for i, c in enumerate(courses):
            if c["context"] != a.context:
                continue
            for prefix in (0, 80):
                row = collect_trial(
                    setup,
                    teacher,
                    c,
                    output=a.output,
                    index=len(records),
                    counts=counts,
                    student=student,
                    student_prefix_controls=prefix,
                )
                oldroot, old = (
                    (Path(teachers[i]["source_root"]), originals[i])
                    if prefix == 0
                    else (npth.parent, native["records"][i])
                )
                fields = compare_trajectory(
                    a.output, row, oldroot, old, prefix=None if prefix == 0 else 80
                )
                if prefix == 0 and row["outcome"] != old["outcome"]:
                    raise ValueError("full teacher control outcome differs")
                records.append(
                    row
                    | dict(
                        original_task_index=i,
                        identity_fields_compared=fields,
                        actual_continuation_controlled_pass=controlled_pass_label(row["outcome"]),
                    )
                )
        if (
            len(records) != 40
            or counts["native_steps"] > 160000
            or counts["actor_calls"] > 32000
            or counts["shadow_actor_calls"] > 16000
        ):
            raise ValueError("whole bounded shard required")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="STUDENT_PASS_CONTINUATION_SHARD_COLLECTED_PENDING_WHOLE_OWN_REPLAY",
                source_head=a.expected_head,
                input_file_hashes=pins,
                context=a.context,
                declaration_hash=digest(a.output / "declaration.json"),
                preflight_hash=digest(a.output / "preflight.json"),
                scene_hash=digest(a.output / "scene.xml"),
                records=records,
                counts=counts,
                prefix0_controlled=sum(
                    r["actual_continuation_controlled_pass"]
                    for r in records
                    if r["student_prefix_controls"] == 0
                ),
                prefix80_controlled=sum(
                    r["actual_continuation_controlled_pass"]
                    for r in records
                    if r["student_prefix_controls"] == 80
                ),
                optimizer_updates=0,
                PPO_actor_rows=0,
                correction_bank_ready=False,
                TRAIN_only=True,
                Fresh_verified=False,
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
                status="STUDENT_CONTINUATION_FAILED_PRESERVE_NO_RETRY",
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
