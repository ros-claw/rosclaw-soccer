"""Fixed-A theta12,16 physical starts x2 B teachers, own-native audited separately.

All placements happen before the clock starts; paired arms share identical
initials. No live reset, launcher, learned B deployment, new optimizer,
replay action playback or automatic promotion.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("experience-root", "own-root", "training-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("own-hash", "training-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--audit-source-root", type=Path)
    p.add_argument("--producer-head")
    a = p.parse_args()
    if (a.audit_source_root is None) != (a.producer_head is None):
        p.error("explicit own-native source/head pair required")
    a.state_id = 26
    import numpy as np
    import torch

    from rosclaw_soccer.rsi.broad_receiving_role import (
        MODES,
        OWN,
        SOURCE,
        config,
        sampling_seed,
        select_courses,
    )
    from rosclaw_soccer.rsi.continuing_refinement_contract import KIND
    from rosclaw_soccer.sim.contracts import hash_json
    from rosclaw_soccer.training.native_pass_courses import NativePassCourse
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_learned_refinement_common import admit, open_models
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused once-only broad role output required")
    admitted = admit(a)
    pins, scope = admitted[1:3]
    trained = json.loads((a.training_root / "complete.json").read_text())
    if (
        trained["checkpoint_kind"] != KIND
        or trained["config"]["generation_index"] != 2
        or trained["cumulative_actor_steps"] != 12
        or trained["cumulative_critic_steps"] != 12
    ):
        raise ValueError("fixed generation2 theta12 A required, not rejected theta16")
    courses = select_courses(scope["global_courses"])
    source = None
    if a.audit_source_root is not None:
        root = a.audit_source_root
        path = root / "complete.json"
        if root.is_symlink() or path.is_symlink() or (root / "failed.json").exists():
            raise ValueError("ordinary closed broad source required")
        source = json.loads(path.read_text())
        if (
            source["status"] != SOURCE
            or source["config"] != config()
            or source["source_head"] != a.producer_head
            or source["training_hash"] != a.training_hash
            or source["courses"] != courses
            or len(source["records"]) != 32
        ):
            raise ValueError("exact entire broad source commitment required")
        for key, value in source["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting broad source ancestry")
            pins[key] = value
        pins[str(path)] = digest(path)
        for row in source["records"]:
            for art in row["artifacts"].values():
                path = root / art["path"]
                if (
                    path.is_symlink()
                    or path.resolve().parent != (root / f"trial{row['index']}").resolve()
                    or digest(path) != art["hash"]
                ):
                    raise ValueError("confined exact actual raw primitives required")
                pins[str(path)] = art["hash"]
    source_pins(pins)
    check_pins(pins)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    setup, _, actor, _, _ = open_models(a, admitted)
    typed = []
    for course in courses:
        _, w, forward, home, _ = setup.prepare(
            course["context"],
            course["side"],
            ball_planar_offset_world=np.array(course["ball_planar_offset_world"]),
        )
        q, _ = w.layouts["a/"].canonical(w.data)
        rel, lat = home - q[36:38], np.array([-forward[1], forward[0]])
        typed.append(
            NativePassCourse(
                course["state_id"],
                float(rel @ forward),
                float(rel @ lat),
                hash_json(dict(qpos=w.data.qpos.tolist(), qvel=w.data.qvel.tolist())),
            )
        )
        if w.native_steps:
            raise ValueError("all16 initial-state preflight must precede any integration")
    binding = preflight_receiving_courses(tuple(typed))
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            config=config(),
            courses=courses,
            source_head=a.expected_head,
            input_file_hashes=pins,
            training_hash=a.training_hash,
            course_binding=binding,
            ball_dimensions=setup.ball_dimensions,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )
    counts, records, previous = dict(native_steps=0, actor_calls=0), [], None
    try:
        for i, course in enumerate(courses):
            for mode in MODES:
                index = len(records)
                common = dict(
                    output=a.output,
                    index=index,
                    counts=counts,
                    emit_course_rows=False,
                    sampling_seed=sampling_seed(course["state_id"]),
                    sampling_profile="contact-v1",
                    sampling_include_recovery=True,
                    receiver_intent_mode=mode,
                    receiver_role_ablation=True,
                )
                if source is None:
                    from scripts.rsi_pass_transfer_trial import collect_trial

                    row = collect_trial(setup, actor, course, **common)
                else:
                    from scripts.rsi_pass_transfer_replay_trial import replay_trial

                    row, previous = replay_trial(
                        setup,
                        actor,
                        course,
                        source["records"][index],
                        raw_root=a.audit_source_root,
                        prior_initial=previous,
                        **common,
                    )
                if (
                    row["index"] != 2 * i + MODES.index(mode)
                    or row["course"] != course
                    or row["sampling_seed"] != sampling_seed(course["state_id"])
                ):
                    raise ValueError("all32 fixed course/arm/noise executions required")
                records.append(row)
                write_once(a.output / f"record{index}.json", row)
        if (
            counts
            != dict(
                native_steps=sum(r["native_steps"] for r in records),
                actor_calls=sum(r["actor_calls"] for r in records),
            )
            or counts["native_steps"] > 128000
            or counts["actor_calls"] > 25600
        ):
            raise ValueError("all charged actual native and neural work required")
        if source is not None and (
            records != [{k: v for k, v in r.items() if k != "artifacts"} for r in source["records"]]
            or counts != source["counts"]
        ):
            raise ValueError("all32 own-native physical and law outcomes must match")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status=SOURCE if source is None else OWN,
                config=config(),
                source_head=a.expected_head,
                source_hash=None
                if source is None
                else digest(a.audit_source_root / "complete.json"),
                independent_whole_world_replay=source is not None,
                training_hash=a.training_hash,
                courses=courses,
                counts=counts,
                records=records,
                course_binding=binding,
                input_file_hashes=pins,
                optimizer_updates=0,
                teacher_geometry_not_learning=True,
                Fresh_verified=False,
                promotion_authorized=False,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                counts=counts,
                completed_records=len(records),
                incomplete_world_native_steps_unknown=True,
                reserved_native_step_ceiling=128000,
                goal_complete=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
