"""Qualified old neural expert admission and TRAIN-only local course preflight."""

import json

from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import receipt


def admit(root, expected_hash, cfg):
    op = root / "complete.json"
    if (
        root.is_symlink()
        or op.is_symlink()
        or (root / "failed.json").exists()
        or digest(op) != expected_hash
    ):
        raise ValueError("exact qualified global neural expert own80 required")
    own = json.loads(op.read_text())
    pins = dict(own["input_file_hashes"])
    pins[str(op)] = expected_hash
    sp, source = receipt(pins, own["producer_hash"])
    _, trained = receipt(pins, own["training_hash"])
    _, bank = receipt(pins, trained["bank_hash"])
    paths = [sp.parent / name for name in ("declaration.json", "preflight.json", "scene.xml")]
    if (
        own["status"] != "TEAM_PASS_CANDIDATE80_WHOLE_OWN_NATIVE_REPLAY_MATCH"
        or source["checkpoint_kind"] != "consolidated-refinement-v1"
        or source["training_hash"] != own["training_hash"]
        or source["status"] != "TEAM_PASS_CANDIDATE80_COLLECTED_PENDING_OWN_NATIVE_AND_RETENTION"
        or trained["config"]["seed"] != cfg["source_expert_seed"]
        or trained["new_head_steps"] != 2048
        or len(source["records"]) != 80
        or own["records"]
        != [{k: v for k, v in x.items() if k != "artifacts"} for x in source["records"]]
        or own["counts"] != source["counts"]
        or any(
            d[k] is not False
            for d in (own, source, trained)
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
        or any(p.is_symlink() or pins.get(str(p)) != digest(p) for p in paths)
        or source["declaration_hash"] != digest(paths[0])
        or source["preflight_hash"] != digest(paths[1])
        or source["scene_hash"] != digest(paths[2])
    ):
        raise ValueError(
            "unchanged source/own80, exact2048-step expert and ordinary inputs required"
        )
    chosen = [
        r
        for r in source["records"]
        if r["course"]["state_id"] == cfg["state_id"] and r["course"]["task"] == cfg["task"]
    ]
    if len(chosen) != 1 or chosen[0]["course"]["assisted"] is not False:
        raise ValueError("one exact inherited unassisted PUBLIC TRAIN source required")
    for art in chosen[0]["artifacts"].values():
        p = sp.parent / art["path"]
        if (
            p.is_symlink()
            or p.resolve().parent != (sp.parent / f"trial{chosen[0]['index']}").resolve()
            or digest(p) != art["hash"]
        ):
            raise ValueError("confined exact inherited full physical/private traces required")
        pins[str(p)] = art["hash"]
    scope = dict(
        student_training_hash=own["training_hash"],
        student_kind="consolidated-refinement-v1",
        global_courses=[r["course"] for r in source["records"]],
    )
    return pins, scope, bank, json.loads(paths[1].read_text()), sp.parent, chosen[0]


def local_preflight(setup, baseline, cfg, source_root):
    import numpy as np

    from rosclaw_soccer.sim.contracts import hash_json
    from rosclaw_soccer.training.native_pass_courses import NativePassCourse
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

    course = baseline["course"]
    _, w, fwd, _, _ = setup.prepare(
        course["context"],
        course["side"],
        ball_planar_offset_world=np.array(course["ball_planar_offset_world"]),
    )
    with np.load(source_root / baseline["artifacts"]["initial"]["path"], allow_pickle=False) as z:
        if not np.array_equal(z["qpos"], w.data.qpos) or not np.array_equal(z["qvel"], w.data.qvel):
            raise ValueError("fresh original nominal initial state differs from audited source")
    variants = [
        course
        | dict(
            ball_planar_offset_world=(
                np.array(course["ball_planar_offset_world"]) + x * fwd
            ).tolist()
        )
        for x in cfg["unsearched_TRAIN_offsets_forward_m"]
    ]
    typed = []
    for index, c in enumerate([course, *variants]):
        _, w, forward, home, _ = setup.prepare(
            c["context"],
            c["side"],
            ball_planar_offset_world=np.array(c["ball_planar_offset_world"]),
        )
        q, _ = w.layouts["a/"].canonical(w.data)
        rel, lateral = home - q[36:38], np.array([-forward[1], forward[0]])
        typed.append(
            NativePassCourse(
                index,
                float(rel @ forward),
                float(rel @ lateral),
                hash_json(dict(qpos=w.data.qpos.tolist(), qvel=w.data.qvel.tolist())),
            )
        )
        if w.native_steps:
            raise ValueError("ALL declared local course preflight must precede physics")
    return variants, dict(
        course_binding=preflight_receiving_courses(tuple(typed)),
        initial_conditions=3,
        forward=fwd.tolist(),
        validation_courses=variants,
        physical_steps=0,
    )
