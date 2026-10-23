"""Admission/setup for the already SEALED pass-near2D48 study; no sampling."""

import json

from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import receipt

SEAL_STATUS = "PASS_FRESH48_SEALED_AFTER_WEIGHTS_AND_OLD_RETENTION_FREEZE"
SOURCE_STATUS = "PASS_FRESH_CONTEXT_SHARD_COLLECTED_PENDING_WHOLE_OWN"
OWN_STATUS = "PASS_FRESH_CONTEXT_SHARD_WHOLE_OWN_NATIVE_MATCH"


def require_frozen_pass_protocol(pins):
    values = {
        h for p, h in pins.items() if p.endswith("/rosclaw_soccer/rsi/pass_fresh_curriculum.py")
    }
    if len(values) != 1:
        raise ValueError(
            "sampler and screening code must remain byte-identical to pre-sampling seal"
        )


def verify_pass_scene(setup, seal):
    from rosclaw_soccer.sim.contracts import hash_json

    # Native dataclass inertia is a tuple; JSON seal correctly stores a list.
    # Canonicalize serialization, never loosen a physical numeric comparison.
    dimensions = json.loads(json.dumps(setup.ball_dimensions, allow_nan=False))
    if hash_json(setup.xml) != seal["scene_hash"] or dimensions != seal["compiled_ball"]:
        raise ValueError("same compiled shared scene and ball dimensions required")


def admit_pass_seal(root, expected_hash):
    paths = [
        root / n
        for n in (
            "complete.json",
            "declaration.json",
            "private-courses.json",
            "private-initial-states.json",
        )
    ]
    if (
        root.is_symlink()
        or any(p.is_symlink() for p in paths)
        or (root / "failed.json").exists()
        or digest(paths[0]) != expected_hash
    ):
        raise ValueError("exact complete ordinary private pass Fresh seal required")
    seal, decl, private, initial = [json.loads(p.read_text()) for p in paths]
    if (
        seal["status"] != SEAL_STATUS
        or seal["protocol"] != "soccer.pass_near_2D48_sealed.v1"
        or seal["declaration_hash"] != digest(paths[1])
        or any(seal.get(k) != v for k, v in decl.items())
        or seal["courses_hash"] != digest(paths[2])
        or seal["initial_states_hash"] != digest(paths[3])
        or seal["independent_initial_states"] != 48
        or seal["goal_tasks"] != 96
        or seal["shards"] != 4
        or any(seal[k] != 0 for k in ("physical_steps", "fit_rows"))
        or any(
            seal[k] is not True
            for k in ("weights_frozen_before_sampling", "all_states_preflighted_before_workers")
        )
        or any(
            seal[k] is not False
            for k in (
                "sampling_adapts_to_results",
                "private_case_rows_printed",
                "hardware_authorized",
                "promotion_authorized",
                "goal_complete",
            )
        )
        or type(private["seed"]) is not int
        or not 0 <= private["seed"] < 2**64
        or len(private["courses"]) != 96
        or len(initial) != 48
        or len(set(initial)) != 48
    ):
        raise ValueError("immutable whole48 preflighted local pass seal required")
    eligible = [r["seed"] for r in seal["reports"] if r["eligible_for_diagnostic_Fresh"]]
    if (
        [r["seed"] for r in seal["reports"]] != [105023, 105024, 105025]
        or not eligible
        or [m["seed"] for m in seal["models"]] != [65023] + eligible
        or any(
            r["eligible_for_diagnostic_Fresh"]
            != (
                r["retention"]["acquired_frontier_retained"]
                and r["pass_summary"]["candidate_controlled"]
                > r["pass_summary"]["baseline_controlled"]
            )
            for r in seal["reports"]
        )
    ):
        raise ValueError("all and only independently retained improving candidates required")
    pins = dict(seal["input_file_hashes"])
    pins.update({str(p): digest(p) for p in paths})
    return seal, pins, private, initial


def open_pass_seal_setup(seal, pins, private, initial):
    """Load one baseline plus admitted heads, then verify ALL48 before stepping."""
    import numpy as np

    from rosclaw_soccer.rsi.consolidated_adapter_checkpoint import (
        load_motor_candidate_for_sim_audit,
    )
    from rosclaw_soccer.rsi.pass_fresh_curriculum import pass_fresh_courses
    from rosclaw_soccer.rsi.shared_relay_setup import SharedRelaySetup
    from rosclaw_soccer.sim.contracts import hash_json
    from rosclaw_soccer.training.native_pass_courses import NativePassCourse
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
    from scripts.rsi_stroke_search_common import open_motor

    _, data = receipt(pins, seal["data_hash"])
    _, old = receipt(pins, seal["baseline_training_hash"])
    _, original = receipt(pins, data["base_data_hash"])
    contexts = (
        {int(k): v for k, v in data["contexts"].items()}
        if isinstance(data["contexts"], dict)
        else dict(enumerate(data["contexts"]))
    )
    bench, baseline = open_motor(
        pins,
        seal["baseline_training_hash"],
        {i: contexts[i] for i in (1, 4, 20, 21)},
        checkpoint_kind="contact-consolidated-v2",
    )
    models = {65023: baseline}
    for model in seal["models"]:
        tp, trained = receipt(pins, model["training_hash"])
        if (
            str(tp.parent) != model["training_root"]
            or trained["checkpoint_hash"] != model["checkpoint_hash"]
            or trained["config"]["seed"] != model["seed"]
        ):
            raise ValueError("same frozen model identity required")
        if model["seed"] == 65023:
            if model["kind"] != "contact-consolidated-v2":
                raise ValueError("unchanged baseline kind required")
            continue
        if model["kind"] != "consolidated-adapter-v1":
            raise ValueError("explicit new head required")
        actor, _ = load_motor_candidate_for_sim_audit(
            baseline.parent,
            parent_lineage_hash=old["parent_lineage_hash"],
            root=tp.parent,
            expected_receipt_hash=model["training_hash"],
            checkpoint_kind=model["kind"],
        )
        models[model["seed"]] = actor
    setup = SharedRelaySetup(pins, contexts, bench=bench, actor=baseline)
    verify_pass_scene(setup, seal)
    templates = sorted(
        [r["source_course"] for r in original["records"] if r["role"] == "new_unassisted"],
        key=lambda c: (c["state_id"], c["task"]),
    )
    directions = {i: setup.prepare(i, -0.55)[2] for i in (1, 4, 20, 21)}
    if pass_fresh_courses(templates, directions, private["seed"]) != private["courses"]:
        raise ValueError("sealed sample must exactly reproduce; no resampling allowed")
    typed = []
    for c in private["courses"][::2]:
        _, w, fwd, home, _ = setup.prepare(
            c["context"],
            c["side"],
            ball_planar_offset_world=np.array(c["ball_planar_offset_world"]),
        )
        h = hash_json(dict(qpos=w.data.qpos.tolist(), qvel=w.data.qvel.tolist()))
        if h != initial[c["state_id"]] or w.native_steps:
            raise ValueError("compiled pre-physics initial state differs from sealed bank")
        q, _ = w.layouts["a/"].canonical(w.data)
        rel = home - q[36:38]
        lat = np.array([-fwd[1], fwd[0]])
        typed.append(NativePassCourse(c["state_id"], float(rel @ fwd), float(rel @ lat), h))
    if preflight_receiving_courses(tuple(typed)) != seal["course_binding"] or bench.native_steps:
        raise ValueError("entire48 course binding must match before any physical worker step")
    return setup, models


def verify_trial_initial(setup, course, expected_hash):
    import numpy as np

    from rosclaw_soccer.sim.contracts import hash_json

    _, w, _, _, _ = setup.prepare(
        course["context"],
        course["side"],
        ball_planar_offset_world=np.array(course["ball_planar_offset_world"]),
    )
    if (
        w.native_steps
        or hash_json(dict(qpos=w.data.qpos.tolist(), qvel=w.data.qvel.tolist())) != expected_hash
    ):
        raise ValueError("exact sealed pre-physics state required")


def confined_artifact(root, relative, expected_hash):
    path = root / relative
    if (
        path.is_symlink()
        or not path.resolve().is_relative_to(root.resolve())
        or digest(path) != expected_hash
    ):
        raise ValueError("exact confined ordinary physical artifact required")
    return path
