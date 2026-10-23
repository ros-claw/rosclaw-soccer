"""Review ALL three independent models, then seal a NEW local2D pass Fresh bank.

Only models retaining the fixed old skill frontier and improving consumed
controlled passing enter this diagnostic screen. No best-seed selection, fit,
physics stepping, Champion promotion or claim of global zero forgetting.
"""

import argparse
import json
import secrets
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("manifest", "core-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("manifest-hash", "core-head", "expected-head"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    if (
        a.output.exists()
        or a.manifest.is_symlink()
        or a.manifest.stat().st_size > 262144
        or digest(a.manifest) != a.manifest_hash
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=a.core_root, text=True).strip()
        != a.core_head
    ):
        raise ValueError("unused seal and exact reviewed inputs/Core source required")
    manifest = json.loads(a.manifest.read_text())
    if manifest["schema"] != "soccer.pass_fresh_frontier_inputs.v1" or [
        r["seed"] for r in manifest["candidates"]
    ] != [105023, 105024, 105025]:
        raise ValueError("ALL three predeclared trained seeds required")
    pins = {str(a.manifest): a.manifest_hash}

    def admit(entry, status):
        path = Path(entry["root"]) / "complete.json"
        if (
            path.is_symlink()
            or (path.parent / "failed.json").exists()
            or digest(path) != entry["hash"]
        ):
            raise ValueError("exact CLOSED independent evidence required")
        doc = json.loads(path.read_text())
        if (
            doc["status"] != status
            or doc["hardware_authorized"] is not False
            or doc["promotion_authorized"] is not False
        ):
            raise ValueError("actual complete SIM-only physical proof required")
        for key, h in doc["input_file_hashes"].items():
            if key in pins and pins[key] != h:
                raise ValueError("conflicting immutable lineage")
            pins[key] = h
        pins[str(path)] = entry["hash"]
        return path, doc

    old_status = "DUAL_RECURRENT_ALL44_WHOLE_NATIVE_OWN_HISTORY_AUDIT_PASS"
    _, old = admit(manifest["baseline_own"], old_status)
    if old["checkpoint_kind"] != "contact-consolidated-v2" or old["worlds"] != 44:
        raise ValueError("exact learned65023 full44 baseline required")
    _, baseline_training = receipt(pins, old["training_hash"])
    if baseline_training["config"]["seed"] != 65023:
        raise ValueError("fixed accepted research memory65023 required")

    def scope(own):
        if (
            own["worlds"] != 44
            or own["independent_whole_world_replay"] is not True
            or own["identity_intent_only"] is not True
        ):
            raise ValueError("ALL44 own unassisted actual histories required")
        _, source = receipt(pins, own["source_hash"])
        _, decl = receipt_or_declaration(source)
        if (
            own["counts"]["native"] != source["actual_native_steps"]
            or own["counts"]["neural"] != source["actual_actor_decisions"]
        ):
            raise ValueError("complete native/NN work accounting required")
        return decl

    def receipt_or_declaration(source):
        paths = [
            Path(p)
            for p, h in pins.items()
            if h == source["declaration_hash"] and p.endswith("/declaration.json")
        ]
        if len(paths) != 1 or digest(paths[0]) != source["declaration_hash"]:
            raise ValueError("exact original physical context declaration required")
        return paths[0], json.loads(paths[0].read_text())

    old_scope = scope(old)
    candidates = []
    bank = None
    for entry in manifest["candidates"]:
        _, passing = admit(entry["pass_own"], "TEAM_PASS_CANDIDATE80_WHOLE_OWN_NATIVE_REPLAY_MATCH")
        _, retained = admit(entry["retention_own"], old_status)
        _, cpu = admit(
            entry["cpu_own"],
            "DUAL_RECURRENT_TRAINED_CHECKPOINT_CPU_OWN_HISTORY_AUDIT_PASS_NOT_PHYSICS",
        )
        tp, trained = receipt(pins, passing["training_hash"])
        if (
            passing["checkpoint_kind"] != "consolidated-adapter-v1"
            or retained["checkpoint_kind"] != "consolidated-adapter-v1"
            or retained["training_hash"] != passing["training_hash"]
            or cpu["source_hash"] != passing["training_hash"]
            or any(
                d["checkpoint_hash"] != trained["checkpoint_hash"] for d in (passing, retained, cpu)
            )
            or trained["config"]["seed"] != entry["seed"]
            or scope(retained)["contexts"] != old_scope["contexts"]
            or retained["old_protected_cases"] != old["old_protected_cases"]
        ):
            raise ValueError(
                "same model, unchanged old protection and original context scope required"
            )
        _, data = receipt(pins, trained["data_hash"])
        if bank is not None and bank != trained["data_hash"]:
            raise ValueError("same training bank for all seeds required")
        bank = trained["data_hash"]
        candidates.append(
            dict(
                seed=entry["seed"],
                training_root=str(tp.parent),
                training_hash=passing["training_hash"],
                checkpoint_hash=trained["checkpoint_hash"],
                kind="consolidated-adapter-v1",
                native=passing,
                retained=retained,
                witnesses=entry,
            )
        )
    source_pins(pins)
    pins.update({str(p): digest(p) for p in (a.core_root / "src").rglob("*.py")})
    check_pins(pins)
    import rosclaw.continual.stability as stability

    from rosclaw_soccer.rsi.acquired_motor_retention import acquired_motor_retention
    from rosclaw_soccer.rsi.pass_fresh_curriculum import PROTOCOL, pass_fresh_courses
    from rosclaw_soccer.sim.contracts import hash_json

    if (
        Path(stability.__file__).resolve()
        != (a.core_root / "src/rosclaw/continual/stability.py").resolve()
    ):
        raise ValueError("retention must use the exact pinned Core contract")
    scenes = {h for p, h in pins.items() if p.endswith("/cached-standard/declared-scene.xml")}
    if len(scenes) != 1:
        raise ValueError("one unchanged body/ball scene required across all motor histories")
    ids = [
        hash_json(
            dict(
                scene_hash=next(iter(scenes)),
                context=c,
                profile="native_reference_motion.contact10N.post1p2s.v1",
                recovery_controls=100,
            )
        )
        for c in old_scope["contexts"]
    ]
    reports = []
    eligible = []
    for c in candidates:
        retention = acquired_motor_retention(
            context_hashes=ids,
            original_protected=old["old_protected_cases"],
            historical=[(manifest["baseline_own"]["hash"], old["results"])],
            current=c["retained"]["results"],
        )
        summary = c["native"]["summary"]
        allowed = (
            retention["acquired_frontier_retained"]
            and summary["candidate_controlled"] > summary["baseline_controlled"]
        )
        reports.append(
            dict(
                seed=c["seed"],
                retention=retention,
                pass_summary=summary,
                eligible_for_diagnostic_Fresh=allowed,
                accepted_as_improvement=False,
                witnesses=c["witnesses"],
            )
        )
        if allowed:
            eligible.append(
                {
                    k: c[k]
                    for k in ("seed", "training_root", "training_hash", "checkpoint_hash", "kind")
                }
            )
    if not eligible:
        raise ValueError("no eligible model; preserve review evidence, do not sample Fresh")
    _, data = receipt(pins, bank)
    _, original = receipt(pins, data["base_data_hash"])
    templates = sorted(
        [r["source_course"] for r in original["records"] if r["role"] == "new_unassisted"],
        key=lambda c: (c["state_id"], c["task"]),
    )
    import numpy as np

    from rosclaw_soccer.rsi.shared_relay_setup import SharedRelaySetup
    from rosclaw_soccer.training.native_pass_courses import NativePassCourse
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
    from scripts.rsi_stroke_search_common import open_motor

    contexts = (
        {int(k): v for k, v in data["contexts"].items()}
        if isinstance(data["contexts"], dict)
        else dict(enumerate(data["contexts"]))
    )
    bench, actor = open_motor(
        pins,
        old["training_hash"],
        {i: contexts[i] for i in (1, 4, 20, 21)},
        checkpoint_kind="contact-consolidated-v2",
    )
    setup = SharedRelaySetup(pins, contexts, bench=bench, actor=actor)
    directions = {i: setup.prepare(i, -0.55)[2] for i in (1, 4, 20, 21)}
    a.output.mkdir(mode=0o700)
    write_once(
        a.output / "review.json",
        dict(
            reports=reports,
            eligible_seeds=[c["seed"] for c in eligible],
            manifest_hash=a.manifest_hash,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    seed = secrets.randbits(64)
    write_once(a.output / "private-sampling-declaration.json", dict(seed=seed, protocol=PROTOCOL))
    try:
        courses = pass_fresh_courses(templates, directions, seed)
        write_once(a.output / "private-courses.json", dict(seed=seed, courses=courses))
        typed = []
        for c in courses[::2]:
            _, world, fwd, home, _ = setup.prepare(
                c["context"],
                c["side"],
                ball_planar_offset_world=np.array(c["ball_planar_offset_world"]),
            )
            q, _ = world.layouts["a/"].canonical(world.data)
            rel = home - q[36:38]
            lat = np.array([-fwd[1], fwd[0]])
            typed.append(
                NativePassCourse(
                    c["state_id"],
                    float(rel @ fwd),
                    float(rel @ lat),
                    hash_json(dict(qpos=world.data.qpos.tolist(), qvel=world.data.qvel.tolist())),
                )
            )
            if world.native_steps:
                raise ValueError("no physics during seal allowed")
        binding = preflight_receiving_courses(tuple(typed))
        if bench.native_steps:
            raise ValueError("no teacher or candidate physics during seal")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "private-initial-states.json", [c.initial_world_state_hash for c in typed]
        )
        models = [
            dict(
                seed=65023,
                training_root=str(receipt(pins, old["training_hash"])[0].parent),
                training_hash=old["training_hash"],
                checkpoint_hash=old["checkpoint_hash"],
                kind="contact-consolidated-v2",
            )
        ] + eligible
        declaration = dict(
            protocol=PROTOCOL,
            source_head=a.expected_head,
            input_file_hashes=pins,
            manifest_hash=a.manifest_hash,
            core_head=a.core_head,
            reports=reports,
            models=models,
            data_hash=bank,
            baseline_training_hash=old["training_hash"],
            courses_hash=digest(a.output / "private-courses.json"),
            initial_states_hash=digest(a.output / "private-initial-states.json"),
            course_binding=binding,
            compiled_ball=setup.ball_dimensions,
            scene_hash=hash_json(setup.xml),
            independent_initial_states=48,
            goal_tasks=96,
            shards=4,
            physical_steps=0,
            fit_rows=0,
            weights_frozen_before_sampling=True,
            all_states_preflighted_before_workers=True,
            sampling_adapts_to_results=False,
            private_case_rows_printed=False,
            eligibility_rule="ALL_SEEDS_WITH_ZERO_CRITICAL17_LOSS_AND_DEV_NET_GAIN_NOT_BEST_SCORE",
            Fresh_screen_rule="net>=8;each_context_net>=0;state_bootstrap95_lower>0;zero_new_fall/nonfoot/B_excluded",
            existing_TRAIN_pass_regressions_reported=True,
            global_zero_forgetting_proven=False,
            legacy52_equivalence=False,
            strict_point_precision_proven=False,
            team_game_verified=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        )
        write_once(a.output / "declaration.json", declaration)
        write_once(
            a.output / "complete.json",
            dict(
                declaration,
                status="PASS_FRESH48_SEALED_AFTER_WEIGHTS_AND_OLD_RETENTION_FREEZE",
                declaration_hash=digest(a.output / "declaration.json"),
            ),
        )
        print(
            json.dumps(
                dict(
                    status="SEALED_NOT_EXECUTED",
                    initial_states=48,
                    goal_tasks=96,
                    candidate_seeds=[c["seed"] for c in eligible],
                    physical_steps=0,
                )
            ),
            flush=True,
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                physical_steps=bench.native_steps,
                fit_rows=0,
                invalid_cases_replaced=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
