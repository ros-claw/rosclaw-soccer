"""Frozen 44-body public V2 torque PPO study in the reviewed mesh world.

No old-world checkpoint is retagged. All collection outcomes enter learning;
zero accepted optimizer steps is a measured training result, not a retry signal.
"""

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    source, core, root = (Path(cfg[k]) for k in ("source_root", "core_root", "output_root"))
    if (
        root.exists()
        or cfg.get("predecessor") is not None
        or cfg["max_attempts"] != 44
        or cfg["new_physical_budget"] != 44
        or cfg["neural_training_authorized"] is not True
        or cfg["joint_optimizer_steps_per_seed"] != 4
        or cfg["motor_mean_frozen"] is not True
        or Path(__file__).resolve() != source / "scripts/rsi_mesh_cerebellar_three_seed.py"
    ):
        raise ValueError("unused frozen 44-body three-seed new-world study required")
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("exact clean frozen physical and optimizer source required")
    pins = {str(p): digest(p) for p in (Path(__file__), args.config)}
    for project in (source, core):
        for folder in (project / "src", project / "scripts"):
            pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
    sys.path[:0] = [str(source / "src"), str(source), str(core / "src")]
    import numpy as np

    from rosclaw_soccer.rsi.cerebellar_actor_critic import fit_update, initialize_checkpoint
    from rosclaw_soccer.rsi.cerebellar_recovery_return import training_return
    from rosclaw_soccer.rsi.cerebellar_torque_policy import initialize
    from rosclaw_soccer.rsi.cerebellar_training_batch import build_batch
    from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
    from rosclaw_soccer.rsi.first_touch_campaign import FirstTouchCampaign
    from rosclaw_soccer.rsi.first_touch_campaign_worker import prepare, qualified_episode
    from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
    from rosclaw_soccer.rsi.mesh_cerebellar_research_plan import (
        SEEDS,
        build_jobs,
        same_input_torque_drift,
        zero_head_anchor_features,
    )
    from rosclaw_soccer.rsi.portfolio_physical_study import validate_option
    from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
    from rosclaw_soccer.rsi.public_physics_option_planning import planning_contexts
    from rosclaw_soccer.rsi.reviewed_contact_world import ReviewedContactWorld
    from rosclaw_soccer.rsi.sonic_encoder_reference_audit import audit_encoder_reference_consumption
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_first_touch_bundle import allocate_native_first_touch_bundle

    p, assets, manifest, actual = prepare(args.config)
    if actual != cfg:
        raise ValueError("same source-bound actual preparation required")
    world = cfg["mesh_task_contact_world_hash"]
    review_path = Path(cfg["nominal_readback"])
    world_review = ReviewedContactWorld(review_path, cfg["nominal_readback_file_hash"])
    binding = world_review.verify(world)
    nominal_review = p.sealed(review_path)
    nominal_path = Path(cfg["nominal_complete"])
    nominal = p.sealed(nominal_path)
    if (
        nominal_review["report_hash"] != cfg["nominal_readback_hash"]
        or nominal_review["schema"] != "soccer.rsi.mesh_original_public_terminal_readback.v1"
        or nominal_review["complete_hash"] != nominal["report_hash"]
        or len(nominal["results"]) != 4
        or [r["group"] for r in nominal["results"]] != [4, 8, 32, 56]
    ):
        raise ValueError("all four completed new-world original controls required")
    passing, baseline_references = [], {}
    for receipt, row in zip(nominal["results"], nominal_review["rows"], strict=True):
        if receipt["receipt_hash"] != row["receipt_hash"] or receipt["outcome"] != row["outcome"]:
            raise ValueError("raw nominal controls must retain every independently read label")
        for witness in receipt["witnesses"].values():
            path = Path(witness["path"])
            if digest(path) != witness["sha256"]:
                raise ValueError("original nominal physical witness changed")
            pins[str(path)] = witness["sha256"]
        if receipt["candidate"] is not None or "cerebellar_model_hash" in receipt:
            raise ValueError("actual original None-layer reference required")
        with np.load(receipt["witnesses"]["trace"]["path"], allow_pickle=False) as archive:
            trace = dict(archive)
        outcome = label_first_touch(trace)
        if outcome.to_dict() != receipt["outcome"]:
            raise ValueError("all original raw labels must agree")
        if outcome.task_pass:
            passing.append((receipt["group"], trace))
        baseline_references[f"raw-{receipt['group']}"] = receipt["witnesses"]["trace"]
    anchors = zero_head_anchor_features([t for _, t in passing])
    anchor_groups = [g for g, _ in passing]
    del passing, trace
    actors = {seed: initialize(seed=seed, world_hash=world, time_aware=True) for seed in SEEDS}
    checkpoints = {
        seed: initialize_checkpoint(actor, anchor_features=anchors)
        for seed, actor in actors.items()
    }
    plan = PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), assets.MEAN, 18)
    contexts = planning_contexts(plan)
    jobs = build_jobs(contexts, actors)
    if set(cfg["groups"]) != {j["group"] for j in jobs}:
        raise ValueError("configuration must name exactly all declared groups")
    fit_path = Path(cfg["portfolio_fit"])
    fit = p.sealed(fit_path)
    if fit["report_hash"] != cfg["portfolio_fit_hash"]:
        raise ValueError("original fixed public option catalog required, not post-hoc tuning")
    catalog = [validate_option(row["candidate"]) for row in fit["options"]]
    if len(catalog) != 8:
        raise ValueError("whole eight-option prior required")
    for path in (review_path, nominal_path, fit_path):
        pins[str(path)] = digest(path)
    p.stable()
    if args.preflight:
        print(
            "MESH_CEREBELLAR_STATIC_PREFLIGHT_44_JOBS_ZERO_FULL_MODEL_ALLOCATIONS",
            anchor_groups,
            anchors.shape,
            flush=True,
        )
        return
    available = next(
        int(s.split()[1]) * 1024
        for s in Path("/proc/meminfo").read_text().splitlines()
        if s.startswith("MemAvailable:")
    )
    if available < 56 * 1024**3 or time.time() >= cfg["deadline_unix"]:
        raise RuntimeError("56GiB available before allocation and fixed deadline required")
    root.mkdir()
    declaration = dict(
        cfg,
        schema="soccer.rsi.mesh_cerebellar_three_seed_declaration.v1",
        jobs=jobs,
        input_file_hashes=pins,
        prepared_pins=p.PINS,
        source_heads=p.HEADS,
        numerical_world_review=binding,
        anchor_original_passing_groups=anchor_groups,
        anchor_samples=len(anchors),
        anchor_samples_are_frame_starts_not_substeps=True,
        original_failed_nominal_groups_retained_in_evaluation=True,
        required_receipt_witnesses=["intent_witness", "model_witness", "encoder_witness"],
        physical_skill_improvement_claimed=False,
        teacher_qualified=False,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(root / "declaration.json", declaration)
    for seed in SEEDS:
        write_once(root / f"initial-actor-{seed}.json", actors[seed])
        write_once(root / f"initial-checkpoint-{seed}.json", checkpoints[seed])
    with (root / "anchor-features.npz").open("xb") as stream:
        np.savez_compressed(stream, features=anchors)
    campaign = FirstTouchCampaign(root, declaration)

    def phase(name):
        campaign.state["phase"] = name
        campaign.record(name)
        print(name, flush=True)

    try:
        bundle = allocate_native_first_touch_bundle(
            assets, nominal["results"][0], root, record_phase=phase
        )
        results, updates = [], []
        learned = False
        for job in jobs:
            if time.time() >= cfg["deadline_unix"]:
                raise RuntimeError("fixed new-work deadline reached; no implicit extension")
            if job["kind"] in ("evaluation", "retention") and not learned:
                if len(results) != 20:
                    raise ValueError(
                        "all eight controls and twelve actual collections before learning"
                    )
                for seed in SEEDS:
                    expected = [
                        j for j in jobs if j["kind"] == "collection" and j["actor_seed"] == seed
                    ]
                    selected = {j["job"] for j in expected}
                    paths = [
                        root / f"receipt-{i:04d}.json"
                        for i, r in enumerate(results)
                        if r["job"] in selected
                    ]
                    arrays, batch = build_batch(
                        paths,
                        actor=actors[seed],
                        expected_jobs=expected,
                        declaration_hash=declaration["report_hash"],
                        evidence_root=root,
                        physical_payload_reader=bundle.reader,
                    )
                    write_once(root / f"batch-{seed}.json", batch)
                    with (root / f"batch-{seed}.npz").open("xb") as stream:
                        np.savez_compressed(
                            stream, **{k: v for k, v in arrays.items() if isinstance(v, np.ndarray)}
                        )
                    updated = fit_update(
                        checkpoints[seed],
                        arrays,
                        batch_hash=batch["report_hash"],
                        anchor_features=anchors,
                        steps=4,
                    )
                    write_once(root / f"trained-checkpoint-{seed}.json", updated)
                    write_once(root / f"trained-actor-{seed}.json", updated["actor"])
                    drift = dict(
                        collection=same_input_torque_drift(
                            actors[seed], updated["actor"], arrays["features"]
                        ),
                        anchors=same_input_torque_drift(actors[seed], updated["actor"], anchors),
                    )
                    drift["report_hash"] = hash_json(drift)
                    write_once(root / f"same-input-torque-drift-{seed}.json", drift)
                    actors[seed] = updated["actor"]
                    updates.append(
                        dict(
                            seed=seed,
                            checkpoint_hash=updated["checkpoint_hash"],
                            receipt=updated["learning_receipt"],
                            same_input_torque_drift=drift,
                        )
                    )
                    campaign.state["learning_updates"] = updates
                    campaign.record("BOUNDED_PPO_UPDATE_COMMITTED")
                    print(
                        "MESH_CEREBELLAR_PPO_COMMITTED",
                        seed,
                        json.dumps(updated["learning_receipt"]),
                        flush=True,
                    )
                    del arrays, updated
                    gc.collect()
                learned = True
                phase("ALL_THREE_UPDATED_ACTORS_FROZEN_BEFORE_EVALUATION")
            model = actors[job["actor_seed"]]
            context = None if job["context_index"] is None else contexts[job["context_index"]]
            intent = dict(
                job,
                declaration_hash=declaration["report_hash"],
                model_hash=model["model_hash"],
                world_hash=world,
            )
            intent["report_hash"] = hash_json(intent)
            intent_path = root / f"intent-{job['job']}.json"
            write_once(intent_path, intent)
            campaign.state["next_intent_hash"] = intent["report_hash"]
            campaign.record("INTENT_PERSISTED_BEFORE_BODY")
            attempt = campaign.begin(job["job"])
            folder = root / f"attempt-{attempt:04d}"
            receipt = qualified_episode(
                folder,
                job["group"],
                catalog[job["option_index"]],
                plan=plan,
                manifest=manifest,
                assets=assets,
                source=source,
                core=core,
                factory=bundle.factory,
                compiler=bundle.compiler,
                auditor=bundle.auditor,
                physical_payload_reader=bundle.reader,
                entry_source_path=Path(__file__),
                planning_context=context,
                planning_stage="SELECTED_OPTION_EVALUATION"
                if context is not None
                else "PLANNING_SIMULATION",
                cerebellar_model=model,
                cerebellar_sampling_seed=job["sampling_seed"],
                mesh_task_contact_world_hash=world,
                mesh_world_review=world_review,
            )
            with np.load(folder / "physical_trace.npz", allow_pickle=False) as archive:
                trace = dict(archive)
            objective = training_return(trace)
            reference = (
                baseline_references.get(job["pair_id"]) if job["kind"] == "zero_control" else None
            )
            if reference is not None:
                with np.load(reference["path"], allow_pickle=False) as archive:
                    if any(
                        trace[k].dtype != archive[k].dtype
                        or trace[k].shape != archive[k].shape
                        or trace[k].tobytes() != archive[k].tobytes()
                        for k in archive.files
                    ):
                        raise ValueError(
                            "zero V2 head changed original same-world None control bytes"
                        )
                receipt["zero_all_original_common_arrays_byte_equal"] = True
            report = _sealed(folder / "report.json", physical_payload_reader=bundle.reader)
            consumption = audit_encoder_reference_consumption(
                bundle.encoder_session, trace, report["default_locomotion_planner_events"], 2
            )
            encoder_path = root / f"encoder-consumption-{attempt}.json"
            write_once(encoder_path, consumption)
            model_prefix = "initial" if job["model_source"] == "initial_checkpoint" else "trained"
            model_path = root / f"{model_prefix}-actor-{job['actor_seed']}.json"
            receipt.update(
                job=job["job"],
                kind=job["kind"],
                objective=objective,
                intent_witness=dict(path=str(intent_path), sha256=digest(intent_path)),
                encoder_witness=dict(path=str(encoder_path), sha256=digest(encoder_path)),
                model_witness=dict(path=str(model_path), sha256=digest(model_path)),
            )
            receipt["receipt_hash"] = hash_json(
                {k: v for k, v in receipt.items() if k != "receipt_hash"}
            )
            write_once(root / f"receipt-{attempt:04d}.json", receipt)
            campaign.complete(job["job"], receipt)
            results.append(receipt)
            print(
                "MESH_CEREBELLAR_PHYSICS_COMMITTED",
                job["job"],
                json.dumps(receipt["outcome"]),
                flush=True,
            )
            del trace, report
            gc.collect()
        p.stable()
        bundle.factory._stable()
        world_review.verify(world)
        if any(digest(Path(path)) != expected for path, expected in pins.items()):
            raise ValueError("source/model/input changed during experiment")
        complete = dict(
            schema="soccer.rsi.mesh_cerebellar_three_seed_complete.v1",
            declaration_hash=declaration["report_hash"],
            results=results,
            updates=updates,
            native_bodies=44,
            native_substeps=132000,
            independent_cpu_substeps=132000,
            actual_cerebellar_sampling_draws=36000,
            actual_cerebellar_inferences=132000,
            accepted_joint_optimizer_steps=sum(
                u["receipt"]["accepted_joint_optimizer_steps"] for u in updates
            ),
            physical_improvement_claimed=False,
            independent_terminal_readback_required=True,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        complete["report_hash"] = hash_json(complete)
        write_once(root / "complete.json", complete)
        campaign.state["complete_hash"] = complete["report_hash"]
        phase("ALL_DECLARED_LEARNING_AND_PHYSICS_COMPLETE_NO_PROMOTION")
    except BaseException as error:
        campaign.state["phase"] = "FATAL_CONTRACT_VIOLATION"
        campaign.state["fatal_error"] = dict(type=type(error).__name__, message=str(error))
        campaign.record("FATAL_NO_AUTOMATIC_RETRY")
        raise


if __name__ == "__main__":
    main()
