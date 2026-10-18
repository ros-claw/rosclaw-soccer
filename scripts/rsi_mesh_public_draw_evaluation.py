"""Four fixed historical public draws under a numerically reviewed new world.

This is a control/teacher-baseline study, NOT neural training or Fresh testing.
All original None-option failures remain outcomes; old success is not inherited.
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
    config = json.loads(args.config.read_text())
    source, core, root = (Path(config[k]) for k in ("source_root", "core_root", "output_root"))
    if (
        root.exists()
        or config["max_attempts"] != 4
        or config["new_physical_budget"] != 4
        or config["groups"] != [4, 8, 32, 56]
        or config.get("predecessor") is not None
        or config["neural_training_authorized"] is not False
        or Path(__file__).resolve() != source / "scripts/rsi_mesh_public_draw_evaluation.py"
    ):
        raise ValueError("new four-body frozen original-public-draw study required")
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != config["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("exact clean frozen source required")
    pins = {str(path): digest(path) for path in (Path(__file__), args.config)}
    for project in (source, core):
        for folder in (project / "src", project / "scripts"):
            pins.update({str(path): digest(path) for path in folder.rglob("*.py")})
    sys.path[:0] = [str(source / "src"), str(source), str(core / "src")]
    from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
    from rosclaw_soccer.rsi.first_touch_campaign import FirstTouchCampaign
    from rosclaw_soccer.rsi.first_touch_campaign_worker import prepare, qualified_episode
    from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
    from rosclaw_soccer.rsi.sonic_encoder_reference_audit import audit_encoder_reference_consumption
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_first_touch_bundle import allocate_native_first_touch_bundle

    p, assets, manifest, actual = prepare(args.config)
    if actual != config:
        raise ValueError("same frozen configuration required")
    qualified_path = Path(config["contact_world_readback"])
    qualified = p.sealed(qualified_path)
    world = config["mesh_task_contact_world_hash"]
    if (
        qualified["report_hash"] != config["contact_world_readback_hash"]
        or qualified.get("schema") != "soccer.rsi.mesh_contact_terminal_readback.v1"
        or qualified.get("whole_input_and_witness_files_verified") is not True
        or qualified.get("shared_payload_logical_seals_rebuilt") is not True
        or qualified.get("prior_independent_replays_authenticated") is not True
        or len(qualified["rows"]) != 3
        or any(row["world_hash"] != world for row in qualified["rows"])
        or world == assets.WORLD
    ):
        raise ValueError("whole numerical qualification required, not inherited task success")
    completion_path = Path(config["contact_world_complete"])
    completion = p.sealed(completion_path)
    if completion["report_hash"] != qualified["complete_hash"]:
        raise ValueError("readback must bind actual completed three-body world evidence")
    for receipt, row in zip(completion["results"], qualified["rows"], strict=True):
        if receipt["receipt_hash"] != row["receipt_hash"] or receipt["outcome"] != row["outcome"]:
            raise ValueError("all original failures and world receipts must remain bound")
        for witness in receipt["witnesses"].values():
            path = Path(witness["path"])
            if digest(path) != witness["sha256"]:
                raise ValueError("previous world witness changed")
            pins[str(path)] = witness["sha256"]
    old_path = Path(config["old_complete"])
    old_complete = p.sealed(old_path)
    if old_complete["report_hash"] != config["old_complete_hash"]:
        raise ValueError("original reference completion identity required")
    reference = old_complete["results"][0]
    if reference["job"] != "original_none":
        raise ValueError("fixed no-neural-layer reference required")
    for witness in reference["witnesses"].values():
        path = Path(witness["path"])
        if digest(path) != witness["sha256"]:
            raise ValueError("original reference witness changed")
        pins[str(path)] = witness["sha256"]
    for path in (qualified_path, completion_path, old_path):
        pins[str(path)] = digest(path)
    plan = PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), assets.MEAN, 18)
    jobs = [
        dict(
            job=f"original-public-{group}",
            group=group,
            candidate=None,
            course=list(plan.draws()[group].course),
        )
        for group in config["groups"]
    ]
    p.stable()
    if args.preflight:
        print("MESH_PUBLIC_DRAW_STATIC_PREFLIGHT_4_JOBS_ZERO_ALLOCATIONS", flush=True)
        return
    available = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available < 56 * 1024**3 or time.time() >= config["deadline_unix"]:
        raise RuntimeError("56GiB available RAM and fixed start deadline required")
    root.mkdir()
    declaration = dict(
        config,
        schema="soccer.rsi.mesh_public_draw_declaration.v1",
        input_file_hashes=pins,
        prepared_pins=p.PINS,
        source_heads=p.HEADS,
        jobs=jobs,
        neural_optimizer_updates=0,
        historical_success_inherited=False,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(root / "declaration.json", declaration)
    campaign = FirstTouchCampaign(root, declaration)

    def phase(name):
        campaign.state["phase"] = name
        campaign.record(name)
        print(name, flush=True)

    try:
        bundle = allocate_native_first_touch_bundle(assets, reference, root, record_phase=phase)
        results = []
        for job in jobs:
            if time.time() >= config["deadline_unix"]:
                raise RuntimeError("fixed new-work deadline reached; no automatic extension")
            intent = dict(job, declaration_hash=declaration["report_hash"], world_hash=world)
            intent["report_hash"] = hash_json(intent)
            intent_path = root / f"intent-{job['group']}.json"
            write_once(intent_path, intent)
            campaign.state["next_intent_hash"] = intent["report_hash"]
            campaign.record("PUBLIC_DRAW_INTENT_COMMITTED_BEFORE_BODY")
            attempt = campaign.begin(job["job"])
            folder = root / f"attempt-{attempt:04d}"
            receipt = qualified_episode(
                folder,
                job["group"],
                None,
                plan=plan,
                manifest=manifest,
                assets=assets,
                source=source,
                core=core,
                factory=bundle.factory,
                compiler=bundle.compiler,
                auditor=bundle.auditor,
                entry_source_path=Path(__file__),
                physical_payload_reader=bundle.reader,
                mesh_task_contact_world_hash=world,
            )
            import numpy as np

            with np.load(folder / "physical_trace.npz", allow_pickle=False) as archive:
                trace = dict(archive)
            report = _sealed(folder / "report.json", physical_payload_reader=bundle.reader)
            consumption = audit_encoder_reference_consumption(
                bundle.encoder_session, trace, report["default_locomotion_planner_events"], 2
            )
            encoder_path = root / f"encoder-consumption-{attempt}.json"
            write_once(encoder_path, consumption)
            receipt.update(
                job=job["job"],
                intent_witness=dict(path=str(intent_path), sha256=digest(intent_path)),
                encoder_witness=dict(path=str(encoder_path), sha256=digest(encoder_path)),
            )
            receipt["receipt_hash"] = hash_json(receipt)
            write_once(root / f"receipt-{attempt:04d}.json", receipt)
            campaign.complete(job["job"], receipt)
            results.append(receipt)
            print("MESH_PUBLIC_DRAW_COMMITTED", job["job"], receipt["outcome"], flush=True)
            del trace, report
            gc.collect()
        p.stable()
        bundle.factory._stable()
        if any(digest(Path(path)) != expected for path, expected in pins.items()):
            raise ValueError("source/input changed during bounded public evaluation")
        complete = dict(
            schema="soccer.rsi.mesh_public_draw_complete.v1",
            declaration_hash=declaration["report_hash"],
            results=results,
            new_native_bodies=4,
            native_physical_substeps=12000,
            independent_cpu_substeps=12000,
            independent_foundation_calls=1200,
            additional_encoder_reference_calls=1200,
            new_optimizer_updates=0,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        complete["report_hash"] = hash_json(complete)
        write_once(root / "complete.json", complete)
        phase("PUBLIC_DRAW_EVALUATION_COMPLETE_NO_PROMOTION")
    except BaseException as error:
        campaign.state["phase"] = "FATAL_CONTRACT_VIOLATION"
        campaign.state["fatal_error"] = dict(type=type(error).__name__, message=str(error))
        campaign.record("FATAL_NO_AUTOMATIC_RETRY")
        raise


if __name__ == "__main__":
    main()
