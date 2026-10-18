"""Exhaust the existing eight controls on two original hard mesh-world draws.

G2 reachability only, not new neural learning or G3 robust teacher admission.
All sixteen results including misses and unsafe contacts remain in the bank.
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


def declared_jobs():
    return [
        dict(job=f"option-{option}-raw-{group}", option_index=option, group=group)
        for option in range(8)
        for group in (8, 56)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    source, core, root = (Path(cfg[k]) for k in ("source_root", "core_root", "output_root"))
    if (
        root.exists()
        or cfg["groups"] != [8, 56]
        or cfg["max_attempts"] != 16
        or cfg["new_physical_budget"] != 16
        or cfg["neural_training_authorized"] is not False
        or cfg.get("predecessor") is not None
        or cfg.get("generated_seed_episode_binding") is not False
        or Path(__file__).resolve() != source / "scripts/rsi_mesh_contact_catalog_reachability.py"
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("new frozen sixteen-body no-learning control catalog study required")
    sys.path[:0] = [str(source / "src"), str(source), str(core / "src")]
    import numpy as np

    from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
    from rosclaw_soccer.rsi.first_touch_campaign import FirstTouchCampaign
    from rosclaw_soccer.rsi.first_touch_campaign_worker import (
        preflight_population,
        prepare,
        qualified_episode,
    )
    from rosclaw_soccer.rsi.portfolio_physical_study import validate_option
    from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
    from rosclaw_soccer.rsi.reviewed_contact_world import ReviewedContactWorld
    from rosclaw_soccer.rsi.sonic_encoder_reference_audit import audit_encoder_reference_consumption
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_first_touch_bundle import allocate_native_first_touch_bundle

    p, assets, manifest, actual = prepare(args.config)
    if actual != cfg:
        raise ValueError("exact source-bound actual configuration required")
    pins = {str(args.config): digest(args.config)}
    for project in (source, core):
        for folder in (project / "src", project / "scripts"):
            pins.update({str(path): digest(path) for path in folder.rglob("*.py")})
    review = ReviewedContactWorld(Path(cfg["nominal_readback"]), cfg["nominal_readback_file_hash"])
    binding = review.verify(cfg["mesh_task_contact_world_hash"])
    nominal = p.sealed(Path(cfg["nominal_complete"]))
    audited = p.sealed(Path(cfg["nominal_readback"]))
    fit = p.sealed(Path(cfg["portfolio_fit"]))
    if (
        audited["report_hash"] != cfg["nominal_readback_hash"]
        or audited["complete_hash"] != nominal["report_hash"]
        or [r["group"] for r in nominal["results"]] != [4, 8, 32, 56]
        or fit["report_hash"] != cfg["portfolio_fit_hash"]
    ):
        raise ValueError("original exact world, controls and complete option catalog required")
    catalog = [validate_option(row["candidate"]) for row in fit["options"]]
    if len(catalog) != 8 or catalog[0] is not None or any(c is None for c in catalog[1:]):
        raise ValueError("original None plus seven bounded fixed options required")
    # The existing interface preflight expects a four-candidate batch. Exercise
    # all seven options; the final duplicate is numeric preflight only.
    preflight_population(catalog[1:5])
    preflight_population(catalog[5:] + [catalog[7]])
    references = {r["group"]: r for r in nominal["results"]}
    if any(references[g]["outcome"]["task_pass"] for g in cfg["groups"]):
        raise ValueError("original declared difficult draws must remain baseline failures")
    for receipt in nominal["results"]:
        if receipt["candidate"] is not None:
            raise ValueError("original None controls required")
        for witness in receipt["witnesses"].values():
            if digest(Path(witness["path"])) != witness["sha256"]:
                raise ValueError("original witness bytes changed")
            pins[witness["path"]] = witness["sha256"]
    for key in ("nominal_complete", "nominal_readback", "portfolio_fit"):
        pins[cfg[key]] = digest(Path(cfg[key]))
    p.stable()
    if args.preflight:
        print("MESH_CATALOG_16_JOBS_ALL_7_INTERFACES_ZERO_MODEL_ALLOCATIONS", flush=True)
        return
    state = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            cfg["slot_predecessor_unit"],
            "-p",
            "LoadState",
            "-p",
            "MainPID",
            "-p",
            "ActiveState",
        ],
        text=True,
    )
    status = dict(line.split("=", 1) for line in state.splitlines())
    if (
        status["LoadState"] != "loaded"
        or status["MainPID"] != "0"
        or status["ActiveState"] not in ("inactive", "failed")
    ):
        raise ValueError("declared prior worker must relinquish full-model slot")
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
        schema="soccer.rsi.mesh_contact_catalog_declaration.v1",
        jobs=declared_jobs(),
        options=catalog,
        input_file_hashes=pins,
        numerical_world_review=binding,
        independent_course_count=2,
        physical_execution_count=16,
        neural_optimizer_updates=0,
        teacher_qualified=False,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(root / "declaration.json", declaration)
    campaign = FirstTouchCampaign(root, declaration)

    def phase(name):
        campaign.state["phase"] = name
        campaign.record(name)
        print(name, flush=True)

    try:
        bundle = allocate_native_first_touch_bundle(assets, references[4], root, record_phase=phase)
        plan = PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), assets.MEAN, 18)
        results = []
        for job in declaration["jobs"]:
            if time.time() >= cfg["deadline_unix"]:
                raise RuntimeError("fixed catalog deadline reached; no implicit extension")
            candidate = catalog[job["option_index"]]
            intent = dict(job, candidate=candidate, declaration_hash=declaration["report_hash"])
            intent["report_hash"] = hash_json(intent)
            intent_path = root / f"intent-{job['job']}.json"
            write_once(intent_path, intent)
            campaign.state["next_intent_hash"] = intent["report_hash"]
            campaign.record("INTENT_COMMITTED_BEFORE_BODY")
            attempt = campaign.begin(job["job"])
            folder = root / f"attempt-{attempt:04d}"
            receipt = qualified_episode(
                folder,
                job["group"],
                candidate,
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
                mesh_task_contact_world_hash=cfg["mesh_task_contact_world_hash"],
                mesh_world_review=review,
            )
            with np.load(folder / "physical_trace.npz", allow_pickle=False) as archive:
                trace = dict(archive)
            if candidate is None:
                with np.load(
                    references[job["group"]]["witnesses"]["trace"]["path"], allow_pickle=False
                ) as archive:
                    if set(trace) != set(archive.files) or any(
                        trace[k].dtype != archive[k].dtype
                        or trace[k].shape != archive[k].shape
                        or trace[k].tobytes() != archive[k].tobytes()
                        for k in trace
                    ):
                        raise ValueError("all original None-control arrays must remain bit-exact")
            report = _sealed(folder / "report.json", physical_payload_reader=bundle.reader)
            encoder = audit_encoder_reference_consumption(
                bundle.encoder_session, trace, report["default_locomotion_planner_events"], 2
            )
            encoder_path = root / f"encoder-{attempt}.json"
            write_once(encoder_path, encoder)
            receipt.update(
                job=job["job"],
                option_index=job["option_index"],
                intent_witness=dict(path=str(intent_path), sha256=digest(intent_path)),
                encoder_witness=dict(path=str(encoder_path), sha256=digest(encoder_path)),
            )
            receipt["receipt_hash"] = hash_json(receipt)
            write_once(root / f"receipt-{attempt:04d}.json", receipt)
            campaign.complete(job["job"], receipt)
            results.append(receipt)
            print("MESH_CATALOG_BODY_COMMITTED", job["job"], receipt["outcome"], flush=True)
            del trace, report
            gc.collect()
        p.stable()
        if any(digest(Path(name)) != expected for name, expected in pins.items()):
            raise ValueError("source/input bytes changed during bounded catalog study")
        complete = dict(
            schema="soccer.rsi.mesh_contact_catalog_complete.v1",
            declaration_hash=declaration["report_hash"],
            results=results,
            passing_option_indices_by_group={
                str(g): [
                    r["option_index"]
                    for r in results
                    if r["group"] == g and r["outcome"]["task_pass"]
                ]
                for g in (8, 56)
            },
            new_native_bodies=16,
            native_substeps=48000,
            independent_cpu_substeps=48000,
            independent_course_count=2,
            new_optimizer_updates=0,
            multiple_perturbation_teacher_qualified=False,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        complete["report_hash"] = hash_json(complete)
        write_once(root / "complete.json", complete)
        phase("CATALOG_REACHABILITY_COMPLETE_NOT_ROBUST_TEACHER")
    except BaseException as error:
        campaign.state["fatal_error"] = dict(type=type(error).__name__, message=str(error))
        phase("FATAL_CONTRACT_VIOLATION")
        raise


if __name__ == "__main__":
    main()
