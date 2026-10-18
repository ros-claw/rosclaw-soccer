"""Prospective four-source SIM replication of the unchanged catalog2 teacher.

Eight predeclared courses, paired None/catalog2 = sixteen native bodies and
sixteen independent CPU replays. No result-based selection, fitting, retry,
teacher admission, contact-model changes or private exam access.
"""

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.first_touch_campaign import FirstTouchCampaign
from rosclaw_soccer.rsi.first_touch_campaign_worker import (
    preflight_population,
    prepare,
    qualified_episode,
)
from rosclaw_soccer.rsi.paired_learning_assessment import assess_pairs
from rosclaw_soccer.rsi.portfolio_physical_study import validate_option
from rosclaw_soccer.rsi.positive_lateral_episode_inputs import episode_inputs, prospective_split
from rosclaw_soccer.rsi.receding_scope_evidence import authenticate_receipt
from rosclaw_soccer.rsi.reviewed_contact_world import ReviewedContactWorld
from rosclaw_soccer.rsi.sonic_encoder_reference_audit import audit_encoder_reference_consumption
from rosclaw_soccer.rsi.surface_receding_diagnostics import require_endpoints, research_decision
from rosclaw_soccer.rsi.surface_receding_evidence import load_baselines
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_first_touch_bundle import allocate_native_first_touch_bundle
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def require_scope(cfg):
    """Separate finite declaration; old studies are not implicitly extended."""
    if (
        cfg.get("study_profile") != "positive-lateral-sixteen"
        or cfg.get("mode") != "SIM_ONLY"
        or any(
            type(cfg.get(k)) is not int or cfg[k] != n
            for k, n in (
                ("max_attempts", 16),
                ("new_physical_budget", 16),
                ("new_independent_cpu_budget", 16),
                ("combined_body_substep_budget", 96000),
                ("selected_option_index", 2),
            )
        )
        or cfg.get("generated_seed_episode_binding") is not True
        or any(
            cfg.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "private_fresh_accessed",
                "neural_training_authorized",
                "teacher_auto_admission",
                "use_mesh_surface_swing",
                "use_contact_exit_torque_mpc",
                "use_directional_contact_compliance",
                "suppress_postcontact_swing_release",
            )
        )
        or cfg.get("predecessor") is not None
        or cfg.get("required_receipt_witnesses") != ["intent_witness", "encoder_witness"]
    ):
        raise ValueError("fixed original-body-origin catalog2/None sixteen-body SIM study required")


def whole_predecessor(cfg, source, sealed, pins):
    """Do not hide the previous full failure bank by narrowing the new domain."""
    item = cfg["surface_predecessor_review"]
    path = Path(item["configuration"])
    if digest(path) != item["configuration_file_hash"]:
        raise ValueError("exact prior six-body configuration required")
    previous = json.loads(path.read_text())
    pins[str(path)] = digest(path)
    root = Path(previous["output_root"])
    declaration, complete = (sealed(root / n) for n in ("declaration.json", "complete.json"))
    audit = sealed(Path(item["readback"]))
    diagnostic = sealed(Path(item["diagnostic_complete"]))
    paired, _ = require_endpoints(previous, declaration, complete, audit)
    if (
        complete["report_hash"] != item["complete_hash"]
        or audit["report_hash"] != item["readback_hash"]
        or diagnostic["report_hash"] != item["diagnostic_complete_hash"]
        or diagnostic.get("schema") != "soccer.rsi.surface_receding_contact_diagnostic_complete.v1"
        or diagnostic.get("upstream_readback_hash") != audit["report_hash"]
        or diagnostic.get("new_fixed_control_replay_substeps") != 18000
        or diagnostic.get("total_metrics_endpoints") != 28
        or diagnostic.get("research_decision")
        != research_decision(paired, diagnostic["scoped_contact_metrics"])
        or any(
            diagnostic.get(k) is not False
            for k in (
                "teacher_qualified",
                "dataset_admission_authorized",
                "neural_training_authorized",
                "fresh_exam",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
        or previous["mesh_task_contact_world_hash"] != cfg["mesh_task_contact_world_hash"]
        or previous["portfolio_fit_hash"] != cfg["portfolio_fit_hash"]
        or digest(Path(item["reader_source"])) != audit["reader_source_hash"]
    ):
        raise ValueError("whole preceding mechanism study and actual dense diagnosis required")
    for name, h in {**declaration["prepared_pins"], **declaration["input_file_hashes"]}.items():
        if digest(Path(name)) != h:
            raise ValueError("previous whole source/witness bytes changed")
        pins[name] = h
    pins[item["reader_source"]] = digest(Path(item["reader_source"]))
    for index, receipt in enumerate(complete["results"]):
        path = root / f"receipt-{index:04d}.json"
        if json.loads(path.read_text()) != receipt:
            raise ValueError("all six exact preceding receipts required, including failures")
        pins[str(path)] = digest(path)
        authenticate_receipt(
            receipt,
            world_hash=cfg["mesh_task_contact_world_hash"],
            sealed=sealed,
            digest=digest,
            pins=pins,
        )
        for key in ("intent_witness", "prefix_witness", "encoder_witness"):
            witness = receipt[key]
            if digest(Path(witness["path"])) != witness["sha256"]:
                raise ValueError("preceding intent, prefix or encoder witness bytes changed")
            sealed(Path(witness["path"]))
            pins[witness["path"]] = witness["sha256"]
    # All twelve original baseline endpoints, not only C6/C7, remain bound.
    load_baselines(previous, source, sealed, digest, pins)
    return complete, diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    require_scope(cfg)
    source, core, root = (Path(cfg[k]) for k in ("source_root", "core_root", "output_root"))
    if (
        root.exists()
        or Path(__file__).resolve() != source / "scripts/rsi_positive_lateral_physical_pilot.py"
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("unused output and exact frozen physical pilot source required")
    p, assets, _, actual = prepare(args.config)
    if actual != cfg:
        raise ValueError("actual preparation must match declaration")
    pins = {str(args.config): digest(args.config)}
    for project in (source, core):
        for folder in (project / "src", project / "scripts"):
            pins.update({str(path): digest(path) for path in folder.rglob("*.py")})
    registry_path = Path(cfg["sampling_registry"])
    if digest(registry_path) != cfg["sampling_registry_file_hash"]:
        raise ValueError("exact original registry file required")
    registry = p.sealed(registry_path)
    pins[str(registry_path)] = digest(registry_path)
    plan, manifest, contexts = episode_inputs(
        registry,
        expected_registry_hash=cfg["sampling_registry_hash"],
        behavior_model_hash=assets.MEAN,
        digest=digest,
        pins=pins,
    )
    prior, diagnostic = whole_predecessor(cfg, source, p.sealed, pins)
    fit = p.sealed(Path(cfg["portfolio_fit"]))
    if fit["report_hash"] != cfg["portfolio_fit_hash"]:
        raise ValueError("original unchanged catalog required")
    catalog = [validate_option(r["candidate"]) for r in fit["options"]]
    if len(catalog) != 8 or catalog[0] is not None:
        raise ValueError("exact original eight-option catalog required")
    preflight_population([catalog[2]] * 4)
    split = prospective_split(registry["scope"])
    jobs = registry["scope"]["proposed_jobs"]
    if len(jobs) != 16 or [j["group"] for j in jobs[::2]] != [c.source_group for c in contexts]:
        raise ValueError("complete fixed source-context job order required")
    by_group = {c.source_group: c for c in contexts}
    review = ReviewedContactWorld(Path(cfg["nominal_readback"]), cfg["nominal_readback_file_hash"])
    binding = review.verify(cfg["mesh_task_contact_world_hash"])
    nominal = p.sealed(Path(cfg["nominal_complete"]))
    p.stable()
    if args.preflight:
        print("SIXTEEN_FIXED_BODY_JOBS_FOUR_SOURCE_CLUSTERS_ZERO_MODEL_ALLOCATIONS", flush=True)
        return
    require_terminal(cfg["slot_predecessor_unit"])
    available = next(
        int(s.split()[1]) * 1024
        for s in Path("/proc/meminfo").read_text().splitlines()
        if s.startswith("MemAvailable:")
    )
    if available < 56 * 1024**3 or time.time() >= cfg["deadline_unix"]:
        raise RuntimeError("56GiB before full models and fixed deadline required")
    root.mkdir()
    declaration = dict(
        cfg,
        schema="soccer.rsi.positive_lateral_physical_declaration.v1",
        jobs=jobs,
        options=catalog,
        source_plan=plan.contract(),
        scope_hash=registry["scope_hash"],
        prospective_split=split,
        course_cluster_contract=registry["scope"]["cluster_accounting"],
        preceding_complete_hash=prior["report_hash"],
        preceding_diagnostic_hash=diagnostic["report_hash"],
        preceding_research_decision=diagnostic["research_decision"],
        input_file_hashes=pins,
        prepared_pins=dict(p.PINS),
        source_heads=dict(p.HEADS),
        numerical_world_review=binding,
        distinct_derived_course_count=8,
        source_scene_clusters=4,
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
        bundle = allocate_native_first_touch_bundle(
            assets, nominal["results"][0], root, record_phase=phase
        )
        results, controls, evaluations = [], {}, {}
        for job in jobs:
            if time.time() >= cfg["deadline_unix"]:
                raise RuntimeError("fixed sixteen-body deadline reached; no implicit extension")
            candidate = catalog[job["option_index"]]
            context = by_group[job["group"]]
            intent = dict(
                job,
                candidate=candidate,
                declaration_hash=declaration["report_hash"],
                source_plan_hash=plan.contract()["plan_hash"],
                prospective_split_hash=split["split_hash"],
            )
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
                planning_context=context,
                planning_stage="SELECTED_OPTION_EVALUATION",
                generated_seed_episode_binding=True,
            )
            with np.load(folder / "physical_trace.npz", allow_pickle=False) as a:
                trace = dict(a)
            report = _sealed(folder / "report.json", physical_payload_reader=bundle.reader)
            encoder = audit_encoder_reference_consumption(
                bundle.encoder_session, trace, report["default_locomotion_planner_events"], 2
            )
            encoder_path = root / f"encoder-{attempt}.json"
            write_once(encoder_path, encoder)
            receipt.update(
                **job,
                intent_witness=dict(path=str(intent_path), sha256=digest(intent_path)),
                encoder_witness=dict(path=str(encoder_path), sha256=digest(encoder_path)),
            )
            receipt["receipt_hash"] = hash_json(receipt)
            write_once(root / f"receipt-{attempt:04d}.json", receipt)
            campaign.complete(job["job"], receipt)
            results.append(receipt)
            (controls if job["role"] == "control" else evaluations)[job["pair_id"]] = receipt[
                "outcome"
            ]
            print("POSITIVE_LATERAL_BODY_COMMITTED", job["job"], receipt["outcome"], flush=True)
            del trace, report
            gc.collect()
        p.stable()
        if any(digest(Path(path)) != h for path, h in pins.items()):
            raise ValueError("whole frozen source/upstream bytes changed")
        complete = dict(
            schema="soccer.rsi.positive_lateral_physical_complete.v1",
            declaration_hash=declaration["report_hash"],
            results=results,
            paired_development=assess_pairs(controls, evaluations),
            new_native_bodies=16,
            new_independent_cpu_bodies=16,
            native_substeps=48000,
            independent_cpu_substeps=48000,
            distinct_derived_course_count=8,
            source_scene_clusters=4,
            source_plan_hash=plan.contract()["plan_hash"],
            prospective_split=split,
            old_global_failures_retained_not_relabelled=True,
            new_optimizer_updates=0,
            independent_terminal_readback_required=True,
            teacher_qualified=False,
            dataset_admission_authorized=False,
            neural_training_authorized=False,
            fresh_exam=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        complete["report_hash"] = hash_json(complete)
        write_once(root / "complete.json", complete)
        phase("POSITIVE_LATERAL_PHYSICAL_COMPLETE_NOT_QUALIFIED_TEACHER")
    except BaseException as error:
        campaign.state["fatal_error"] = dict(type=type(error).__name__, message=str(error))
        phase("FATAL_CONTRACT_VIOLATION")
        raise


if __name__ == "__main__":
    main()
