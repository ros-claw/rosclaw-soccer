"""Six fixed SIM trials combining catalog2 approach with existing material-point IK.

No new parameters, algorithm, NN training, exam changes or teacher admission.
All six original catalog2 and None controls, including failures, are reused.
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
from rosclaw_soccer.rsi.public_course_cluster_accounting import cluster_contract
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.rsi.receding_teacher_scope import SOURCE_GROUPS, receding_contexts
from rosclaw_soccer.rsi.reviewed_contact_world import ReviewedContactWorld
from rosclaw_soccer.rsi.sonic_encoder_reference_audit import audit_encoder_reference_consumption
from rosclaw_soccer.rsi.surface_receding_evidence import load_baselines, surface_receding_jobs
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_first_touch_bundle import allocate_native_first_touch_bundle
from scripts.rsi_stage_release_contact_replay import digest, require_terminal
from scripts.rsi_swing_release_ablation_pilot import require_original_prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    source, core, root = (Path(cfg[k]) for k in ("source_root", "core_root", "output_root"))
    if (
        root.exists()
        or cfg.get("groups") != list(SOURCE_GROUPS)
        or type(cfg.get("max_attempts")) is not int
        or cfg["max_attempts"] != 6
        or type(cfg.get("new_physical_budget")) is not int
        or cfg["new_physical_budget"] != 6
        or cfg.get("selected_option_index") != 2
        or cfg.get("surface_profile") != "receding-six"
        or cfg.get("use_mesh_surface_swing") is not True
        or any(
            cfg.get(k) is not False
            for k in (
                "neural_training_authorized",
                "teacher_auto_admission",
                "generated_seed_episode_binding",
                "suppress_postcontact_swing_release",
            )
        )
        or any(
            cfg.get(k, False) is not False
            for k in ("use_contact_exit_torque_mpc", "use_directional_contact_compliance")
        )
        or cfg.get("predecessor") is not None
        or cfg.get("upstream_terminal_audit_required") is not True
        or set(cfg.get("required_receipt_witnesses", []))
        != {"intent_witness", "encoder_witness", "prefix_witness"}
        or Path(__file__).resolve() != source / "scripts/rsi_surface_receding_pilot.py"
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("new frozen six-body material-point/candidate2 ablation required")
    p, assets, manifest, actual = prepare(args.config)
    if actual != cfg:
        raise ValueError("actual preparation must match original configuration")
    pins = {str(args.config): digest(args.config)}
    for project in (source, core):
        for folder in (project / "src", project / "scripts"):
            pins.update({str(path): digest(path) for path in folder.rglob("*.py")})
    baselines, original_none = load_baselines(cfg, source, p.sealed, digest, pins)
    fit = p.sealed(Path(cfg["portfolio_fit"]))
    if fit["report_hash"] != cfg["portfolio_fit_hash"]:
        raise ValueError("unchanged original catalog required")
    catalog = [validate_option(row["candidate"]) for row in fit["options"]]
    if (
        len(catalog) != 8
        or catalog[0] is not None
        or any(r["candidate"] != catalog[2] for r in baselines.values())
    ):
        raise ValueError("unchanged actual catalog2 in every original baseline required")
    preflight_population([catalog[2]] * 4)
    plan = PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), assets.MEAN, 18)
    contexts = receding_contexts(plan)
    by_group = {c.source_group: c for c in contexts}
    jobs = surface_receding_jobs()
    review = ReviewedContactWorld(Path(cfg["nominal_readback"]), cfg["nominal_readback_file_hash"])
    binding = review.verify(cfg["mesh_task_contact_world_hash"])
    nominal = p.sealed(Path(cfg["nominal_complete"]))
    reused = [
        dict(role="control", pair_id=j["pair_id"], receipt=baselines[j["pair_id"]]) for j in jobs
    ]
    none_reused = [
        dict(role="control", pair_id=j["pair_id"], receipt=original_none[j["pair_id"]])
        for j in jobs
    ]
    p.stable()
    if args.preflight:
        print("SURFACE_RECEDING_SIX_12_REUSED_ENDPOINTS_ZERO_MODEL_ALLOCATIONS", flush=True)
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
        schema="soccer.rsi.mesh_surface_receding_pilot_declaration.v1",
        jobs=jobs,
        contexts=[dict(context_hash=c.context_hash, course=list(c.course)) for c in contexts],
        options=catalog,
        reused_endpoints=reused,
        reused_original_none_endpoints=none_reused,
        course_cluster_contract=cluster_contract(plan, contexts),
        input_file_hashes=pins,
        prepared_pins=dict(p.PINS),
        source_heads=dict(p.HEADS),
        numerical_world_review=binding,
        distinct_derived_course_count=6,
        source_scene_clusters=3,
        prior_disabled_noninterference_reused_not_rerun=True,
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
        results, evaluations = [], {}
        for job in jobs:
            if time.time() >= cfg["deadline_unix"]:
                raise RuntimeError("fixed six-case deadline reached; no implicit extension")
            baseline = baselines[job["pair_id"]]
            context = by_group[job["group"]]
            intent = dict(
                job,
                candidate=catalog[2],
                context_hash=context.context_hash,
                baseline_receipt_hash=baseline["receipt_hash"],
                declaration_hash=declaration["report_hash"],
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
                catalog[2],
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
                use_mesh_surface_swing=True,
            )
            with np.load(folder / "physical_trace.npz", allow_pickle=False) as a:
                trace = dict(a)
            active = np.flatnonzero(trace["mesh_surface_swing_record"][:, 0, 2] == 1)
            cutoff = int(active[0]) if len(active) else 300
            with np.load(baseline["witnesses"]["trace"]["path"], allow_pickle=False) as a:
                old_trace = dict(a)
            if set(trace) - set(old_trace) != {"mesh_surface_swing_record"}:
                raise ValueError("only the declared material-point record may be added")
            require_original_prefix({k: trace[k] for k in old_trace}, old_trace, cutoff)
            prefix = dict(
                schema="soccer.rsi.surface_pre_activation_prefix_parity.v1",
                baseline_receipt_hash=baseline["receipt_hash"],
                prefix_frames=cutoff,
                prefix_array_hashes={
                    k: hash_bytes(trace[k][:cutoff].tobytes()) for k in sorted(old_trace)
                },
                all_original_array_prefixes_bit_equal=True,
                precontact_is_not_postcontact_success=True,
                hardware_authorized=False,
                promotion_authorized=False,
            )
            prefix["report_hash"] = hash_json(prefix)
            prefix_path = root / f"prefix-{attempt}.json"
            write_once(prefix_path, prefix)
            report = _sealed(folder / "report.json", physical_payload_reader=bundle.reader)
            encoder = audit_encoder_reference_consumption(
                bundle.encoder_session, trace, report["default_locomotion_planner_events"], 2
            )
            encoder_path = root / f"encoder-{attempt}.json"
            write_once(encoder_path, encoder)
            receipt.update(
                **job,
                intent_witness=dict(path=str(intent_path), sha256=digest(intent_path)),
                prefix_witness=dict(path=str(prefix_path), sha256=digest(prefix_path)),
                encoder_witness=dict(path=str(encoder_path), sha256=digest(encoder_path)),
            )
            receipt["receipt_hash"] = hash_json(receipt)
            write_once(root / f"receipt-{attempt:04d}.json", receipt)
            campaign.complete(job["job"], receipt)
            results.append(receipt)
            evaluations[job["pair_id"]] = receipt["outcome"]
            print("SURFACE_RECEDING_BODY_COMMITTED", job["job"], receipt["outcome"], flush=True)
            del trace, old_trace, report
            gc.collect()
        p.stable()
        if any(digest(path) != h for path, h in pins.items()):
            raise ValueError("whole frozen source/upstream bytes changed")
        complete = dict(
            schema="soccer.rsi.mesh_surface_receding_pilot_complete.v1",
            declaration_hash=declaration["report_hash"],
            results=results,
            reused_endpoints=reused,
            reused_original_none_endpoints=none_reused,
            paired_development=assess_pairs(
                {k: r["outcome"] for k, r in baselines.items()}, evaluations
            ),
            original_none_context_retention=assess_pairs(
                {k: r["outcome"] for k, r in original_none.items()}, evaluations
            ),
            new_native_bodies=6,
            native_substeps=18000,
            independent_cpu_substeps=18000,
            distinct_derived_course_count=6,
            source_scene_clusters=3,
            paired_endpoint_count=12,
            new_optimizer_updates=0,
            prior_disabled_noninterference_reused_not_rerun=True,
            independent_terminal_readback_required=True,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        complete["report_hash"] = hash_json(complete)
        write_once(root / "complete.json", complete)
        phase("MESH_SURFACE_RECEDING_COMPLETE_NOT_QUALIFIED_TEACHER")
    except BaseException as error:
        campaign.state["fatal_error"] = dict(type=type(error).__name__, message=str(error))
        phase("FATAL_CONTRACT_VIOLATION")
        raise


if __name__ == "__main__":
    main()
