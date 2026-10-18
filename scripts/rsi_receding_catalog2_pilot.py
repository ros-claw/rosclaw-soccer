"""Ten bounded SIM bodies: reproduce archived C7, then six-course catalog2 scope.

No new control law, parameter sweep, training, Fresh access or policy admission.
Reuses two original None endpoints; all four new controls and failures stay.
"""

import argparse
import gc
import hashlib
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
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.rsi.receding_scope_evidence import load_scope_evidence
from rosclaw_soccer.rsi.receding_teacher_scope import (
    SOURCE_GROUPS,
    receding_contexts,
    receding_jobs,
    reproduce_zero_control,
    scope_contract,
)
from rosclaw_soccer.rsi.reviewed_contact_world import ReviewedContactWorld
from rosclaw_soccer.rsi.sonic_encoder_reference_audit import audit_encoder_reference_consumption
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_first_touch_bundle import allocate_native_first_touch_bundle


def digest(path):
    with Path(path).open("rb") as stream:
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
        or cfg.get("groups") != list(SOURCE_GROUPS)
        or cfg.get("max_attempts") != 10
        or cfg.get("new_physical_budget") != 10
        or cfg.get("selected_option_index") != 2
        or cfg.get("neural_training_authorized") is not False
        or cfg.get("teacher_auto_admission") is not False
        or cfg.get("generated_seed_episode_binding") is not False
        or cfg.get("predecessor") is not None
        or any(
            cfg.get(k, False) is not False
            for k in (
                "use_mesh_surface_swing",
                "use_contact_exit_torque_mpc",
                "use_directional_contact_compliance",
                "suppress_postcontact_swing_release",
            )
        )
        or Path(__file__).resolve() != source / "scripts/rsi_receding_catalog2_pilot.py"
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("new frozen10-body receding catalog2 declaration required")
    p, assets, manifest, actual = prepare(args.config)
    if actual != cfg:
        raise ValueError("same prepared declaration required")
    pins = {str(args.config): digest(args.config)}
    for project in (source, core):
        for folder in (project / "src", project / "scripts"):
            pins.update({str(path): digest(path) for path in folder.rglob("*.py")})
    fit = p.sealed(Path(cfg["portfolio_fit"]))
    if fit["report_hash"] != cfg["portfolio_fit_hash"]:
        raise ValueError("original catalog required; no parameter substitution")
    catalog = [validate_option(r["candidate"]) for r in fit["options"]]
    if len(catalog) != 8 or catalog[0] is not None:
        raise ValueError("whole original catalog required")
    preflight_population([catalog[2]] * 4)  # Interface check only, not physical bodies.
    reused, archived = load_scope_evidence(cfg, p.sealed, digest, pins)
    if archived["candidate"] != catalog[2]:
        raise ValueError("same archived candidate2, not a modified faster controller required")
    nominal = p.sealed(Path(cfg["nominal_complete"]))
    reference = next(r for r in nominal["results"] if r["group"] == 4)
    review = ReviewedContactWorld(Path(cfg["nominal_readback"]), cfg["nominal_readback_file_hash"])
    binding = review.verify(cfg["mesh_task_contact_world_hash"])
    plan = PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), assets.MEAN, 18)
    contexts = receding_contexts(plan)
    by_group = {c.source_group: c for c in contexts}
    jobs, scope = receding_jobs(), scope_contract(plan)
    for item in reused:
        if (
            item["receipt"]["planning_context_hash"]
            != by_group[item["receipt"]["group"]].context_hash
        ):
            raise ValueError("old None controls must retain original exact context identity")
    for key in ("portfolio_fit", "nominal_complete", "nominal_readback"):
        pins[cfg[key]] = digest(cfg[key])
    p.stable()
    if args.preflight:
        print("RECEDING_CATALOG2_10_BODIES_6_COURSES_2_REUSED_ZERO_ALLOCATIONS", flush=True)
        return
    status = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            cfg["slot_predecessor_unit"],
            "-p",
            "MainPID",
            "-p",
            "ActiveState",
            "-p",
            "Result",
            "-p",
            "LoadState",
        ],
        text=True,
    )
    available = next(
        int(row.split()[1]) * 1024
        for row in Path("/proc/meminfo").read_text().splitlines()
        if row.startswith("MemAvailable:")
    )
    if (
        set(status.splitlines())
        != {"MainPID=0", "ActiveState=inactive", "Result=success", "LoadState=loaded"}
        or available < 56 * 1024**3
        or time.time() >= cfg["deadline_unix"]
    ):
        raise ValueError("real predecessor terminal,56GiB RAM and declared deadline required")
    root.mkdir()
    declaration = dict(
        cfg,
        schema="soccer.rsi.receding_catalog2_declaration.v1",
        jobs=jobs,
        contexts=[dict(context_hash=c.context_hash, course=list(c.course)) for c in contexts],
        scope_contract=scope,
        options=catalog,
        reused_endpoints=reused,
        archived_reference_receipt_hash=archived["receipt_hash"],
        input_file_hashes=pins,
        prepared_pins=dict(p.PINS),
        source_heads=dict(p.HEADS),
        numerical_world_review=binding,
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
        bundle = allocate_native_first_touch_bundle(assets, reference, root, record_phase=phase)
        controls = {r["pair_id"]: r["receipt"]["outcome"] for r in reused}
        evaluations, results = {}, []
        for job in jobs:
            if time.time() >= cfg["deadline_unix"]:
                raise RuntimeError(
                    "declared receding pilot deadline reached; no implicit extension"
                )
            candidate = catalog[job["option_index"]]
            context = by_group[job["group"]]
            intent = dict(
                job,
                candidate=candidate,
                context_hash=context.context_hash,
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
            )
            with np.load(folder / "physical_trace.npz", allow_pickle=False) as a:
                trace = dict(a)
            receipt.update(
                **job, intent_witness=dict(path=str(intent_path), sha256=digest(intent_path))
            )
            if attempt == 0:
                with np.load(archived["witnesses"]["trace"]["path"], allow_pickle=False) as a:
                    reproduction = reproduce_zero_control(trace, dict(a))
                if receipt["outcome"] != archived["outcome"]:
                    raise ValueError("full archived zero-layer C7 outcome must reproduce")
                reproduction.update(archived_receipt_hash=archived["receipt_hash"])
                reproduction["report_hash"] = hash_json(reproduction)
                path = root / "archived-c7-reproduction.json"
                write_once(path, reproduction)
                receipt["reproduction_witness"] = dict(path=str(path), sha256=digest(path))
            report = _sealed(folder / "report.json", physical_payload_reader=bundle.reader)
            encoder = audit_encoder_reference_consumption(
                bundle.encoder_session, trace, report["default_locomotion_planner_events"], 2
            )
            path = root / f"encoder-{attempt}.json"
            write_once(path, encoder)
            receipt["encoder_witness"] = dict(path=str(path), sha256=digest(path))
            receipt["receipt_hash"] = hash_json(receipt)
            write_once(root / f"receipt-{attempt:04d}.json", receipt)
            campaign.complete(job["job"], receipt)
            results.append(receipt)
            (controls if job["role"] == "control" else evaluations)[job["pair_id"]] = receipt[
                "outcome"
            ]
            print("RECEDING_BODY_COMMITTED", job["job"], receipt["outcome"], flush=True)
            del trace, report
            gc.collect()
        p.stable()
        if any(digest(path) != h for path, h in pins.items()):
            raise ValueError("original frozen source/input bytes changed")
        complete = dict(
            schema="soccer.rsi.receding_catalog2_complete.v1",
            declaration_hash=declaration["report_hash"],
            results=results,
            reused_endpoints=reused,
            paired_development=assess_pairs(controls, evaluations),
            new_native_bodies=10,
            native_substeps=30000,
            independent_cpu_substeps=30000,
            independent_course_count=6,
            paired_endpoint_count=12,
            new_optimizer_updates=0,
            archived_c7_full_numeric_reproduction=True,
            teacher_qualified=False,
            independent_terminal_readback_required=True,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        complete["report_hash"] = hash_json(complete)
        write_once(root / "complete.json", complete)
        phase("RECEDING_CATALOG2_COMPLETE_NOT_QUALIFIED_TEACHER")
    except BaseException as error:
        campaign.state["fatal_error"] = dict(type=type(error).__name__, message=str(error))
        phase("FATAL_CONTRACT_VIOLATION")
        raise


if __name__ == "__main__":
    main()
