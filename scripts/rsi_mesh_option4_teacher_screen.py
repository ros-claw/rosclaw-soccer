"""Bounded 18-body expansion of two public option4 rescues, no new learning.

Selection uses a disclosed consumed catalog prefix, not blind holdout evidence.
Upstream and new terminal readbacks remain mandatory before teacher admission.
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
    with Path(path).open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def declared_jobs(contexts):
    if len(contexts) != 8 or [c.context_index for c in contexts] != list(range(8)):
        raise ValueError("all eight fixed public planning contexts required")
    return [
        dict(
            job=f"{role}-context-{c.context_index}",
            role=role,
            pair_id=f"context-{c.context_index}",
            context_index=c.context_index,
            group=c.source_group,
            option_index=option,
        )
        for c in contexts
        for role, option in (("control", 0), ("teacher", 4))
    ] + [
        dict(
            job=f"teacher-raw-{g}",
            role="teacher",
            pair_id=f"raw-{g}",
            context_index=None,
            group=g,
            option_index=4,
        )
        for g in (4, 32)
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
        or cfg["max_attempts"] != 18
        or cfg["new_physical_budget"] != 18
        or cfg["neural_training_authorized"] is not False
        or cfg["selected_option_index"] != 4
        or cfg.get("generated_seed_episode_binding") is not False
        or cfg.get("predecessor") is not None
        or cfg["upstream_terminal_audit_required"] is not True
        or cfg["teacher_auto_admission"] is not False
        or Path(__file__).resolve() != source / "scripts/rsi_mesh_option4_teacher_screen.py"
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("new frozen 18-body zero-learning option4 screen required")
    sys.path[:0] = [str(source / "src"), str(source), str(core / "src")]
    import numpy as np

    from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
    from rosclaw_soccer.rsi.first_touch_campaign import FirstTouchCampaign
    from rosclaw_soccer.rsi.first_touch_campaign_worker import (
        preflight_population,
        prepare,
        qualified_episode,
    )
    from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
    from rosclaw_soccer.rsi.first_touch_qualification import require_first_touch_qualification
    from rosclaw_soccer.rsi.paired_learning_assessment import assess_pairs
    from rosclaw_soccer.rsi.portfolio_physical_study import validate_option
    from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
    from rosclaw_soccer.rsi.public_physics_option_planning import (
        planning_contexts,
        validate_context_source,
    )
    from rosclaw_soccer.rsi.reviewed_contact_world import ReviewedContactWorld
    from rosclaw_soccer.rsi.sonic_encoder_reference_audit import audit_encoder_reference_consumption
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_first_touch_bundle import allocate_native_first_touch_bundle

    p, assets, manifest, actual = prepare(args.config)
    if actual != cfg:
        raise ValueError("actual prepared declaration differs")
    pins = {str(args.config): digest(args.config)}
    for project in (source, core):
        for folder in (project / "src", project / "scripts"):
            pins.update({str(path): digest(path) for path in folder.rglob("*.py")})
    review = ReviewedContactWorld(Path(cfg["nominal_readback"]), cfg["nominal_readback_file_hash"])
    binding = review.verify(cfg["mesh_task_contact_world_hash"])
    nominal = p.sealed(Path(cfg["nominal_complete"]))
    audited = p.sealed(Path(cfg["nominal_readback"]))
    fit = p.sealed(Path(cfg["portfolio_fit"]))
    selection_declaration = p.sealed(Path(cfg["selection_declaration"]))
    if (
        audited["report_hash"] != cfg["nominal_readback_hash"]
        or audited["complete_hash"] != nominal["report_hash"]
        or [r["group"] for r in nominal["results"]] != [4, 8, 32, 56]
        or fit["report_hash"] != cfg["portfolio_fit_hash"]
        or selection_declaration["report_hash"] != cfg["selection_declaration_hash"]
        or selection_declaration["source_head"] != cfg["selection_source_head"]
        or selection_declaration["max_attempts"] != 16
        or selection_declaration["neural_training_authorized"] is not False
    ):
        raise ValueError("exact original world, catalog and disclosed selection study required")
    catalog = [validate_option(row["candidate"]) for row in fit["options"]]
    if len(catalog) != 8 or catalog[0] is not None or selection_declaration["options"] != catalog:
        raise ValueError("unchanged original catalog and selected option4 required")
    candidate = catalog[4]
    preflight_population([candidate] * 4)  # Interface check only, not four physical bodies.
    plan = PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), assets.MEAN, 18)
    contexts = planning_contexts(plan)
    for context in contexts:
        validate_context_source(context, plan, group=context.source_group)
    jobs = declared_jobs(contexts)
    references = {r["group"]: r for r in nominal["results"]}
    controls = {f"raw-{g}": r["outcome"] for g, r in references.items()}
    evaluations, reused = {}, []

    def authenticate_receipt(receipt):
        if receipt.get("receipt_hash") != hash_json(
            {k: v for k, v in receipt.items() if k != "receipt_hash"}
        ):
            raise ValueError("whole original receipt seal differs")
        if receipt["witnesses"]["world"]["sha256"] != cfg["mesh_task_contact_world_hash"]:
            raise ValueError("same physical contact world required")
        for witness in receipt["witnesses"].values():
            if digest(witness["path"]) != witness["sha256"]:
                raise ValueError("immutable original witness bytes changed")
            pins[witness["path"]] = witness["sha256"]
        with np.load(receipt["witnesses"]["trace"]["path"], allow_pickle=False) as archive:
            outcome = label_first_touch(dict(archive))
        independent = p.sealed(Path(receipt["witnesses"]["independent_review"]["path"]))
        foundation = p.sealed(Path(receipt["witnesses"]["foundation_review"]["path"]))
        if (
            outcome.to_dict() != receipt["outcome"]
            or require_first_touch_qualification(independent, foundation, outcome)
            != receipt["qualification_alignment"]
        ):
            raise ValueError("original primitive and independent qualification labels differ")

    for g, receipt in references.items():
        if receipt["candidate"] is not None:
            raise ValueError("all four original controls must remain None")
        authenticate_receipt(receipt)
        reused.append(dict(role="control", pair_id=f"raw-{g}", receipt=receipt))
    if len(cfg["selection_receipts"]) != 2:
        raise ValueError("two fixed public-prefix selection receipts required")
    for expected, g in zip(cfg["selection_receipts"], (8, 56), strict=True):
        path = Path(expected["path"])
        receipt = json.loads(path.read_text())
        if (
            digest(path) != expected["file_hash"]
            or receipt.get("receipt_hash") != expected["receipt_hash"]
            or receipt.get("job") != f"option-4-raw-{g}"
            or receipt.get("option_index") != 4
            or receipt.get("candidate") != candidate
            or not receipt["outcome"]["task_pass"]
            or not receipt["outcome"]["safe"]
        ):
            raise ValueError("exact disclosed public selection positives required")
        pins[str(path)] = expected["file_hash"]
        authenticate_receipt(receipt)
        evaluations[f"raw-{g}"] = receipt["outcome"]
        reused.append(dict(role="teacher", pair_id=f"raw-{g}", receipt=receipt))
    for key in ("nominal_complete", "nominal_readback", "portfolio_fit", "selection_declaration"):
        pins[cfg[key]] = digest(cfg[key])
    p.stable()
    if args.preflight:
        print(
            "OPTION4_SCREEN_18_JOBS_12_PAIRED_COURSES_6_REUSED_ZERO_MODEL_ALLOCATIONS", flush=True
        )
        return
    status = dict(
        line.split("=", 1)
        for line in subprocess.check_output(
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
                "-p",
                "Result",
            ],
            text=True,
        ).splitlines()
    )
    if status != dict(LoadState="loaded", MainPID="0", ActiveState="inactive", Result="success"):
        raise ValueError("completed old study must relinquish full-model slot")
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
        schema="soccer.rsi.mesh_option4_teacher_screen_declaration.v1",
        jobs=jobs,
        contexts=[dict(context_hash=c.context_hash, course=list(c.course)) for c in contexts],
        options=catalog,
        reused_endpoints=reused,
        input_file_hashes=pins,
        prepared_pins=dict(p.PINS),
        source_heads=dict(p.HEADS),
        numerical_world_review=binding,
        new_native_bodies=18,
        independent_course_count=12,
        paired_endpoint_count=24,
        selection_scope="two consumed positive receipts from still-in-progress public catalog",
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
        results = []
        for job in jobs:
            if time.time() >= cfg["deadline_unix"]:
                raise RuntimeError("fixed screen deadline reached; no implicit extension")
            option = catalog[job["option_index"]]
            context = None if job["context_index"] is None else contexts[job["context_index"]]
            intent = dict(
                job,
                candidate=option,
                context_hash=None if context is None else context.context_hash,
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
                option,
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
                planning_stage="SELECTED_OPTION_EVALUATION" if context else "PLANNING_SIMULATION",
            )
            with np.load(folder / "physical_trace.npz", allow_pickle=False) as archive:
                trace = dict(archive)
            report = _sealed(folder / "report.json", physical_payload_reader=bundle.reader)
            encoder = audit_encoder_reference_consumption(
                bundle.encoder_session, trace, report["default_locomotion_planner_events"], 2
            )
            encoder_path = root / f"encoder-{attempt}.json"
            write_once(encoder_path, encoder)
            receipt.update(
                job=job["job"],
                role=job["role"],
                pair_id=job["pair_id"],
                option_index=job["option_index"],
                context_index=job["context_index"],
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
            print("OPTION4_SCREEN_BODY_COMMITTED", job["job"], receipt["outcome"], flush=True)
            del trace, report
            gc.collect()
        paired = assess_pairs(controls, evaluations)
        p.stable()
        if any(digest(name) != h for name, h in pins.items()):
            raise ValueError("frozen source/input bytes changed")
        complete = dict(
            schema="soccer.rsi.mesh_option4_teacher_screen_complete.v1",
            declaration_hash=declaration["report_hash"],
            results=results,
            reused_endpoints=reused,
            paired_development=paired,
            new_native_bodies=18,
            native_substeps=54000,
            independent_cpu_substeps=54000,
            independent_course_count=12,
            paired_endpoint_count=24,
            new_optimizer_updates=0,
            teacher_qualified=False,
            independent_terminal_readback_required=True,
            upstream_terminal_audit_required=True,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        complete["report_hash"] = hash_json(complete)
        write_once(root / "complete.json", complete)
        phase("TEACHER_SCREEN_COMPLETE_NOT_AUTOMATIC_ADMISSION")
    except BaseException as error:
        campaign.state["fatal_error"] = dict(type=type(error).__name__, message=str(error))
        phase("FATAL_CONTRACT_VIOLATION")
        raise


if __name__ == "__main__":
    main()
