"""Bounded full-loop SIM controls with complete original endpoint comparisons.

Four-case release/MPC/compliance studies retain first-contact prefixes. The
separate seven-body surface study verifies disabled full parity first, then
six precontact controls with parity before their first activation. No learning,
trajectory repair, parameter sweep or automatic admission. All failures stay.
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
        raise ValueError("all original contexts required before selecting the fixed four courses")
    return [
        dict(
            job=f"release-zero-{pair}",
            role="ablation",
            pair_id=pair,
            context_index=context,
            group=group,
            option_index=4,
        )
        for pair, group, context in (
            ("raw-8", 8, None),
            ("raw-56", 56, None),
            ("context-0", 64, 0),
            ("context-1", 65, 1),
        )
    ]


def require_original_prefix(trace, baseline_trace, cutoff):
    """Compare every original array, including dtype and full shape, before divergence."""
    if type(cutoff) is not int or not 1 <= cutoff <= 300:
        raise ValueError("observed first-contact prefix required")
    if set(trace) != set(baseline_trace) or not trace:
        raise ValueError("all original arrays through first contact must be bit-exact")
    if any(
        trace[k].dtype != baseline_trace[k].dtype
        or trace[k].shape != baseline_trace[k].shape
        or trace[k].ndim == 0
        or trace[k].shape[0] != 300
        or trace[k][:cutoff].tobytes() != baseline_trace[k][:cutoff].tobytes()
        for k in trace
    ):
        raise ValueError("all original arrays through first contact must be bit-exact")


def declared_surface_jobs(contexts):
    original = declared_jobs(contexts)
    rows = [dict(j, job="surface-" + j["pair_id"], role="teacher") for j in original]
    rows += [
        dict(
            job=f"surface-raw-{g}",
            role="teacher",
            pair_id=f"raw-{g}",
            context_index=None,
            group=g,
            option_index=4,
        )
        for g in (4, 32)
    ]
    return [dict(rows[0], job="surface-off-raw-8", role="noninterference"), *rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    torque_mpc = cfg.get("use_contact_exit_torque_mpc", False)
    directional = cfg.get("use_directional_contact_compliance", False)
    surface = cfg.get("use_mesh_surface_swing", False)
    if (
        type(torque_mpc) is not bool
        or type(directional) is not bool
        or type(surface) is not bool
        or sum((torque_mpc, directional, surface)) > 1
    ):
        raise ValueError("explicit boolean diagnostic selection required")
    study_kind = (
        "mesh_surface_swing"
        if surface
        else "directional_contact_compliance"
        if directional
        else "contact_exit_torque_mpc"
        if torque_mpc
        else "swing_release_ablation"
    )
    source, core, root = (Path(cfg[k]) for k in ("source_root", "core_root", "output_root"))
    if (
        root.exists()
        or cfg["max_attempts"] != (7 if surface else 4)
        or cfg["new_physical_budget"] != (7 if surface else 4)
        or cfg["neural_training_authorized"] is not False
        or cfg["selected_option_index"] != 4
        or cfg.get("generated_seed_episode_binding") is not False
        or cfg.get("predecessor") is not None
        or cfg["upstream_terminal_audit_required"] is not True
        or cfg["teacher_auto_admission"] is not False
        or cfg.get("suppress_postcontact_swing_release")
        is not (not (torque_mpc or directional or surface))
        or set(cfg.get("required_receipt_witnesses", []))
        != {"intent_witness", "encoder_witness", "prefix_witness"}
        or Path(__file__).resolve() != source / "scripts/rsi_swing_release_ablation_pilot.py"
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("new frozen four-body zero-learning option4 screen required")
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
    from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
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
    if surface:
        jobs = declared_surface_jobs(contexts)
    if torque_mpc:
        if cfg.get("maximum_forecast_steps_per_episode") != 6000:
            raise ValueError("fixed bounded contact-exit planning budget required")
        jobs = [dict(job, job="contact-exit-" + job["pair_id"], role="teacher") for job in jobs]
    if directional:
        if cfg.get("maximum_forecast_steps_per_episode") != 700:
            raise ValueError("fixed bounded directional gain forecast budget required")
        jobs = [dict(job, job="directional-exit-" + job["pair_id"], role="teacher") for job in jobs]
    references = {r["group"]: r for r in nominal["results"]}
    controls = {}
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

    for receipt in references.values():
        if receipt["candidate"] is not None:
            raise ValueError("original numerical anchors must remain None")
        authenticate_receipt(receipt)
    baselines = {}
    if len(cfg["baseline_receipts"]) != (6 if surface else 4):
        raise ValueError("all four predeclared passing and unsafe baseline endpoints required")
    for item, job in zip(cfg["baseline_receipts"], jobs[1:] if surface else jobs, strict=True):
        path = Path(item["path"])
        receipt = json.loads(path.read_text())
        context = None if job["context_index"] is None else contexts[job["context_index"]]
        expected_job = (
            f"option-4-raw-{job['group']}"
            if job["group"] in (8, 56)
            else f"teacher-raw-{job['group']}"
            if context is None
            else f"teacher-context-{job['context_index']}"
        )
        if (
            item["pair_id"] != job["pair_id"]
            or digest(path) != item["file_hash"]
            or receipt["receipt_hash"] != item["receipt_hash"]
            or receipt["candidate"] != candidate
            or receipt["group"] != job["group"]
            or receipt["job"] != expected_job
            or receipt.get("planning_context_hash")
            != (None if context is None else context.context_hash)
            or "swing_release_ablation" in receipt
        ):
            raise ValueError("exact original candidate4 baseline required, including its failures")
        pins[str(path)] = item["file_hash"]
        authenticate_receipt(receipt)
        with np.load(receipt["witnesses"]["trace"]["path"], allow_pickle=False) as archive:
            force = archive["force_n"][:, 0]
        first = np.flatnonzero(np.any(force > 1, axis=1))
        if not surface and (
            not len(first) or int(first[0]) != receipt["outcome"]["first_foot_contact_frame"]
        ):
            raise ValueError("baseline must have an observed foot-first contact for this ablation")
        commitment = json.loads(Path(receipt["witnesses"]["commitment"]["path"]).read_text())
        if "swing_release_ablation" in commitment:
            raise ValueError("original baseline must retain the original release")
        baselines[job["pair_id"]] = receipt
        controls[job["pair_id"]] = receipt["outcome"]
        reused.append(dict(role="control", pair_id=job["pair_id"], receipt=receipt))
    for project in cfg["baseline_sources"]:
        previous = Path(project["root"])
        if (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=previous, text=True).strip()
            != project["head"]
        ):
            raise ValueError("original baseline source head changed")
        p.HEADS[str(previous)] = project["head"]
        for folder in (previous / "src", previous / "scripts"):
            captured = {str(path): digest(path) for path in folder.rglob("*.py")}
            p.add_pins(captured)
            pins.update(captured)
    for key in ("nominal_complete", "nominal_readback", "portfolio_fit", "selection_declaration"):
        pins[cfg[key]] = digest(cfg[key])
    p.stable()
    if args.preflight:
        print(
            study_kind.upper(),
            len(jobs),
            "JOBS_ORIGINAL_BASELINES_ZERO_MODEL_ALLOCATIONS",
            flush=True,
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
        schema=f"soccer.rsi.{study_kind}_pilot_declaration.v1",
        jobs=jobs,
        contexts=[dict(context_hash=c.context_hash, course=list(c.course)) for c in contexts],
        options=catalog,
        reused_endpoints=reused,
        input_file_hashes=pins,
        prepared_pins=dict(p.PINS),
        source_heads=dict(p.HEADS),
        numerical_world_review=binding,
        new_native_bodies=len(jobs),
        independent_course_count=6 if surface else 4,
        paired_endpoint_count=12 if surface else 8,
        selection_scope=(
            "six consumed option4 endpoints plus one disabled noninterference body"
            if surface
            else "four fixed consumed candidate4 endpoints, including unsafe failure"
        ),
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
                baseline_receipt_hash=baselines[job["pair_id"]]["receipt_hash"],
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
                suppress_postcontact_swing_release=not (torque_mpc or directional or surface),
                use_contact_exit_torque_mpc=torque_mpc,
                use_directional_contact_compliance=directional,
                use_mesh_surface_swing=surface and job["role"] != "noninterference",
            )
            with np.load(folder / "physical_trace.npz", allow_pickle=False) as archive:
                trace = dict(archive)
            baseline = baselines[job["pair_id"]]
            cutoff = 300 if surface else baseline["outcome"]["first_foot_contact_frame"] + 1
            if surface:
                if job["role"] == "noninterference":
                    cutoff = 300
                    if receipt["outcome"] != baseline["outcome"]:
                        raise ValueError(
                            "disabled new source must preserve original complete outcome"
                        )
                else:
                    active = np.flatnonzero(trace["mesh_surface_swing_record"][:, 0, 2] == 1)
                    cutoff = int(active[0]) if len(active) else 300
            with np.load(baseline["witnesses"]["trace"]["path"], allow_pickle=False) as archive:
                baseline_trace = dict(archive)
                original_trace = {k: v for k, v in trace.items() if k in baseline_trace}
                if torque_mpc or directional or surface:
                    new_fields = (
                        (
                            {"mesh_surface_swing_record"}
                            if job["role"] != "noninterference"
                            else set()
                        )
                        if surface
                        else {
                            "contact_exit_mpc_requested_offset_nm",
                            "contact_exit_mpc_applied_offset_nm",
                            "contact_exit_mpc_forecast_steps",
                        }
                        if torque_mpc
                        else {
                            "directional_compliance_stiffness_nm_per_rad",
                            "directional_compliance_requested_delta_nm",
                            "directional_compliance_applied_delta_nm",
                            "directional_compliance_forecast_steps",
                        }
                    )
                    if set(trace) - set(baseline_trace) != new_fields:
                        raise ValueError("only explicitly declared teacher capture may be added")
                elif set(trace) != set(baseline_trace):
                    raise ValueError("ablation may not add capture fields")
                require_original_prefix(original_trace, baseline_trace, cutoff)
            if not surface and receipt["outcome"]["first_foot_contact_frame"] != cutoff - 1:
                raise ValueError("postcontact ablation cannot change the original first contact")
            prefix = dict(
                schema="soccer.rsi.surface_pre_activation_prefix_parity.v1"
                if surface
                else "soccer.rsi.first_contact_prefix_parity.v1",
                baseline_receipt_hash=baseline["receipt_hash"],
                prefix_frames=cutoff,
                prefix_array_hashes={
                    k: hash_bytes(trace[k][:cutoff].tobytes()) for k in sorted(original_trace)
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
                job=job["job"],
                role=job["role"],
                pair_id=job["pair_id"],
                option_index=job["option_index"],
                context_index=job["context_index"],
                intent_witness=dict(path=str(intent_path), sha256=digest(intent_path)),
                encoder_witness=dict(path=str(encoder_path), sha256=digest(encoder_path)),
                prefix_witness=dict(path=str(prefix_path), sha256=digest(prefix_path)),
            )
            receipt["receipt_hash"] = hash_json(receipt)
            write_once(root / f"receipt-{attempt:04d}.json", receipt)
            campaign.complete(job["job"], receipt)
            results.append(receipt)
            if job["role"] != "noninterference":
                (controls if job["role"] == "control" else evaluations)[job["pair_id"]] = receipt[
                    "outcome"
                ]
            print(study_kind.upper(), "BODY_COMMITTED", job["job"], receipt["outcome"], flush=True)
            del trace, report
            gc.collect()
        paired = assess_pairs(controls, evaluations)
        p.stable()
        if any(digest(name) != h for name, h in pins.items()):
            raise ValueError("frozen source/input bytes changed")
        complete = dict(
            schema=f"soccer.rsi.{study_kind}_pilot_complete.v1",
            declaration_hash=declaration["report_hash"],
            results=results,
            reused_endpoints=reused,
            paired_development=paired,
            new_native_bodies=len(jobs),
            native_substeps=3000 * len(jobs),
            independent_cpu_substeps=3000 * len(jobs),
            independent_course_count=6 if surface else 4,
            paired_endpoint_count=12 if surface else 8,
            new_optimizer_updates=0,
            teacher_qualified=False,
            independent_terminal_readback_required=True,
            upstream_terminal_audit_required=True,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        if surface:
            complete["original_none_raw_retention"] = assess_pairs(
                {f"raw-{g}": references[g]["outcome"] for g in (4, 8, 32, 56)},
                {f"raw-{g}": evaluations[f"raw-{g}"] for g in (4, 8, 32, 56)},
            )
            complete["disabled_complete_300_frame_noninterference_verified"] = True
        if torque_mpc or directional:
            complete["native_forecast_substeps"] = sum(
                row["forecast_physical_substeps"] for row in results
            )
            complete["independent_optimization_forecast_substeps"] = 0
            complete["independent_chosen_control_forecast_substeps"] = 0
            for row in results:
                audit = p.sealed(Path(row["witnesses"]["independent_review"]["path"]))
                reconstruction = audit[
                    "contact_exit_torque_mpc_review"
                    if torque_mpc
                    else "directional_contact_compliance_review"
                ]
                complete["independent_optimization_forecast_substeps"] += reconstruction[
                    "planning_counters"
                ]["integrated_forecast_steps"]
                complete["independent_chosen_control_forecast_substeps"] += reconstruction[
                    "chosen_control_integrated_forecast_steps"
                ]
        complete["report_hash"] = hash_json(complete)
        write_once(root / "complete.json", complete)
        phase(
            "MESH_SURFACE_SWING_COMPLETE_NOT_QUALIFIED_TEACHER"
            if surface
            else "DIRECTIONAL_CONTACT_COMPLIANCE_COMPLETE_NOT_QUALIFIED_TEACHER"
            if directional
            else "CONTACT_EXIT_TORQUE_MPC_COMPLETE_NOT_QUALIFIED_TEACHER"
            if torque_mpc
            else "SWING_RELEASE_ABLATION_COMPLETE_NOT_OPTIMIZED_RECOVERY"
        )
    except BaseException as error:
        campaign.state["fatal_error"] = dict(type=type(error).__name__, message=str(error))
        phase("FATAL_CONTRACT_VIOLATION")
        raise


if __name__ == "__main__":
    main()
