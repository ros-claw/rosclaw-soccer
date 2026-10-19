"""Thirteen finite own-body SIM episodes, not teacher-state reconstruction.

OFF must fully reproduce a prior prospective teacher before student execution.
Two raw retention courses get both zero and fitted reference models. No fitting,
private exam access, implicit retries, control-parameter changes or promotion.
"""

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from rosclaw.growth.causal_residual_memory import initial_parameters

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.first_touch_campaign import FirstTouchCampaign
from rosclaw_soccer.rsi.first_touch_campaign_worker import prepare, qualified_episode
from rosclaw_soccer.rsi.portfolio_physical_study import validate_option
from rosclaw_soccer.rsi.positive_lateral_episode_inputs import episode_inputs
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.rsi.receding_scope_evidence import authenticate_receipt
from rosclaw_soccer.rsi.reference_exam_profiles import (
    assess_reference,
    profile_info,
    reference_scope,
)
from rosclaw_soccer.rsi.reference_student_exam_scope import zero_model_like
from rosclaw_soccer.rsi.reference_student_execution import own_model
from rosclaw_soccer.rsi.reference_student_probe import require_parity
from rosclaw_soccer.rsi.reviewed_contact_world import ReviewedContactWorld
from rosclaw_soccer.rsi.sonic_encoder_reference_audit import audit_encoder_reference_consumption
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_fit_pre_parent_reference import FIT_SEED
from scripts.rsi_fit_pre_parent_reference import require_scope as require_fit_scope
from scripts.rsi_native_first_touch_bundle import allocate_native_first_touch_bundle
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def require_scope(cfg):
    info = profile_info(cfg.get("exam_profile"))
    if (
        cfg.get("mode") != "SIM_ONLY"
        or cfg.get("required_receipt_witnesses")
        != ["intent_witness", "encoder_witness", "prefix_witness"]
        or any(
            type(cfg.get(k)) is not int or cfg[k] != n
            for k, n in (
                ("max_attempts", info["bodies"]),
                ("new_physical_budget", info["bodies"]),
                ("new_independent_cpu_budget", info["bodies"]),
                ("combined_body_substep_budget", 2 * info["steps"]),
                ("new_optimizer_updates", 0),
            )
        )
        or ("matched_ablation" in cfg and "actual_winner_fit" in cfg)
        or any(
            cfg.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "private_fresh_accessed",
                "neural_training_authorized",
                "teacher_auto_admission",
                "use_mesh_surface_swing",
                "suppress_postcontact_swing_release",
                "use_contact_exit_torque_mpc",
                "use_directional_contact_compliance",
            )
        )
    ):
        raise ValueError("fixed thirteen-body own-student, retention and OFF SIM exam required")


def load_inputs(config_path, *, pulse_research=False, original_ik_research=False):
    cfg = json.loads(config_path.read_text())
    if (
        type(pulse_research) is not bool
        or type(original_ik_research) is not bool
        or (pulse_research and original_ik_research)
    ):
        raise ValueError("explicit fixed pulse research loader mode required")
    if original_ik_research:
        from rosclaw_soccer.rsi.original_ik_planner_scope import require_ik_scope

        require_ik_scope(cfg)
    elif pulse_research:
        from rosclaw_soccer.rsi.planner_navigation_pulse_scope import require_pulse_scope

        require_pulse_scope(cfg)
    else:
        require_scope(cfg)
    source, core, root = (Path(cfg[k]) for k in ("source_root", "core_root", "output_root"))
    if (
        Path(__file__).resolve() != source / "scripts/rsi_reference_student_exam.py"
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("exact immutable own-body exam source required")
    require_terminal(cfg["fit_unit"])
    p, assets, old_manifest, actual = prepare(config_path)
    if actual != cfg:
        raise ValueError("same original preparation required")
    pins = {str(config_path): digest(config_path)}
    for project in (source, core):
        for folder in (project / "src", project / "scripts"):
            pins.update({str(path): digest(path) for path in folder.rglob("*.py")})

    def sealed(path, key="report_hash"):
        path = Path(path)
        value = json.loads(path.read_text())
        if value[key] != hash_json({k: v for k, v in value.items() if k != key}):
            raise ValueError("original complete evidence seal required")
        pins[str(path)] = digest(path)
        return value

    path = Path(cfg["fit_configuration"])
    if digest(path) != cfg["fit_configuration_file_hash"]:
        raise ValueError("original fit stage configuration required")
    fc = json.loads(path.read_text())
    require_fit_scope(fc)
    pins[str(path)] = digest(path)
    fitroot = Path(fc["output_root"])
    stage, fit = (sealed(fitroot / n) for n in ("complete.json", "numeric-fit.json"))
    if (
        stage.get("schema") != "soccer.rsi.reference_fit_stage_complete.v1"
        or stage.get("status") != "NUMERIC_FIT_COMPLETE_REQUIRES_OWN_BODY_EXAM"
        or stage.get("configuration") != fc
        or stage.get("numeric_fit_hash") != fit["report_hash"]
        or stage.get("optimizer_updates") != 128
        or fit.get("schema") != "soccer.rsi.reference_numeric_imitation_fit.v1"
        or any(
            stage.get(k) is not False
            for k in (
                "own_body_student_evaluated",
                "teacher_qualified",
                "dataset_admission_authorized",
                "promotion_authorized",
                "hardware_authorized",
                "fresh_exam",
            )
        )
    ):
        raise ValueError(
            "actual completed finite numeric fit, not refusal or untrained probe, required"
        )
    for name, h in stage["input_file_hashes"].items():
        if digest(name) != h:
            raise ValueError("original training evidence or source bytes changed")
        pins[name] = h
    student = own_model(fit["numeric_model"])
    lr = fit["learning_receipt"]
    if (
        student["parent_model_hash"] != assets.MEAN
        or student["world_hash"] != cfg["mesh_task_contact_world_hash"]
        or lr["fitted_parameters_hash"] != hash_json(student["parameters"])
        or lr["parameters"] != student["parameters"]
        or lr["completed_optimizer_steps"] != 128
        or lr["config"]
        != dict(
            steps=128,
            batch_episodes=4,
            learning_rate=0.0004,
            residual_cap=0.2,
            seed=FIT_SEED,
            compute_device="cpu",
        )
        or fit["initial_parameters_hash"]
        != hash_json(initial_parameters(135, 12, hidden_dimension=64, seed=FIT_SEED))
        or fit.get("validation_excluded_from_gradient_and_normalization") is not True
        or fit.get("own_body_student_evaluated") is not False
    ):
        raise ValueError("exact fitted numeric parameters and original train-only receipt required")
    pcpath = Path(fc["positive_configuration"])
    if digest(pcpath) != fc["positive_configuration_file_hash"]:
        raise ValueError("original positive source configuration required")
    pc = json.loads(pcpath.read_text())
    pd, pos = (sealed(Path(pc["output_root"]) / n) for n in ("declaration.json", "complete.json"))
    if pos["report_hash"] != stage["positive_complete_hash"] or len(pos["results"]) != 16:
        raise ValueError("all sixteen original teacher/control successes and failures required")
    registry = sealed(pc["sampling_registry"])
    if digest(pc["sampling_registry"]) != pc["sampling_registry_file_hash"]:
        raise ValueError("unchanged original source registry required")
    positive_plan, positive_manifest, contexts = episode_inputs(
        registry,
        expected_registry_hash=pc["sampling_registry_hash"],
        behavior_model_hash=assets.MEAN,
        digest=digest,
        pins=pins,
    )
    teachers = {r["group"]: r for r in pos["results"] if r["role"] == "teacher"}
    candidate = validate_option(pd["options"][2])
    if any(r["candidate"] != candidate for r in teachers.values()):
        raise ValueError("unchanged original approach parameters required in all bodies")
    nominal = sealed(cfg["nominal_complete"])
    retained = {r["group"]: r for r in nominal["results"] if r["group"] in (4, 32)}
    if set(retained) != {4, 32} or any(
        r["outcome"]["task_pass"] is not True for r in retained.values()
    ):
        raise ValueError("both original historical successes required")
    for r in (*teachers.values(), *retained.values()):
        authenticate_receipt(
            r,
            world_hash=cfg["mesh_task_contact_world_hash"],
            sealed=sealed,
            digest=digest,
            pins=pins,
        )
    review = ReviewedContactWorld(Path(cfg["nominal_readback"]), cfg["nominal_readback_file_hash"])
    binding = review.verify(cfg["mesh_task_contact_world_hash"])
    p.stable()
    if "matched_ablation" in cfg:
        from scripts.rsi_load_reference_ablation_exam import load_ablation

        student, stage, fit = load_ablation(
            cfg,
            sealed=sealed,
            digest=digest,
            pins=pins,
            original_stage=stage,
            original_fit=fit,
        )
    elif "actual_winner_fit" in cfg:
        from scripts.rsi_load_actual_winner_exam import load_actual_winner

        student, stage, fit = load_actual_winner(cfg, sealed=sealed, digest=digest, pins=pins)
    if original_ik_research:
        from rosclaw_soccer.rsi.original_ik_planner_scope import ik_scope

        scope = ik_scope(assets.MEAN)
    elif pulse_research:
        from rosclaw_soccer.rsi.planner_navigation_pulse_scope import pulse_scope

        scope = pulse_scope(assets.MEAN, profile=cfg["exam_profile"])
    else:
        scope = reference_scope(cfg["exam_profile"], assets.MEAN)
    return SimpleNamespace(
        cfg=cfg,
        source=source,
        core=core,
        root=root,
        p=p,
        assets=assets,
        pins=pins,
        sealed=sealed,
        old_manifest=old_manifest,
        positive_manifest=positive_manifest,
        positive_plan=positive_plan,
        old_plan=PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), assets.MEAN, 18),
        contexts={c.source_group: c for c in contexts},
        teachers=teachers,
        retained=retained,
        candidate=candidate,
        review=review,
        binding=binding,
        nominal=nominal,
        models=dict(student=student, zero=zero_model_like(student, seed=FIT_SEED), off=None),
        scope=scope,
        fit_stage=stage,
        fit_report=fit,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    c = load_inputs(args.config)
    info = profile_info(c.cfg["exam_profile"])
    if c.root.exists():
        raise ValueError("unused once-only physical output required")
    if args.preflight:
        print(info["preflight"], flush=True)
        return
    available = next(
        int(s.split()[1]) * 1024
        for s in Path("/proc/meminfo").read_text().splitlines()
        if s.startswith("MemAvailable:")
    )
    if available < 56 * 1024**3 or time.time() >= c.cfg["deadline_unix"]:
        raise RuntimeError("56GiB before full models and fixed exam deadline required")
    units = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "list-units",
            "--type=service",
            "--state=active,activating",
            "--plain",
            "--no-legend",
        ],
        text=True,
    )
    if (
        sum(
            line.split()[0].startswith("rosclaw-") and "-native-" in line.split()[0]
            for line in units.splitlines()
            if line.split()
        )
        > 2
    ):
        raise RuntimeError("at most two own full native workers required")
    c.root.mkdir()
    declaration = dict(
        c.cfg,
        schema=info["declaration_schema"],
        scope=c.scope,
        models=c.models,
        candidate=c.candidate,
        fit_stage_hash=c.fit_stage["report_hash"],
        fit_report_hash=c.fit_report["report_hash"],
        reused_teachers={str(k): v for k, v in c.teachers.items()},
        reused_legacy_retention={str(k): v for k, v in c.retained.items()},
        input_file_hashes=c.pins,
        prepared_pins=dict(c.p.PINS),
        source_heads=dict(c.p.HEADS),
        numerical_world_review=c.binding,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(c.root / "declaration.json", declaration)
    campaign = FirstTouchCampaign(c.root, declaration)

    def phase(name):
        campaign.state["phase"] = name
        campaign.record(name)
        print(name, flush=True)

    try:
        bundle = allocate_native_first_touch_bundle(
            c.assets, c.nominal["results"][0], c.root, record_phase=phase
        )
        results, raw_zero = [], {}
        for job in c.scope["jobs"]:
            if time.time() >= c.cfg["deadline_unix"]:
                raise RuntimeError("fixed thirteen-body deadline reached")
            positive = job["plan"] == "positive19"
            plan, manifest = (
                (c.positive_plan, c.positive_manifest) if positive else (c.old_plan, c.old_manifest)
            )
            context = c.contexts[job["group"]] if positive else None
            model = c.models[job["role"]]
            intent = dict(
                job,
                model_hash=None if model is None else model["model_hash"],
                source_plan_hash=plan.contract()["plan_hash"],
                candidate=c.candidate,
                declaration_hash=declaration["report_hash"],
            )
            intent["report_hash"] = hash_json(intent)
            intent_path = c.root / f"intent-{job['job']}.json"
            write_once(intent_path, intent)
            campaign.state["next_intent_hash"] = intent["report_hash"]
            campaign.record("INTENT_COMMITTED_BEFORE_BODY")
            attempt = campaign.begin(job["job"])
            folder = c.root / f"attempt-{attempt:04d}"
            receipt = qualified_episode(
                folder,
                job["group"],
                c.candidate,
                plan=plan,
                manifest=manifest,
                assets=c.assets,
                source=c.source,
                core=c.core,
                factory=bundle.factory,
                compiler=bundle.compiler,
                auditor=bundle.auditor,
                entry_source_path=Path(__file__),
                physical_payload_reader=bundle.reader,
                mesh_task_contact_world_hash=c.cfg["mesh_task_contact_world_hash"],
                mesh_world_review=c.review,
                planning_context=context,
                planning_stage="SELECTED_OPTION_EVALUATION" if positive else "PLANNING_SIMULATION",
                generated_seed_episode_binding=job["use_generated_binding"],
                reference_student_model=model,
            )
            with np.load(folder / "physical_trace.npz", allow_pickle=False) as a:
                trace = dict(a)
            if positive:
                with np.load(
                    c.teachers[job["group"]]["witnesses"]["trace"]["path"], allow_pickle=False
                ) as a:
                    baseline = dict(a)
                compared = 300 if job["role"] == "off" else 30
            elif job["role"] == "zero":
                raw_zero[job["group"]] = trace
                baseline, compared = trace, 30
            else:
                baseline, compared = raw_zero[job["group"]], 30
            prefix = dict(
                schema="soccer.rsi.reference_exam_prefix_parity.v1",
                compared_frames=compared,
                array_hashes=require_parity(trace, baseline, compared),
                zero_control_self_prefix=not positive and job["role"] == "zero",
                hardware_authorized=False,
                promotion_authorized=False,
            )
            prefix["report_hash"] = hash_json(prefix)
            pp = c.root / f"prefix-{attempt}.json"
            write_once(pp, prefix)
            report = _sealed(folder / "report.json", physical_payload_reader=bundle.reader)
            encoder = audit_encoder_reference_consumption(
                bundle.encoder_session, trace, report["default_locomotion_planner_events"], 2
            )
            ep = c.root / f"encoder-{attempt}.json"
            write_once(ep, encoder)
            receipt.update(
                job,
                intent_witness=dict(path=str(intent_path), sha256=digest(intent_path)),
                prefix_witness=dict(path=str(pp), sha256=digest(pp)),
                encoder_witness=dict(path=str(ep), sha256=digest(ep)),
            )
            receipt["receipt_hash"] = hash_json(receipt)
            write_once(c.root / f"receipt-{attempt:04d}.json", receipt)
            campaign.complete(job["job"], receipt)
            results.append(receipt)
            print(
                "REFERENCE_STUDENT_OWN_BODY_COMMITTED", job["job"], receipt["outcome"], flush=True
            )
            del trace, baseline, report
            gc.collect()
        paired = assess_reference(
            c.cfg["exam_profile"],
            c.scope,
            results,
            teachers={g: r["outcome"] for g, r in c.teachers.items()},
            original_retention={g: r["outcome"] for g, r in c.retained.items()},
        )
        c.p.stable()
        if any(digest(p) != h for p, h in c.pins.items()):
            raise ValueError("frozen source or original input changed during exam")
        complete = dict(
            schema=info["complete_schema"],
            declaration_hash=declaration["report_hash"],
            results=results,
            paired_results=paired,
            new_native_bodies=info["bodies"],
            new_independent_cpu_bodies=info["bodies"],
            native_substeps=info["steps"],
            independent_cpu_substeps=info["steps"],
            new_optimizer_updates=0,
            independent_terminal_readback_required=True,
            teacher_qualified=False,
            fresh_exam=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        complete["report_hash"] = hash_json(complete)
        write_once(c.root / "complete.json", complete)
        phase("REFERENCE_STUDENT_OWN_BODY_EXAM_COMPLETE_NOT_PROMOTION")
    except BaseException as error:
        campaign.state["fatal_error"] = dict(type=type(error).__name__, message=str(error))
        phase("FATAL_CONTRACT_VIOLATION")
        raise


if __name__ == "__main__":
    main()
