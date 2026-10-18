"""Three once-only full-CPU G1 interface probes; no fitting or teacher admission."""

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.first_touch_campaign import FirstTouchCampaign
from rosclaw_soccer.rsi.first_touch_campaign_worker import prepare, qualified_episode
from rosclaw_soccer.rsi.portfolio_physical_study import validate_option
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.rsi.receding_teacher_scope import receding_contexts
from rosclaw_soccer.rsi.reference_student_probe import (
    JOBS,
    SEED,
    measured_authority,
    probe_models,
    require_parity,
)
from rosclaw_soccer.rsi.reviewed_contact_world import ReviewedContactWorld
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_first_touch_bundle import allocate_native_first_touch_bundle
from scripts.rsi_positive_lateral_physical_pilot import whole_predecessor
from scripts.rsi_stage_release_contact_replay import digest


def require_scope(cfg):
    if (
        cfg.get("reference_probe_profile") != "three-body-interface-authority"
        or cfg.get("groups") != [76]
        or cfg.get("probe_seed") != SEED
        or cfg.get("probe_head_bias") != 0.2
        or cfg.get("probe_joint_index") != 4
        or any(
            type(cfg.get(k)) is not int or cfg[k] != n
            for k, n in (
                ("max_attempts", 3),
                ("new_physical_budget", 3),
                ("new_independent_cpu_budget", 3),
                ("combined_body_substep_budget", 18000),
                ("max_own_full_nn_workers", 2),
            )
        )
        or any(
            cfg.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "private_fresh_accessed",
                "neural_training_authorized",
                "teacher_auto_admission",
                "generated_seed_episode_binding",
                "use_mesh_surface_swing",
                "suppress_postcontact_swing_release",
                "use_contact_exit_torque_mpc",
                "use_directional_contact_compliance",
            )
        )
        or cfg.get("mode") != "SIM_ONLY"
        or cfg.get("required_receipt_witnesses") != ["intent_witness", "parity_witness"]
    ):
        raise ValueError("fixed three-body untrained interface-authority SIM protocol required")


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
        or Path(__file__).resolve() != source / "scripts/rsi_reference_student_three_pilot.py"
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("unused output and exact immutable probe source required")
    p, assets, manifest, actual = prepare(args.config)
    if actual != cfg:
        raise ValueError("same original preparation configuration required")
    pins = {str(args.config): digest(args.config)}
    for project in (source, core):
        for folder in (project / "src", project / "scripts"):
            pins.update({str(path): digest(path) for path in folder.rglob("*.py")})
    owner_path = Path(cfg["evidence_owner_configuration"])
    if digest(owner_path) != cfg["evidence_owner_configuration_file_hash"]:
        raise ValueError("exact prior evidence owner required")
    owner = json.loads(owner_path.read_text())
    pins[str(owner_path)] = digest(owner_path)
    # Authenticate historical evidence with its ORIGINAL, unchanged control
    # source. This does NOT transfer its parity to the changed student kernel.
    # New case0 must prove full physical OFF parity before either new body runs.
    prior, diagnostic = whole_predecessor(owner, Path(owner["source_root"]), p.sealed, pins)
    baseline = next(
        r["receipt"] for r in prior["reused_endpoints"] if r["pair_id"] == "receding-g76"
    )
    if (
        baseline["receipt_hash"] != cfg["baseline_receipt_hash"]
        or baseline["outcome"]["task_pass"] is not True
    ):
        raise ValueError("exact completed original C6 teacher required for default OFF parity")
    fit = p.sealed(Path(cfg["portfolio_fit"]))
    candidate = validate_option(fit["options"][2]["candidate"])
    if fit["report_hash"] != cfg["portfolio_fit_hash"] or candidate != baseline["candidate"]:
        raise ValueError("unchanged actual catalog2 required")
    plan = PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), assets.MEAN, 18)
    context = next(c for c in receding_contexts(plan) if c.source_group == 76)
    models = probe_models(cfg["mesh_task_contact_world_hash"], assets.MEAN)
    review = ReviewedContactWorld(Path(cfg["nominal_readback"]), cfg["nominal_readback_file_hash"])
    binding = review.verify(cfg["mesh_task_contact_world_hash"])
    nominal = p.sealed(Path(cfg["nominal_complete"]))
    p.stable()
    if args.preflight:
        print(
            "THREE_FIXED_REFERENCE_PROBES_ORIGINAL_C6_ZERO_PHYSICS_FOUNDATION_ALLOCATIONS",
            flush=True,
        )
        return
    available = next(
        int(s.split()[1]) * 1024
        for s in Path("/proc/meminfo").read_text().splitlines()
        if s.startswith("MemAvailable:")
    )
    if available < 56 * 1024**3 or time.time() >= cfg["deadline_unix"]:
        raise RuntimeError("56GiB before full models and fixed deadline required")
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
    own_native = [
        line.split()[0]
        for line in units.splitlines()
        if line.split() and line.split()[0].startswith("rosclaw-") and "-native-" in line.split()[0]
    ]
    if len(own_native) > 2:
        raise RuntimeError("at most two own full native NN workers; no automatic extra allocation")
    root.mkdir()
    declaration = dict(
        cfg,
        schema="soccer.rsi.reference_student_three_declaration.v1",
        jobs=list(JOBS),
        models=models,
        candidate=candidate,
        source_plan=plan.contract(),
        context_hash=context.context_hash,
        baseline_receipt=baseline,
        prior_complete_hash=prior["report_hash"],
        prior_diagnostic_hash=diagnostic["report_hash"],
        numerical_world_review=binding,
        input_file_hashes=pins,
        prepared_pins=dict(p.PINS),
        source_heads=dict(p.HEADS),
        teacher_qualified=False,
        probe_is_trained_policy=False,
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
        results, traces = [], []
        with np.load(baseline["witnesses"]["trace"]["path"], allow_pickle=False) as a:
            original = dict(a)
        for job in JOBS:
            if time.time() >= cfg["deadline_unix"]:
                raise RuntimeError("fixed three-probe deadline reached; no extension")
            model = None if job["model_index"] is None else models[job["model_index"]]
            intent = dict(
                job,
                candidate=candidate,
                context_hash=context.context_hash,
                reference_model_hash=None if model is None else model["model_hash"],
                declaration_hash=declaration["report_hash"],
            )
            intent["report_hash"] = hash_json(intent)
            path = root / f"intent-{job['job']}.json"
            write_once(path, intent)
            campaign.state["next_intent_hash"] = intent["report_hash"]
            campaign.record("INTENT_COMMITTED_BEFORE_BODY")
            attempt = campaign.begin(job["job"])
            folder = root / f"attempt-{attempt:04d}"
            receipt = qualified_episode(
                folder,
                76,
                candidate,
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
                planning_stage="SELECTED_OPTION_EVALUATION",
                mesh_task_contact_world_hash=cfg["mesh_task_contact_world_hash"],
                mesh_world_review=review,
                reference_student_model=model,
            )
            with np.load(folder / "physical_trace.npz", allow_pickle=False) as a:
                trace = dict(a)
            frames = 300 if model is None else 30
            parity = dict(
                schema="soccer.rsi.reference_probe_array_parity.v1",
                compared_frames=frames,
                baseline_receipt_hash=baseline["receipt_hash"],
                all_original_array_hashes=require_parity(trace, original, frames),
                physical_parity_here_is_not_learning=True,
                hardware_authorized=False,
                promotion_authorized=False,
            )
            parity["report_hash"] = hash_json(parity)
            parity_path = root / f"parity-{attempt}.json"
            write_once(parity_path, parity)
            receipt.update(
                **job,
                intent_witness=dict(path=str(path), sha256=digest(path)),
                parity_witness=dict(path=str(parity_path), sha256=digest(parity_path)),
            )
            receipt["receipt_hash"] = hash_json(receipt)
            write_once(root / f"receipt-{attempt:04d}.json", receipt)
            campaign.complete(job["job"], receipt)
            results.append(receipt)
            traces.append(trace)
            print("REFERENCE_PROBE_BODY_COMMITTED", job["job"], receipt["outcome"], flush=True)
            gc.collect()
        authority = measured_authority(traces[1], traces[2])
        p.stable()
        if any(digest(Path(path)) != h for path, h in pins.items()):
            raise ValueError("frozen source and original evidence changed during physical probe")
        complete = dict(
            schema="soccer.rsi.reference_student_three_complete.v1",
            declaration_hash=declaration["report_hash"],
            results=results,
            original_off_all33_arrays_bit_exact=True,
            observed_authority=authority,
            new_native_bodies=3,
            new_independent_cpu_bodies=3,
            native_substeps=9000,
            independent_cpu_substeps=9000,
            reference_student_native_inferences=600,
            new_optimizer_updates=0,
            distinct_derived_course_count=1,
            source_scene_clusters=1,
            independent_terminal_readback_required=True,
            probe_is_trained_policy=False,
            teacher_qualified=False,
            dataset_admission_authorized=False,
            neural_training_authorized=False,
            fresh_exam=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        complete["report_hash"] = hash_json(complete)
        write_once(root / "complete.json", complete)
        phase("REFERENCE_STUDENT_PROBE_COMPLETE_NOT_TRAINED_MODEL")
    except BaseException as error:
        campaign.state["fatal_error"] = dict(type=type(error).__name__, message=str(error))
        phase("FATAL_CONTRACT_VIOLATION")
        raise


if __name__ == "__main__":
    main()
