"""Three frozen public G1 contact-world diagnostics, no optimization/promotion."""

import argparse
import gc
import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
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
        or config["max_attempts"] != 3
        or config["new_physical_budget"] != 3
        or config.get("predecessor") is not None
        or config["neural_training_authorized"] is not False
        or Path(__file__).resolve() != source / "scripts/rsi_mesh_contact_native_pilot.py"
    ):
        raise ValueError("unused three-body new-world diagnostic source/config required")
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != config["source_head"]
    ):
        raise ValueError("exact frozen source head required")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip():
        raise ValueError("clean frozen source required")
    pins = {str(p): digest(p) for p in (Path(__file__), args.config)}
    for project in (source, core):
        for folder in (project / "src", project / "scripts"):
            pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
    sys.path[:0] = [str(source / "src"), str(core / "src"), str(source)]
    from rosclaw_soccer.rsi.first_touch_campaign import FirstTouchCampaign
    from rosclaw_soccer.rsi.first_touch_campaign_worker import prepare, qualified_episode
    from rosclaw_soccer.rsi.portfolio_physical_study import validate_option
    from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
    from rosclaw_soccer.rsi.public_physics_option_planning import planning_contexts
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once

    p, assets, manifest, actual = prepare(args.config)
    if actual != config:
        raise ValueError("same pinned configuration required")
    old_path = Path(config["old_complete"])
    old_complete = p.sealed(old_path)
    if old_complete["report_hash"] != config["old_complete_hash"]:
        raise ValueError("exact historical three-body completion required")
    old = old_complete["results"][0]
    if old["job"] != "original_none":
        raise ValueError("complete old no-neural-layer reference required")
    fit_path = Path(config["portfolio_fit"])
    fit = p.sealed(fit_path)
    if fit["report_hash"] != config["portfolio_fit_hash"]:
        raise ValueError("same original frozen candidate catalog required")
    catalog = tuple(validate_option(row["candidate"]) for row in fit["options"])
    for path in (old_path, fit_path):
        pins[str(path)] = digest(path)
    for witness in old["witnesses"].values():
        path = Path(witness["path"])
        if digest(path) != witness["sha256"]:
            raise ValueError("historical physical witness changed")
        pins[str(path)] = witness["sha256"]
    plan = PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), assets.MEAN, 18)
    old_context = planning_contexts(plan)[0]
    new_contexts = planning_contexts(plan, base_groups=(96, 100, 104, 108))
    jobs = [
        ("old64-option3", old_context, 3),
        ("new96-option3", new_contexts[0], 3),
        ("unsafe104-option7", new_contexts[4], 7),
    ]
    if [context.source_group for _, context, _ in jobs] != [64, 96, 104]:
        raise ValueError("fixed historical, new and unsafe diagnostic contexts required")
    expected = "sha256:df743dfc121a2649bc9159fa38cf2b6a1c7d618113484b882bce25cbd3e1601e"
    if config["mesh_task_contact_world_hash"] != expected or expected == assets.WORLD:
        raise ValueError("distinct preconstructed task contact world required")
    p.stable()
    if args.preflight:
        print("MESH_TASK_CONTACT_STATIC_PREFLIGHT_3_JOBS_ZERO_ALLOCATIONS", flush=True)
        return
    available = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available < 56 * 1024**3 or time.time() >= config["deadline_unix"]:
        raise RuntimeError("56GiB RAM before full-model allocation and fixed deadline required")
    root.mkdir()
    declaration = dict(
        config,
        schema="soccer.rsi.mesh_contact_three_body_declaration.v1",
        input_file_hashes=pins,
        prepared_pins=p.PINS,
        source_heads=p.HEADS,
        jobs=[
            dict(job=name, context=asdict(context), option_index=index, candidate=catalog[index])
            for name, context, index in jobs
        ],
        old_world_qualification_transferred=False,
        new_optimizer_updates=0,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(root / "declaration.json", declaration)
    campaign = FirstTouchCampaign(root, declaration)
    campaign.state["phase"] = "NEW_CONTACT_WORLD_MODEL_INITIALIZATION"
    campaign.record("NEW_WORLD_DECLARED_BEFORE_ALLOCATION")
    try:
        import numpy as np
        import onnxruntime as ort
        import torch

        from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
        from rosclaw_soccer.rsi.fixed_physical_payload_reader import FixedPhysicalPayloadReader
        from rosclaw_soccer.rsi.foundation_torch_capture_audit import TorchCallAuditor
        from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
        from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import (
            RecurrentSamplingEpisodeFactory,
        )
        from rosclaw_soccer.rsi.sampling_reference_audit import SamplingReferenceAuditCompiler
        from rosclaw_soccer.rsi.sonic_encoder_reference_audit import (
            audit_encoder_reference_consumption,
        )

        torch.set_num_threads(1)
        print("MESH_TASK_CONTACT_ALLOCATING_FIXED_MEAN", flush=True)
        mean = load_json_artifact(assets.EXPORT / "model-0.json.gz")
        factory = RecurrentSamplingEpisodeFactory(mean)
        shared = root / ".shared-models"
        shared.mkdir()
        payload = Path(old["witnesses"]["proof_payload"]["path"])
        shutil.copyfile(payload, shared / payload.name)
        reader = FixedPhysicalPayloadReader(
            shared / payload.name, expected_file_hash=old["witnesses"]["proof_payload"]["sha256"]
        )
        old_report = _sealed(
            Path(old["witnesses"]["report"]["path"]), physical_payload_reader=reader
        )
        if old_report["report_hash"] != old["physical_report_hash"]:
            raise ValueError("old policy identity must bind complete physical evidence")
        compiler = SamplingReferenceAuditCompiler(old_report["executed_motor_policy"])
        del mean, old_report
        gc.collect()
        auditor = TorchCallAuditor(assets.SONIC)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(assets.SONIC / "planner_sonic.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        campaign.record("MODEL_AND_INDEPENDENT_REFERENCE_READY")
        results = []
        for name, context, index in jobs:
            if time.time() >= config["deadline_unix"]:
                raise RuntimeError("fixed diagnostic deadline reached; no automatic extension")
            intent = dict(
                job=name,
                context_hash=context.context_hash,
                option_index=index,
                candidate_hash=hash_json(catalog[index]),
                world_hash=expected,
                declaration_hash=declaration["report_hash"],
            )
            intent["report_hash"] = hash_json(intent)
            intent_path = root / f"intent-{name}.json"
            write_once(intent_path, intent)
            campaign.state["next_intent_hash"] = intent["report_hash"]
            campaign.record("NEW_WORLD_INTENT_COMMITTED_BEFORE_BODY")
            attempt = campaign.begin(name)
            folder = root / f"attempt-{attempt:04d}"
            receipt = qualified_episode(
                folder,
                context.source_group,
                catalog[index],
                plan=plan,
                manifest=manifest,
                assets=assets,
                source=source,
                core=core,
                factory=factory,
                compiler=compiler,
                auditor=auditor,
                entry_source_path=Path(__file__),
                physical_payload_reader=reader,
                planning_context=context,
                planning_stage="SELECTED_OPTION_EVALUATION",
                mesh_task_contact_world_hash=expected,
            )
            with np.load(folder / "physical_trace.npz", allow_pickle=False) as archive:
                trace = dict(archive)
            report = _sealed(folder / "report.json", physical_payload_reader=reader)
            consumption = audit_encoder_reference_consumption(
                session, trace, report["default_locomotion_planner_events"], 2
            )
            encoder_path = root / f"encoder-consumption-{attempt}.json"
            write_once(encoder_path, consumption)
            receipt.update(
                job=name,
                option_index=index,
                intent_witness=dict(path=str(intent_path), sha256=digest(intent_path)),
                encoder_witness=dict(path=str(encoder_path), sha256=digest(encoder_path)),
            )
            receipt["receipt_hash"] = hash_json(receipt)
            write_once(root / f"receipt-{attempt:04d}.json", receipt)
            campaign.complete(name, receipt)
            results.append(receipt)
            print("MESH_TASK_CONTACT_PHYSICS_COMMITTED", name, receipt["outcome"], flush=True)
            del trace, report
            gc.collect()
        p.stable()
        factory._stable()
        if any(digest(Path(path)) != expected_hash for path, expected_hash in pins.items()):
            raise ValueError("source/input changed during experiment")
        complete = dict(
            schema="soccer.rsi.mesh_contact_three_body_complete.v1",
            declaration_hash=declaration["report_hash"],
            results=results,
            new_native_bodies=3,
            native_physical_substeps=9000,
            independent_cpu_substeps=9000,
            independent_foundation_calls=900,
            new_optimizer_updates=0,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        complete["report_hash"] = hash_json(complete)
        write_once(root / "complete.json", complete)
        campaign.state["phase"] = "CONTACT_WORLD_DIAGNOSTIC_COMPLETE_NO_PROMOTION"
        campaign.record("THREE_BODY_DIAGNOSTIC_COMPLETE")
    except BaseException as error:
        campaign.state["phase"] = "FATAL_CONTRACT_VIOLATION"
        campaign.state["fatal_error"] = dict(type=type(error).__name__, message=str(error))
        campaign.record("FATAL_NO_AUTOMATIC_RETRY")
        raise


if __name__ == "__main__":
    main()
