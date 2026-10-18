"""Four fixed paired bodies qualify optional generated binding, not a new skill.

Same source, world, controls and original reference compiler for both paths.
One original passing draw and one original failing draw, opposite path orders.
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
    cfg = json.loads(args.config.read_text())
    source, core, root = (Path(cfg[k]) for k in ("source_root", "core_root", "output_root"))
    if (
        root.exists()
        or cfg["groups"] != [4, 8]
        or cfg["new_physical_budget"] != 4
        or cfg["max_attempts"] != 4
        or cfg["neural_training_authorized"] is not False
        or cfg.get("predecessor") is not None
        or Path(__file__).resolve() != source / "scripts/rsi_qualify_generated_binding_native.py"
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("unused clean source-pinned four-body noninterference study required")
    sys.path[:0] = [str(source / "src"), str(source), str(core / "src")]
    import numpy as np

    from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
    from rosclaw_soccer.rsi.first_touch_campaign import FirstTouchCampaign
    from rosclaw_soccer.rsi.first_touch_campaign_worker import prepare, qualified_episode
    from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
    from rosclaw_soccer.rsi.reviewed_contact_world import ReviewedContactWorld
    from rosclaw_soccer.rsi.sonic_encoder_reference_audit import audit_encoder_reference_consumption
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_first_touch_bundle import allocate_native_first_touch_bundle

    p, assets, manifest, actual = prepare(args.config)
    if actual != cfg:
        raise ValueError("same actual preparation configuration required")
    pins = {str(args.config): digest(args.config)}
    for project in (source, core):
        for folder in (project / "src", project / "scripts"):
            pins.update({str(path): digest(path) for path in folder.rglob("*.py")})
    review = ReviewedContactWorld(Path(cfg["nominal_readback"]), cfg["nominal_readback_file_hash"])
    binding = review.verify(cfg["mesh_task_contact_world_hash"])
    original = p.sealed(Path(cfg["nominal_complete"]))
    readback = p.sealed(Path(cfg["nominal_readback"]))
    if (
        original["report_hash"] != readback["complete_hash"]
        or readback["report_hash"] != cfg["nominal_readback_hash"]
        or [r["group"] for r in original["results"]] != [4, 8, 32, 56]
    ):
        raise ValueError("all original four new-world controls and readback required")
    for key in ("nominal_complete", "nominal_readback"):
        pins[cfg[key]] = digest(Path(cfg[key]))
    references = {r["group"]: r for r in original["results"]}
    for group in cfg["groups"]:
        reference = references[group]
        if reference["candidate"] is not None or reference["outcome"]["task_pass"] != (group == 4):
            raise ValueError("original None passing/failing draw required")
        for witness in reference["witnesses"].values():
            if digest(Path(witness["path"])) != witness["sha256"]:
                raise ValueError("original reference witness bytes changed")
            pins[witness["path"]] = witness["sha256"]
    jobs = [
        dict(job=f"g{g}-generated-{enabled}", group=g, generated=enabled)
        for g, enabled in ((4, False), (4, True), (8, True), (8, False))
    ]
    p.stable()
    if args.preflight:
        print("GENERATED_NATIVE_FOUR_BODY_PREFLIGHT_ZERO_MODEL_ALLOCATIONS", flush=True)
        return
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
        schema="soccer.rsi.generated_binding_native_declaration.v1",
        jobs=jobs,
        input_file_hashes=pins,
        prepared_pins=dict(p.PINS),
        source_heads=dict(p.HEADS),
        numerical_world_review=binding,
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
        receipts, traces, timings = [], {}, {}
        for job in jobs:
            if time.time() >= cfg["deadline_unix"]:
                raise RuntimeError("fixed noninterference deadline, no retry or extension")
            intent = dict(job, declaration_hash=declaration["report_hash"])
            intent["report_hash"] = hash_json(intent)
            intent_path = root / f"intent-{job['job']}.json"
            write_once(intent_path, intent)
            campaign.state["next_intent_hash"] = intent["report_hash"]
            campaign.record("INTENT_COMMITTED_BEFORE_BODY")
            attempt = campaign.begin(job["job"])
            folder = root / f"attempt-{attempt:04d}"
            start = time.perf_counter()
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
                mesh_task_contact_world_hash=cfg["mesh_task_contact_world_hash"],
                mesh_world_review=review,
                generated_seed_episode_binding=job["generated"],
            )
            timings[job["job"]] = time.perf_counter() - start
            with np.load(folder / "physical_trace.npz", allow_pickle=False) as archive:
                trace = dict(archive)
            with np.load(
                references[job["group"]]["witnesses"]["trace"]["path"], allow_pickle=False
            ) as archive:
                if set(trace) != set(archive.files) or any(
                    trace[k].shape != archive[k].shape
                    or trace[k].dtype != archive[k].dtype
                    or trace[k].tobytes() != archive[k].tobytes()
                    for k in trace
                ):
                    raise ValueError(
                        "every original same-world control array must remain bit-exact"
                    )
            key = job["group"]
            if key in traces and (
                set(trace) != set(traces[key])
                or any(trace[k].tobytes() != traces[key][k].tobytes() for k in trace)
            ):
                raise ValueError("all paired old/new native arrays must be bit-exact")
            traces[key] = trace
            report = _sealed(folder / "report.json", physical_payload_reader=bundle.reader)
            encoder = audit_encoder_reference_consumption(
                bundle.encoder_session, trace, report["default_locomotion_planner_events"], 2
            )
            encoder_path = root / f"encoder-{attempt}.json"
            write_once(encoder_path, encoder)
            if receipt["outcome"] != references[key]["outcome"]:
                raise ValueError("original passing and failing labels must both remain unchanged")
            receipt.update(
                job=job["job"],
                generated_seed_episode_binding=job["generated"],
                intent_witness=dict(path=str(intent_path), sha256=digest(intent_path)),
                encoder_witness=dict(path=str(encoder_path), sha256=digest(encoder_path)),
                all_original_arrays_bit_equal=True,
                compared_arrays=len(trace),
            )
            receipt["receipt_hash"] = hash_json(receipt)
            write_once(root / f"receipt-{attempt:04d}.json", receipt)
            campaign.complete(job["job"], receipt)
            receipts.append(receipt)
            print("GENERATED_BINDING_NATIVE_COMMITTED", job["job"], flush=True)
            del report
            gc.collect()
        p.stable()
        if any(digest(Path(name)) != expected for name, expected in pins.items()):
            raise ValueError("original source/input bytes changed during qualification")
        complete = dict(
            schema="soccer.rsi.generated_binding_native_complete.v1",
            declaration_hash=declaration["report_hash"],
            results=receipts,
            qualified_episode_wall_seconds=timings,
            new_native_bodies=4,
            native_substeps=12000,
            independent_cpu_substeps=12000,
            original_passing_and_failing_controls_preserved=True,
            all_original_and_paired_native_arrays_bit_equal=True,
            new_optimizer_updates=0,
            new_football_improvement_claimed=False,
            hardware_authorized=False,
            promotion_authorized=False,
            teacher_qualified=False,
        )
        complete["report_hash"] = hash_json(complete)
        write_once(root / "complete.json", complete)
        phase("OPTIONAL_BINDING_NUMERIC_NONINTERFERENCE_COMPLETE_NOT_SKILL_GROWTH")
    except BaseException as error:
        campaign.state["fatal_error"] = dict(type=type(error).__name__, message=str(error))
        phase("FATAL_CONTRACT_VIOLATION")
        raise


if __name__ == "__main__":
    main()
