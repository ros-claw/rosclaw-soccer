"""Whole three-body reference-authority readback; zero new physics or training."""

import argparse
import json
import sqlite3
import subprocess
from pathlib import Path

import numpy as np

from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.first_touch_campaign_worker import prepare
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.first_touch_qualification import require_first_touch_qualification
from rosclaw_soccer.rsi.fixed_physical_payload_reader import FixedPhysicalPayloadReader
from rosclaw_soccer.rsi.native_first_touch_initial_state import NativeFirstTouchInitialState
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.rsi.receding_teacher_scope import receding_contexts
from rosclaw_soccer.rsi.reference_student_probe import (
    JOBS,
    measured_authority,
    probe_models,
    require_parity,
)
from rosclaw_soccer.rsi.reviewed_contact_world import ReviewedContactWorld
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_positive_lateral_physical_pilot import whole_predecessor
from scripts.rsi_reference_student_three_pilot import require_scope
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_terminal(args.unit)
    started = args.output.with_suffix(".started.json")
    if args.output.exists() or started.exists():
        raise ValueError("unused once-only whole readback output required")
    cfg = json.loads(args.config.read_text())
    require_scope(cfg)
    source, root = Path(cfg["source_root"]), Path(cfg["output_root"])
    if (
        Path(__file__).resolve() != source / "scripts/rsi_read_reference_student_three.py"
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("reader must execute exact immutable producer source")
    p, assets, _, actual = prepare(args.config)
    if actual != cfg:
        raise ValueError("same actual preparation required")
    pins = {str(args.config): digest(args.config)}

    def sealed(path, key="report_hash"):
        path = Path(path)
        value = json.loads(path.read_text())
        if value[key] != hash_json({k: v for k, v in value.items() if k != key}):
            raise ValueError("whole original evidence seal required")
        pins[str(path)] = digest(path)
        return value

    declaration, complete = (sealed(root / n) for n in ("declaration.json", "complete.json"))
    if (
        declaration.get("schema") != "soccer.rsi.reference_student_three_declaration.v1"
        or complete.get("schema") != "soccer.rsi.reference_student_three_complete.v1"
        or complete.get("declaration_hash") != declaration["report_hash"]
        or declaration.get("jobs") != list(JOBS)
        or len(complete.get("results", [])) != 3
        or any(k not in declaration or declaration[k] != v for k, v in cfg.items())
        or declaration.get("prepared_pins") != p.PINS
        or declaration.get("source_heads") != p.HEADS
        or any(
            type(complete.get(k)) is not int or complete[k] != n
            for k, n in (
                ("new_native_bodies", 3),
                ("new_independent_cpu_bodies", 3),
                ("native_substeps", 9000),
                ("independent_cpu_substeps", 9000),
                ("reference_student_native_inferences", 600),
                ("new_optimizer_updates", 0),
                ("distinct_derived_course_count", 1),
                ("source_scene_clusters", 1),
            )
        )
        or complete.get("original_off_all33_arrays_bit_exact") is not True
        or any(
            complete.get(k) is not False
            for k in (
                "probe_is_trained_policy",
                "teacher_qualified",
                "dataset_admission_authorized",
                "neural_training_authorized",
                "fresh_exam",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
    ):
        raise ValueError("whole three-body original protocol and exact budgets required")
    write_once(
        started,
        dict(
            schema="soccer.rsi.reference_probe_readback_started.v1",
            reader_source_hash=digest(Path(__file__)),
            automatic_retry_authorized=False,
            hardware_authorized=False,
        ),
    )
    for name, h in {**declaration["prepared_pins"], **declaration["input_file_hashes"]}.items():
        if digest(name) != h:
            raise ValueError("source or original input bytes changed")
        pins[name] = h
    owner_path = Path(cfg["evidence_owner_configuration"])
    if digest(owner_path) != cfg["evidence_owner_configuration_file_hash"]:
        raise ValueError("unchanged historical evidence owner required")
    owner = json.loads(owner_path.read_text())
    prior, diagnostic = whole_predecessor(owner, Path(owner["source_root"]), sealed, pins)
    binding = ReviewedContactWorld(
        Path(cfg["nominal_readback"]), cfg["nominal_readback_file_hash"]
    ).verify(cfg["mesh_task_contact_world_hash"])
    if declaration["numerical_world_review"] != binding:
        raise ValueError("actual original numerical world review must remain bound")
    baseline = next(
        r["receipt"] for r in prior["reused_endpoints"] if r["pair_id"] == "receding-g76"
    )
    if (
        baseline != declaration["baseline_receipt"]
        or baseline["receipt_hash"] != cfg["baseline_receipt_hash"]
        or declaration["prior_complete_hash"] != prior["report_hash"]
        or declaration["prior_diagnostic_hash"] != diagnostic["report_hash"]
    ):
        raise ValueError("exact complete prior study and original C6 control required")
    models = probe_models(cfg["mesh_task_contact_world_hash"], assets.MEAN)
    if declaration["models"] != models or declaration["candidate"] != baseline["candidate"]:
        raise ValueError("only predeclared numeric zero and single-ankle parameters permitted")
    plan = PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), assets.MEAN, 18)
    context = next(c for c in receding_contexts(plan) if c.source_group == 76)
    if (
        hash_json(declaration["source_plan"]) != hash_json(plan.contract())
        or declaration["context_hash"] != context.context_hash
    ):
        raise ValueError("original plan18 C6 identity, not new plan19 group76, required")
    events, previous = [], "GENESIS"
    with sqlite3.connect((root / "campaign.sqlite").as_uri() + "?mode=ro", uri=True) as db:
        db.execute("BEGIN")
        for seq, link, body, h in db.execute("SELECT * FROM events ORDER BY seq"):
            value = json.loads(body)
            if (
                seq != len(events) + 1
                or link != previous
                or h != hash_json(dict(seq=seq, previous=link, body=value))
            ):
                raise ValueError("whole durable journal chain required")
            events.append(value)
            previous = h
    if not events:
        raise ValueError("nonempty original durable journal required")
    final = events[-1]["state"]
    if (
        final["phase"] != "REFERENCE_STUDENT_PROBE_COMPLETE_NOT_TRAINED_MODEL"
        or final["active"] is not None
        or final["attempts"] != 3
        or set(final["completed"]) != {j["job"] for j in JOBS}
        or final["declaration_hash"] != hash_json(declaration)
    ):
        raise ValueError("all original charged jobs and terminal journal required")
    with np.load(baseline["witnesses"]["trace"]["path"], allow_pickle=False) as a:
        original = dict(a)
    rows, traces, payload = [], [], None
    for index, job in enumerate(JOBS):
        receipt = sealed(root / f"receipt-{index:04d}.json", "receipt_hash")
        if (
            receipt != complete["results"][index]
            or receipt != final["completed"][job["job"]]
            or any(receipt.get(k) != v for k, v in job.items())
            or receipt["candidate"] != declaration["candidate"]
        ):
            raise ValueError("ordered whole original receipts required")
        witnesses = dict(receipt["witnesses"])
        witnesses.update(
            {
                k: receipt[k]
                for k in ("intent_witness", "parity_witness", "planning_context_witness")
            }
        )
        for witness in witnesses.values():
            path = Path(witness["path"])
            if (
                not path.resolve().is_relative_to(root.resolve())
                or path.is_symlink()
                or digest(path) != witness["sha256"]
            ):
                raise ValueError("ordinary owned unchanged physical witnesses required")
            pins[str(path)] = witness["sha256"]
        if payload is None:
            payload = FixedPhysicalPayloadReader(
                Path(witnesses["proof_payload"]["path"]),
                expected_file_hash=witnesses["proof_payload"]["sha256"],
            )
            payload_witness = witnesses["proof_payload"]
        if witnesses["proof_payload"] != payload_witness:
            raise ValueError("unchanged whole parent payload required in all three probes")
        if payload_witness["sha256"] != baseline["witnesses"]["proof_payload"]["sha256"]:
            raise ValueError("actual mean payload bytes must equal original C6 parent")
        report = _sealed(Path(witnesses["report"]["path"]), physical_payload_reader=payload)
        commitment = json.loads(Path(witnesses["commitment"]["path"]).read_text())
        view = report["executed_motor_policy"]["step_motor_proof"]["model"]
        if (
            view["mean_model"]["model_hash"] != assets.MEAN
            or view["seed"] != plan.draws()[76].sampling_seed
            or report["step_model_hash"] != view["model_hash"]
            or any(
                type(receipt.get(k)) is not int or receipt[k] != n
                for k, n in (
                    ("physical_substeps", 3000),
                    ("foundation_calls", 300),
                    ("sampling_draws", 270),
                )
            )
            or any(
                k in doc
                for doc in (report, commitment)
                for k in (
                    "mesh_surface_swing",
                    "contact_exit_torque_mpc",
                    "directional_contact_compliance",
                    "swing_release_ablation",
                    "cerebellar_torque_contract",
                    "generated_seed_episode_binding",
                )
            )
        ):
            raise ValueError(
                "original parent, sampling seed, primitive counts and unmixed control required"
            )
        if (
            hash_json(commitment) != report["commitment_hash"]
            or report["report_hash"] != receipt["physical_report_hash"]
            or report["compiled_model_hash"] != cfg["mesh_task_contact_world_hash"]
            or witnesses["world"]["sha256"] != cfg["mesh_task_contact_world_hash"]
            or report["physical_trace_hash"] != witnesses["trace"]["sha256"]
            or report["source_hash"]
            != digest(source / "src/rosclaw_soccer/rsi/native_first_touch_episode.py")
        ):
            raise ValueError("actual physical report, world, trace and source identity required")
        model = None if job["model_index"] is None else models[job["model_index"]]
        expected_hash = None if model is None else model["model_hash"]
        if (
            report.get("reference_student_model") != model
            or receipt.get("reference_student_model_hash") != expected_hash
            or report["course"] != list(context.course)
        ):
            raise ValueError("actual model and original course must match charged intent")
        with np.load(witnesses["trace"]["path"], allow_pickle=False) as a:
            trace = dict(a)
        NativeFirstTouchInitialState(context.course, G1SonicRunupController.default_angles).verify(
            report["course"], trace["canonical_qpos"][0, 0], trace["canonical_qvel"][0, 0]
        )
        outcome = label_first_touch(trace)
        review, foundation = (
            sealed(witnesses[k]["path"]) for k in ("independent_review", "foundation_review")
        )
        alignment = require_first_touch_qualification(
            review, foundation, outcome, reference_student_model_hash=expected_hash
        )
        if (
            alignment != receipt["qualification_alignment"]
            or outcome.to_dict() != receipt["outcome"]
            or review["reviewed_report_hash"] != report["report_hash"]
            or foundation["physical_report_hash"] != report["report_hash"]
            or foundation["physical_trace_hash"] != witnesses["trace"]["sha256"]
            or review["source_hash"]
            != digest(source / "src/rosclaw_soccer/rsi/cpu_motor_transfer_evidence.py")
            or foundation["audit_source_hash"]
            != digest(source / "src/rosclaw_soccer/rsi/foundation_torch_capture_audit.py")
        ):
            raise ValueError("actual independent CPU and Foundation reconstruction required")
        parity = sealed(witnesses["parity_witness"]["path"])
        frames = 300 if model is None else 30
        if (
            parity["compared_frames"] != frames
            or parity["baseline_receipt_hash"] != baseline["receipt_hash"]
            or parity["all_original_array_hashes"] != require_parity(trace, original, frames)
        ):
            raise ValueError(
                "full physical OFF parity and both pre-action prefixes must reconstruct"
            )
        intent = sealed(witnesses["intent_witness"]["path"])
        expected = dict(
            job,
            candidate=declaration["candidate"],
            context_hash=context.context_hash,
            reference_model_hash=expected_hash,
            declaration_hash=declaration["report_hash"],
        )
        if {k: v for k, v in intent.items() if k != "report_hash"} != expected:
            raise ValueError("exact intent must precede actual body execution")
        starts = [
            i
            for i, e in enumerate(events)
            if e["event"] == "ATTEMPT_STARTED"
            and e["state"].get("active") == dict(job=job["job"], attempt=index)
        ]
        if (
            len(starts) != 1
            or starts[0] == 0
            or events[starts[0] - 1]["event"] != "INTENT_COMMITTED_BEFORE_BODY"
            or events[starts[0] - 1]["state"].get("next_intent_hash") != intent["report_hash"]
        ):
            raise ValueError("one prior committed intent per charged body required")
        rows.append(
            dict(
                job=job["job"],
                receipt_hash=receipt["receipt_hash"],
                outcome=outcome.to_dict(),
                reference_model_hash=expected_hash,
            )
        )
        traces.append(trace)
    authority = measured_authority(traces[1], traces[2])
    if authority != complete["observed_authority"]:
        raise ValueError("actual proposed, parent and physical state differences must reconstruct")
    p.stable()
    if any(digest(path) != h for path, h in pins.items()):
        raise ValueError("whole source/witness bytes changed during readback")
    result = dict(
        schema="soccer.rsi.reference_student_three_terminal_readback.v1",
        complete_hash=complete["report_hash"],
        ledger_tip=previous,
        rows=rows,
        reader_source_hash=digest(Path(__file__)),
        observed_authority=authority,
        original_off_all33_arrays_bit_exact=True,
        all_pre_action_prefixes_bit_exact=True,
        prior_independent_replays_authenticated=True,
        whole_input_and_witness_files_verified=True,
        shared_payload_logical_seals_rebuilt=True,
        source_native_bodies=3,
        new_physical_steps=0,
        new_optimizer_updates=0,
        distinct_derived_course_count=1,
        source_scene_clusters=1,
        probe_is_trained_policy=False,
        teacher_qualified=False,
        dataset_admission_authorized=False,
        neural_training_authorized=False,
        fresh_exam=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output, result)
    print(
        "REFERENCE_STUDENT_THREE_READBACK_COMPLETE_NOT_LEARNING", result["report_hash"], flush=True
    )


if __name__ == "__main__":
    main()
