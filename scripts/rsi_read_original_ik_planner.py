"""Authenticate all original-IK composition receipts; no new physics or inference."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.first_touch_qualification import require_first_touch_qualification
from rosclaw_soccer.rsi.fixed_physical_payload_reader import FixedPhysicalPayloadReader
from rosclaw_soccer.rsi.original_ik_planner_evidence import TAG, require_original_ik_capture
from rosclaw_soccer.rsi.original_ik_planner_pulse import OriginalIkPlannerPulse
from rosclaw_soccer.rsi.original_ik_planner_scope import assess_ik_scope
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_original_ik_planner_pilot import intent_for, prefix_for, prepare
from scripts.rsi_read_pre_planner_navigation import read_journal
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def require_complete(c, declaration, complete):
    if (
        declaration.get("schema") != "soccer.rsi.original_ik_planner_six_declaration.v1"
        or complete.get("schema") != "soccer.rsi.original_ik_planner_six_complete.v1"
        or any(declaration.get(k) != v for k, v in c.cfg.items())
        or hash_json(declaration["scope"]) != hash_json(c.scope)
        or declaration["candidate"] != c.candidate
        or declaration["reused_teachers"] != {str(g): r for g, r in c.teachers.items()}
        or declaration["signed_predecessor_reviews"] != c.signed_predecessors
        or declaration["prepared_pins"] != c.p.PINS
        or declaration["source_heads"] != c.p.HEADS
        or complete["declaration_hash"] != declaration["report_hash"]
        or len(complete["results"]) != 6
        or any(
            type(complete.get(k)) is not int or complete[k] != n
            for k, n in (
                ("new_native_bodies", 6),
                ("new_independent_cpu_bodies", 6),
                ("native_substeps", 18000),
                ("independent_cpu_substeps", 18000),
                ("new_optimizer_updates", 0),
            )
        )
        or complete.get("independent_terminal_readback_required") is not True
        or any(
            complete.get(k) is not False
            for k in ("teacher_qualified", "hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("whole original-IK declaration and all six completed bodies required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require_terminal(args.unit)
    started = args.output.with_suffix(".started.json")
    if args.output.exists() or started.exists():
        raise ValueError("unused once-only original-IK readback required")
    c = prepare(args.config, closed=True)
    if Path(__file__).resolve() != c.source / "scripts/rsi_read_original_ik_planner.py":
        raise ValueError("original frozen producer-owned original-IK reader required")
    declaration, complete = (c.sealed(c.root / n) for n in ("declaration.json", "complete.json"))
    require_complete(c, declaration, complete)
    for path, h in {**declaration["input_file_hashes"], **declaration["prepared_pins"]}.items():
        if digest(path) != h or (path in c.pins and c.pins[path] != h):
            raise ValueError("original source, input or teacher witness changed")
        c.pins[path] = h
    tip, state = read_journal(c.root, declaration)
    write_once(
        started,
        dict(
            source_complete_hash=complete["report_hash"],
            automatic_retry_authorized=False,
            new_physical_steps=0,
            hardware_authorized=False,
        ),
    )
    reader, payload, rows = None, None, []
    for i, job in enumerate(c.scope["jobs"]):
        r = c.sealed(c.root / f"receipt-{i:04d}.json", "receipt_hash")
        if (
            r != complete["results"][i]
            or r != state["completed"][job["job"]]
            or any(r.get(k) != v for k, v in job.items())
            or r.get("candidate") != c.candidate
            or any(k.startswith("reference_student_") or k.startswith("cerebellar_") for k in r)
            or any(
                type(r.get(k)) is not int or r[k] != n
                for k, n in (
                    ("physical_substeps", 3000),
                    ("foundation_calls", 300),
                    ("sampling_draws", 270),
                    ("neural_optimizer_updates", 0),
                )
            )
            or any(
                r.get(k) is not False
                for k in (
                    "hardware_authorized",
                    "promotion_authorized",
                    "private_fresh_accessed",
                    "teacher_qualified",
                )
            )
        ):
            raise ValueError("all original ordered unmixed IK receipts required")
        witnesses = dict(r["witnesses"])
        witnesses.update({k: v for k, v in r.items() if k.endswith("_witness")})
        for w in witnesses.values():
            path = Path(w["path"])
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(c.root.resolve())
                or digest(path) != w["sha256"]
            ):
                raise ValueError("ordinary producer-owned unchanged witness files required")
            c.pins[str(path)] = w["sha256"]
        if reader is None:
            payload = witnesses["proof_payload"]
            if payload["sha256"] != c.nominal["results"][0]["witnesses"]["proof_payload"]["sha256"]:
                raise ValueError("same unchanged complete original parent payload required")
            reader = FixedPhysicalPayloadReader(
                Path(payload["path"]), expected_file_hash=payload["sha256"]
            )
        if witnesses["proof_payload"] != payload:
            raise ValueError("one unchanged original parent payload required")
        report = _sealed(Path(witnesses["report"]["path"]), physical_payload_reader=reader)
        commitment = json.loads(Path(witnesses["commitment"]["path"]).read_text())
        pulse = OriginalIkPlannerPulse(job["lateral_delta_m_s"])
        view = report["executed_motor_policy"]["step_motor_proof"]["model"]
        if (
            report["commitment_hash"] != hash_json(commitment)
            or report["report_hash"] != r["physical_report_hash"]
            or report["physical_trace_hash"] != witnesses["trace"]["sha256"]
            or witnesses["world"]["sha256"] != c.cfg["mesh_task_contact_world_hash"]
            or report["compiled_model_hash"] != c.cfg["mesh_task_contact_world_hash"]
            or report["source_hash"]
            != digest(c.source / "src/rosclaw_soccer/rsi/native_first_touch_episode.py")
            or commitment[TAG] != pulse.contract()
            or view["mean_model"]["model_hash"] != c.assets.MEAN
            or view["seed"] != c.positive_plan.draws()[job["group"]].sampling_seed
            or report["step_model_hash"] != view["model_hash"]
            or report["course"] != list(c.contexts[job["group"]].course)
        ):
            raise ValueError("original source-bound IK, parent, seed, world and course required")
        with np.load(witnesses["trace"]["path"], allow_pickle=False) as a:
            trace = dict(a)
        prefix = prefix_for(job, trace, c.teachers[job["group"]], c.pins)
        if c.sealed(witnesses["prefix_witness"]["path"]) != prefix:
            raise ValueError("original-teacher OFF300 or active100 bitwise prefix required")
        if c.sealed(witnesses["intent_witness"]["path"]) != intent_for(
            job, declaration["report_hash"]
        ):
            raise ValueError("original before-body IK request intent required")
        encoder = c.sealed(witnesses["encoder_witness"]["path"])
        if (
            encoder.get("schema") != "soccer.rsi.sonic_encoder_reference_consumption.v1"
            or type(encoder.get("actual_cpu_onnx_planner_calls")) is not int
            or encoder["actual_cpu_onnx_planner_calls"] != 15
            or type(encoder.get("compared_encoder_frames")) is not int
            or encoder["compared_encoder_frames"] != 300
            or encoder.get("physical_improvement_proven") is not False
        ):
            raise ValueError("all planner and encoder consumption evidence required")
        review = c.sealed(witnesses["independent_review"]["path"])
        foundation = c.sealed(witnesses["foundation_review"]["path"])
        scalar = require_original_ik_capture(report, commitment, trace)
        if (
            review["source_hash"]
            != digest(c.source / "src/rosclaw_soccer/rsi/cpu_motor_transfer_evidence.py")
            or review["original_ik_planner_pulse_review"] != scalar
            or review["report_hash"] != r["independent_review_hash"]
            or foundation["report_hash"] != r["foundation_review_hash"]
        ):
            raise ValueError(
                "original independent scalar and actual CPU reconstruction reviews required"
            )
        outcome = label_first_touch(trace)
        alignment = require_first_touch_qualification(
            review,
            foundation,
            outcome,
            original_ik_planner_pulse_contract_hash=pulse.contract()["contract_hash"],
        )
        if outcome.to_dict() != r["outcome"] or alignment != r["qualification_alignment"]:
            raise ValueError("original primitive M0 quality plus safety labels required")
        rows.append(
            dict(job=job, receipt_hash=r["receipt_hash"], outcome=r["outcome"], alignment=alignment)
        )
    assessment = assess_ik_scope(c.scope, complete["results"])
    c.p.stable()
    if assessment != complete["assessment"] or any(digest(n) != h for n, h in c.pins.items()):
        raise ValueError("whole assessment and unchanged frozen sources/evidence required")
    result = dict(
        schema="soccer.rsi.original_ik_planner_six_terminal_readback.v1",
        complete_hash=complete["report_hash"],
        ledger_tip=tip,
        rows=rows,
        assessment=assessment,
        input_file_hashes=c.pins,
        reader_source_hash=digest(Path(__file__)),
        source_native_bodies=6,
        source_independent_cpu_bodies=6,
        prior_actual_cpu_ik_reconstructions_authenticated=6,
        new_physical_steps=0,
        new_foundation_inferences=0,
        optimizer_updates=0,
        teacher_qualified=False,
        dataset_admission_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output, result)
    print("ORIGINAL_IK_SIX_READBACK_COMPLETE_NOT_PROMOTION", result["report_hash"], flush=True)


if __name__ == "__main__":
    main()
