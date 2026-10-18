"""Whole six-body terminal readback; no simulator state allocation or physics."""

import argparse
import json
import sqlite3
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.first_touch_qualification import require_first_touch_qualification
from rosclaw_soccer.rsi.fixed_physical_payload_reader import FixedPhysicalPayloadReader
from rosclaw_soccer.rsi.planner_navigation_pulse import PlannerNavigationPulse
from rosclaw_soccer.rsi.planner_navigation_pulse_evidence import (
    require_navigation_pulse_capture,
    require_pulse_physical_parity,
)
from rosclaw_soccer.rsi.planner_navigation_pulse_scope import assess_pulse_scope
from rosclaw_soccer.rsi.reference_student_execution import reconstruct
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_pre_planner_navigation_pilot import prepare
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def read_journal(root, declaration):
    tip, events = "GENESIS", []
    with sqlite3.connect((root / "campaign.sqlite").as_uri() + "?mode=ro", uri=True) as db:
        db.execute("BEGIN")
        for seq, previous, body, h in db.execute("SELECT * FROM events ORDER BY seq"):
            v = json.loads(body)
            if (
                seq != len(events) + 1
                or previous != tip
                or h != hash_json(dict(seq=seq, previous=previous, body=v))
            ):
                raise ValueError("whole original six-body journal chain required")
            events.append(v)
            tip = h
    if not events:
        raise ValueError("nonempty original pulse journal required")
    state = events[-1]["state"]
    if (
        state["phase"] != "PILOT_COMPLETE_EXPANDED_VALIDATION_REQUIRED"
        or state["active"] is not None
        or type(state["attempts"]) is not int
        or state["attempts"] != 6
        or set(state["completed"]) != {j["job"] for j in declaration["scope"]["jobs"]}
        or state["declaration_hash"] != hash_json(declaration)
    ):
        raise ValueError("all six charged bodies and exact terminal journal required")
    return tip, state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--unit", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    require_terminal(args.unit)
    started = args.output.with_suffix(".started.json")
    if args.output.exists() or started.exists():
        raise ValueError("unused once-only pulse readback required")
    c = prepare(args.config, closed=True)
    if Path(__file__).resolve() != c.source / "scripts/rsi_read_pre_planner_navigation.py":
        raise ValueError("original producer-owned pulse readback source required")
    declaration, complete = (c.sealed(c.root / n) for n in ("declaration.json", "complete.json"))
    if (
        declaration.get("schema") != "soccer.rsi.pre_planner_pulse_six_declaration.v1"
        or complete.get("schema") != "soccer.rsi.pre_planner_pulse_six_complete.v1"
        or any(declaration.get(k) != v for k, v in c.cfg.items())
        or declaration["scope"] != c.scope
        or declaration["candidate"] != c.candidate
        or declaration["zero_reference_model"] != c.models["zero"]
        or declaration["reused_zero_controls"] != {str(g): r for g, r in c.recorded_zeros.items()}
        or declaration["recorded_zero_complete_hash"] != c.recorded_zero_complete_hash
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
        or any(
            complete.get(k) is not False
            for k in ("teacher_qualified", "hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("whole original six-body declaration and result required")
    for name, h in {**declaration["input_file_hashes"], **declaration["prepared_pins"]}.items():
        if digest(name) != h:
            raise ValueError("original pulse inputs or source bytes changed")
        c.pins[name] = h
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
    reader, payload, limits, rows = None, None, None, []
    for i, job in enumerate(c.scope["jobs"]):
        r = c.sealed(c.root / f"receipt-{i:04d}.json", "receipt_hash")
        if (
            r != complete["results"][i]
            or r != state["completed"][job["job"]]
            or any(r.get(k) != v for k, v in job.items())
            or r.get("candidate") != c.candidate
            or r.get("reference_student_model_hash") != c.models["zero"]["model_hash"]
            or any(
                type(r.get(k)) is not int or r[k] != n
                for k, n in (
                    ("physical_substeps", 3000),
                    ("foundation_calls", 300),
                    ("sampling_draws", 270),
                    ("reference_student_inference_calls", 300),
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
            raise ValueError("all original ordered pulse receipts required")
        witnesses = dict(r["witnesses"])
        witnesses.update({k: v for k, v in r.items() if k.endswith("_witness")})
        for w in witnesses.values():
            path = Path(w["path"])
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(c.root.resolve())
                or digest(path) != w["sha256"]
            ):
                raise ValueError("ordinary unchanged producer-owned witnesses required")
            c.pins[str(path)] = w["sha256"]
        if reader is None:
            payload = witnesses["proof_payload"]
            if payload["sha256"] != c.nominal["results"][0]["witnesses"]["proof_payload"]["sha256"]:
                raise ValueError("same unchanged whole parent payload required")
            reader = FixedPhysicalPayloadReader(
                Path(payload["path"]), expected_file_hash=payload["sha256"]
            )
        if witnesses["proof_payload"] != payload:
            raise ValueError("one unchanged shared payload required")
        report = _sealed(Path(witnesses["report"]["path"]), physical_payload_reader=reader)
        commitment = json.loads(Path(witnesses["commitment"]["path"]).read_text())
        pulse = PlannerNavigationPulse(job["lateral_delta_m_s"])
        view = report["executed_motor_policy"]["step_motor_proof"]["model"]
        if (
            report["commitment_hash"] != hash_json(commitment)
            or report["report_hash"] != r["physical_report_hash"]
            or report["physical_trace_hash"] != witnesses["trace"]["sha256"]
            or witnesses["world"]["sha256"] != c.cfg["mesh_task_contact_world_hash"]
            or report["compiled_model_hash"] != c.cfg["mesh_task_contact_world_hash"]
            or report["source_hash"]
            != digest(c.source / "src/rosclaw_soccer/rsi/native_first_touch_episode.py")
            or report["reference_student_model"] != c.models["zero"]
            or commitment["pre_planner_navigation_pulse"] != pulse.contract()
            or view["mean_model"]["model_hash"] != c.assets.MEAN
            or view["seed"] != c.positive_plan.draws()[job["group"]].sampling_seed
            or report["step_model_hash"] != view["model_hash"]
            or report["course"] != list(c.contexts[job["group"]].course)
        ):
            raise ValueError("original source-bound zero model, pulse, world and report required")
        with np.load(witnesses["trace"]["path"], allow_pickle=False) as a:
            trace = dict(a)
        nav = c.recorded_zeros[job["group"]]
        with np.load(nav["witnesses"]["trace"]["path"], allow_pickle=False) as a:
            baseline = dict(a)
        prefix = require_pulse_physical_parity(
            trace, baseline, frames=300 if job["role"] == "pulse_off" else 100
        )
        prefix.update(
            original_nav_receipt_hash=nav["receipt_hash"],
            original_nav_trace_hash=nav["witnesses"]["trace"]["sha256"],
            hardware_authorized=False,
            promotion_authorized=False,
        )
        prefix["report_hash"] = hash_json(prefix)
        if c.sealed(witnesses["prefix_witness"]["path"]) != prefix:
            raise ValueError("original full OFF or before-pulse prefix witness required")
        intent = dict(
            job,
            zero_reference_model_hash=c.models["zero"]["model_hash"],
            pulse_contract_hash=pulse.contract()["contract_hash"],
            declaration_hash=declaration["report_hash"],
        )
        intent["report_hash"] = hash_json(intent)
        if c.sealed(witnesses["intent_witness"]["path"]) != intent:
            raise ValueError("original before-body model/pulse/job intent required")
        encoder = c.sealed(witnesses["encoder_witness"]["path"])
        if (
            encoder.get("schema") != "soccer.rsi.sonic_encoder_reference_consumption.v1"
            or encoder.get("actual_cpu_onnx_planner_calls") != 15
            or encoder.get("compared_encoder_frames") != 300
            or encoder.get("physical_improvement_proven") is not False
        ):
            raise ValueError("all original planner calls and encoder feature comparisons required")
        review = c.sealed(witnesses["independent_review"]["path"])
        foundation = c.sealed(witnesses["foundation_review"]["path"])
        pulse_review = require_navigation_pulse_capture(report, commitment, trace)
        if limits is None:
            model = mujoco.MjModel.from_binary_path(witnesses["world"]["path"])
            limits = np.array([model.jnt_range[model.joint(n).id] for n in G1_DDS_JOINT_NAMES])
        reference_review = reconstruct(
            commitment, trace, joint_limits=limits, parent_model_hash=c.assets.MEAN
        )
        if (
            review["source_hash"]
            != digest(c.source / "src/rosclaw_soccer/rsi/cpu_motor_transfer_evidence.py")
            or review["planner_navigation_pulse_review"] != pulse_review
            or review["reference_student_review"] != reference_review
            or review["report_hash"] != r["independent_review_hash"]
            or foundation["report_hash"] != r["foundation_review_hash"]
        ):
            raise ValueError(
                "original independent CPU reviews and whole zero-reference reconstruction required"
            )
        outcome = label_first_touch(trace)
        alignment = require_first_touch_qualification(
            review,
            foundation,
            outcome,
            reference_student_model_hash=c.models["zero"]["model_hash"],
            planner_navigation_pulse_contract_hash=pulse.contract()["contract_hash"],
        )
        if outcome.to_dict() != r["outcome"] or alignment != r["qualification_alignment"]:
            raise ValueError("original primitive safety and outcome labels required")
        rows.append(
            dict(
                job=job,
                receipt_hash=r["receipt_hash"],
                outcome=r["outcome"],
                pulse_review_hash=pulse_review["report_hash"],
                alignment=alignment,
            )
        )
    assessment = assess_pulse_scope(c.scope, complete["results"])
    if assessment != complete["assessment"] or any(digest(n) != h for n, h in c.pins.items()):
        raise ValueError("whole outcome assessment and unchanged source/witness files required")
    result = dict(
        schema="soccer.rsi.pre_planner_pulse_six_terminal_readback.v1",
        complete_hash=complete["report_hash"],
        ledger_tip=tip,
        rows=rows,
        assessment=assessment,
        input_file_hashes=c.pins,
        reader_source_hash=digest(Path(__file__)),
        source_native_bodies=6,
        zero_reference_frames_reconstructed=1800,
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
    print("SIX_PULSE_BODIES_READBACK_COMPLETE_NOT_PROMOTION", result["report_hash"], flush=True)


if __name__ == "__main__":
    main()
