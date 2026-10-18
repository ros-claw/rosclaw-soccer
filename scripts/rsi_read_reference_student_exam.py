"""Whole own-body exam readback; no physical integration, fitting or promotion."""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.first_touch_qualification import require_first_touch_qualification
from rosclaw_soccer.rsi.first_touch_recovery_observations import observe_first_touch_recovery
from rosclaw_soccer.rsi.fixed_physical_payload_reader import FixedPhysicalPayloadReader
from rosclaw_soccer.rsi.native_first_touch_initial_state import NativeFirstTouchInitialState
from rosclaw_soccer.rsi.reference_exam_profiles import (
    ZERO,
    assess_reference,
    profile_info,
    require_zero_navigation_trace,
)
from rosclaw_soccer.rsi.reference_student_execution import PREFIX, reconstruct
from rosclaw_soccer.rsi.reference_student_probe import require_parity
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_reference_student_exam import load_inputs
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def journal(root, declaration):
    info = profile_info(declaration["exam_profile"])
    events, tip = [], "GENESIS"
    with sqlite3.connect((root / "campaign.sqlite").as_uri() + "?mode=ro", uri=True) as db:
        db.execute("BEGIN")
        for seq, previous, body, h in db.execute("SELECT * FROM events ORDER BY seq"):
            value = json.loads(body)
            if (
                seq != len(events) + 1
                or previous != tip
                or h != hash_json(dict(seq=seq, previous=previous, body=value))
            ):
                raise ValueError("whole original journal chain required")
            events.append(value)
            tip = h
    if not events:
        raise ValueError("nonempty original own-body journal required")
    final = events[-1]["state"]
    if (
        final["phase"] != "REFERENCE_STUDENT_OWN_BODY_EXAM_COMPLETE_NOT_PROMOTION"
        or final["active"] is not None
        or final["attempts"] != info["bodies"]
        or set(final["completed"]) != {j["job"] for j in declaration["scope"]["jobs"]}
        or final["declaration_hash"] != hash_json(declaration)
    ):
        raise ValueError("all thirteen charged jobs and terminal journal required")
    return events, tip, final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_terminal(args.unit)
    started = args.output.with_suffix(".started.json")
    if args.output.exists() or started.exists():
        raise ValueError("unused once-only readback output required")
    c = load_inputs(args.config)
    info = profile_info(c.cfg["exam_profile"])
    declaration, complete = (c.sealed(c.root / n) for n in ("declaration.json", "complete.json"))
    if (
        declaration.get("schema") != info["declaration_schema"]
        or complete.get("schema") != info["complete_schema"]
        or any(declaration.get(k) != v for k, v in c.cfg.items())
        or hash_json(declaration["scope"]) != hash_json(c.scope)
        or declaration["models"] != c.models
        or declaration["candidate"] != c.candidate
        or declaration["fit_stage_hash"] != c.fit_stage["report_hash"]
        or declaration["fit_report_hash"] != c.fit_report["report_hash"]
        or declaration["reused_teachers"] != {str(k): v for k, v in c.teachers.items()}
        or declaration["reused_legacy_retention"] != {str(k): v for k, v in c.retained.items()}
        or declaration["prepared_pins"] != c.p.PINS
        or declaration["source_heads"] != c.p.HEADS
        or declaration["numerical_world_review"] != c.binding
        or complete["declaration_hash"] != declaration["report_hash"]
        or len(complete.get("results", [])) != info["bodies"]
        or any(
            type(complete.get(k)) is not int or complete[k] != n
            for k, n in (
                ("new_native_bodies", info["bodies"]),
                ("new_independent_cpu_bodies", info["bodies"]),
                ("native_substeps", info["steps"]),
                ("independent_cpu_substeps", info["steps"]),
                ("new_optimizer_updates", 0),
            )
        )
        or any(
            complete.get(k) is not False
            for k in (
                "teacher_qualified",
                "hardware_authorized",
                "promotion_authorized",
                "fresh_exam",
            )
        )
    ):
        raise ValueError("whole original thirteen-body declaration and complete required")
    write_once(
        started,
        dict(
            schema="soccer.rsi.reference_exam_readback_started.v1",
            reader_source_hash=digest(Path(__file__)),
            automatic_retry_authorized=False,
            hardware_authorized=False,
        ),
    )
    for name, h in {**declaration["prepared_pins"], **declaration["input_file_hashes"]}.items():
        if digest(name) != h:
            raise ValueError("original source or input bytes changed")
        c.pins[name] = h
    events, tip, final = journal(c.root, declaration)
    rows, receipts, zero_traces = [], [], {}
    reader, payload_witness, limits = None, None, None
    reconstructed_frames = 0
    for index, job in enumerate(c.scope["jobs"]):
        receipt = c.sealed(c.root / f"receipt-{index:04d}.json", "receipt_hash")
        if (
            receipt != complete["results"][index]
            or receipt != final["completed"][job["job"]]
            or any(receipt.get(k) != v for k, v in job.items())
            or receipt["candidate"] != c.candidate
        ):
            raise ValueError("ordered original charged receipt and control parameters required")
        witnesses = dict(receipt["witnesses"])
        witnesses.update({k: v for k, v in receipt.items() if k.endswith("_witness")})
        for witness in witnesses.values():
            path = Path(witness["path"])
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(c.root.resolve())
                or digest(path) != witness["sha256"]
            ):
                raise ValueError("unchanged ordinary owned physical witnesses required")
            c.pins[str(path)] = witness["sha256"]
        if reader is None:
            payload_witness = witnesses["proof_payload"]
            if (
                payload_witness["sha256"]
                != c.nominal["results"][0]["witnesses"]["proof_payload"]["sha256"]
            ):
                raise ValueError("original whole parent payload bytes required")
            reader = FixedPhysicalPayloadReader(
                Path(payload_witness["path"]), expected_file_hash=payload_witness["sha256"]
            )
        if witnesses["proof_payload"] != payload_witness:
            raise ValueError("one unchanged shared whole parent payload required")
        report = _sealed(Path(witnesses["report"]["path"]), physical_payload_reader=reader)
        commitment = json.loads(Path(witnesses["commitment"]["path"]).read_text())
        positive = job["plan"] == "positive19"
        plan = c.positive_plan if positive else c.old_plan
        context = c.contexts[job["group"]] if positive else None
        course = plan.draws()[job["group"]].course if context is None else context.course
        view = report["executed_motor_policy"]["step_motor_proof"]["model"]
        model = c.models[job["role"]]
        model_hash = None if model is None else model["model_hash"]
        if (
            hash_json(commitment) != report["commitment_hash"]
            or report["report_hash"] != receipt["physical_report_hash"]
            or report["physical_trace_hash"] != witnesses["trace"]["sha256"]
            or report["compiled_model_hash"] != c.cfg["mesh_task_contact_world_hash"]
            or witnesses["world"]["sha256"] != c.cfg["mesh_task_contact_world_hash"]
            or report["source_hash"]
            != digest(c.source / "src/rosclaw_soccer/rsi/native_first_touch_episode.py")
            or view["mean_model"]["model_hash"] != c.assets.MEAN
            or view["seed"] != plan.draws()[job["group"]].sampling_seed
            or report["step_model_hash"] != view["model_hash"]
            or report["course"] != list(course)
            or any(
                type(receipt.get(k)) is not int or receipt[k] != n
                for k, n in (
                    ("physical_substeps", 3000),
                    ("foundation_calls", 300),
                    ("sampling_draws", 270),
                )
            )
            or any(
                report.get(k) is not False for k in ("hardware_authorized", "promotion_authorized")
            )
            or any(
                k in doc
                for doc in (report, commitment)
                for k in (
                    "mesh_surface_swing",
                    "contact_exit_torque_mpc",
                    "directional_contact_compliance",
                    "cerebellar_torque_contract",
                    "swing_release_ablation",
                )
            )
            or any(
                doc.get("reference_student_model") != model
                or ("reference_student_model" in doc) != (model is not None)
                for doc in (report, commitment)
            )
            or receipt.get("reference_student_model_hash") != model_hash
        ):
            raise ValueError(
                "exact original world, parent, model, course, source and primitive counts required"
            )
        for doc in (report, commitment):
            if job["use_generated_binding"]:
                bound = doc.get("generated_seed_episode_binding", {})
                if (
                    bound.get("seed") != view["seed"]
                    or any(
                        bound.get(k) is not True
                        for k in (
                            "full_external_sampling_view_restored_and_validated",
                            "complete_generated_model_equals_external_view",
                            "full_original_factory_constructor_retained",
                        )
                    )
                    or bound.get("sampling_law_changed") is not False
                ):
                    raise ValueError("actual complete private-generated source binding required")
            elif "generated_seed_episode_binding" in doc:
                raise ValueError("original raw retention path must not acquire generated binding")
        with np.load(witnesses["trace"]["path"], allow_pickle=False) as a:
            trace = dict(a)
        if c.cfg["exam_profile"] == ZERO and job["role"] == "zero":
            require_zero_navigation_trace(trace)
        NativeFirstTouchInitialState(course, G1SonicRunupController.default_angles).verify(
            report["course"], trace["canonical_qpos"][0, 0], trace["canonical_qvel"][0, 0]
        )
        outcome = label_first_touch(trace)
        review, foundation = (
            c.sealed(witnesses[k]["path"]) for k in ("independent_review", "foundation_review")
        )
        aligned = require_first_touch_qualification(
            review, foundation, outcome, reference_student_model_hash=model_hash
        )
        if (
            outcome.to_dict() != receipt["outcome"]
            or aligned != receipt["qualification_alignment"]
            or review["reviewed_report_hash"] != report["report_hash"]
            or foundation["physical_report_hash"] != report["report_hash"]
            or foundation["physical_trace_hash"] != witnesses["trace"]["sha256"]
            or review["source_hash"]
            != digest(c.source / "src/rosclaw_soccer/rsi/cpu_motor_transfer_evidence.py")
            or foundation["audit_source_hash"]
            != digest(c.source / "src/rosclaw_soccer/rsi/foundation_torch_capture_audit.py")
        ):
            raise ValueError("same original CPU dynamics/parent/Foundation qualification required")
        if model is not None:
            if limits is None:
                import mujoco

                mjmodel = mujoco.MjModel.from_binary_path(witnesses["world"]["path"])
                if (mjmodel.nq, mjmodel.nv, mjmodel.nu) != (43, 41, 29):
                    raise ValueError("canonical G1 compiled world required")
                limits = np.stack(
                    [mjmodel.jnt_range[mjmodel.joint(n).id] for n in G1_DDS_JOINT_NAMES]
                )
            actual = reconstruct(
                commitment, trace, joint_limits=limits, parent_model_hash=c.assets.MEAN
            )
            if actual != review["reference_student_review"]:
                raise ValueError("whole own-action causal reference reconstruction required")
            reconstructed_frames += 300
        elif any(k.startswith(PREFIX) for k in trace):
            raise ValueError("OFF trace cannot contain student fields")
        if positive:
            with np.load(
                c.teachers[job["group"]]["witnesses"]["trace"]["path"], allow_pickle=False
            ) as a:
                baseline = dict(a)
            compared = 300 if job["role"] == "off" else 30
        elif job["role"] == "zero":
            zero_traces[job["group"]] = trace
            baseline, compared = trace, 30
        else:
            baseline, compared = zero_traces[job["group"]], 30
        prefix = c.sealed(witnesses["prefix_witness"]["path"])
        if (
            prefix["compared_frames"] != compared
            or prefix["array_hashes"] != require_parity(trace, baseline, compared)
            or prefix["zero_control_self_prefix"] != (not positive and job["role"] == "zero")
        ):
            raise ValueError("full OFF and both kinds of pre-action prefixes must reconstruct")
        encoder = c.sealed(witnesses["encoder_witness"]["path"])
        if (
            encoder.get("schema") != "soccer.rsi.sonic_encoder_reference_consumption.v1"
            or encoder.get("actual_cpu_onnx_planner_calls") != 15
            or encoder.get("compared_encoder_frames") != 300
            or encoder.get("physical_improvement_proven") is not False
        ):
            raise ValueError("complete original 15-call planner/300-frame encoder witness required")
        intent = c.sealed(witnesses["intent_witness"]["path"])
        expected = dict(
            job,
            model_hash=model_hash,
            source_plan_hash=plan.contract()["plan_hash"],
            candidate=c.candidate,
            declaration_hash=declaration["report_hash"],
        )
        if {k: v for k, v in intent.items() if k != "report_hash"} != expected:
            raise ValueError("exact prior charged intent required")
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
            raise ValueError("one committed intent before each actual body required")
        rows.append(
            dict(
                job=job,
                receipt_hash=receipt["receipt_hash"],
                outcome=outcome.to_dict(),
                model_hash=model_hash,
                recovery_observations=observe_first_touch_recovery(trace),
            )
        )
        receipts.append(receipt)
    paired = assess_reference(
        c.cfg["exam_profile"],
        c.scope,
        receipts,
        teachers={g: r["outcome"] for g, r in c.teachers.items()},
        original_retention={g: r["outcome"] for g, r in c.retained.items()},
    )
    if paired != complete["paired_results"] or reconstructed_frames != info["reference_frames"]:
        raise ValueError(
            "whole original paired outcomes and all own-action reconstructions required"
        )
    c.p.stable()
    if any(digest(p) != h for p, h in c.pins.items()):
        raise ValueError("input/source bytes changed during readback")
    result = dict(
        schema=info["readback_schema"],
        complete_hash=complete["report_hash"],
        ledger_tip=tip,
        rows=rows,
        paired_results=paired,
        reader_source_hash=digest(Path(__file__)),
        input_file_hashes=c.pins,
        whole_input_and_witness_files_verified=True,
        shared_payload_logical_seals_rebuilt=True,
        original_off_all33_arrays_bit_exact=True,
        all_pre_action_prefixes_bit_exact=True,
        causal_reference_frames_reconstructed=info["reference_frames"],
        source_native_bodies=info["bodies"],
        new_physical_steps=0,
        new_optimizer_updates=0,
        successor_readiness_qualified=False,
        official_52_case_exam_completed=False,
        fresh_exam=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    if c.cfg["exam_profile"] == ZERO:
        result.update(
            matched_zero_all300_reference_increments_exactly_zero=True,
            fitted_student_policy_evaluated=False,
        )
    result["report_hash"] = hash_json(result)
    write_once(args.output, result)
    print(
        "REFERENCE_STUDENT_OWN_BODY_READBACK_COMPLETE_NOT_PROMOTION",
        result["report_hash"],
        flush=True,
    )


if __name__ == "__main__":
    main()
