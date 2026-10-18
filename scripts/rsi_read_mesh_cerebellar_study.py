"""Terminal 44-body evidence readback and exact numeric update reconstruction.

No simulator or new candidate is allocated. Recomputed Adam steps are audit
work, explicitly counted, not new training achievements or additional data.
"""

import argparse
import hashlib
import importlib.util
import json
import sqlite3
import subprocess
import sys
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def optimizer_audit_accounting(receipts):
    """Count actual replay attempts, including a rejected final trial."""
    if len(receipts) != 3:
        raise ValueError("all three reconstructed optimizer receipts required")
    attempted, accepted = 0, 0
    for receipt in receipts:
        counts = [
            receipt[key]
            for key in (
                "requested_optimizer_steps",
                "attempted_optimizer_steps",
                "accepted_joint_optimizer_steps",
            )
        ]
        requested, tried, kept = counts
        if (
            any(type(value) is not int for value in counts)
            or requested != 4
            or not 0 <= kept <= tried <= requested
        ):
            raise ValueError("actual bounded integer optimizer accounting required")
        attempted += tried
        accepted += kept
    return dict(
        accepted_training_optimizer_steps=accepted,
        additional_audit_accepted_optimizer_step_recomputations=accepted,
        additional_audit_optimizer_step_attempts=attempted,
        additional_audit_optimizer_step_attempt_budget=12,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started_path = args.output.with_suffix(".started.json")
    state = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            args.unit,
            "-p",
            "MainPID",
            "-p",
            "ActiveState",
            "-p",
            "Result",
            "-p",
            "LoadState",
        ],
        text=True,
    )
    if (
        args.output.exists()
        or started_path.exists()
        or set(state.splitlines())
        != {
            "MainPID=0",
            "ActiveState=inactive",
            "Result=success",
            "LoadState=loaded",
        }
    ):
        raise ValueError("terminal successful worker and unused audit path required")
    config = json.loads(args.config.read_text())
    source, core, root = (Path(config[k]) for k in ("source_root", "core_root", "output_root"))
    # The frozen producer's policy/replay/optimizer code owns its source hashes.
    # The independent audit script and pure pairing module have separate hashes.
    sys.path[:0] = [str(source / "src"), str(source), str(core / "src")]
    import numpy as np

    from rosclaw_soccer.rsi.cerebellar_actor_critic import fit_update, validate_checkpoint
    from rosclaw_soccer.rsi.cerebellar_recovery_return import training_return
    from rosclaw_soccer.rsi.cerebellar_torque_audit import reconstruct_native_cerebellar
    from rosclaw_soccer.rsi.cerebellar_training_batch import build_batch
    from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
    from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
    from rosclaw_soccer.rsi.first_touch_qualification import require_first_touch_qualification
    from rosclaw_soccer.rsi.fixed_physical_payload_reader import FixedPhysicalPayloadReader
    from rosclaw_soccer.rsi.mesh_cerebellar_research_plan import (
        SEEDS,
        build_jobs,
        same_input_torque_drift,
        zero_head_anchor_features,
    )
    from rosclaw_soccer.rsi.portfolio_physical_study import validate_option
    from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
    from rosclaw_soccer.rsi.public_physics_option_planning import planning_contexts
    from rosclaw_soccer.rsi.reviewed_contact_world import ReviewedContactWorld
    from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
    from scripts.rsi_atomic_artifacts import write_once

    write_once(
        started_path,
        dict(
            schema="soccer.rsi.mesh_cerebellar_terminal_audit_start.v1",
            audit_source_hash=digest(Path(__file__)),
            configuration_file_hash=digest(args.config),
            worker_unit=args.unit,
            automatic_retry_authorized=False,
            hardware_authorized=False,
        ),
    )

    # Deliberately use the auditor checkout's source for new pairing logic.
    pairing_path = (
        Path(__file__).resolve().parents[1] / "src/rosclaw_soccer/rsi/paired_learning_assessment.py"
    )
    spec = importlib.util.spec_from_file_location("independent_pairing", pairing_path)
    if spec is None or spec.loader is None:
        raise ValueError("ordinary pinned audit pairing source required")
    pairing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pairing)

    def sealed(path, key="report_hash"):
        value = json.loads(path.read_text())
        if value[key] != hash_json({k: v for k, v in value.items() if k != key}):
            raise ValueError("whole JSON seal changed")
        return value

    declaration = sealed(root / "declaration.json")
    complete = sealed(root / "complete.json")
    if (
        declaration["schema"] != "soccer.rsi.mesh_cerebellar_three_seed_declaration.v1"
        or complete["schema"] != "soccer.rsi.mesh_cerebellar_three_seed_complete.v1"
        or complete["declaration_hash"] != declaration["report_hash"]
        or len(declaration["jobs"]) != 44
        or len(complete["results"]) != 44
        or any(declaration.get(k) != v for k, v in config.items())
        or any(
            complete.get(k) is not False
            for k in (
                "physical_improvement_claimed",
                "teacher_qualified",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
    ):
        raise ValueError("complete exact SIM research declaration and accounting required")
    pins = dict(declaration["input_file_hashes"])
    pins.update(declaration["prepared_pins"])
    for name, expected in pins.items():
        if digest(Path(name)) != expected:
            raise ValueError("frozen source/input changed")
    for name, expected in declaration["source_heads"].items():
        if (
            subprocess.check_output(["git", "-C", name, "rev-parse", "HEAD"], text=True).strip()
            != expected
        ):
            raise ValueError("frozen checkout head changed")
        if subprocess.check_output(["git", "-C", name, "status", "--porcelain"], text=True).strip():
            raise ValueError("frozen checkout dirty")
    binding = ReviewedContactWorld(
        Path(config["nominal_readback"]), config["nominal_readback_file_hash"]
    ).verify(config["mesh_task_contact_world_hash"])
    if binding != declaration["numerical_world_review"]:
        raise ValueError("same actual numerical review required")
    catalog = [
        validate_option(row["candidate"])
        for row in sealed(Path(config["portfolio_fit"]))["options"]
    ]
    previous, events = "GENESIS", []
    with sqlite3.connect((root / "campaign.sqlite").as_uri() + "?mode=ro", uri=True) as db:
        db.execute("BEGIN")
        for seq, link, body, value in db.execute("SELECT * FROM events ORDER BY seq"):
            event = json.loads(body)
            if (
                seq != len(events) + 1
                or link != previous
                or value != hash_json(dict(seq=seq, previous=link, body=event))
            ):
                raise ValueError("complete append-only ledger required")
            events.append(event)
            previous = value
    final = events[-1]["state"]
    jobs = declaration["jobs"]
    if (
        final["phase"] != "ALL_DECLARED_LEARNING_AND_PHYSICS_COMPLETE_NO_PROMOTION"
        or final["active"] is not None
        or final["attempts"] != 44
        or set(final["completed"]) != {j["job"] for j in jobs}
        or final.get("complete_hash") != complete["report_hash"]
    ):
        raise ValueError("all original charged attempts must be terminal")
    initial, trained, checkpoints = {}, {}, {}
    for seed in SEEDS:
        initial[seed] = sealed(root / f"initial-actor-{seed}.json", "model_hash")
        trained[seed] = sealed(root / f"trained-actor-{seed}.json", "model_hash")
        checkpoints[seed] = sealed(root / f"trained-checkpoint-{seed}.json", "checkpoint_hash")
        validate_checkpoint(checkpoints[seed])
        if checkpoints[seed]["actor"] != trained[seed]:
            raise ValueError("trained actor must be the exact persisted checkpoint actor")
    nominal = sealed(Path(config["nominal_complete"]))
    passing = []
    baseline_traces = {}
    for receipt in nominal["results"]:
        witness = receipt["witnesses"]["trace"]
        if digest(Path(witness["path"])) != witness["sha256"]:
            raise ValueError("original full nominal reference trace changed")
        with np.load(witness["path"], allow_pickle=False) as archive:
            original = dict(archive)
        if label_first_touch(original).to_dict() != receipt["outcome"]:
            raise ValueError("original passing/failing controls changed")
        baseline_traces[f"raw-{receipt['group']}"] = witness
        if receipt["outcome"]["task_pass"]:
            passing.append(original)
    expected_anchors = zero_head_anchor_features(passing)
    del passing, original
    if [u["seed"] for u in complete["updates"]] != list(SEEDS):
        raise ValueError("all three ordered persistent updates required")
    for update in complete["updates"]:
        checkpoint = checkpoints[update["seed"]]
        if (
            update["checkpoint_hash"] != checkpoint["checkpoint_hash"]
            or update["receipt"] != checkpoint["learning_receipt"]
        ):
            raise ValueError("completion learning receipts differ from actual checkpoint")
    reader, rows, controls, evaluations = None, [], {}, {s: {} for s in SEEDS}
    draws = inferences = 0
    for index, job in enumerate(jobs):
        receipt = sealed(root / f"receipt-{index:04d}.json", "receipt_hash")
        if receipt != complete["results"][index] or receipt != final["completed"][job["job"]]:
            raise ValueError("receipt, ledger and completion disagree")
        model = (initial if job["model_source"] == "initial_checkpoint" else trained)[
            job["actor_seed"]
        ]
        witnesses = dict(receipt["witnesses"])
        witnesses.update(
            {k: receipt[k] for k in ("intent_witness", "encoder_witness", "model_witness")}
        )
        for witness in witnesses.values():
            path = Path(witness["path"])
            if (
                not path.is_absolute()
                or path.is_symlink()
                or any(p.is_symlink() for p in path.parents)
                or not path.resolve().is_relative_to(root.resolve())
            ):
                raise ValueError("ordinary study-owned physical/model/intent witnesses required")
            if digest(path) != witness["sha256"]:
                raise ValueError("complete witness bytes changed")
        intent = sealed(Path(witnesses["intent_witness"]["path"]))
        expected_intent = dict(
            job,
            model_hash=model["model_hash"],
            world_hash=config["mesh_task_contact_world_hash"],
            declaration_hash=declaration["report_hash"],
        )
        if {k: v for k, v in intent.items() if k != "report_hash"} != expected_intent:
            raise ValueError("actual model and pre-physical job intent differ")
        starts = [
            i
            for i, e in enumerate(events)
            if e["event"] == "ATTEMPT_STARTED"
            and e["state"].get("active") == dict(job=job["job"], attempt=index)
        ]
        if len(starts) != 1 or starts[0] < 1:
            raise ValueError("one charged start per original job required")
        prior = events[starts[0] - 1]
        if (
            prior["event"] != "INTENT_PERSISTED_BEFORE_BODY"
            or prior["state"].get("next_intent_hash") != intent["report_hash"]
        ):
            raise ValueError("intent must be journaled before physical execution")
        if job["kind"] in ("evaluation", "retention"):
            freezes = [
                i
                for i, e in enumerate(events)
                if e["event"] == "ALL_THREE_UPDATED_ACTORS_FROZEN_BEFORE_EVALUATION"
            ]
            if len(freezes) != 1 or freezes[0] >= starts[0]:
                raise ValueError("all updated actors must be frozen before any evaluation")
        payload = witnesses["proof_payload"]
        if reader is None:
            reader = FixedPhysicalPayloadReader(
                Path(payload["path"]), expected_file_hash=payload["sha256"]
            )
        report = _sealed(Path(witnesses["report"]["path"]), physical_payload_reader=reader)
        commitment = json.loads(Path(witnesses["commitment"]["path"]).read_text())
        if (
            report["commitment_hash"] != hash_json(commitment)
            or report["report_hash"] != receipt["physical_report_hash"]
            or report["physical_trace_hash"] != witnesses["trace"]["sha256"]
            or witnesses["world"]["sha256"] != model["world_hash"]
            or sealed(Path(witnesses["model_witness"]["path"]), "model_hash") != model
            or commitment["cerebellar_torque_contract"]["model"] != model
            or commitment["mesh_task_contact_protocol"]["numerical_review"] != binding
            or commitment["cerebellar_torque_contract"]["sampling"]["sampling_seed"]
            != job["sampling_seed"]
        ):
            raise ValueError("report/model/world/sampling/review commitments differ")
        if (
            receipt["job"] != job["job"]
            or receipt["kind"] != job["kind"]
            or receipt["group"] != job["group"]
            or receipt.get("planning_context_hash") != job["context_hash"]
        ):
            raise ValueError("actual complete ordered context/job identity required")
        motor = report["executed_motor_policy"]["step_motor_proof"]["model"]
        plan = PublicFirstTouchTrainingPlan(
            tuple(range(20261060, 20261070)), motor["mean_model"]["model_hash"], 18
        )
        draw = plan.draws()[job["group"]]
        context = (
            None if job["context_index"] is None else planning_contexts(plan)[job["context_index"]]
        )
        if index == 0 and build_jobs(planning_contexts(plan), initial) != jobs:
            raise ValueError(
                "exact predefined three-seed jobs required, not an outcome-selected bank"
            )
        course = draw.course if context is None else context.course
        if (
            report["course"] != list(course)
            or report["seed"] != draw.seed
            or report["lane"] != draw.lane
            or motor["seed"] != draw.sampling_seed
            or receipt["candidate"] != catalog[job["option_index"]]
            or (None if context is None else context.context_hash) != job["context_hash"]
        ):
            raise ValueError("declared public course, motor RNG, context and option required")
        with np.load(witnesses["trace"]["path"], allow_pickle=False) as archive:
            trace = dict(archive)
        if job["kind"] == "zero_control" and job["pair_id"] in baseline_traces:
            with np.load(baseline_traces[job["pair_id"]]["path"], allow_pickle=False) as archive:
                if any(
                    trace[k].dtype != archive[k].dtype
                    or trace[k].shape != archive[k].shape
                    or trace[k].tobytes() != archive[k].tobytes()
                    for k in archive.files
                ):
                    raise ValueError(
                        "zero head changed actual original None-layer common trace bytes"
                    )
        outcome = label_first_touch(trace)
        review = sealed(Path(witnesses["independent_review"]["path"]))
        foundation = sealed(Path(witnesses["foundation_review"]["path"]))
        expected_draws = 3000 if job["kind"] == "collection" else 0
        alignment = require_first_touch_qualification(
            review,
            foundation,
            outcome,
            cerebellar_model_hash=model["model_hash"],
            cerebellar_sampling_draws=expected_draws,
        )
        numeric = reconstruct_native_cerebellar(report, commitment, trace)
        objective = training_return(trace)
        if (
            outcome.to_dict() != receipt["outcome"]
            or alignment != receipt["qualification_alignment"]
            or objective != receipt["objective"]
            or numeric is None
            or numeric["sampling_draws"] != expected_draws
            or numeric["independently_recomputed_substeps"] != 3000
            or receipt["cerebellar_sampling_draws"] != expected_draws
            or receipt["cerebellar_inference_calls"] != 3000
            or review["reviewed_report_hash"] != report["report_hash"]
            or foundation["physical_report_hash"] != report["report_hash"]
        ):
            raise ValueError("primitive outcome, full neural reconstruction and replay differ")
        encoder = sealed(Path(witnesses["encoder_witness"]["path"]))
        recorded = trace["foundation_neural_encoder_features"]
        if (
            recorded.shape != (300, 1, 640)
            or recorded.dtype != np.float32
            or encoder["recorded_features_hash"] != hash_bytes(recorded.tobytes())
            or encoder["compared_encoder_frames"] != 300
            or encoder["compared_features_per_frame"] != 640
            or encoder["actual_cpu_onnx_planner_calls"] != 15
            or not 0 <= encoder["maximum_absolute_error"] <= 1e-5
        ):
            raise ValueError("full planner/encoder consumption proof required")
        expected_events = [
            {k: e[k] for k in ("frame", "planner_mode", "reference_hash")}
            for e in report["default_locomotion_planner_events"]
        ]
        if encoder["planner_events"] != expected_events or [
            e["frame"] for e in expected_events
        ] != list(range(0, 300, 20)):
            raise ValueError("encoder comparison must bind complete actual planner history")
        if job["kind"] == "zero_control":
            controls[job["pair_id"]] = outcome.to_dict()
        elif job["kind"] in ("evaluation", "retention"):
            evaluations[job["actor_seed"]][job["pair_id"]] = outcome.to_dict()
        draws += expected_draws
        inferences += 3000
        rows.append(
            dict(
                job=job["job"],
                receipt_hash=receipt["receipt_hash"],
                outcome=outcome.to_dict(),
                objective=objective,
            )
        )
        del trace, report
    # Reconstruct each complete admitted batch and the exact persistent numeric
    # checkpoint. This is repeat computation, not additional learning data.
    with np.load(root / "anchor-features.npz", allow_pickle=False) as archive:
        anchors = archive["features"]
    if (
        anchors.dtype != expected_anchors.dtype
        or anchors.shape != expected_anchors.shape
        or anchors.tobytes() != expected_anchors.tobytes()
    ):
        raise ValueError("anchors must be exact inputs from passing original None controls")
    reconstructed_learning_receipts = []
    for seed in SEEDS:
        expected = [j for j in jobs if j["kind"] == "collection" and j["actor_seed"] == seed]
        paths = [root / f"receipt-{i:04d}.json" for i, j in enumerate(jobs) if j in expected]
        arrays, batch = build_batch(
            paths,
            actor=initial[seed],
            expected_jobs=expected,
            declaration_hash=declaration["report_hash"],
            evidence_root=root,
            physical_payload_reader=reader,
        )
        if batch != sealed(root / f"batch-{seed}.json"):
            raise ValueError("independently reconstructed complete training batch differs")
        before = sealed(root / f"initial-checkpoint-{seed}.json", "checkpoint_hash")
        rebuilt = fit_update(
            before, arrays, batch_hash=batch["report_hash"], anchor_features=anchors, steps=4
        )
        if rebuilt != checkpoints[seed]:
            raise ValueError("exact numeric actor/critic/Adam update reconstruction differs")
        drift = dict(
            collection=same_input_torque_drift(initial[seed], trained[seed], arrays["features"]),
            anchors=same_input_torque_drift(initial[seed], trained[seed], anchors),
        )
        drift["report_hash"] = hash_json(drift)
        if drift != sealed(root / f"same-input-torque-drift-{seed}.json"):
            raise ValueError("actual same-input torque drift changed")
        reconstructed_learning_receipts.append(rebuilt["learning_receipt"])
        del arrays, rebuilt
    accounting = optimizer_audit_accounting(reconstructed_learning_receipts)
    if (
        complete["native_bodies"] != 44
        or complete["native_substeps"] != 132000
        or complete["independent_cpu_substeps"] != 132000
        or complete["actual_cerebellar_sampling_draws"] != draws
        or draws != 36000
        or complete["actual_cerebellar_inferences"] != inferences
        or inferences != 132000
        or complete["accepted_joint_optimizer_steps"]
        != accounting["accepted_training_optimizer_steps"]
    ):
        raise ValueError("actual whole-study physical/update accounting differs")
    for name, expected in pins.items():
        if digest(Path(name)) != expected:
            raise ValueError("frozen inputs changed during audit")
    report = dict(
        schema="soccer.rsi.mesh_cerebellar_terminal_readback.v1",
        complete_hash=complete["report_hash"],
        ledger_tip=previous,
        rows=rows,
        paired_development={str(s): pairing.assess_pairs(controls, evaluations[s]) for s in SEEDS},
        audit_source_hash=digest(Path(__file__)),
        pairing_source_hash=digest(pairing_path),
        **accounting,
        independently_recomputed_neural_substeps=inferences,
        full_source_and_witness_bytes_verified=True,
        actual_persistent_actor_critic_adam_reconstructed=True,
        new_physical_replays=0,
        new_training_episodes=0,
        new_candidate_models=0,
        promotion_authorized=False,
        hardware_authorized=False,
        teacher_qualified=False,
    )
    report["report_hash"] = hash_json(report)
    write_once(args.output, report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
