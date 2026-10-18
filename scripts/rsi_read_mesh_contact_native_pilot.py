"""Independent terminal receipt/ledger readback, zero new integration steps."""

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=("contact-three", "original-public-four"), default="contact-three"
    )
    args = parser.parse_args()
    public_four = args.profile == "original-public-four"
    count = 4 if public_four else 3
    status = subprocess.check_output(
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
        ],
        text=True,
    )
    if set(status.splitlines()) != {"MainPID=0", "ActiveState=inactive", "Result=success"}:
        raise ValueError("finished worker required; partial output is not completion")
    if args.output.exists():
        raise ValueError("unused terminal audit output required")
    config = json.loads(args.config.read_text())
    source, core, root = (Path(config[k]) for k in ("source_root", "core_root", "output_root"))
    sys.path[:0] = [str(source / "src"), str(source), str(core / "src")]
    import numpy as np

    from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
    from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
    from rosclaw_soccer.rsi.first_touch_qualification import require_first_touch_qualification
    from rosclaw_soccer.rsi.fixed_physical_payload_reader import FixedPhysicalPayloadReader
    from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
    from scripts.rsi_atomic_artifacts import write_once

    def sealed(path, key="report_hash"):
        value = json.loads(path.read_text())
        if value[key] != hash_json({k: v for k, v in value.items() if k != key}):
            raise ValueError("source document seal changed")
        return value

    declaration, complete = (sealed(root / n) for n in ("declaration.json", "complete.json"))
    if (
        complete["declaration_hash"] != declaration["report_hash"]
        or complete["new_native_bodies"] != count
        or complete["native_physical_substeps"] != count * 3000
        or complete["independent_cpu_substeps"] != count * 3000
        or complete["independent_foundation_calls"] != count * 300
        or complete["new_optimizer_updates"] != 0
    ):
        raise ValueError("exact declared whole-body accounting required")
    if public_four and (
        complete.get("schema") != "soccer.rsi.mesh_public_draw_complete.v1"
        or declaration.get("schema") != "soccer.rsi.mesh_public_draw_declaration.v1"
        or config.get("groups") != [4, 8, 32, 56]
        or config.get("max_attempts") != 4
        or config.get("new_physical_budget") != 4
        or complete.get("additional_encoder_reference_calls") != 1200
    ):
        raise ValueError("exact original-public-four schema, groups and evidence counts required")
    if any(k not in declaration or declaration[k] != v for k, v in config.items()):
        raise ValueError("same frozen configuration required")
    pins = dict(declaration["input_file_hashes"])
    pins.update(declaration["prepared_pins"])
    for name, expected in pins.items():
        if digest(Path(name)) != expected:
            raise ValueError("source/input pin changed")
    for name, expected in declaration["source_heads"].items():
        if (
            subprocess.check_output(["git", "-C", name, "rev-parse", "HEAD"], text=True).strip()
            != expected
        ):
            raise ValueError("source checkout head changed")
    previous = "GENESIS"
    events = []
    with sqlite3.connect((root / "campaign.sqlite").as_uri() + "?mode=ro", uri=True) as db:
        db.execute("BEGIN")
        for seq, link, body, value in db.execute("SELECT * FROM events ORDER BY seq"):
            document = json.loads(body)
            if (
                seq != len(events) + 1
                or link != previous
                or value != hash_json(dict(seq=seq, previous=link, body=document))
            ):
                raise ValueError("whole durable event chain required")
            previous = value
            events.append(document)
    if not events:
        raise ValueError("nonempty durable ledger required")
    final = events[-1]["state"]
    names = (
        tuple(f"original-public-{g}" for g in (4, 8, 32, 56))
        if public_four
        else ("old64-option3", "new96-option3", "unsafe104-option7")
    )
    terminal_phase = (
        "PUBLIC_DRAW_EVALUATION_COMPLETE_NO_PROMOTION"
        if public_four
        else "CONTACT_WORLD_DIAGNOSTIC_COMPLETE_NO_PROMOTION"
    )
    if (
        final["phase"] != terminal_phase
        or final["active"] is not None
        or final["attempts"] != count
        or set(final["completed"]) != set(names)
    ):
        raise ValueError("all charged attempts must finish in original order")
    rows = []
    for index, name in enumerate(names):
        receipt = sealed(root / f"receipt-{index:04d}.json", "receipt_hash")
        if (
            receipt != complete["results"][index]
            or receipt != final["completed"][name]
            or receipt["job"] != name
        ):
            raise ValueError("completion, ledger and actual ordered receipts must agree")
        witnesses = dict(receipt["witnesses"])
        witnesses.update({k: receipt[k] for k in ("intent_witness", "encoder_witness")})
        for witness in witnesses.values():
            if digest(Path(witness["path"])) != witness["sha256"]:
                raise ValueError("whole physical witness file changed")
        reader = FixedPhysicalPayloadReader(
            Path(witnesses["proof_payload"]["path"]),
            expected_file_hash=witnesses["proof_payload"]["sha256"],
        )
        report = _sealed(Path(witnesses["report"]["path"]), physical_payload_reader=reader)
        commitment = json.loads(Path(witnesses["commitment"]["path"]).read_text())
        if hash_json(commitment) != report["commitment_hash"]:
            raise ValueError("actual commitment hash must bind the sealed physical report")
        expected_world = config["mesh_task_contact_world_hash"]
        for doc in (report, commitment):
            if (
                doc["compiled_model_hash"] != expected_world
                or doc.get("mesh_task_contact_protocol", {}).get("expected_world_hash")
                != expected_world
            ):
                raise ValueError("new world must bind actual report and commitment")
        if (
            report["report_hash"] != receipt["physical_report_hash"]
            or digest(Path(witnesses["world"]["path"])) != expected_world
            or report["physical_trace_hash"] != witnesses["trace"]["sha256"]
        ):
            raise ValueError("actual report/world/trace identity differs")
        with np.load(witnesses["trace"]["path"], allow_pickle=False) as archive:
            trace = dict(archive)
        if public_four:
            encoder = sealed(Path(witnesses["encoder_witness"]["path"]))
            recorded = np.asarray(trace.get("foundation_neural_encoder_features"))
            required_counts = dict(
                actual_cpu_onnx_planner_calls=15,
                compared_encoder_frames=300,
                compared_features_per_frame=640,
            )
            if (
                encoder.get("schema") != "soccer.rsi.sonic_encoder_reference_consumption.v1"
                or any(
                    type(encoder.get(k)) is not int or encoder[k] != v
                    for k, v in required_counts.items()
                )
                or recorded.shape != (300, 1, 640)
                or recorded.dtype != np.float32
                or not np.isfinite(recorded).all()
                or encoder.get("recorded_features_hash") != hash_bytes(recorded.tobytes())
                or encoder.get("recorded_features_dtype") != "float32"
                or type(encoder.get("maximum_absolute_error")) is not float
                or not np.isfinite(encoder["maximum_absolute_error"])
                or not 0 <= encoder["maximum_absolute_error"] <= 1e-5
            ):
                raise ValueError(
                    "whole raw encoder feature bytes and independent consumption counts required"
                )
            expected_events = [
                {k: event[k] for k in ("frame", "planner_mode", "reference_hash")}
                for event in report["default_locomotion_planner_events"]
            ]
            if encoder.get("planner_events") != expected_events or [
                e["frame"] for e in expected_events
            ] != list(range(0, 300, 20)):
                raise ValueError(
                    "encoder reference audit must bind actual complete planner history"
                )
        outcome = label_first_touch(trace)
        review = sealed(Path(witnesses["independent_review"]["path"]))
        foundation = sealed(Path(witnesses["foundation_review"]["path"]))
        alignment = require_first_touch_qualification(review, foundation, outcome)
        if (
            outcome.to_dict() != receipt["outcome"]
            or alignment != receipt["qualification_alignment"]
            or review["reviewed_report_hash"] != report["report_hash"]
            or foundation["physical_report_hash"] != report["report_hash"]
            or foundation["physical_trace_hash"] != witnesses["trace"]["sha256"]
        ):
            raise ValueError("raw outcome and independently reconstructed body/Foundation differ")
        intended = declaration["jobs"][index]
        intent = sealed(Path(witnesses["intent_witness"]["path"]))
        if public_four:
            from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
            from rosclaw_soccer.rsi.native_first_touch_initial_state import (
                NativeFirstTouchInitialState,
            )
            from rosclaw_soccer.rsi.public_first_touch_training_plan import (
                PublicFirstTouchTrainingPlan,
            )

            view = report["executed_motor_policy"]["step_motor_proof"]["model"]
            plan = PublicFirstTouchTrainingPlan(
                tuple(range(20261060, 20261070)), view["mean_model"]["model_hash"], 18
            )
            group = (4, 8, 32, 56)[index]
            draw = plan.draws()[group]
            if (
                intended != dict(job=name, group=group, candidate=None, course=list(draw.course))
                or receipt["group"] != group
                or receipt["candidate"] is not None
                or report.get("bounded_control_profile") is not None
                or report["approach_mode"] != "legacy"
                or report["swing_mode"] != "legacy"
                or view["seed"] != draw.sampling_seed
            ):
                raise ValueError("unperturbed public draw, seed and original None control required")
            context = dict(course=list(draw.course), source_seed=draw.seed, source_lane=draw.lane)
            NativeFirstTouchInitialState(draw.course, G1SonicRunupController.default_angles).verify(
                report["course"], trace["canonical_qpos"][0, 0], trace["canonical_qvel"][0, 0]
            )
            expected_intent = dict(
                intended, world_hash=expected_world, declaration_hash=declaration["report_hash"]
            )
        else:
            context = intended["context"]
            expected_intent = dict(
                job=name,
                context_hash=receipt["planning_context_hash"],
                option_index=intended["option_index"],
                candidate_hash=hash_json(intended["candidate"]),
                world_hash=expected_world,
                declaration_hash=declaration["report_hash"],
            )
        if {k: v for k, v in intent.items() if k != "report_hash"} != expected_intent:
            raise ValueError("candidate, context and world must be fixed before body execution")
        starts = [
            i
            for i, event in enumerate(events)
            if event["event"] == "ATTEMPT_STARTED"
            and event["state"].get("active") == dict(job=name, attempt=index)
        ]
        if len(starts) != 1 or starts[0] < 1:
            raise ValueError("one charged body start required per declared job")
        prior = events[starts[0] - 1]
        if (
            prior["event"]
            != (
                "PUBLIC_DRAW_INTENT_COMMITTED_BEFORE_BODY"
                if public_four
                else "NEW_WORLD_INTENT_COMMITTED_BEFORE_BODY"
            )
            or prior["state"].get("next_intent_hash") != intent["report_hash"]
            or prior["state"].get("active") is not None
        ):
            raise ValueError("intent must be journaled before the charged body start")
        if receipt["candidate"] != intended["candidate"]:
            raise ValueError("actual candidate differs from the frozen job")
        if (
            report["course"] != context["course"]
            or report["seed"] != context["source_seed"]
            or report["lane"] != context["source_lane"]
            or not np.array_equal(
                trace["canonical_qpos"][0, 0, 36:39], [*context["course"][:2], 0.13]
            )
        ):
            raise ValueError("declared initial course must enter physical trace")
        rows.append(
            dict(
                job=name,
                receipt_hash=receipt["receipt_hash"],
                outcome=outcome.to_dict(),
                world_hash=expected_world,
                physical_report_hash=report["report_hash"],
            )
        )
    report = dict(
        schema=(
            "soccer.rsi.mesh_original_public_terminal_readback.v1"
            if public_four
            else "soccer.rsi.mesh_contact_terminal_readback.v1"
        ),
        complete_hash=complete["report_hash"],
        ledger_tip=previous,
        rows=rows,
        reader_source_hash=digest(Path(__file__)),
        whole_input_and_witness_files_verified=True,
        shared_payload_logical_seals_rebuilt=True,
        prior_independent_replays_authenticated=True,
        new_physical_replays=0,
        new_optimizer_updates=0,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    if public_four:
        report.update(
            source_native_bodies=4,
            source_encoder_compared_frames=1200,
            source_independent_planner_onnx_calls=60,
            source_neural_encoder_recomputations=1200,
        )
    report["report_hash"] = hash_json(report)
    write_once(args.output, report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
