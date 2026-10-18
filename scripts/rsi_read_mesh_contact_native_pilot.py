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


def bounded_profile(profile):
    if profile == "contact-exit-four":
        spec = bounded_profile("release-four")
        spec.update(
            declaration_schema="soccer.rsi.contact_exit_torque_mpc_pilot_declaration.v1",
            complete_schema="soccer.rsi.contact_exit_torque_mpc_pilot_complete.v1",
            readback_schema="soccer.rsi.contact_exit_torque_mpc_terminal_readback.v1",
            phase="CONTACT_EXIT_TORQUE_MPC_COMPLETE_NOT_QUALIFIED_TEACHER",
            jobs=[
                dict(job, job="contact-exit-" + job["pair_id"], role="teacher")
                for job in spec["jobs"]
            ],
        )
        return spec
    if profile == "release-four":
        return dict(
            count=4,
            groups=[8, 56, 64, 65],
            declaration_schema="soccer.rsi.swing_release_ablation_pilot_declaration.v1",
            complete_schema="soccer.rsi.swing_release_ablation_pilot_complete.v1",
            readback_schema="soccer.rsi.swing_release_ablation_terminal_readback.v1",
            phase="SWING_RELEASE_ABLATION_COMPLETE_NOT_OPTIMIZED_RECOVERY",
            jobs=[
                dict(
                    job=f"release-zero-{pair}",
                    role="ablation",
                    pair_id=pair,
                    context_index=context,
                    group=group,
                    option_index=4,
                )
                for pair, group, context in (
                    ("raw-8", 8, None),
                    ("raw-56", 56, None),
                    ("context-0", 64, 0),
                    ("context-1", 65, 1),
                )
            ],
        )
    if profile == "binding-four":
        return dict(
            count=4,
            groups=[4, 8],
            declaration_schema="soccer.rsi.generated_binding_native_declaration.v1",
            complete_schema="soccer.rsi.generated_binding_native_complete.v1",
            readback_schema="soccer.rsi.generated_binding_native_terminal_readback.v1",
            phase="OPTIONAL_BINDING_NUMERIC_NONINTERFERENCE_COMPLETE_NOT_SKILL_GROWTH",
            jobs=[
                dict(job=f"g{g}-generated-{flag}", group=g, generated=flag)
                for g, flag in ((4, False), (4, True), (8, True), (8, False))
            ],
        )
    if profile == "catalog-sixteen":
        return dict(
            count=16,
            groups=[8, 56],
            declaration_schema="soccer.rsi.mesh_contact_catalog_declaration.v1",
            complete_schema="soccer.rsi.mesh_contact_catalog_complete.v1",
            readback_schema="soccer.rsi.mesh_contact_catalog_terminal_readback.v1",
            phase="CATALOG_REACHABILITY_COMPLETE_NOT_ROBUST_TEACHER",
            jobs=[
                dict(job=f"option-{o}-raw-{g}", option_index=o, group=g)
                for o in range(8)
                for g in (8, 56)
            ],
        )
    if profile == "teacher-eighteen":
        return dict(
            count=18,
            groups=[4, 8, 32, 56],
            declaration_schema="soccer.rsi.mesh_option4_teacher_screen_declaration.v1",
            complete_schema="soccer.rsi.mesh_option4_teacher_screen_complete.v1",
            readback_schema="soccer.rsi.mesh_option4_teacher_screen_terminal_readback.v1",
            phase="TEACHER_SCREEN_COMPLETE_NOT_AUTOMATIC_ADMISSION",
            jobs=[
                dict(
                    job=f"{role}-context-{i}",
                    role=role,
                    pair_id=f"context-{i}",
                    context_index=i,
                    group=g,
                    option_index=o,
                )
                for i, g in enumerate((64, 65, 68, 69, 72, 73, 76, 77))
                for role, o in (("control", 0), ("teacher", 4))
            ]
            + [
                dict(
                    job=f"teacher-raw-{g}",
                    role="teacher",
                    pair_id=f"raw-{g}",
                    context_index=None,
                    group=g,
                    option_index=4,
                )
                for g in (4, 32)
            ],
        )
    raise ValueError("explicit new bounded study profile required")


def verify_persisted_preparation(declaration, prepared):
    fields = ("prepared_pins", "source_heads")
    present = [key in declaration for key in fields]
    if any(present) and not all(present):
        raise ValueError("complete prepared pins and source heads required together")
    if not any(present):
        return False  # Explicitly disclosed legacy producer limitation.
    if (
        declaration["prepared_pins"] != prepared.PINS
        or declaration["source_heads"] != prepared.HEADS
    ):
        raise ValueError("original persisted preparation differs from current exact reconstruction")
    return True


def verify_release_prefix(trace, original_trace, original_outcome, outcome, prefix, baseline_hash):
    """Independent byte readback, not the producer's assertion or its comparison helper."""
    import numpy as np

    from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
    from rosclaw_soccer.sim.contracts import hash_bytes

    cutoff = original_outcome["first_foot_contact_frame"]
    if type(cutoff) is not int or not 0 <= cutoff < 300:
        raise ValueError("original observed foot-first contact required")
    cutoff += 1
    first = np.flatnonzero(np.any(original_trace["force_n"][:, 0] > 1, axis=1))
    if (
        not len(first)
        or int(first[0]) != cutoff - 1
        or outcome.first_foot_contact_frame != cutoff - 1
        or set(trace) != set(original_trace)
        or any(
            trace[k].shape != original_trace[k].shape
            or trace[k].dtype != original_trace[k].dtype
            or trace[k].ndim == 0
            or trace[k].shape[0] != 300
            or trace[k][:cutoff].tobytes() != original_trace[k][:cutoff].tobytes()
            for k in trace
        )
        or label_first_touch(original_trace).to_dict() != original_outcome
    ):
        raise ValueError("independently compared complete first-contact prefixes differ")
    if (
        prefix.get("schema") != "soccer.rsi.first_contact_prefix_parity.v1"
        or prefix.get("baseline_receipt_hash") != baseline_hash
        or prefix.get("prefix_frames") != cutoff
        or prefix.get("prefix_array_hashes")
        != {k: hash_bytes(trace[k][:cutoff].tobytes()) for k in sorted(trace)}
        or prefix.get("all_original_array_prefixes_bit_equal") is not True
    ):
        raise ValueError("prefix witness must bind the actual original array bytes")


def verify_contact_exit_reconstruction(report, commitment, receipt, review, source):
    """Authenticate original replay evidence; this reader performs no physics."""
    contract = commitment["contact_exit_torque_mpc"]
    expected_sources = {
        "teacher": source / "src/rosclaw_soccer/rsi/contact_exit_torque_mpc.py",
        "forecast": source / "src/rosclaw_soccer/physics/g1_pd_torque_forecast.py",
        "kinematics": source / "src/rosclaw_soccer/physics/g1_joint_task_kinematics.py",
    }
    counters = report["contact_exit_mpc_counters"]
    active = sum(p["mode"] != "INACTIVE" for p in report["contact_exit_mpc_plans"])
    reconstruction = review.get("contact_exit_torque_mpc_review", {})
    if (
        contract.get("source_hashes") != {k: digest(p) for k, p in expected_sources.items()}
        or contract.get("forecast_budget_steps") != 6000
        or contract.get("maximum_offset_nm") != 12.0
        or contract.get("maximum_hard_limit_fraction") != 0.2
        or contract.get("maximum_offset_change_per_control_frame_nm") != 2.0
        or receipt.get("contact_exit_mpc_counters") != counters
        or receipt.get("forecast_physical_substeps") != counters["integrated_forecast_steps"]
        or reconstruction.get("schema") != "soccer.rsi.contact_exit_torque_mpc_reconstruction.v1"
        or reconstruction.get("contract") != contract
        or reconstruction.get("planning_counters") != counters
        or reconstruction.get("frames_reconstructed") != 300
        or reconstruction.get("optimization_reconstructed_with_same_pinned_implementation")
        is not True
        or reconstruction.get("independent_optimization_implementation") is not False
        or reconstruction.get("locally_bitexact_active_intervals") != active
        or reconstruction.get("chosen_control_forecast_attempts") != active
        or reconstruction.get("chosen_control_attempted_forecast_steps") != active * 10
        or reconstruction.get("chosen_control_integrated_forecast_steps") != active * 10
        or reconstruction.get("source_numeric_replay_tolerance") != 1e-8
        or reconstruction.get("hardware_authorized") is not False
        or reconstruction.get("promotion_authorized") is not False
        or type(reconstruction.get("source_end_state_hash_matches")) is not int
        or not 0 <= reconstruction["source_end_state_hash_matches"] <= active
    ):
        raise ValueError("source-bound contact-exit optimization/actual replay accounting required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream-audit-source", type=Path)
    parser.add_argument("--baseline-screen-readback", type=Path)
    parser.add_argument("--baseline-screen-audit-source", type=Path)
    parser.add_argument(
        "--profile",
        choices=(
            "contact-three",
            "original-public-four",
            "binding-four",
            "catalog-sixteen",
            "teacher-eighteen",
            "release-four",
            "contact-exit-four",
        ),
        default="contact-three",
    )
    args = parser.parse_args()
    public_four = args.profile == "original-public-four"
    teacher_screen = args.profile == "teacher-eighteen"
    torque_mpc = args.profile == "contact-exit-four"
    release_ablation = args.profile in {"release-four", "contact-exit-four"}
    context_study = teacher_screen or release_ablation
    spec = (
        bounded_profile(args.profile)
        if args.profile
        in (
            "binding-four",
            "catalog-sixteen",
            "teacher-eighteen",
            "release-four",
            "contact-exit-four",
        )
        else None
    )
    count = spec["count"] if spec is not None else 4 if public_four else 3
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
            "-p",
            "LoadState",
        ],
        text=True,
    )
    if set(status.splitlines()) != {
        "MainPID=0",
        "ActiveState=inactive",
        "Result=success",
        "LoadState=loaded",
    }:
        raise ValueError("finished worker required; partial output is not completion")
    started_path = args.output.with_suffix(".started.json")
    if args.output.exists() or started_path.exists():
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

    write_once(
        started_path,
        dict(
            schema="soccer.rsi.mesh_terminal_audit_started.v1",
            profile=args.profile,
            reader_source_hash=digest(Path(__file__)),
            configuration_file_hash=digest(args.config),
            automatic_retry_authorized=False,
            hardware_authorized=False,
        ),
    )

    audit_inputs = {}

    def sealed(path, key="report_hash"):
        audit_inputs[str(path)] = digest(path)
        value = json.loads(path.read_text())
        if value[key] != hash_json({k: v for k, v in value.items() if k != key}):
            raise ValueError("source document seal changed")
        return value

    declaration, complete = (sealed(root / n) for n in ("declaration.json", "complete.json"))
    if (
        complete.get("declaration_hash") != declaration["report_hash"]
        or complete.get("new_native_bodies") != count
        or complete.get("native_substeps" if spec else "native_physical_substeps") != count * 3000
        or complete.get("independent_cpu_substeps") != count * 3000
        or (spec is None and complete.get("independent_foundation_calls") != count * 300)
        or complete.get("new_optimizer_updates") != 0
    ):
        raise ValueError("exact declared whole-body accounting required")
    if spec is not None and (
        complete.get("schema") != spec["complete_schema"]
        or declaration.get("schema") != spec["declaration_schema"]
        or config.get("groups") != spec["groups"]
        or config.get("max_attempts") != count
        or config.get("new_physical_budget") != count
        or declaration.get("jobs") != spec["jobs"]
        or len(complete.get("results", [])) != count
        or any(
            complete.get(k) is not False
            for k in ("teacher_qualified", "hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError(
            "exact complete bounded study, not a partial or retagged protocol required"
        )
    if teacher_screen and (
        args.upstream_audit_source is None
        or config.get("upstream_terminal_audit_required") is not True
        or config.get("selected_option_index") != 4
        or complete.get("independent_course_count") != 12
        or complete.get("paired_endpoint_count") != 24
        or complete.get("reused_endpoints") != declaration.get("reused_endpoints")
    ):
        raise ValueError("exact expanded teacher screen and upstream audit binding required")
    if release_ablation and (
        args.upstream_audit_source is None
        or args.baseline_screen_readback is None
        or args.baseline_screen_audit_source is None
        or config.get("suppress_postcontact_swing_release") is not (not torque_mpc)
        or config.get("use_contact_exit_torque_mpc", False) is not torque_mpc
        or config.get("selected_option_index") != 4
        or config.get("upstream_terminal_audit_required") is not True
        or complete.get("independent_course_count") != 4
        or complete.get("paired_endpoint_count") != 8
        or complete.get("reused_endpoints") != declaration.get("reused_endpoints")
        or set(config.get("required_receipt_witnesses", []))
        != {"intent_witness", "encoder_witness", "prefix_witness"}
    ):
        raise ValueError("exact four-course ablation with both terminal upstream audits required")
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
    prepared, preparation_was_persisted = None, False
    if spec is not None:
        from rosclaw_soccer.rsi.first_touch_campaign_worker import prepare
        from rosclaw_soccer.rsi.reviewed_contact_world import ReviewedContactWorld

        prepared, _, source_manifest, actual = prepare(args.config)
        if actual != config:
            raise ValueError("same original preparation configuration required")
        if release_ablation:
            for project in config["baseline_sources"]:
                previous_source = Path(project["root"])
                prepared.HEADS[str(previous_source)] = project["head"]
                prepared.add_pins(
                    {
                        str(path): digest(path)
                        for folder in (previous_source / "src", previous_source / "scripts")
                        for path in folder.rglob("*.py")
                    }
                )
        preparation_was_persisted = verify_persisted_preparation(declaration, prepared)
        pins.update(prepared.PINS)
        heads = prepared.HEADS
        binding = ReviewedContactWorld(
            Path(config["nominal_readback"]), config["nominal_readback_file_hash"]
        ).verify(config["mesh_task_contact_world_hash"])
        if binding != declaration["numerical_world_review"]:
            raise ValueError("original numerical world review binding changed")
    else:
        pins.update(declaration["prepared_pins"])
        heads = declaration["source_heads"]
    for name, expected in pins.items():
        if digest(Path(name)) != expected:
            raise ValueError("source/input pin changed")
    for name, expected in heads.items():
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
        tuple(j["job"] for j in spec["jobs"])
        if spec is not None
        else tuple(f"original-public-{g}" for g in (4, 8, 32, 56))
        if public_four
        else ("old64-option3", "new96-option3", "unsafe104-option7")
    )
    terminal_phase = (
        spec["phase"]
        if spec is not None
        else "PUBLIC_DRAW_EVALUATION_COMPLETE_NO_PROMOTION"
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
    if context_study and final.get("declaration_hash") != hash_json(declaration):
        raise ValueError("teacher-screen ledger must bind the whole original declaration")
    rows = []
    shared_reader, shared_reader_witness = None, None
    original = sealed(Path(config["nominal_complete"])) if spec is not None else None
    references = {} if original is None else {r["group"]: r for r in original["results"]}
    if spec is not None:
        audited = sealed(Path(config["nominal_readback"]))
        if (
            original["report_hash"] != audited["complete_hash"]
            or audited["report_hash"] != config["nominal_readback_hash"]
        ):
            raise ValueError("original four controls must bind original numerical review")
        from rosclaw_soccer.rsi.portfolio_physical_study import validate_option

        fit = sealed(Path(config["portfolio_fit"]))
        if fit["report_hash"] != config["portfolio_fit_hash"]:
            raise ValueError("original option catalog seal required")
        catalog = [validate_option(r["candidate"]) for r in fit["options"]]
        if len(catalog) != 8 or catalog[0] is not None:
            raise ValueError("whole original catalog including None required")
        if (
            args.profile
            in ("catalog-sixteen", "teacher-eighteen", "release-four", "contact-exit-four")
            and declaration["options"] != catalog
        ):
            raise ValueError("actual entire catalog differs")
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
        if release_ablation:
            witnesses["prefix_witness"] = receipt["prefix_witness"]
        if context_study and spec["jobs"][index]["context_index"] is not None:
            witnesses["planning_context_witness"] = receipt["planning_context_witness"]
        for witness in witnesses.values():
            if spec is not None and (
                not Path(witness["path"]).resolve().is_relative_to(root.resolve())
                or Path(witness["path"]).is_symlink()
            ):
                raise ValueError("actual study-owned ordinary witnesses required")
            if digest(Path(witness["path"])) != witness["sha256"]:
                raise ValueError("whole physical witness file changed")
            if spec is not None:
                if witness["path"] in pins and pins[witness["path"]] != witness["sha256"]:
                    raise ValueError("one ordinary witness cannot have two identities")
                pins[witness["path"]] = witness["sha256"]
        if spec is not None:
            if shared_reader_witness is None:
                shared_reader_witness = witnesses["proof_payload"]
                shared_reader = FixedPhysicalPayloadReader(
                    Path(shared_reader_witness["path"]),
                    expected_file_hash=shared_reader_witness["sha256"],
                )
            if witnesses["proof_payload"] != shared_reader_witness:
                raise ValueError(
                    "all new bounded-study episodes require the same fixed numerical mean"
                )
            reader = shared_reader
        else:
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
        if public_four or spec is not None:
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
        if release_ablation:
            from rosclaw_soccer.rsi.swing_release_ablation import (
                release_ablation_contract,
                require_zero_postcontact_swing,
            )

            if torque_mpc:
                from rosclaw_soccer.rsi.contact_exit_torque_evidence import (
                    require_contact_exit_capture,
                )

                require_contact_exit_capture(report, commitment, trace)
                if receipt.get("contact_exit_torque_mpc") != commitment["contact_exit_torque_mpc"]:
                    raise ValueError("contact-exit receipt must bind actual contract")
            else:
                tag = release_ablation_contract(expected_world)
                if any(
                    doc.get("swing_release_ablation") != tag
                    for doc in (report, commitment, receipt)
                ):
                    raise ValueError("actual report, commitment and receipt must bind intervention")
                require_zero_postcontact_swing(commitment, trace)
            baseline_item = config["baseline_receipts"][index]
            baseline = sealed(Path(baseline_item["path"]), "receipt_hash")
            if (
                baseline_item["pair_id"] != spec["jobs"][index]["pair_id"]
                or digest(Path(baseline_item["path"])) != baseline_item["file_hash"]
                or baseline["receipt_hash"] != baseline_item["receipt_hash"]
                or "swing_release_ablation" in baseline
            ):
                raise ValueError("exact original passing and unsafe baseline receipts required")
            for witness in baseline["witnesses"].values():
                if digest(Path(witness["path"])) != witness["sha256"]:
                    raise ValueError("original baseline witness bytes changed")
                pins[witness["path"]] = witness["sha256"]
            with np.load(baseline["witnesses"]["trace"]["path"], allow_pickle=False) as archive:
                original_trace = dict(archive)
            prefix = sealed(Path(witnesses["prefix_witness"]["path"]))
            prefix_trace = trace
            if torque_mpc:
                if set(trace) - set(original_trace) != {
                    "contact_exit_mpc_requested_offset_nm",
                    "contact_exit_mpc_applied_offset_nm",
                    "contact_exit_mpc_forecast_steps",
                }:
                    raise ValueError(
                        "only declared contact-exit capture may extend original arrays"
                    )
                prefix_trace = {k: trace[k] for k in original_trace if k in trace}
            verify_release_prefix(
                prefix_trace,
                original_trace,
                baseline["outcome"],
                outcome,
                prefix,
                baseline["receipt_hash"],
            )
        review = sealed(Path(witnesses["independent_review"]["path"]))
        if torque_mpc:
            verify_contact_exit_reconstruction(report, commitment, receipt, review, source)
        foundation = sealed(Path(witnesses["foundation_review"]["path"]))
        alignment = require_first_touch_qualification(review, foundation, outcome)
        if spec is not None and (
            report["source_hash"]
            != digest(source / "src/rosclaw_soccer/rsi/native_first_touch_episode.py")
            or review["source_hash"]
            != digest(source / "src/rosclaw_soccer/rsi/cpu_motor_transfer_evidence.py")
            or foundation["audit_source_hash"]
            != digest(source / "src/rosclaw_soccer/rsi/foundation_torch_capture_audit.py")
            or type(receipt["physical_substeps"]) is not int
            or receipt["physical_substeps"] != 3000
            or type(receipt["foundation_calls"]) is not int
            or receipt["foundation_calls"] != 300
            or "cerebellar_torque_contract" in report
        ):
            raise ValueError("exact producer physics, CPU and Foundation audit sources required")
        if (
            outcome.to_dict() != receipt["outcome"]
            or alignment != receipt["qualification_alignment"]
            or review["reviewed_report_hash"] != report["report_hash"]
            or foundation["physical_report_hash"] != report["report_hash"]
            or foundation["physical_trace_hash"] != witnesses["trace"]["sha256"]
        ):
            raise ValueError("raw outcome and independently reconstructed body/Foundation differ")
        intended = declaration["jobs"][index]
        if context_study and any(
            receipt.get(k) != intended[k]
            for k in ("role", "pair_id", "context_index", "option_index")
        ):
            raise ValueError("actual receipt must retain its predeclared role and paired course")
        intent = sealed(Path(witnesses["intent_witness"]["path"]))
        if public_four or spec is not None:
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
            group = spec["jobs"][index]["group"] if spec is not None else (4, 8, 32, 56)[index]
            draw = plan.draws()[group]
            planning = None
            if context_study:
                from rosclaw_soccer.rsi.public_physics_option_planning import planning_contexts

                if declaration.get("contexts") != [
                    dict(context_hash=c.context_hash, course=list(c.course))
                    for c in planning_contexts(plan)
                ]:
                    raise ValueError("whole predeclared eight-context source bank differs")
            if context_study and intended["context_index"] is not None:
                from dataclasses import asdict

                from rosclaw_soccer.rsi.public_physics_option_planning import (
                    planning_contexts,
                    validate_context_source,
                )

                planning = planning_contexts(plan)[intended["context_index"]]
                validate_context_source(planning, plan, group=group)
                document = sealed(Path(witnesses["planning_context_witness"]["path"]))
                expected_planning = dict(
                    schema="soccer.rsi.public_planning_execution_context.v1",
                    context=json.loads(json.dumps(asdict(planning))),
                    context_hash=planning.context_hash,
                    source_plan_hash=plan.contract()["plan_hash"],
                    stage="SELECTED_OPTION_EVALUATION",
                    source_sampling_view_hash=digest(
                        Path(
                            source_manifest["records"][group]["witnesses"]["sampling_view"]["path"]
                        )
                    ),
                    candidate_hash=hash_json(catalog[intended["option_index"]]),
                    outcome_contract_changed=False,
                    promotion_authorized=False,
                    hardware_authorized=False,
                )
                if {
                    k: v for k, v in document.items() if k != "report_hash"
                } != expected_planning or receipt.get(
                    "planning_context_hash"
                ) != planning.context_hash:
                    raise ValueError("actual full public planning execution context differs")
            if spec is not None:
                expected_candidate = (
                    None if args.profile == "binding-four" else catalog[intended["option_index"]]
                )
                if (
                    receipt["group"] != group
                    or receipt["candidate"] != expected_candidate
                    or view["seed"] != draw.sampling_seed
                ):
                    raise ValueError("actual raw draw, sampling seed and original candidate differ")
                if args.profile == "binding-four":
                    enabled = intended["generated"]
                    expected_tag = dict(
                        schema="soccer.rsi.native_private_generated_episode_binding.v1",
                        seed=draw.sampling_seed,
                        full_external_sampling_view_restored_and_validated=True,
                        complete_generated_model_equals_external_view=True,
                        full_original_factory_constructor_retained=True,
                        sampling_law_changed=False,
                    )
                    if (
                        enabled and report.get("generated_seed_episode_binding") != expected_tag
                    ) or (not enabled and "generated_seed_episode_binding" in report):
                        raise ValueError(
                            "original versus generated binding must match actual commitment"
                        )
                    expected_intent = dict(intended, declaration_hash=declaration["report_hash"])
                else:
                    if "generated_seed_episode_binding" in report:
                        raise ValueError("catalog must retain original native binding path")
                    expected_intent = dict(
                        intended,
                        candidate=expected_candidate,
                        declaration_hash=declaration["report_hash"],
                    )
                    if context_study:
                        expected_intent["context_hash"] = (
                            None if planning is None else planning.context_hash
                        )
                    if release_ablation:
                        expected_intent["baseline_receipt_hash"] = baseline["receipt_hash"]
                if expected_candidate is None:
                    if (
                        report.get("bounded_control_profile") is not None
                        or report["approach_mode"] != "legacy"
                        or report["swing_mode"] != "legacy"
                    ):
                        raise ValueError("None must retain original control law")
                    if planning is None:
                        reference = references[group]
                        for witness in reference["witnesses"].values():
                            if digest(Path(witness["path"])) != witness["sha256"]:
                                raise ValueError("original numerical-control witness changed")
                        with np.load(
                            reference["witnesses"]["trace"]["path"], allow_pickle=False
                        ) as archive:
                            if set(trace) != set(archive.files) or any(
                                trace[k].dtype != archive[k].dtype
                                or trace[k].shape != archive[k].shape
                                or trace[k].tobytes() != archive[k].tobytes()
                                for k in trace
                            ):
                                raise ValueError(
                                    "every original same-world None array must remain bit-exact"
                                )
                        if outcome.to_dict() != reference["outcome"]:
                            raise ValueError(
                                "original passing and failing labels must remain unchanged"
                            )
                else:
                    from rosclaw_soccer.rsi.bounded_first_touch_control import (
                        BoundedFirstTouchControl,
                    )
                    from rosclaw_soccer.rsi.postcontact_navigation_probe import (
                        PostcontactNavigationProbe,
                    )

                    if (
                        report.get("bounded_control_profile")
                        != BoundedFirstTouchControl(**expected_candidate["parameters"]).contract()
                        or report["approach_mode"] != "bounded_approach_diagnostic"
                        or report["swing_mode"] != expected_candidate["swing"]
                    ):
                        raise ValueError("exact bounded option must bind actual execution")
                    postcontact = (
                        PostcontactNavigationProbe(**expected_candidate["postcontact"]).contract()
                        if "postcontact" in expected_candidate
                        else None
                    )
                    if report.get("postcontact_navigation_probe") != postcontact:
                        raise ValueError("exact postcontact option must bind actual execution")
            if spec is None and (
                intended != dict(job=name, group=group, candidate=None, course=list(draw.course))
                or receipt["group"] != group
                or receipt["candidate"] is not None
                or report.get("bounded_control_profile") is not None
                or report["approach_mode"] != "legacy"
                or report["swing_mode"] != "legacy"
                or view["seed"] != draw.sampling_seed
            ):
                raise ValueError("unperturbed public draw, seed and original None control required")
            actual_course = draw.course if planning is None else planning.course
            context = dict(course=list(actual_course), source_seed=draw.seed, source_lane=draw.lane)
            NativeFirstTouchInitialState(
                actual_course, G1SonicRunupController.default_angles
            ).verify(report["course"], trace["canonical_qpos"][0, 0], trace["canonical_qvel"][0, 0])
            if spec is None:
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
                else "INTENT_COMMITTED_BEFORE_BODY"
                if spec is not None
                else "NEW_WORLD_INTENT_COMMITTED_BEFORE_BODY"
            )
            or prior["state"].get("next_intent_hash") != intent["report_hash"]
            or prior["state"].get("active") is not None
        ):
            raise ValueError("intent must be journaled before the charged body start")
        if spec is None and receipt["candidate"] != intended["candidate"]:
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
            spec["readback_schema"]
            if spec is not None
            else "soccer.rsi.mesh_original_public_terminal_readback.v1"
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
    if public_four or spec is not None:
        report.update(
            source_native_bodies=count,
            source_encoder_compared_frames=count * 300,
            source_independent_planner_onnx_calls=count * 15,
            source_neural_encoder_recomputations=count * 300,
        )
    if prepared is not None:
        prepared.stable()
        pins.update(audit_inputs)
        for name, expected in pins.items():
            if digest(Path(name)) != expected:
                raise ValueError("source/input bytes changed during terminal readback")
        report[
            "prepared_pins_rederived_from_exact_original_preparation"
        ] = not preparation_was_persisted
        report["original_prepared_pins_persisted_by_producer"] = preparation_was_persisted
        if args.profile == "binding-four":
            report["noninterference_scope"] = dict(
                world_hash=config["mesh_task_contact_world_hash"],
                groups=[4, 8],
                candidate=None,
                cerebellar_model=None,
                full_original_arrays_bit_equal=True,
                all_other_control_profiles_qualified=False,
                all_other_worlds_qualified=False,
            )
        elif args.profile == "catalog-sixteen":
            report["passing_option_indices_by_group"] = {
                str(g): [
                    j["option_index"]
                    for j, r in zip(spec["jobs"], rows, strict=True)
                    if j["group"] == g and r["outcome"]["task_pass"]
                ]
                for g in (8, 56)
            }
            if (
                report["passing_option_indices_by_group"]
                != complete["passing_option_indices_by_group"]
            ):
                raise ValueError("whole independently reconstructed option outcome matrix differs")
        elif release_ablation:
            from rosclaw_soccer.rsi.paired_learning_assessment import assess_pairs

            upstream = sealed(Path(config["upstream_catalog_readback"]))
            selection = sealed(Path(config["selection_declaration"]))
            catalog_complete = sealed(
                Path(config["selection_declaration"]).parent / "complete.json"
            )
            screen = sealed(args.baseline_screen_readback)
            screen_root = Path(config["baseline_receipts"][2]["path"]).parent
            screen_complete = sealed(screen_root / "complete.json")
            if (
                upstream.get("schema") != "soccer.rsi.mesh_contact_catalog_terminal_readback.v1"
                or upstream.get("complete_hash") != catalog_complete["report_hash"]
                or catalog_complete.get("declaration_hash") != selection["report_hash"]
                or selection["report_hash"] != config["selection_declaration_hash"]
                or upstream.get("source_native_bodies") != 16
                or upstream.get("reader_source_hash") != digest(args.upstream_audit_source)
                or screen.get("schema")
                != "soccer.rsi.mesh_option4_teacher_screen_terminal_readback.v1"
                or screen.get("complete_hash") != screen_complete["report_hash"]
                or screen.get("source_native_bodies") != 18
                or screen.get("reader_source_hash") != digest(args.baseline_screen_audit_source)
                or any(
                    doc.get(k) is not True
                    for doc in (upstream, screen)
                    for k in (
                        "whole_input_and_witness_files_verified",
                        "shared_payload_logical_seals_rebuilt",
                        "prior_independent_replays_authenticated",
                    )
                )
            ):
                raise ValueError("both complete independently audited baseline studies required")
            for path in (args.upstream_audit_source, args.baseline_screen_audit_source):
                pins[str(path)] = digest(path)
            reused = []
            for index, item in enumerate(config["baseline_receipts"]):
                prior = sealed(Path(item["path"]), "receipt_hash")
                prior_complete, prior_audit, prior_index = (
                    (catalog_complete, upstream, 8 + index)
                    if index < 2
                    else (screen_complete, screen, 1 if index == 2 else 3)
                )
                if prior != prior_complete["results"][prior_index] or not any(
                    row["job"] == prior["job"]
                    and row["receipt_hash"] == prior["receipt_hash"]
                    and row["outcome"] == prior["outcome"]
                    for row in prior_audit["rows"]
                ):
                    raise ValueError("exact baseline endpoint must belong to its terminal audit")
                job = spec["jobs"][index]
                if (
                    item["pair_id"] != job["pair_id"]
                    or prior["candidate"] != catalog[4]
                    or prior["group"] != job["group"]
                    or prior.get("planning_context_hash")
                    != complete["results"][index].get("planning_context_hash")
                ):
                    raise ValueError("same option4 and course, not a substituted baseline required")
                reused.append(dict(role="control", pair_id=item["pair_id"], receipt=prior))
            if reused != declaration["reused_endpoints"]:
                raise ValueError(
                    "all four original endpoints including the unsafe failure required"
                )
            paired = assess_pairs(
                {r["pair_id"]: r["receipt"]["outcome"] for r in reused},
                {j["pair_id"]: r["outcome"] for j, r in zip(spec["jobs"], rows, strict=True)},
            )
            if paired != complete["paired_development"]:
                raise ValueError("complete independently reconstructed four-course pairing differs")
            report.update(
                paired_development=paired,
                reused_endpoint_count=4,
                independent_course_count=4,
                paired_endpoint_count=8,
                all_original_first_contact_prefixes_bit_equal=True,
                optimized_recovery_claimed=False,
                reused_endpoints_authenticated_via_prior_complete_terminal_readbacks=True,
                reused_endpoint_physics_rerun_here=False,
                upstream_terminal_readback_hash=upstream["report_hash"],
                baseline_screen_terminal_readback_hash=screen["report_hash"],
            )
            if torque_mpc:
                totals = dict(
                    native_forecast_substeps=0,
                    independent_optimization_forecast_substeps=0,
                    independent_chosen_control_forecast_substeps=0,
                )
                for item in complete["results"]:
                    audit = sealed(Path(item["witnesses"]["independent_review"]["path"]))
                    reconstruction = audit["contact_exit_torque_mpc_review"]
                    totals["native_forecast_substeps"] += item["forecast_physical_substeps"]
                    totals["independent_optimization_forecast_substeps"] += reconstruction[
                        "planning_counters"
                    ]["integrated_forecast_steps"]
                    totals["independent_chosen_control_forecast_substeps"] += reconstruction[
                        "chosen_control_integrated_forecast_steps"
                    ]
                if any(complete.get(k) != v for k, v in totals.items()):
                    raise ValueError("all native and independent forecast integrations must agree")
                report.update(totals)
                report["optimization_reconstruction_uses_same_pinned_implementation"] = True
                report["independent_optimization_implementation"] = False
            else:
                report["actual_postcontact_taskspace_increment_exactly_zero"] = True
        elif teacher_screen:
            from rosclaw_soccer.rsi.paired_learning_assessment import assess_pairs

            upstream = sealed(Path(config["upstream_catalog_readback"]))
            selection = sealed(Path(config["selection_declaration"]))
            upstream_complete = sealed(
                Path(config["selection_declaration"]).parent / "complete.json"
            )
            if (
                upstream.get("schema") != "soccer.rsi.mesh_contact_catalog_terminal_readback.v1"
                or upstream.get("complete_hash") != upstream_complete["report_hash"]
                or upstream_complete.get("declaration_hash") != selection["report_hash"]
                or selection["report_hash"] != config["selection_declaration_hash"]
                or upstream.get("source_native_bodies") != 16
                or upstream.get("reader_source_hash") != digest(args.upstream_audit_source)
                or any(
                    upstream.get(k) is not True
                    for k in (
                        "whole_input_and_witness_files_verified",
                        "shared_payload_logical_seals_rebuilt",
                        "prior_independent_replays_authenticated",
                    )
                )
            ):
                raise ValueError("complete independently audited original selection study required")
            pins[str(args.upstream_audit_source)] = digest(args.upstream_audit_source)
            reused = declaration["reused_endpoints"]
            expected_reused = [
                dict(role="control", pair_id=f"raw-{g}", receipt=references[g])
                for g in (4, 8, 32, 56)
            ]
            for item, g in zip(config["selection_receipts"], (8, 56), strict=True):
                selected = sealed(Path(item["path"]), "receipt_hash")
                if (
                    digest(Path(item["path"])) != item["file_hash"]
                    or selected["receipt_hash"] != item["receipt_hash"]
                ):
                    raise ValueError("exact original selection receipt required")
                if selected != upstream_complete["results"][8 if g == 8 else 9] or not any(
                    row["job"] == f"option-4-raw-{g}"
                    and row["receipt_hash"] == selected["receipt_hash"]
                    and row["outcome"] == selected["outcome"]
                    for row in upstream["rows"]
                ):
                    raise ValueError("selected endpoint differs from independent upstream readback")
                expected_reused.append(dict(role="teacher", pair_id=f"raw-{g}", receipt=selected))
            if reused != expected_reused:
                raise ValueError("exact six predeclared reused endpoints required")
            controls = {
                r["pair_id"]: r["receipt"]["outcome"] for r in reused if r["role"] == "control"
            }
            evaluations = {
                r["pair_id"]: r["receipt"]["outcome"] for r in reused if r["role"] == "teacher"
            }
            for job, row in zip(spec["jobs"], rows, strict=True):
                (controls if job["role"] == "control" else evaluations)[job["pair_id"]] = row[
                    "outcome"
                ]
            paired = assess_pairs(controls, evaluations)
            if paired != complete["paired_development"]:
                raise ValueError("independent complete 12-course pairing differs")
            report.update(
                paired_development=paired,
                reused_endpoint_count=6,
                independent_course_count=12,
                paired_endpoint_count=24,
                reused_endpoints_authenticated_via_prior_complete_terminal_readbacks=True,
                reused_endpoint_physics_rerun_here=False,
                upstream_terminal_readback_hash=upstream["report_hash"],
            )
        pins.update(audit_inputs)
        prepared.stable()
        if any(digest(Path(name)) != expected for name, expected in pins.items()):
            raise ValueError("all source, upstream and reused inputs must remain unchanged")
    report["report_hash"] = hash_json(report)
    write_once(args.output, report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
