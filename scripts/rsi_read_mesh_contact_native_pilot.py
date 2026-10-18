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
    if profile == "surface-receding-six":
        from rosclaw_soccer.rsi.receding_teacher_scope import SOURCE_GROUPS
        from rosclaw_soccer.rsi.surface_receding_evidence import surface_receding_jobs

        return dict(
            count=6,
            groups=list(SOURCE_GROUPS),
            declaration_schema="soccer.rsi.mesh_surface_receding_pilot_declaration.v1",
            complete_schema="soccer.rsi.mesh_surface_receding_pilot_complete.v1",
            readback_schema="soccer.rsi.mesh_surface_receding_terminal_readback.v1",
            phase="MESH_SURFACE_RECEDING_COMPLETE_NOT_QUALIFIED_TEACHER",
            jobs=surface_receding_jobs(),
        )
    if profile == "receding-catalog2-ten":
        from rosclaw_soccer.rsi.receding_teacher_scope import SOURCE_GROUPS, receding_jobs

        return dict(
            count=10,
            groups=list(SOURCE_GROUPS),
            declaration_schema="soccer.rsi.receding_catalog2_declaration.v1",
            complete_schema="soccer.rsi.receding_catalog2_complete.v1",
            readback_schema="soccer.rsi.receding_catalog2_terminal_readback.v1",
            phase="RECEDING_CATALOG2_COMPLETE_NOT_QUALIFIED_TEACHER",
            jobs=receding_jobs(),
        )
    if profile == "surface-context-six":
        return dict(
            count=6,
            groups=[68, 69, 72, 73, 76, 77],
            declaration_schema="soccer.rsi.mesh_surface_context_expansion_pilot_declaration.v1",
            complete_schema="soccer.rsi.mesh_surface_context_expansion_pilot_complete.v1",
            readback_schema="soccer.rsi.mesh_surface_context_expansion_terminal_readback.v1",
            phase="MESH_SURFACE_CONTEXT_EXPANSION_COMPLETE_NOT_QUALIFIED_TEACHER",
            jobs=[
                dict(
                    job=f"surface-context-{i}",
                    role="teacher",
                    pair_id=f"context-{i}",
                    context_index=i,
                    group=g,
                    option_index=4,
                )
                for i, g in zip(range(2, 8), (68, 69, 72, 73, 76, 77), strict=True)
            ],
        )
    if profile == "surface-seven":
        spec = bounded_profile("release-four")
        rows = [dict(j, job="surface-" + j["pair_id"], role="teacher") for j in spec["jobs"]]
        rows += [
            dict(
                job=f"surface-raw-{g}",
                role="teacher",
                pair_id=f"raw-{g}",
                context_index=None,
                group=g,
                option_index=4,
            )
            for g in (4, 32)
        ]
        spec.update(
            count=7,
            groups=[8, 56, 64, 65, 4, 32],
            declaration_schema="soccer.rsi.mesh_surface_swing_pilot_declaration.v1",
            complete_schema="soccer.rsi.mesh_surface_swing_pilot_complete.v1",
            readback_schema="soccer.rsi.mesh_surface_swing_terminal_readback.v1",
            phase="MESH_SURFACE_SWING_COMPLETE_NOT_QUALIFIED_TEACHER",
            jobs=[dict(rows[0], job="surface-off-raw-8", role="noninterference"), *rows],
        )
        return spec
    if profile == "directional-four":
        spec = bounded_profile("release-four")
        spec.update(
            declaration_schema="soccer.rsi.directional_contact_compliance_pilot_declaration.v1",
            complete_schema="soccer.rsi.directional_contact_compliance_pilot_complete.v1",
            readback_schema="soccer.rsi.directional_contact_compliance_terminal_readback.v1",
            phase="DIRECTIONAL_CONTACT_COMPLIANCE_COMPLETE_NOT_QUALIFIED_TEACHER",
            jobs=[
                dict(job, job="directional-exit-" + job["pair_id"], role="teacher")
                for job in spec["jobs"]
            ],
        )
        return spec
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


def verify_surface_prefix(trace, original_trace, prefix, baseline_hash, *, enabled):
    import numpy as np

    from rosclaw_soccer.sim.contracts import hash_bytes

    extra = {"mesh_surface_swing_record"} if enabled else set()
    if set(trace) != set(original_trace) | extra:
        raise ValueError("surface study may only add its one declared record")
    active = np.flatnonzero(trace["mesh_surface_swing_record"][:, 0, 2] == 1) if enabled else []
    cutoff = int(active[0]) if len(active) else 300
    if not 30 <= cutoff <= 300 or any(
        trace[k].dtype != v.dtype
        or trace[k].shape != v.shape
        or trace[k][:cutoff].tobytes() != v[:cutoff].tobytes()
        for k, v in original_trace.items()
    ):
        raise ValueError("all original arrays before surface activation must be bit-exact")
    if (
        prefix.get("schema") != "soccer.rsi.surface_pre_activation_prefix_parity.v1"
        or prefix.get("baseline_receipt_hash") != baseline_hash
        or prefix.get("prefix_frames") != cutoff
        or prefix.get("all_original_array_prefixes_bit_equal") is not True
        or prefix.get("prefix_array_hashes")
        != {k: hash_bytes(trace[k][:cutoff].tobytes()) for k in sorted(original_trace)}
    ):
        raise ValueError("surface prefix witness must bind every original array")


def verify_surface_reconstruction(report, commitment, receipt, review, source, trace):
    from rosclaw_soccer.sim.contracts import hash_json

    contract = commitment["mesh_surface_swing"]
    r = review.get("mesh_surface_swing_review", {})
    if (
        receipt.get("mesh_surface_swing") != contract
        or contract.get("contract_hash")
        != hash_json({k: v for k, v in contract.items() if k != "contract_hash"})
        or contract.get("source_hashes")
        != {
            k: digest(source / "src/rosclaw_soccer" / p)
            for k, p in dict(
                adapter="rsi/mesh_surface_swing_probe.py",
                solver="rsi/mesh_surface_swing.py",
                geometry="physics/convex_surface_distance.py",
            ).items()
        }
        or r.get("schema") != "soccer.rsi.mesh_surface_swing_reconstruction.v1"
        or r.get("contract") != contract
        or r.get("frames_reconstructed") != 300
        or type(r.get("active_frames")) is not int
        or not 0 <= r["active_frames"] <= 270
        or r["active_frames"] != int(trace["mesh_surface_swing_record"][:, 0, 2].sum())
        or r.get("reconstruction_uses_same_pinned_geometry_and_ik") is not True
        or r.get("independent_ik_implementation") is not False
        or r.get("side_and_release_reconstructed_from_replayed_history") is not True
        or r.get("source_numeric_replay_tolerance") != 1e-8
        or any(r.get(k) is not False for k in ("hardware_authorized", "promotion_authorized"))
        or report.get("mesh_surface_swing") != contract
    ):
        raise ValueError("source-bound causal surface reconstruction and honest scope required")


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


def verify_directional_reconstruction(report, commitment, receipt, review, source):
    contract = commitment["directional_contact_compliance"]
    expected_sources = {
        "controller": "rsi/directional_contact_compliance.py",
        "shaping": "rsi/directional_contact_impedance.py",
        "forecast": "physics/g1_pd_torque_forecast.py",
        "kinematics": "physics/g1_joint_task_kinematics.py",
        "pressure": "rsi/contact_exit_torque_mpc.py",
    }
    counters = report["directional_compliance_counters"]
    calls = sum(not p["identity"] for p in report["directional_compliance_plans"])
    r = review.get("directional_contact_compliance_review", {})
    if (
        contract.get("source_hashes")
        != {k: digest(source / "src/rosclaw_soccer" / p) for k, p in expected_sources.items()}
        or receipt.get("directional_compliance_counters") != counters
        or receipt.get("forecast_physical_substeps") != counters["integrated_forecast_steps"]
        or r.get("schema") != "soccer.rsi.directional_compliance_reconstruction.v1"
        or r.get("contract") != contract
        or r.get("frames_reconstructed") != 300
        or r.get("planning_counters") != counters
        or r.get("gain_law_reconstructed_with_same_pinned_implementation") is not True
        or r.get("independent_gain_law_implementation") is not False
        or r.get("chosen_control_forecast_attempts") != calls
        or r.get("chosen_control_integrated_forecast_steps") != calls * 10
        or r.get("locally_bitexact_active_intervals") != calls
        or r.get("source_numeric_replay_tolerance") != 1e-8
        or type(r.get("source_end_state_hash_matches")) is not int
        or not 0 <= r["source_end_state_hash_matches"] <= calls
        or r.get("hardware_authorized") is not False
        or r.get("promotion_authorized") is not False
    ):
        raise ValueError("source-bound gain reconstruction and local actual replay required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream-audit-source", type=Path)
    parser.add_argument("--baseline-screen-readback", type=Path)
    parser.add_argument("--baseline-screen-audit-source", type=Path)
    parser.add_argument("--collected-unit-journal", type=Path)
    parser.add_argument("--collected-unit-journal-hash")
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
            "directional-four",
            "surface-seven",
            "surface-context-six",
            "receding-catalog2-ten",
            "surface-receding-six",
        ),
        default="contact-three",
    )
    args = parser.parse_args()
    public_four = args.profile == "original-public-four"
    teacher_screen = args.profile == "teacher-eighteen"
    receding = args.profile == "receding-catalog2-ten"
    surface_receding = args.profile == "surface-receding-six"
    torque_mpc = args.profile == "contact-exit-four"
    directional = args.profile == "directional-four"
    expansion = args.profile == "surface-context-six"
    surface = args.profile in ("surface-seven", "surface-context-six", "surface-receding-six")
    surface_off = surface and not expansion and not surface_receding
    release_ablation = args.profile in {
        "release-four",
        "contact-exit-four",
        "directional-four",
        "surface-seven",
        "surface-context-six",
    }
    context_study = teacher_screen or release_ablation or receding or surface_receding
    spec = (
        bounded_profile(args.profile)
        if args.profile
        in (
            "binding-four",
            "catalog-sixteen",
            "teacher-eighteen",
            "release-four",
            "contact-exit-four",
            "directional-four",
            "surface-seven",
            "surface-context-six",
            "receding-catalog2-ten",
            "surface-receding-six",
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
    collected_handoff = None
    collected_helper = None
    collected_helper_hash = None
    if args.collected_unit_journal is not None:
        import runpy

        # Do not import the reviewer Soccer package before binding the producer
        # below: doing so would poison subsequent exact-source reconstruction.
        collected_helper = (
            Path(__file__).resolve().parents[1]
            / "src/rosclaw_soccer/rsi/collected_sim_stage_evidence.py"
        )
        collected_helper_hash = digest(collected_helper)
        verify_collected_sim_journal = runpy.run_path(str(collected_helper))[
            "verify_collected_sim_journal"
        ]

        if (
            not torque_mpc
            or digest(args.collected_unit_journal) != args.collected_unit_journal_hash
        ):
            raise ValueError("explicit pinned collected MPC journal required")
        records = json.loads(args.collected_unit_journal.read_text())
        collected_handoff = verify_collected_sim_journal(
            records,
            status=dict(line.split("=", 1) for line in status.splitlines()),
            unit=args.unit,
            expected_command=[
                sys.executable,
                "-u",
                "scripts/rsi_swing_release_ablation_pilot.py",
                "--config",
                str(args.config),
            ],
            terminal_phase=spec["phase"],
        )
    elif set(status.splitlines()) != {
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
    if receding and (
        config.get("selected_option_index") != 2
        or config.get("neural_training_authorized") is not False
        or config.get("teacher_auto_admission") is not False
        or complete.get("independent_course_count") != 6
        or complete.get("paired_endpoint_count") != 12
        or complete.get("archived_c7_full_numeric_reproduction") is not True
        or complete.get("reused_endpoints") != declaration.get("reused_endpoints")
        or any(
            config.get(k, False) is not False
            for k in (
                "use_mesh_surface_swing",
                "use_contact_exit_torque_mpc",
                "use_directional_contact_compliance",
                "suppress_postcontact_swing_release",
            )
        )
    ):
        raise ValueError(
            "exact receding catalog2 scope, reproduction and original controls required"
        )
    if surface_receding and (
        config.get("selected_option_index") != 2
        or config.get("surface_profile") != "receding-six"
        or config.get("use_mesh_surface_swing") is not True
        or any(
            config.get(k) is not False
            for k in (
                "neural_training_authorized",
                "teacher_auto_admission",
                "generated_seed_episode_binding",
                "suppress_postcontact_swing_release",
            )
        )
        or any(
            config.get(k, False) is not False
            for k in ("use_contact_exit_torque_mpc", "use_directional_contact_compliance")
        )
        or complete.get("distinct_derived_course_count") != 6
        or complete.get("source_scene_clusters") != 3
        or complete.get("paired_endpoint_count") != 12
        or complete.get("prior_disabled_noninterference_reused_not_rerun") is not True
        or complete.get("reused_endpoints") != declaration.get("reused_endpoints")
        or complete.get("reused_original_none_endpoints")
        != declaration.get("reused_original_none_endpoints")
        or set(config.get("required_receipt_witnesses", []))
        != {"intent_witness", "encoder_witness", "prefix_witness"}
    ):
        raise ValueError("exact six-course catalog2/material-point intervention required")
    if release_ablation and (
        args.upstream_audit_source is None
        or args.baseline_screen_readback is None
        or args.baseline_screen_audit_source is None
        or config.get("suppress_postcontact_swing_release")
        is not (not (torque_mpc or directional or surface))
        or config.get("use_contact_exit_torque_mpc", False) is not torque_mpc
        or config.get("use_directional_contact_compliance", False) is not directional
        or config.get("use_mesh_surface_swing", False) is not surface
        or config.get("selected_option_index") != 4
        or config.get("upstream_terminal_audit_required") is not True
        or complete.get("independent_course_count") != (6 if surface else 4)
        or complete.get("paired_endpoint_count") != (12 if surface else 8)
        or len(config.get("baseline_receipts", [])) != (6 if surface else 4)
        or [i["pair_id"] for i in config["baseline_receipts"]]
        != [j["pair_id"] for j in (spec["jobs"][1:] if surface_off else spec["jobs"])]
        or config.get("surface_profile", "original-seven")
        != ("context-six" if expansion else "original-seven")
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
    if collected_handoff is not None:
        pins[str(args.collected_unit_journal)] = args.collected_unit_journal_hash
        pins[str(collected_helper)] = collected_helper_hash
    prepared, preparation_was_persisted = None, False
    receding_reused, receding_archived = None, None
    surface_receding_baselines, surface_receding_none = None, None
    if spec is not None:
        from rosclaw_soccer.rsi.first_touch_campaign_worker import prepare
        from rosclaw_soccer.rsi.reviewed_contact_world import ReviewedContactWorld

        prepared, _, source_manifest, actual = prepare(args.config)
        if actual != config:
            raise ValueError("same original preparation configuration required")
        if receding:
            from rosclaw_soccer.rsi.receding_scope_evidence import load_scope_evidence

            receding_reused, receding_archived = load_scope_evidence(config, sealed, digest, pins)
            if receding_reused != declaration["reused_endpoints"]:
                raise ValueError("original None receding endpoints must match declaration")
        if surface_receding:
            from rosclaw_soccer.rsi.surface_receding_evidence import load_baselines

            surface_receding_baselines, surface_receding_none = load_baselines(
                config, source, sealed, digest, pins
            )
        if expansion:
            from rosclaw_soccer.rsi.surface_context_expansion import require_previous_surface

            require_previous_surface(config, source, sealed, digest, pins)
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
            in (
                "catalog-sixteen",
                "teacher-eighteen",
                "release-four",
                "contact-exit-four",
                "directional-four",
                "surface-seven",
                "surface-context-six",
                "receding-catalog2-ten",
                "surface-receding-six",
            )
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
        if receding and index == 0:
            witnesses["reproduction_witness"] = receipt["reproduction_witness"]
        if release_ablation or surface_receding:
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
        if receding and index == 0:
            from rosclaw_soccer.rsi.receding_teacher_scope import reproduce_zero_control

            with np.load(
                receding_archived["witnesses"]["trace"]["path"], allow_pickle=False
            ) as old_archive:
                numeric = reproduce_zero_control(trace, dict(old_archive))
            numeric["archived_receipt_hash"] = receding_archived["receipt_hash"]
            numeric["report_hash"] = hash_json(numeric)
            if (
                sealed(Path(receipt["reproduction_witness"]["path"])) != numeric
                or outcome.to_dict() != receding_archived["outcome"]
            ):
                raise ValueError("independent complete original zero-layer reproduction differs")
        if release_ablation or surface_receding:
            from rosclaw_soccer.rsi.swing_release_ablation import (
                release_ablation_contract,
                require_zero_postcontact_swing,
            )

            if surface:
                if spec["jobs"][index]["role"] != "noninterference":
                    from rosclaw_soccer.rsi.mesh_surface_swing_evidence import (
                        require_surface_swing_capture,
                    )

                    require_surface_swing_capture(report, commitment, trace)
                elif "mesh_surface_swing" in commitment or "mesh_surface_swing" in receipt:
                    raise ValueError(
                        "disabled source noninterference body may not enable new control"
                    )
            elif directional:
                from rosclaw_soccer.rsi.directional_compliance_evidence import (
                    require_directional_capture,
                )

                require_directional_capture(report, commitment, trace)
                if (
                    receipt.get("directional_contact_compliance")
                    != commitment["directional_contact_compliance"]
                ):
                    raise ValueError("gain receipt must bind actual physical contract")
            elif torque_mpc:
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
            baseline_item = (
                next(
                    i
                    for i in config["baseline_receipts"]
                    if i["pair_id"] == spec["jobs"][index]["pair_id"]
                )
                if surface
                else config["baseline_receipts"][index]
            )
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
            if torque_mpc or directional:
                extra = (
                    {
                        "contact_exit_mpc_requested_offset_nm",
                        "contact_exit_mpc_applied_offset_nm",
                        "contact_exit_mpc_forecast_steps",
                    }
                    if torque_mpc
                    else {
                        "directional_compliance_stiffness_nm_per_rad",
                        "directional_compliance_requested_delta_nm",
                        "directional_compliance_applied_delta_nm",
                        "directional_compliance_forecast_steps",
                    }
                )
                if set(trace) - set(original_trace) != extra:
                    raise ValueError(
                        "only declared contact-exit capture may extend original arrays"
                    )
                prefix_trace = {k: trace[k] for k in original_trace if k in trace}
            if surface:
                verify_surface_prefix(
                    trace,
                    original_trace,
                    prefix,
                    baseline["receipt_hash"],
                    enabled=spec["jobs"][index]["role"] != "noninterference",
                )
                if (
                    spec["jobs"][index]["role"] == "noninterference"
                    and outcome.to_dict() != baseline["outcome"]
                ):
                    raise ValueError("disabled complete outcome must match original baseline")
            else:
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
        if directional:
            verify_directional_reconstruction(report, commitment, receipt, review, source)
        if surface and spec["jobs"][index]["role"] != "noninterference":
            verify_surface_reconstruction(report, commitment, receipt, review, source, trace)
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

                if receding or surface_receding:
                    from rosclaw_soccer.rsi.receding_teacher_scope import receding_contexts

                    study_contexts = receding_contexts(plan)
                else:
                    study_contexts = planning_contexts(plan)
                if declaration.get("contexts") != [
                    dict(context_hash=c.context_hash, course=list(c.course)) for c in study_contexts
                ]:
                    raise ValueError("whole predeclared eight-context source bank differs")
            if context_study and intended["context_index"] is not None:
                from dataclasses import asdict

                from rosclaw_soccer.rsi.public_physics_option_planning import (
                    planning_contexts,
                    validate_context_source,
                )

                planning = (
                    next(c for c in study_contexts if c.source_group == group)
                    if receding or surface_receding
                    else planning_contexts(plan)[intended["context_index"]]
                )
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
                    if release_ablation or surface_receding:
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
    if collected_handoff is not None:
        report["collected_sim_stage_handoff"] = collected_handoff
        report["worker_os_exit_code_proven"] = False
        report["collected_unit_journal"] = dict(
            path=str(args.collected_unit_journal), sha256=args.collected_unit_journal_hash
        )
        report["collected_handoff_helper"] = dict(
            path=str(collected_helper), sha256=collected_helper_hash
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
        elif surface_receding:
            from rosclaw_soccer.rsi.paired_learning_assessment import assess_pairs
            from rosclaw_soccer.rsi.public_course_cluster_accounting import cluster_contract

            jobs = spec["jobs"]
            reused = [
                dict(
                    role="control",
                    pair_id=j["pair_id"],
                    receipt=surface_receding_baselines[j["pair_id"]],
                )
                for j in jobs
            ]
            none_reused = [
                dict(
                    role="control",
                    pair_id=j["pair_id"],
                    receipt=surface_receding_none[j["pair_id"]],
                )
                for j in jobs
            ]
            evaluations = {j["pair_id"]: r["outcome"] for j, r in zip(jobs, rows, strict=True)}
            paired = assess_pairs(
                {k: r["outcome"] for k, r in surface_receding_baselines.items()}, evaluations
            )
            retention = assess_pairs(
                {k: r["outcome"] for k, r in surface_receding_none.items()}, evaluations
            )
            clusters = cluster_contract(plan, study_contexts)
            if (
                reused != declaration["reused_endpoints"]
                or none_reused != declaration["reused_original_none_endpoints"]
                or paired != complete["paired_development"]
                or retention != complete["original_none_context_retention"]
                or declaration.get("course_cluster_contract") != clusters
            ):
                raise ValueError(
                    "whole six pairs, None retention and source clusters must reconstruct"
                )
            report.update(
                paired_development=paired,
                original_none_context_retention=retention,
                reused_endpoint_count=6,
                reused_original_none_endpoint_count=6,
                distinct_derived_course_count=6,
                source_scene_clusters=3,
                course_cluster_contract=clusters,
                paired_endpoint_count=12,
                prior_disabled_noninterference_reused_not_rerun=True,
                all_original_pre_activation_prefixes_bit_equal=True,
                geometry_and_ik_reconstruction_uses_same_pinned_implementation=True,
                independent_ik_implementation=False,
                fresh_exam=False,
                reused_endpoints_authenticated_via_prior_complete_terminal_readbacks=True,
                reused_endpoint_physics_rerun_here=False,
            )
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
                    (screen_complete, screen, 2 * spec["jobs"][index]["context_index"] + 1)
                    if expansion
                    else (catalog_complete, upstream, 8 + index)
                    if index < 2
                    else (
                        screen_complete,
                        screen,
                        (1, 3, 16, 17)[index - 2] if surface else 1 if index == 2 else 3,
                    )
                )
                if prior != prior_complete["results"][prior_index] or not any(
                    row["job"] == prior["job"]
                    and row["receipt_hash"] == prior["receipt_hash"]
                    and row["outcome"] == prior["outcome"]
                    for row in prior_audit["rows"]
                ):
                    raise ValueError("exact baseline endpoint must belong to its terminal audit")
                job = spec["jobs"][index + 1 if surface_off else index]
                if (
                    item["pair_id"] != job["pair_id"]
                    or prior["candidate"] != catalog[4]
                    or prior["group"] != job["group"]
                    or prior.get("planning_context_hash")
                    != complete["results"][index + 1 if surface_off else index].get(
                        "planning_context_hash"
                    )
                ):
                    raise ValueError("same option4 and course, not a substituted baseline required")
                reused.append(dict(role="control", pair_id=item["pair_id"], receipt=prior))
            if reused != declaration["reused_endpoints"]:
                raise ValueError(
                    "all four original endpoints including the unsafe failure required"
                )
            paired = assess_pairs(
                {r["pair_id"]: r["receipt"]["outcome"] for r in reused},
                {
                    j["pair_id"]: r["outcome"]
                    for j, r in zip(
                        spec["jobs"][1:] if surface_off else spec["jobs"],
                        rows[1:] if surface_off else rows,
                        strict=True,
                    )
                },
            )
            if paired != complete["paired_development"]:
                raise ValueError("complete independently reconstructed four-course pairing differs")
            report.update(
                paired_development=paired,
                reused_endpoint_count=6 if surface else 4,
                independent_course_count=6 if surface else 4,
                paired_endpoint_count=12 if surface else 8,
                optimized_recovery_claimed=False,
                reused_endpoints_authenticated_via_prior_complete_terminal_readbacks=True,
                reused_endpoint_physics_rerun_here=False,
                upstream_terminal_readback_hash=upstream["report_hash"],
                baseline_screen_terminal_readback_hash=screen["report_hash"],
            )
            if expansion:
                original_none = {}
                for job in spec["jobs"]:
                    prior = screen_complete["results"][2 * job["context_index"]]
                    if (
                        prior["candidate"] is not None
                        or prior["group"] != job["group"]
                        or prior["job"] != "control-" + job["pair_id"]
                        or not any(
                            r["receipt_hash"] == prior["receipt_hash"]
                            and r["outcome"] == prior["outcome"]
                            for r in screen["rows"]
                        )
                    ):
                        raise ValueError(
                            "original None contexts must belong to whole baseline audit"
                        )
                    original_none[job["pair_id"]] = prior["outcome"]
                retention = assess_pairs(
                    original_none,
                    {j["pair_id"]: r["outcome"] for j, r in zip(spec["jobs"], rows, strict=True)},
                )
                if (
                    retention != complete.get("original_none_context_retention")
                    or complete.get("prior_disabled_noninterference_reused_not_rerun") is not True
                ):
                    raise ValueError("all None context outcomes and prior OFF reuse required")
                report.update(
                    original_none_context_retention=retention,
                    prior_disabled_noninterference_reused_not_rerun=True,
                    all_original_pre_activation_prefixes_bit_equal=True,
                    geometry_and_ik_reconstruction_uses_same_pinned_implementation=True,
                    independent_ik_implementation=False,
                )
            elif surface:
                raw_evaluations = {
                    j["pair_id"]: r["outcome"]
                    for j, r in zip(spec["jobs"][1:], rows[1:], strict=True)
                    if j["context_index"] is None
                }
                retention = assess_pairs(
                    {f"raw-{g}": references[g]["outcome"] for g in (4, 8, 32, 56)},
                    raw_evaluations,
                )
                if (
                    retention != complete.get("original_none_raw_retention")
                    or complete.get("disabled_complete_300_frame_noninterference_verified")
                    is not True
                ):
                    raise ValueError(
                        "original None raw retention and actual disabled noninterference required"
                    )
                report.update(
                    original_none_raw_retention=retention,
                    all_original_pre_activation_prefixes_bit_equal=True,
                    disabled_complete_300_frame_noninterference_verified=True,
                    geometry_and_ik_reconstruction_uses_same_pinned_implementation=True,
                    independent_ik_implementation=False,
                )
            else:
                report["all_original_first_contact_prefixes_bit_equal"] = True
            if torque_mpc or directional:
                totals = dict(
                    native_forecast_substeps=0,
                    independent_optimization_forecast_substeps=0,
                    independent_chosen_control_forecast_substeps=0,
                )
                for item in complete["results"]:
                    audit = sealed(Path(item["witnesses"]["independent_review"]["path"]))
                    reconstruction = audit[
                        "contact_exit_torque_mpc_review"
                        if torque_mpc
                        else "directional_contact_compliance_review"
                    ]
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
                if torque_mpc:
                    report["optimization_reconstruction_uses_same_pinned_implementation"] = True
                    report["independent_optimization_implementation"] = False
                else:
                    report["gain_law_reconstruction_uses_same_pinned_implementation"] = True
                    report["independent_gain_law_implementation"] = False
            elif not surface:
                report["actual_postcontact_taskspace_increment_exactly_zero"] = True
        elif receding:
            from rosclaw_soccer.rsi.paired_learning_assessment import assess_pairs
            from rosclaw_soccer.rsi.receding_teacher_scope import scope_contract

            if declaration.get("scope_contract") != scope_contract(plan):
                raise ValueError("whole six-course receding scope must match original source plan")
            if (
                declaration.get("archived_reference_receipt_hash")
                != receding_archived["receipt_hash"]
                or receding_archived["candidate"] != catalog[2]
            ):
                raise ValueError("candidate2 must match exact archived original success")
            controls = {r["pair_id"]: r["receipt"]["outcome"] for r in receding_reused}
            evaluations = {}
            for job, row in zip(spec["jobs"], rows, strict=True):
                (controls if job["role"] == "control" else evaluations)[job["pair_id"]] = row[
                    "outcome"
                ]
            paired = assess_pairs(controls, evaluations)
            if paired != complete["paired_development"]:
                raise ValueError("whole original and new receding pairing differs")
            report.update(
                paired_development=paired,
                reused_endpoint_count=2,
                independent_course_count=6,
                paired_endpoint_count=12,
                archived_c7_full_numeric_reproduction=True,
                reused_endpoints_authenticated_via_prior_complete_terminal_readbacks=True,
                reused_endpoint_physics_rerun_here=False,
                single_high_x_receding_stratum_only=True,
                fresh_exam=False,
            )
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
