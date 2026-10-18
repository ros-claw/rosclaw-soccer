"""Full synthetic four-endpoint reader path. Numerical/model verification is mocked."""

import copy
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi import first_touch_campaign_worker
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.first_touch_qualification import require_first_touch_qualification
from rosclaw_soccer.rsi.paired_learning_assessment import assess_pairs
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.rsi.swing_release_ablation import release_ablation_contract
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts import rsi_read_mesh_contact_native_pilot as audit
from tests.rsi.test_mesh_contact_terminal_readback import write
from tests.rsi.test_mesh_teacher_terminal_readback import expanded_fixture


def fixture(
    tmp_path, monkeypatch, tamper=None, *, torque_mpc=False, directional=False, surface=False
):
    expanded_fixture(tmp_path, monkeypatch)
    prepared, _, manifest, config = first_touch_campaign_worker.prepare(None)
    screen_root = tmp_path / "teacher"
    templates = [
        json.loads((screen_root / f"receipt-{i:04d}.json").read_text())
        for i in ((16, 17, 1, 3, 16, 17) if surface else (16, 17, 1, 3))
    ]
    root = tmp_path / "release"
    root.mkdir()
    profile = (
        "surface-seven"
        if surface
        else "directional-four"
        if directional
        else "contact-exit-four"
        if torque_mpc
        else "release-four"
    )
    spec = audit.bounded_profile(profile)
    if surface:
        from tests.rsi.test_mesh_surface_swing_evidence import capture as surface_capture

        surface_contract = surface_capture()[1]["mesh_surface_swing"]
        source = Path(config["source_root"])
        source_files = dict(
            adapter="rsi/mesh_surface_swing_probe.py",
            solver="rsi/mesh_surface_swing.py",
            geometry="physics/convex_surface_distance.py",
        )
        for name in source_files.values():
            path = source / "src/rosclaw_soccer" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# synthetic surface source, not physical validation\n")
        surface_contract.update(
            world_hash=config["mesh_task_contact_world_hash"],
            source_hashes={
                k: audit.digest(source / "src/rosclaw_soccer" / p) for k, p in source_files.items()
            },
        )
        surface_contract.pop("contract_hash")
        surface_contract["contract_hash"] = hash_json(surface_contract)
    if torque_mpc or directional:
        if directional:
            from tests.rsi.test_directional_compliance_evidence import capture
        else:
            from tests.rsi.test_contact_exit_torque_evidence import capture
        contract_key = (
            "directional_contact_compliance" if directional else "contact_exit_torque_mpc"
        )
        counter_key = (
            "directional_compliance_counters" if directional else "contact_exit_mpc_counters"
        )
        mpc_report, mpc_commitment, mpc_trace = capture()
        contract = mpc_commitment[contract_key]
        source = Path(config["source_root"])
        source_files = {
            "teacher": "rsi/contact_exit_torque_mpc.py",
            "forecast": "physics/g1_pd_torque_forecast.py",
            "kinematics": "physics/g1_joint_task_kinematics.py",
        }
        if directional:
            source_files = {
                "controller": "rsi/directional_contact_compliance.py",
                "shaping": "rsi/directional_contact_impedance.py",
                "forecast": "physics/g1_pd_torque_forecast.py",
                "kinematics": "physics/g1_joint_task_kinematics.py",
                "pressure": "rsi/contact_exit_torque_mpc.py",
            }
        for name in source_files.values():
            path = source / "src/rosclaw_soccer" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# synthetic teacher source, not executable physics\n")
        contract.update(
            world_hash=config["mesh_task_contact_world_hash"],
            source_hashes={
                k: audit.digest(source / "src/rosclaw_soccer" / name)
                for k, name in source_files.items()
            },
            maximum_offset_nm=12.0,
            maximum_hard_limit_fraction=0.2,
            maximum_offset_change_per_control_frame_nm=2.0,
        )
        contract["contract_hash"] = hash_json(
            {k: v for k, v in contract.items() if k != "contract_hash"}
        )
        mpc_report[contract_key] = copy.deepcopy(contract)
    plan = PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), "sha256:" + "a" * 64, 18)

    def endpoint(index, *, ablated=False, declaration=None, baseline=None):
        job = spec["jobs"][index + 1 if surface and not ablated else index]
        template_index = max(index - 1, 0) if surface and ablated else index
        r = copy.deepcopy(templates[template_index])
        folder = root / (f"new-{index}" if ablated else f"baseline-{index}")
        folder.mkdir()
        with np.load(r["witnesses"]["trace"]["path"], allow_pickle=False) as archive:
            trace = dict(archive)
        trace["force_n"][74, 0, 0] = 20
        if job["pair_id"] == "context-0":
            trace["force_n"][99, 0, 4] = 20
        else:
            trace["ball_position_after_step_m"][134, 0, 0] = 2
        trace["taskspace_joint_increment_rad"] = np.zeros((300, 1, 29))
        trace["taskspace_joint_increment_rad"][74:95] = 0.1
        if torque_mpc or directional or surface:
            for key in ("pd_requested_torque_nm", "torque_nm"):
                trace[key] = np.zeros((300, 1, 10, 29)) if surface else mpc_trace[key].copy()
            trace["root_pose_xyzw_m"] = np.zeros((300, 1, 7))
            trace["root_pose_xyzw_m"][:, 0, 6] = 1
            trace["ball_linear_velocity_before_step_m_s"] = np.zeros((300, 1, 3))
        c = json.loads(Path(r["witnesses"]["commitment"]["path"]).read_text())
        if directional:
            c["physics"] = copy.deepcopy(mpc_commitment["physics"])
            for key in ("joint_target_rad", "joint_position_rad", "joint_velocity_rad_s"):
                trace[key] = mpc_trace[key].copy()
        draw = plan.draws()[job["group"]]
        if job["context_index"] is None:
            c["executed_motor_policy"]["step_motor_proof"]["model"]["seed"] = draw.sampling_seed
            trace["canonical_qpos"][0, 0, 36:43] = (*draw.course[:2], 0.13, 1, 0, 0, 0)
            trace["canonical_qvel"][0, 0, 35:41] = (
                draw.course[2],
                0,
                0,
                0,
                draw.course[2] / 0.11,
                0,
            )
            r.pop("planning_context_hash", None)
        if ablated:
            if not (torque_mpc or directional or surface):
                trace["taskspace_joint_increment_rad"][75:] = 0
            if tamper == "postcontact_nonzero" and index == 0:
                trace["taskspace_joint_increment_rad"][76] = 0.01
            c.update(
                foundation_observation_capture=dict(synthetic=True),
                execution_profile="taskspace_plus_motor",
            )
            if surface:
                c["default_locomotion_reference_capture"] = dict(synthetic=True)
                if job["role"] != "noninterference":
                    from rosclaw_soccer.rsi.mesh_surface_swing_probe import (
                        inactive_surface_probe_record,
                    )

                    trace["mesh_surface_swing_record"] = np.stack(
                        [inactive_surface_probe_record(f) for f in range(300)]
                    )[:, None]
                    trace["mesh_surface_swing_record"][40, 0, 1:3] = [0, 1]
                    c["mesh_surface_swing"] = surface_contract
                    r["mesh_surface_swing"] = surface_contract
                    if tamper == "surface_nonfoot_active":
                        trace["mesh_surface_swing_record"][100, 0, 1:3] = [0, 1]
                elif tamper == "surface_disabled_enabled":
                    c["mesh_surface_swing"] = surface_contract
            elif torque_mpc or directional:
                c[contract_key] = contract
                trace.update({k: v.copy() for k, v in mpc_trace.items()})
                r[contract_key] = contract
                r[counter_key] = mpc_report[counter_key]
                r["forecast_physical_substeps"] = 0
            else:
                c["swing_release_ablation"] = release_ablation_contract(
                    config["mesh_task_contact_world_hash"]
                )
                r["swing_release_ablation"] = c["swing_release_ablation"]
        write(folder / "commitment.json", c, key=None)
        np.savez(folder / "trace.npz", **trace)
        old_report = json.loads(Path(r["witnesses"]["report"]["path"]).read_text())
        report = write(
            folder / "report.json",
            dict(
                old_report,
                **c,
                **(
                    {k: v for k, v in mpc_report.items() if k != contract_key}
                    if (torque_mpc or directional) and ablated
                    else {}
                ),
                commitment_hash=hash_json(c),
                physical_trace_hash=audit.digest(folder / "trace.npz"),
                **(
                    dict(course=list(draw.course), seed=draw.seed, lane=draw.lane)
                    if job["context_index"] is None
                    else {}
                ),
            ),
        )
        outcome = label_first_touch(trace)
        review = json.loads(Path(r["witnesses"]["independent_review"]["path"]).read_text())
        review = write(
            folder / "independent.json",
            dict(
                review,
                reviewed_report_hash=report["report_hash"],
                high_quality=outcome.task_pass,
                clean_foot_only=outcome.safe,
                **(
                    dict(
                        contact_exit_torque_mpc_review=dict(
                            schema="soccer.rsi.contact_exit_torque_mpc_reconstruction.v1",
                            contract=contract,
                            planning_counters=mpc_report[counter_key],
                            frames_reconstructed=300,
                            optimization_reconstructed_with_same_pinned_implementation=True,
                            independent_optimization_implementation=False,
                            locally_bitexact_active_intervals=0,
                            chosen_control_forecast_attempts=0,
                            chosen_control_attempted_forecast_steps=0,
                            chosen_control_integrated_forecast_steps=0,
                            source_end_state_hash_matches=0,
                            source_numeric_replay_tolerance=1e-8,
                            hardware_authorized=False,
                            promotion_authorized=False,
                        )
                    )
                    if (torque_mpc or directional) and ablated
                    else {}
                ),
            ),
        )
        if directional and ablated:
            gain_review = copy.deepcopy(review["contact_exit_torque_mpc_review"])
            gain_review["schema"] = "soccer.rsi.directional_compliance_reconstruction.v1"
            gain_review["gain_law_reconstructed_with_same_pinned_implementation"] = True
            gain_review["independent_gain_law_implementation"] = False
            gain_review["planning_counters"] = mpc_report[counter_key]
            review.pop("contact_exit_torque_mpc_review")
            review["directional_contact_compliance_review"] = gain_review
            review = write(folder / "independent.json", review)
        if surface and ablated and job["role"] != "noninterference":
            review["mesh_surface_swing_review"] = dict(
                schema="soccer.rsi.mesh_surface_swing_reconstruction.v1",
                contract=surface_contract,
                frames_reconstructed=300,
                active_frames=1,
                reconstruction_uses_same_pinned_geometry_and_ik=True,
                independent_ik_implementation=False,
                side_and_release_reconstructed_from_replayed_history=True,
                source_numeric_replay_tolerance=1e-8,
                hardware_authorized=False,
                promotion_authorized=False,
            )
            if tamper == "surface_replay_claim":
                review["mesh_surface_swing_review"]["independent_ik_implementation"] = True
            if tamper == "surface_frame_count":
                review["mesh_surface_swing_review"]["active_frames"] = 0
            review = write(folder / "independent.json", review)
        foundation = json.loads(Path(r["witnesses"]["foundation_review"]["path"]).read_text())
        foundation = write(
            folder / "foundation.json",
            dict(
                foundation,
                physical_report_hash=report["report_hash"],
                physical_trace_hash=report["physical_trace_hash"],
            ),
        )

        def witness(path):
            return dict(path=str(path), sha256=audit.digest(path))

        r["witnesses"].update(
            {
                k: witness(folder / filename)
                for k, filename in (
                    ("trace", "trace.npz"),
                    ("report", "report.json"),
                    ("commitment", "commitment.json"),
                    ("independent_review", "independent.json"),
                    ("foundation_review", "foundation.json"),
                )
            }
        )
        # Reused MJB/payload are already owned by the synthetic screen. The real
        # reader requires all new witnesses to be owned by the new study.
        for key in ("world", "proof_payload"):
            dest = root / Path(r["witnesses"][key]["path"]).name
            if not dest.exists():
                dest.write_bytes(Path(r["witnesses"][key]["path"]).read_bytes())
            r["witnesses"][key] = witness(dest)
        r.update(
            **job,
            outcome=outcome.to_dict(),
            physical_report_hash=report["report_hash"],
            qualification_alignment=require_first_touch_qualification(review, foundation, outcome),
        )
        if not ablated:
            r["job"] = (
                f"option-4-raw-{job['group']}"
                if job["group"] in (8, 56)
                else f"teacher-raw-{job['group']}"
                if job["context_index"] is None
                else f"teacher-context-{job['context_index']}"
            )
        else:
            intent = write(
                folder / "intent.json",
                dict(
                    job,
                    candidate=r["candidate"],
                    context_hash=r.get("planning_context_hash"),
                    baseline_receipt_hash=baseline["receipt_hash"],
                    declaration_hash=declaration["report_hash"],
                ),
            )
            if tamper == "intent_baseline" and index == 0:
                intent = write(
                    folder / "intent.json", dict(intent, baseline_receipt_hash=hash_json("other"))
                )
            r["intent_witness"] = witness(folder / "intent.json")
            for key in ("encoder_witness", "planning_context_witness"):
                if key in r:
                    dest = folder / Path(r[key]["path"]).name
                    dest.write_bytes(Path(r[key]["path"]).read_bytes())
                    r[key] = witness(dest)
            cutoff = (300 if job["role"] == "noninterference" else 40) if surface else 75
            prefix = dict(
                schema="soccer.rsi.surface_pre_activation_prefix_parity.v1"
                if surface
                else "soccer.rsi.first_contact_prefix_parity.v1",
                baseline_receipt_hash=baseline["receipt_hash"],
                prefix_frames=cutoff,
                all_original_array_prefixes_bit_equal=True,
                prefix_array_hashes={
                    k: hash_bytes(v[:cutoff].tobytes())
                    for k, v in trace.items()
                    if not k.startswith(
                        ("contact_exit_mpc_", "directional_compliance_", "mesh_surface_")
                    )
                },
            )
            if tamper == "prefix_hash" and index == 0:
                prefix["prefix_array_hashes"]["force_n"] = hash_json("other")
            write(folder / "prefix.json", prefix)
            r["prefix_witness"] = witness(folder / "prefix.json")
        return write(
            root / f"{'receipt' if ablated else 'baseline'}-{index:04d}.json", r, "receipt_hash"
        )

    baselines = [endpoint(i) for i in range(6 if surface else 4)]
    for i in range(2, len(baselines)):
        write(screen_root / f"baseline-{i:04d}.json", baselines[i], "receipt_hash")
    baseline_items = [
        dict(
            path=str((screen_root if i >= 2 else root) / f"baseline-{i:04d}.json"),
            file_hash=audit.digest((screen_root if i >= 2 else root) / f"baseline-{i:04d}.json"),
            receipt_hash=r["receipt_hash"],
            pair_id=spec["jobs"][i + 1 if surface else i]["pair_id"],
        )
        for i, r in enumerate(baselines)
    ]
    config.update(
        output_root=str(root),
        groups=spec["groups"],
        max_attempts=spec["count"],
        new_physical_budget=spec["count"],
        suppress_postcontact_swing_release=not (torque_mpc or directional or surface),
        baseline_receipts=baseline_items,
        baseline_sources=[],
        required_receipt_witnesses=["intent_witness", "encoder_witness", "prefix_witness"],
    )
    if torque_mpc:
        config["use_contact_exit_torque_mpc"] = True
    if directional:
        config["use_directional_contact_compliance"] = True
    if surface:
        config["use_mesh_surface_swing"] = True
    config_path = root / "config.json"
    write(config_path, config, key=None)
    pins = {str(config_path): audit.digest(config_path)}
    prepared = SimpleNamespace(PINS=pins, HEADS={}, stable=lambda: None)
    monkeypatch.setattr(
        first_touch_campaign_worker, "prepare", lambda p: (prepared, None, manifest, config)
    )
    old_decl = json.loads((screen_root / "declaration.json").read_text())
    reused = [
        dict(role="control", pair_id=i["pair_id"], receipt=r)
        for i, r in zip(baseline_items, baselines, strict=True)
    ]
    declaration = write(
        root / "declaration.json",
        dict(
            old_decl,
            **config,
            schema=spec["declaration_schema"],
            jobs=spec["jobs"],
            reused_endpoints=reused,
            input_file_hashes=pins,
            prepared_pins=pins,
            source_heads={},
        ),
    )
    receipts = [
        endpoint(
            i,
            ablated=True,
            declaration=declaration,
            baseline=baselines[max(i - 1, 0) if surface else i],
        )
        for i in range(spec["count"])
    ]
    # Synthetic prior audit rows bind precisely the source endpoints; payload
    # reconstruction remains mocked by expanded_fixture, not physical evidence.
    catalog_path = Path(config["selection_declaration"]).parent / "complete.json"
    catalog_complete = json.loads(catalog_path.read_text())
    catalog_complete["results"][8:10] = baselines[:2]
    catalog_complete = write(catalog_path, catalog_complete)
    upstream_path = Path(config["upstream_catalog_readback"])
    upstream = json.loads(upstream_path.read_text())
    write(
        upstream_path,
        dict(
            upstream,
            complete_hash=catalog_complete["report_hash"],
            rows=[
                dict(job=r["job"], receipt_hash=r["receipt_hash"], outcome=r["outcome"])
                for r in baselines[:2]
            ],
        ),
    )
    screen_complete = json.loads((screen_root / "complete.json").read_text())
    screen_complete["results"][1], screen_complete["results"][3] = baselines[2:4]
    if surface:
        screen_complete["results"][16:18] = baselines[4:]
    screen_complete = write(screen_root / "complete.json", screen_complete)
    screen_source = tmp_path / "screen-reader.py"
    screen_source.write_text("synthetic prior reader")
    screen_readback = tmp_path / "screen-readback.json"
    write(
        screen_readback,
        dict(
            schema="soccer.rsi.mesh_option4_teacher_screen_terminal_readback.v1",
            complete_hash=screen_complete["report_hash"],
            source_native_bodies=18,
            reader_source_hash=audit.digest(screen_source),
            whole_input_and_witness_files_verified=tamper != "upstream_unverified",
            shared_payload_logical_seals_rebuilt=True,
            prior_independent_replays_authenticated=True,
            rows=[
                dict(job=r["job"], receipt_hash=r["receipt_hash"], outcome=r["outcome"])
                for r in baselines[2:]
            ],
        ),
    )
    state = dict(active=None, attempts=0, completed={}, declaration_hash=hash_json(declaration))
    events = []
    for i, r in enumerate(receipts):
        intent = json.loads(Path(r["intent_witness"]["path"]).read_text())
        state.update(next_intent_hash=intent["report_hash"], active=None)
        events.append(dict(event="INTENT_COMMITTED_BEFORE_BODY", state=copy.deepcopy(state)))
        state.update(attempts=i + 1, active=dict(job=r["job"], attempt=i))
        events.append(dict(event="ATTEMPT_STARTED", state=copy.deepcopy(state)))
        state["completed"][r["job"]] = r
        state["active"] = None
    state["phase"] = spec["phase"]
    events.append(dict(event=spec["phase"], state=copy.deepcopy(state)))
    with sqlite3.connect(root / "campaign.sqlite") as db:
        db.execute(
            "CREATE TABLE events(seq INTEGER PRIMARY KEY, previous TEXT, body TEXT, digest TEXT)"
        )
        previous = "GENESIS"
        for i, e in enumerate(events, 1):
            h = hash_json(dict(seq=i, previous=previous, body=e))
            db.execute("INSERT INTO events VALUES(?,?,?,?)", (i, previous, json.dumps(e), h))
            previous = h
    paired = assess_pairs(
        {i["pair_id"]: r["outcome"] for i, r in zip(baseline_items, baselines, strict=True)},
        {
            i["pair_id"]: r["outcome"]
            for i, r in zip(
                spec["jobs"][1:] if surface else spec["jobs"],
                receipts[1:] if surface else receipts,
                strict=True,
            )
        },
    )
    write(
        root / "complete.json",
        dict(
            schema=spec["complete_schema"],
            declaration_hash=declaration["report_hash"],
            new_native_bodies=spec["count"],
            native_substeps=3000 * spec["count"],
            independent_cpu_substeps=3000 * spec["count"],
            new_optimizer_updates=0,
            independent_course_count=6 if surface else 4,
            paired_endpoint_count=12 if surface else 8,
            reused_endpoints=reused,
            results=receipts,
            paired_development=paired,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
            **(
                dict(
                    disabled_complete_300_frame_noninterference_verified=True,
                    original_none_raw_retention=assess_pairs(
                        {
                            f"raw-{r['group']}": r["outcome"]
                            for r in json.loads(Path(config["nominal_complete"]).read_text())[
                                "results"
                            ]
                        },
                        {
                            j["pair_id"]: r["outcome"]
                            for j, r in zip(spec["jobs"][1:], receipts[1:], strict=True)
                            if j["context_index"] is None
                        },
                    ),
                )
                if surface
                else {}
            ),
            **(
                dict(
                    native_forecast_substeps=0,
                    independent_optimization_forecast_substeps=0,
                    independent_chosen_control_forecast_substeps=0,
                )
                if torque_mpc or directional
                else {}
            ),
        ),
    )
    output = root / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--config",
            str(config_path),
            "--unit",
            "fake.service",
            "--output",
            str(output),
            "--profile",
            profile,
            "--upstream-audit-source",
            str(tmp_path / "upstream-reader.py"),
            "--baseline-screen-readback",
            str(screen_readback),
            "--baseline-screen-audit-source",
            str(screen_source),
        ],
    )
    return output


def test_full_four_reader_including_unsafe_baseline_and_both_prior_audits(tmp_path, monkeypatch):
    output = fixture(tmp_path, monkeypatch)
    audit.main()
    report = json.loads(output.read_text())
    assert len(report["rows"]) == 4 and report["reused_endpoint_count"] == 4
    assert report["all_original_first_contact_prefixes_bit_equal"] is True
    assert report["actual_postcontact_taskspace_increment_exactly_zero"] is True
    assert report["rows"][2]["outcome"]["safe"] is False
    assert report["optimized_recovery_claimed"] is False


def test_full_contact_exit_reader_retains_release_and_separates_forecast_accounting(
    tmp_path, monkeypatch
):
    output = fixture(tmp_path, monkeypatch, torque_mpc=True)
    audit.main()
    report = json.loads(output.read_text())
    assert report["schema"] == "soccer.rsi.contact_exit_torque_mpc_terminal_readback.v1"
    assert len(report["rows"]) == 4 and report["reused_endpoint_count"] == 4
    assert report["all_original_first_contact_prefixes_bit_equal"] is True
    assert "actual_postcontact_taskspace_increment_exactly_zero" not in report
    assert report["independent_optimization_implementation"] is False
    assert report["native_forecast_substeps"] == 0
    assert report["independent_chosen_control_forecast_substeps"] == 0
    assert report["rows"][2]["outcome"]["safe"] is False


def test_full_directional_reader_keeps_old_unsafe_case_and_does_not_claim_independent_gain_law(
    tmp_path, monkeypatch
):
    output = fixture(tmp_path, monkeypatch, directional=True)
    audit.main()
    report = json.loads(output.read_text())
    assert report["schema"] == "soccer.rsi.directional_contact_compliance_terminal_readback.v1"
    assert len(report["rows"]) == 4 and report["rows"][2]["outcome"]["safe"] is False
    assert report["independent_gain_law_implementation"] is False
    assert report["gain_law_reconstruction_uses_same_pinned_implementation"] is True
    assert "actual_postcontact_taskspace_increment_exactly_zero" not in report


def test_full_surface_seven_keeps_disabled_body_six_pairs_and_original_none_retention(
    tmp_path, monkeypatch
):
    output = fixture(tmp_path, monkeypatch, surface=True)
    audit.main()
    report = json.loads(output.read_text())
    assert report["schema"] == "soccer.rsi.mesh_surface_swing_terminal_readback.v1"
    assert len(report["rows"]) == 7
    assert report["reused_endpoint_count"] == 6
    assert report["independent_course_count"] == 6
    assert report["paired_endpoint_count"] == 12
    assert report["disabled_complete_300_frame_noninterference_verified"] is True
    assert report["all_original_pre_activation_prefixes_bit_equal"] is True
    assert "all_original_first_contact_prefixes_bit_equal" not in report
    assert "actual_postcontact_taskspace_increment_exactly_zero" not in report
    assert report["independent_ik_implementation"] is False
    assert report["rows"][3]["outcome"]["safe"] is False
    assert report["teacher_qualified"] is False


@pytest.mark.parametrize(
    "tamper",
    [
        "surface_nonfoot_active",
        "surface_disabled_enabled",
        "surface_replay_claim",
        "surface_frame_count",
        "intent_baseline",
        "prefix_hash",
        "upstream_unverified",
    ],
)
def test_surface_reader_rejects_resealed_inconsistent_evidence(tmp_path, monkeypatch, tamper):
    output = fixture(tmp_path, monkeypatch, tamper, surface=True)
    with pytest.raises(ValueError):
        audit.main()
    assert not output.exists()


@pytest.mark.parametrize("tamper", ["intent_baseline", "prefix_hash", "upstream_unverified"])
def test_contact_exit_reader_rejects_resealed_lineage_and_prefix_tampering(
    tmp_path, monkeypatch, tamper
):
    output = fixture(tmp_path, monkeypatch, tamper, torque_mpc=True)
    with pytest.raises(ValueError):
        audit.main()
    assert not output.exists()


@pytest.mark.parametrize(
    "tamper", ["postcontact_nonzero", "intent_baseline", "prefix_hash", "upstream_unverified"]
)
def test_four_reader_rejects_resealed_intervention_and_upstream_failures(
    tmp_path, monkeypatch, tamper
):
    output = fixture(tmp_path, monkeypatch, tamper)
    with pytest.raises(ValueError):
        audit.main()
    assert not output.exists()
