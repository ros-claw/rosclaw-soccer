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


def fixture(tmp_path, monkeypatch, tamper=None, *, torque_mpc=False):
    expanded_fixture(tmp_path, monkeypatch)
    prepared, _, manifest, config = first_touch_campaign_worker.prepare(None)
    screen_root = tmp_path / "teacher"
    templates = [
        json.loads((screen_root / f"receipt-{i:04d}.json").read_text()) for i in (16, 17, 1, 3)
    ]
    root = tmp_path / "release"
    root.mkdir()
    profile = "contact-exit-four" if torque_mpc else "release-four"
    spec = audit.bounded_profile(profile)
    if torque_mpc:
        from tests.rsi.test_contact_exit_torque_evidence import capture

        mpc_report, mpc_commitment, mpc_trace = capture()
        contract = mpc_commitment["contact_exit_torque_mpc"]
        source = Path(config["source_root"])
        source_files = {
            "teacher": "rsi/contact_exit_torque_mpc.py",
            "forecast": "physics/g1_pd_torque_forecast.py",
            "kinematics": "physics/g1_joint_task_kinematics.py",
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
        mpc_report["contact_exit_torque_mpc"] = copy.deepcopy(contract)
    plan = PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), "sha256:" + "a" * 64, 18)

    def endpoint(index, *, ablated=False, declaration=None, baseline=None):
        job = spec["jobs"][index]
        r = copy.deepcopy(templates[index])
        folder = root / (f"new-{index}" if ablated else f"baseline-{index}")
        folder.mkdir()
        with np.load(r["witnesses"]["trace"]["path"], allow_pickle=False) as archive:
            trace = dict(archive)
        trace["force_n"][74, 0, 0] = 20
        if index == 2:
            trace["force_n"][99, 0, 4] = 20
        else:
            trace["ball_position_after_step_m"][134, 0, 0] = 2
        trace["taskspace_joint_increment_rad"] = np.zeros((300, 1, 29))
        trace["taskspace_joint_increment_rad"][74:95] = 0.1
        if torque_mpc:
            for key in ("pd_requested_torque_nm", "torque_nm"):
                trace[key] = mpc_trace[key].copy()
        c = json.loads(Path(r["witnesses"]["commitment"]["path"]).read_text())
        draw = plan.draws()[job["group"]]
        if index < 2:
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
            if not torque_mpc:
                trace["taskspace_joint_increment_rad"][75:] = 0
            if tamper == "postcontact_nonzero" and index == 0:
                trace["taskspace_joint_increment_rad"][76] = 0.01
            c.update(
                foundation_observation_capture=dict(synthetic=True),
                execution_profile="taskspace_plus_motor",
            )
            if torque_mpc:
                c["contact_exit_torque_mpc"] = contract
                trace.update({k: v.copy() for k, v in mpc_trace.items()})
                r.update(
                    contact_exit_torque_mpc=contract,
                    contact_exit_mpc_counters=mpc_report["contact_exit_mpc_counters"],
                    forecast_physical_substeps=0,
                )
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
                    {k: v for k, v in mpc_report.items() if k != "contact_exit_torque_mpc"}
                    if torque_mpc and ablated
                    else {}
                ),
                commitment_hash=hash_json(c),
                physical_trace_hash=audit.digest(folder / "trace.npz"),
                **(
                    dict(course=list(draw.course), seed=draw.seed, lane=draw.lane)
                    if index < 2
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
                            planning_counters=mpc_report["contact_exit_mpc_counters"],
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
                    if torque_mpc and ablated
                    else {}
                ),
            ),
        )
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
                f"option-4-raw-{job['group']}" if index < 2 else f"teacher-context-{index - 2}"
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
            prefix = dict(
                schema="soccer.rsi.first_contact_prefix_parity.v1",
                baseline_receipt_hash=baseline["receipt_hash"],
                prefix_frames=75,
                all_original_array_prefixes_bit_equal=True,
                prefix_array_hashes={
                    k: hash_bytes(v[:75].tobytes())
                    for k, v in trace.items()
                    if not k.startswith("contact_exit_mpc_")
                },
            )
            if tamper == "prefix_hash" and index == 0:
                prefix["prefix_array_hashes"]["force_n"] = hash_json("other")
            write(folder / "prefix.json", prefix)
            r["prefix_witness"] = witness(folder / "prefix.json")
        return write(
            root / f"{'receipt' if ablated else 'baseline'}-{index:04d}.json", r, "receipt_hash"
        )

    baselines = [endpoint(i) for i in range(4)]
    for i in (2, 3):
        write(screen_root / f"baseline-{i:04d}.json", baselines[i], "receipt_hash")
    baseline_items = [
        dict(
            path=str((screen_root if i >= 2 else root) / f"baseline-{i:04d}.json"),
            file_hash=audit.digest((screen_root if i >= 2 else root) / f"baseline-{i:04d}.json"),
            receipt_hash=r["receipt_hash"],
            pair_id=spec["jobs"][i]["pair_id"],
        )
        for i, r in enumerate(baselines)
    ]
    config.update(
        output_root=str(root),
        groups=spec["groups"],
        max_attempts=4,
        new_physical_budget=4,
        suppress_postcontact_swing_release=not torque_mpc,
        baseline_receipts=baseline_items,
        baseline_sources=[],
        required_receipt_witnesses=["intent_witness", "encoder_witness", "prefix_witness"],
    )
    if torque_mpc:
        config["use_contact_exit_torque_mpc"] = True
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
        endpoint(i, ablated=True, declaration=declaration, baseline=baselines[i]) for i in range(4)
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
    screen_complete["results"][1], screen_complete["results"][3] = baselines[2:]
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
        {i["pair_id"]: r["outcome"] for i, r in zip(spec["jobs"], receipts, strict=True)},
    )
    write(
        root / "complete.json",
        dict(
            schema=spec["complete_schema"],
            declaration_hash=declaration["report_hash"],
            new_native_bodies=4,
            native_substeps=12000,
            independent_cpu_substeps=12000,
            new_optimizer_updates=0,
            independent_course_count=4,
            paired_endpoint_count=8,
            reused_endpoints=reused,
            results=receipts,
            paired_development=paired,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
            **(
                dict(
                    native_forecast_substeps=0,
                    independent_optimization_forecast_substeps=0,
                    independent_chosen_control_forecast_substeps=0,
                )
                if torque_mpc
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
