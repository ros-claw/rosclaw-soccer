"""Synthetic file/ledger admission tests. Shared payload reconstruction is mocked.

These tests DO NOT qualify any physics or independent physical replay.
"""

import copy
import json
import sqlite3
import sys

import numpy as np
import pytest

from rosclaw_soccer.rsi import failure_curriculum_evidence, fixed_physical_payload_reader
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.first_touch_qualification import require_first_touch_qualification
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts import rsi_read_mesh_contact_native_pilot as audit


def write(path, value, key="report_hash"):
    if key:
        value[key] = hash_json({k: v for k, v in value.items() if k != key})
    path.write_text(json.dumps(value))
    return value


def fixture(tmp_path, monkeypatch, *, tamper=None, public_four=False):
    root = tmp_path / "run"
    root.mkdir()
    output = tmp_path / "audit.json"
    world = root / "mesh.mjb"
    world.write_bytes(b"synthetic model bytes, not an actual MJB")
    world_hash = audit.digest(world)
    config = dict(
        source_root=str(tmp_path),
        core_root=str(tmp_path),
        output_root=str(root),
        mesh_task_contact_world_hash=world_hash,
    )
    config_path = tmp_path / "config.json"
    if public_four:
        config.update(groups=[4, 8, 32, 56], max_attempts=4, new_physical_budget=4)
    write(config_path, config, key=None)
    names = ("old64-option3", "new96-option3", "unsafe104-option7")
    jobs = [
        dict(
            job=name,
            option_index=i,
            candidate=dict(index=i),
            context=dict(course=[2.3, 0.01, -0.4], source_seed=1, source_lane=0),
        )
        for i, name in enumerate(names)
    ]
    draws = None
    if public_four:
        names = tuple(f"original-public-{g}" for g in (4, 8, 32, 56))
        draws = PublicFirstTouchTrainingPlan(
            tuple(range(20261060, 20261070)), "sha256:" + "a" * 64, 18
        ).draws()
        jobs = [
            dict(job=name, group=g, candidate=None, course=list(draws[g].course))
            for name, g in zip(names, (4, 8, 32, 56), strict=True)
        ]
        if tamper == "perturbed_course":
            jobs[0]["course"][0] += 0.001
    declaration = write(
        root / "declaration.json",
        dict(
            config,
            jobs=jobs,
            input_file_hashes={str(config_path): audit.digest(config_path)},
            prepared_pins={},
            source_heads={},
            **(dict(schema="soccer.rsi.mesh_public_draw_declaration.v1") if public_four else {}),
        ),
    )
    events = []
    receipts = []
    state = dict(phase="READY_FOR_NEXT_JOB", active=None, attempts=0, completed={})
    for index, name in enumerate(names):
        folder = root / str(index)
        folder.mkdir()
        trace = dict(
            force_n=np.zeros((300, 1, 6)),
            pelvis_z_per_substep_m=np.full((300, 1, 10), 0.7),
            ball_position_after_step_m=np.zeros((300, 1, 3)),
            canonical_qpos=np.zeros((300, 1, 43)),
        )
        trace["canonical_qpos"][0, 0, 36:39] = [2.3, 0.01, 0.13]
        course, seed, lane = [2.3, 0.01, -0.4], 1, 0
        view = None
        if public_four:
            from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController

            draw = draws[jobs[index]["group"]]
            course, seed, lane = jobs[index]["course"], draw.seed, draw.lane
            q = trace["canonical_qpos"][0, 0]
            q[:7] = (0, 0, 0.793, 1, 0, 0, 0)
            q[7:36] = G1SonicRunupController.default_angles
            q[36:43] = (*course[:2], 0.13, 1, 0, 0, 0)
            trace["canonical_qvel"] = np.zeros((300, 1, 41))
            trace["canonical_qvel"][0, 0, 35:41] = (course[2], 0, 0, 0, course[2] / 0.11, 0)
            if tamper == "initial_spin":
                trace["canonical_qvel"][0, 0, 39] += 0.1
            view = dict(mean_model=dict(model_hash="sha256:" + "a" * 64), seed=draw.sampling_seed)
            if tamper == "sampling_seed":
                view["seed"] += 1
            trace["foundation_neural_encoder_features"] = np.zeros((300, 1, 640), dtype=np.float32)
            if tamper == "feature_dtype":
                trace["foundation_neural_encoder_features"] = trace[
                    "foundation_neural_encoder_features"
                ].astype(np.float64)
        np.savez(folder / "trace.npz", **trace)
        outcome = label_first_touch(trace)
        commitment = dict(
            compiled_model_hash=world_hash,
            mesh_task_contact_protocol=dict(expected_world_hash=world_hash),
        )
        if public_four:
            commitment.update(
                approach_mode="legacy",
                swing_mode="legacy",
                executed_motor_policy=dict(step_motor_proof=dict(model=view)),
            )
            commitment["default_locomotion_planner_events"] = [
                dict(frame=i, planner_mode=2, reference_hash=hash_json(i))
                for i in range(0, 300, 20)
            ]
        write(folder / "commitment.json", commitment, key=None)
        report = write(
            folder / "report.json",
            dict(
                commitment,
                commitment_hash=hash_json(commitment),
                course=course,
                seed=seed,
                lane=lane,
                physical_trace_hash=audit.digest(folder / "trace.npz"),
            ),
        )
        review = write(
            folder / "independent_review.json",
            dict(
                physical_substeps=3000,
                actual_recurrent_sampling_draws_reconstructed=270,
                high_quality=False,
                safety_passed=True,
                clean_foot_only=False,
                minimum_pelvis_z_m=0.7,
                reviewed_report_hash=report["report_hash"],
            ),
        )
        foundation = write(
            folder / "foundation_review.json",
            dict(
                recomputed_executed_calls=300,
                physical_report_hash=report["report_hash"],
                physical_trace_hash=report["physical_trace_hash"],
            ),
        )
        expected_intent = (
            dict(jobs[index], world_hash=world_hash, declaration_hash=declaration["report_hash"])
            if public_four
            else dict(
                job=name,
                context_hash=hash_json(jobs[index]["context"]),
                option_index=index,
                candidate_hash=hash_json(jobs[index]["candidate"]),
                world_hash=world_hash,
                declaration_hash=declaration["report_hash"],
            )
        )
        intent = write(
            folder / "intent.json",
            expected_intent,
        )
        encoder = dict(synthetic=True)
        if public_four:
            encoder = dict(
                schema="soccer.rsi.sonic_encoder_reference_consumption.v1",
                actual_cpu_onnx_planner_calls=15,
                compared_encoder_frames=300,
                compared_features_per_frame=640,
                recorded_features_hash=hash_bytes(
                    trace["foundation_neural_encoder_features"].tobytes()
                ),
                recorded_features_dtype="float32",
                maximum_absolute_error=0.0,
                planner_events=commitment["default_locomotion_planner_events"],
            )
            if tamper == "encoder_reference":
                encoder["recorded_features_hash"] = hash_json("wrong features")
        write(folder / "encoder.json", encoder)
        (folder / "proof.json").write_bytes(b"synthetic shared payload")
        files = dict(
            trace=folder / "trace.npz",
            report=folder / "report.json",
            commitment=folder / "commitment.json",
            independent_review=folder / "independent_review.json",
            foundation_review=folder / "foundation_review.json",
            proof_payload=folder / "proof.json",
            world=world,
        )

        def witness(path):
            return dict(path=str(path), sha256=audit.digest(path))

        receipt = write(
            root / f"receipt-{index:04d}.json",
            dict(
                job=name,
                physical_report_hash=report["report_hash"],
                outcome=outcome.to_dict(),
                qualification_alignment=require_first_touch_qualification(
                    review, foundation, outcome
                ),
                witnesses={k: witness(p) for k, p in files.items()},
                intent_witness=witness(folder / "intent.json"),
                encoder_witness=witness(folder / "encoder.json"),
                candidate=jobs[index]["candidate"],
                **(
                    dict(group=jobs[index]["group"])
                    if public_four
                    else dict(planning_context_hash=intent["context_hash"])
                ),
            ),
            "receipt_hash",
        )
        state["next_intent_hash"] = intent["report_hash"]
        events.append(
            dict(
                event=(
                    "PUBLIC_DRAW_INTENT_COMMITTED_BEFORE_BODY"
                    if public_four
                    else "NEW_WORLD_INTENT_COMMITTED_BEFORE_BODY"
                ),
                state=copy.deepcopy(state),
            )
        )
        state.update(active=dict(job=name, attempt=index), attempts=index + 1)
        events.append(dict(event="ATTEMPT_STARTED", state=copy.deepcopy(state)))
        state["completed"][name] = receipt
        state["active"] = None
        receipts.append(receipt)
    state["phase"] = (
        "PUBLIC_DRAW_EVALUATION_COMPLETE_NO_PROMOTION"
        if public_four
        else "CONTACT_WORLD_DIAGNOSTIC_COMPLETE_NO_PROMOTION"
    )
    events.append(dict(event="THREE_BODY_DIAGNOSTIC_COMPLETE", state=copy.deepcopy(state)))
    write(
        root / "complete.json",
        dict(
            declaration_hash=declaration["report_hash"],
            new_native_bodies=len(names),
            native_physical_substeps=3000 * len(names),
            independent_cpu_substeps=3000 * len(names),
            independent_foundation_calls=300 * len(names),
            new_optimizer_updates=0,
            results=receipts,
            **(
                dict(
                    schema="soccer.rsi.mesh_public_draw_complete.v1",
                    additional_encoder_reference_calls=1200,
                )
                if public_four
                else {}
            ),
        ),
    )
    if tamper == "intent_order":
        events[0]["state"]["next_intent_hash"] = hash_json("wrong")
    previous = "GENESIS"
    with sqlite3.connect(root / "campaign.sqlite") as db:
        db.execute("CREATE TABLE events(seq INTEGER, previous TEXT, body TEXT, digest TEXT)")
        for seq, event in enumerate(events, 1):
            link = hash_json(dict(seq=seq, previous=previous, body=event))
            db.execute(
                "INSERT INTO events VALUES(?,?,?,?)", (seq, previous, json.dumps(event), link)
            )
            previous = link
        if tamper == "chain":
            db.execute("UPDATE events SET digest='broken' WHERE seq=1")
    if tamper == "trace":
        (root / "0/trace.npz").write_bytes(b"changed")
    if tamper == "config":
        config_path.write_text(json.dumps(dict(config, extra=True)))
    monkeypatch.setattr(
        audit.subprocess,
        "check_output",
        lambda *a, **k: "MainPID=0\nActiveState=inactive\nResult=success\n",
    )
    # Only shared-payload loading is mocked: all byte pins, ledger links,
    # actual primitive labels and qualification compositions remain exercised.
    monkeypatch.setattr(
        fixed_physical_payload_reader, "FixedPhysicalPayloadReader", lambda *a, **k: None
    )

    def fake_sealed(path, **kwargs):
        value = json.loads(path.read_text())
        assert value["report_hash"] == hash_json(
            {k: v for k, v in value.items() if k != "report_hash"}
        )
        return value

    monkeypatch.setattr(failure_curriculum_evidence, "_sealed", fake_sealed)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--config",
            str(config_path),
            "--unit",
            "synthetic.service",
            "--output",
            str(output),
            *(["--profile", "original-public-four"] if public_four else []),
        ],
    )
    return output


def test_synthetic_complete_chain_and_failed_task_labels_are_preserved(tmp_path, monkeypatch):
    output = fixture(tmp_path, monkeypatch)
    audit.main()
    report = json.loads(output.read_text())
    assert len(report["rows"]) == 3
    assert all(not row["outcome"]["task_pass"] for row in report["rows"])
    assert report["new_physical_replays"] == report["new_optimizer_updates"] == 0


@pytest.mark.parametrize("tamper", ["chain", "trace", "config", "intent_order"])
def test_synthetic_tampering_rejected_without_output(tmp_path, monkeypatch, tamper):
    output = fixture(tmp_path, monkeypatch, tamper=tamper)
    with pytest.raises(ValueError):
        audit.main()
    assert not output.exists()


def test_nonterminal_unit_is_not_read_as_completed(tmp_path, monkeypatch):
    output = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        audit.subprocess,
        "check_output",
        lambda *a, **k: "MainPID=12\nActiveState=active\nResult=success\n",
    )
    with pytest.raises(ValueError, match="finished worker"):
        audit.main()
    assert not output.exists()


def test_four_original_public_draws_have_distinct_terminal_profile(tmp_path, monkeypatch):
    output = fixture(tmp_path, monkeypatch, public_four=True)
    audit.main()
    report = json.loads(output.read_text())
    assert len(report["rows"]) == 4
    assert report["schema"] == "soccer.rsi.mesh_original_public_terminal_readback.v1"
    assert all(not row["outcome"]["task_pass"] for row in report["rows"])


@pytest.mark.parametrize(
    "tamper",
    [
        "chain",
        "trace",
        "config",
        "intent_order",
        "initial_spin",
        "perturbed_course",
        "sampling_seed",
        "encoder_reference",
        "feature_dtype",
    ],
)
def test_four_case_binding_rejects_consistently_resealed_but_wrong_public_states(
    tmp_path, monkeypatch, tamper
):
    output = fixture(tmp_path, monkeypatch, public_four=True, tamper=tamper)
    with pytest.raises(ValueError):
        audit.main()
    assert not output.exists()


def test_three_body_profile_cannot_certify_four_case_campaign(tmp_path, monkeypatch):
    output = fixture(tmp_path, monkeypatch, public_four=True)
    monkeypatch.setattr(sys, "argv", sys.argv[:-2])
    with pytest.raises(ValueError, match="whole-body accounting"):
        audit.main()
    assert not output.exists()
