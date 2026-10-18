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
from rosclaw_soccer.sim.contracts import hash_json
from scripts import rsi_read_mesh_contact_native_pilot as audit


def write(path, value, key="report_hash"):
    if key:
        value[key] = hash_json({k: v for k, v in value.items() if k != key})
    path.write_text(json.dumps(value))
    return value


def fixture(tmp_path, monkeypatch, *, tamper=None):
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
    declaration = write(
        root / "declaration.json",
        dict(
            config,
            jobs=jobs,
            input_file_hashes={str(config_path): audit.digest(config_path)},
            prepared_pins={},
            source_heads={},
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
        np.savez(folder / "trace.npz", **trace)
        outcome = label_first_touch(trace)
        commitment = dict(
            compiled_model_hash=world_hash,
            mesh_task_contact_protocol=dict(expected_world_hash=world_hash),
        )
        write(folder / "commitment.json", commitment, key=None)
        report = write(
            folder / "report.json",
            dict(
                commitment,
                commitment_hash=hash_json(commitment),
                course=[2.3, 0.01, -0.4],
                seed=1,
                lane=0,
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
        intent = write(
            folder / "intent.json",
            dict(
                job=name,
                context_hash=hash_json(jobs[index]["context"]),
                option_index=index,
                candidate_hash=hash_json(jobs[index]["candidate"]),
                world_hash=world_hash,
                declaration_hash=declaration["report_hash"],
            ),
        )
        write(folder / "encoder.json", dict(synthetic=True))
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
                planning_context_hash=intent["context_hash"],
            ),
            "receipt_hash",
        )
        state["next_intent_hash"] = intent["report_hash"]
        events.append(
            dict(event="NEW_WORLD_INTENT_COMMITTED_BEFORE_BODY", state=copy.deepcopy(state))
        )
        state.update(active=dict(job=name, attempt=index), attempts=index + 1)
        events.append(dict(event="ATTEMPT_STARTED", state=copy.deepcopy(state)))
        state["completed"][name] = receipt
        state["active"] = None
        receipts.append(receipt)
    state["phase"] = "CONTACT_WORLD_DIAGNOSTIC_COMPLETE_NO_PROMOTION"
    events.append(dict(event="THREE_BODY_DIAGNOSTIC_COMPLETE", state=copy.deepcopy(state)))
    write(
        root / "complete.json",
        dict(
            declaration_hash=declaration["report_hash"],
            new_native_bodies=3,
            native_physical_substeps=9000,
            independent_cpu_substeps=9000,
            independent_foundation_calls=900,
            new_optimizer_updates=0,
            results=receipts,
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
