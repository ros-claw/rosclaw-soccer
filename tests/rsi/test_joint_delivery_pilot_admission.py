import copy
import json
import subprocess
import sys

import pytest

from rosclaw_soccer.rsi.joint_delivery_teacher import pilot_config
from scripts.rsi_joint_delivery_pilot_common import admit
from scripts.rsi_native_context_recovery_common import digest


def fixture(tmp_path):
    """Receipt/namespace fixture ONLY, not a physical qualification proof."""
    cfg = pilot_config(25)
    source_root = tmp_path / "source"
    source_root.mkdir()
    own_root = tmp_path / "own"
    own_root.mkdir()
    flags = dict(hardware_authorized=False, promotion_authorized=False, goal_complete=False)
    pins = {}
    for name in ("declaration.json", "preflight.json", "scene.xml"):
        p = source_root / name
        p.write_text("{}")
        pins[str(p)] = digest(p)
    trained = dict(**flags, config=dict(seed=115025), new_head_steps=2048, bank_hash="bank")
    records = [
        dict(index=i, course=dict(state_id=i // 2, task=i % 2, assisted=False), artifacts={})
        for i in range(80)
    ]
    tr = source_root / "trial51"
    tr.mkdir()
    for name in ("initial", "physical", "A", "B"):
        p = tr / (name + ".npz")
        p.write_bytes(b"namespace-only")
        records[51]["artifacts"][name] = dict(path=str(p.relative_to(source_root)), hash=digest(p))
    source = dict(
        **flags,
        checkpoint_kind="consolidated-refinement-v1",
        training_hash="trained",
        status="TEAM_PASS_CANDIDATE80_COLLECTED_PENDING_OWN_NATIVE_AND_RETENTION",
        records=records,
        counts={},
        declaration_hash=pins[str(source_root / "declaration.json")],
        preflight_hash=pins[str(source_root / "preflight.json")],
        scene_hash=pins[str(source_root / "scene.xml")],
    )
    docs = {
        "source": (source_root / "complete.json", source),
        "trained": (tmp_path / "trained/complete.json", trained),
        "bank": (tmp_path / "bank/complete.json", {}),
    }
    own = dict(
        **flags,
        input_file_hashes=pins,
        producer_hash="source",
        training_hash="trained",
        status="TEAM_PASS_CANDIDATE80_WHOLE_OWN_NATIVE_REPLAY_MATCH",
        counts={},
        records=[{k: v for k, v in r.items() if k != "artifacts"} for r in copy.deepcopy(records)],
    )
    return cfg, own_root, own, docs


def test_receipt_admission_selects_exact_expert_task_without_allocating_a_model(
    tmp_path, monkeypatch
):
    import scripts.rsi_joint_delivery_pilot_common as common

    cfg, root, own, docs = fixture(tmp_path)
    monkeypatch.setattr(common, "receipt", lambda pins, h: docs[h])
    op = root / "complete.json"
    op.write_text(json.dumps(own))
    _, scope, _, _, _, baseline = admit(root, digest(op), cfg)
    assert scope["student_kind"] == "consolidated-refinement-v1"
    assert baseline["index"] == 51 and len(scope["global_courses"]) == 80


@pytest.mark.parametrize(
    "fault", ["status", "seed", "missing-row", "borrowed", "permission", "namespace"]
)
def test_partial_or_wrong_expert_evidence_rejected_before_physics(tmp_path, monkeypatch, fault):
    import scripts.rsi_joint_delivery_pilot_common as common

    cfg, root, own, docs = fixture(tmp_path)
    monkeypatch.setattr(common, "receipt", lambda pins, h: docs[h])
    if fault == "status":
        own["status"] = "pending"
    elif fault == "seed":
        docs["trained"][1]["config"]["seed"] = 115023
    elif fault == "missing-row":
        docs["source"][1]["records"].pop()
    elif fault == "borrowed":
        own["records"][0]["course"]["task"] = 1
    elif fault == "permission":
        own["hardware_authorized"] = True
    else:
        docs["source"][1]["records"][51]["artifacts"]["A"]["path"] = "../escape.npz"
    op = root / "complete.json"
    op.write_text(json.dumps(own))
    with pytest.raises((ValueError, FileNotFoundError)):
        admit(root, digest(op), cfg)


@pytest.mark.parametrize(
    "extra",
    [
        ["--source-root", "x"],
        ["--source-root", "x", "--source-hash", "h", "--preflight-only"],
        ["--producer-head", "h"],
    ],
)
def test_invalid_audit_cli_does_not_create_output(tmp_path, extra):
    out = tmp_path / "no output"
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_run_joint_delivery_pilot",
            "--expert-own-root",
            "x",
            "--expert-own-hash",
            "h",
            "--expected-head",
            "h",
            "--state-id",
            "25",
            "--output",
            str(out),
            *extra,
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 2 and not out.exists()
