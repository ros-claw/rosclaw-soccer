"""Fixture receipt tests, not simulated skills."""

import json
import sys

import pytest

from scripts import rsi_collect_shadow_motor, rsi_motion_portfolio_common
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_review_kinematic_motion_exam import main


def setup(tmp_path, monkeypatch, *, keep_old=True):
    bd = dict(
        rows=[
            dict(
                split="validation" if i in (7, 13, 20, 23, 25, 28, 30, 38) else "train",
                course=dict(course=[0.3, -0.2, 0.0]),
            )
            for i in range(40)
        ]
    )
    cases = [{i: (dict(score=dict(qualified=i in (0, 10))), None) for i in range(40)}]
    monkeypatch.setattr(rsi_motion_portfolio_common, "admit", lambda a: ({}, bd, {}, cases))
    monkeypatch.setattr(
        rsi_collect_shadow_motor, "wait_for_terminal_collection", lambda *a, **k: None
    )
    args = [
        "review",
        "--inputs",
        "unused",
        "--core-source",
        "unused",
        "--expected-head",
        "review",
        "--expected-inputs-hash",
        "unused",
        "--expected-exam-source",
        "exam",
        "--output",
        str(tmp_path / "out"),
    ]
    roots = []
    for part in range(4):
        root = tmp_path / f"part{part}"
        root.mkdir()
        roots.append(root)
        args += ["--campaign", f"unit{part}", str(root)]
        indices = list(range(10 * part, 10 * part + 10))
        d = dict(
            source_head="exam", candidate_hash="weights", indices=indices, input_file_hashes={}
        )
        (root / "declaration.json").write_text(json.dumps(d))
        rows = []
        for i in indices:
            folder = root / f"course{i:02d}"
            folder.mkdir()
            for name in ("native.npz", "sampling.npz"):
                (folder / name).write_bytes(b"fixture only")
            r = dict(
                course_index=i,
                split=bd["rows"][i]["split"],
                native_steps=6000,
                native_hash=digest(folder / "native.npz"),
                sampling_hash=digest(folder / "sampling.npz"),
                regenerated_noise_and_prequant_likelihood_bit_exact=True,
                score=dict(
                    qualified=i in ((0, 10, 20) if keep_old else (10, 20)),
                    minimum_pelvis_z_m=0.75,
                    minimum_upright=0.99,
                    maximum_nonfoot_force_n=0.0,
                ),
                prefix_diagnostics=dict(
                    prefix_physical_safe=True, post_boundary_control_can_cause_first_touch=True
                ),
                replay=dict(
                    additional_native_steps=3000,
                    student_neural_and_muscle_native_bit_exact=True,
                    intention_network_calls=600,
                    muscle_decoder_calls=280,
                    teacher_encoder_calls=0,
                    teacher_planner_calls=0,
                ),
            )
            rows.append(r)
            (folder / "complete.json").write_text(json.dumps(r))
        c = dict(
            source_head="exam",
            candidate_hash="weights",
            status="KINEMATIC_COLUMN_NATIVE_EPISODES_AND_REPLAY_VERIFIED",
            new_native_steps=60000,
            optimizer_steps=0,
            results=dict(
                status="ALL_DECLARED_EPISODES_AND_NOISE_REPLAY_VERIFIED",
                results=rows,
                new_native_steps=60000,
            ),
        )
        (root / "complete.json").write_text(json.dumps(c))
    monkeypatch.setattr(sys, "argv", args)
    return roots


@pytest.mark.parametrize("keep_old", [True, False])
def test_retention_and_validation_are_explicit_but_never_promote(tmp_path, monkeypatch, keep_old):
    setup(tmp_path, monkeypatch, keep_old=keep_old)
    main()
    r = json.loads((tmp_path / "out/complete.json").read_text())
    assert r["validation_success"] == 1
    assert r["development_necessary_gate_pass"] is keep_old
    assert r["regressions"] == ([] if keep_old else [0])
    assert r["promotion_authorized"] is r["fresh_exam_pass"] is False
    assert r["new_native_steps"] == r["optimizer_steps"] == 0


@pytest.mark.parametrize("fault", ["worker", "raw", "candidate"])
def test_review_blocks_faults_without_physics_retry(tmp_path, monkeypatch, fault):
    roots = setup(tmp_path, monkeypatch)
    if fault == "worker":

        def fail(*args, **kwargs):
            raise ValueError("fixture worker failed")

        monkeypatch.setattr(rsi_collect_shadow_motor, "wait_for_terminal_collection", fail)
    elif fault == "raw":
        (roots[0] / "course00/native.npz").write_bytes(b"tampered")
    else:
        path = roots[1] / "complete.json"
        d = json.loads(path.read_text())
        d["candidate_hash"] = "unexpected"
        path.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        main()
    assert not (tmp_path / "out/complete.json").exists()
    assert (
        json.loads((tmp_path / "out/failure.json").read_text())["automatic_retry_authorized"]
        is False
    )
