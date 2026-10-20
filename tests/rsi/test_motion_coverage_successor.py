import json
import sys

import pytest

from scripts import rsi_review_motion_coverage as review
from scripts.rsi_native_context_recovery_common import digest


def prepare(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        review.subprocess,
        "check_output",
        lambda command, **kw: "head\n" if "rev-parse" in command else b"",
    )
    monkeypatch.setattr(review, "wait_for_terminal_collection", lambda *a, **k: None)
    args = [
        "review",
        "--expected-head",
        "head",
        "--expected-actor-head",
        "actor",
        "--output",
        str(tmp_path / "out"),
    ]
    roots = []
    for owner in range(1, 5):
        root = tmp_path / f"owner{owner}"
        root.mkdir()
        roots.append(root)
        args.extend(["--campaign", f"unit{owner}", str(root)])
        declaration = dict(
            source_head="actor",
            input_file_hashes={},
            owners=[owner],
            indices=review.diagnostic_indices(),
            reused_consumed_holdout=True,
            fresh_test=False,
        )
        (root / "declaration.json").write_text(json.dumps(declaration))
        rows = []
        for i in review.diagnostic_indices():
            folder = root / f"course{i:03d}-owner{owner}"
            folder.mkdir()
            for name in ("native.npz", "sampling.npz"):
                (folder / name).write_bytes(b"fixture raw artifact; no physics performed")
            passed = owner == 1 and i == 0
            row = dict(
                course_index=i,
                owner=owner,
                baseline_success=False,
                deployed_portfolio_success=False,
                deployed_owner=0,
                stratum="fixture",
                causal_qualified=passed,
                new_native_steps=6000,
                native_hash=digest(folder / "native.npz"),
                sampling_hash=digest(folder / "sampling.npz"),
                score=dict(qualified=passed),
                prefix_diagnostics=dict(
                    prefix_physical_safe=True, post_boundary_control_can_cause_first_touch=True
                ),
                replay=dict(
                    additional_native_steps=3000,
                    student_neural_and_muscle_native_bit_exact=True,
                    intention_network_calls=300,
                    muscle_decoder_calls=280,
                    teacher_encoder_calls=0,
                    teacher_planner_calls=0,
                ),
            )
            (folder / "complete.json").write_text(json.dumps(row))
            rows.append(row)
        (root / "complete.json").write_text(
            json.dumps(
                dict(
                    source_head="actor",
                    status="COUNTERFACTUAL_FROZEN_EXPERT_COVERAGE_VERIFIED",
                    optimizer_steps=0,
                    new_native_steps=144000,
                    results=rows,
                    hardware_authorized=False,
                    promotion_authorized=False,
                )
            )
        )
    monkeypatch.setattr(sys, "argv", args)
    return roots


def test_declared_successor_closes_but_never_deploys_oracle(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch)
    review.main()
    result = json.loads((tmp_path / "out/complete.json").read_text())
    assert result["counterfactual_oracle_union_success"] == 1
    assert result["counterfactual_oracle_missed_by_deployed"] == 1
    assert result["new_native_steps"] == result["optimizer_steps"] == 0
    assert result["oracle_deployable"] is result["fresh_test"] is False


@pytest.mark.parametrize("fault", ["worker", "raw_hash", "source", "duplicate"])
def test_failed_successor_preserves_failure_without_retry(tmp_path, monkeypatch, fault):
    roots = prepare(tmp_path, monkeypatch)
    if fault == "worker":

        def fail(*args, **kwargs):
            raise ValueError("fixture producer failed")

        monkeypatch.setattr(review, "wait_for_terminal_collection", fail)
    elif fault == "raw_hash":
        (roots[0] / "course000-owner1/native.npz").write_bytes(b"tampered")
    else:
        path = roots[0] / "complete.json"
        value = json.loads(path.read_text())
        if fault == "source":
            value["source_head"] = "unexpected"
        else:
            value["results"][1] = value["results"][0]
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        review.main()
    assert not (tmp_path / "out/complete.json").exists()
    failure = json.loads((tmp_path / "out/failure.json").read_text())
    assert failure["automatic_retry_authorized"] is False
    assert failure["new_native_steps"] == 0
