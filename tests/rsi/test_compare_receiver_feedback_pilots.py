"""Tests for complete matched pilot accounting, using synthetic physics fixtures."""

import json
import sys

import numpy as np
import pytest

from scripts import rsi_compare_receiver_feedback_pilots as review


def fixture_run(tmp_path, monkeypatch, *, same_baseline=True):
    roots = [tmp_path / "dense", tmp_path / "motion"]
    for root in roots:
        source = root / "source"
        source.mkdir(parents=True)
        (root / "audit").mkdir()
        (source / "declaration.json").write_text(json.dumps({"development_context_block": 0}))
    records = []
    for context in (2, 3, 5, 7):
        for side in (-0.35, 0.35):
            for learned in (False, True):
                records.append(
                    dict(
                        course=dict(
                            context=context, lateral_m=side, arm=dict(receiver_learned=learned)
                        ),
                        termination="WORLD_HORIZON_COMPLETE",
                        controls=1,
                        native_steps=10,
                        result=dict(handoff_executed=True),
                        artifacts=dict(initial={"hash": "same"}),
                    )
                )

    def report(root, audit, digest, status, *, worlds):
        assert worlds == 16
        return dict(source_head="head-" + root.parent.name, records=records), {}, "source-hash"

    def bind(root, row, name):
        return root / "physics.npz"

    for root in roots:
        forces = np.zeros((1, 10, 2, 3))
        forces[0, 2, 0, 0] = 11
        forces[0, 5, 1, 0] = 11
        np.savez(
            root / "source/physics.npz",
            ball_forces=forces,
            body_failed=np.zeros((1, 2), dtype=bool),
        )
    monkeypatch.setattr(review, "clean", lambda _: None)
    monkeypatch.setattr(review, "digest", lambda _: "audit-hash")
    monkeypatch.setattr(review, "report", report)
    monkeypatch.setattr(review, "bind", bind)
    monkeypatch.setattr(review, "handoff_equal", lambda *args: True)
    monkeypatch.setattr(review, "artifact_equal", lambda *args: same_baseline)
    monkeypatch.setattr(review, "measured", lambda *args: {"return_0p1m_and_healthy": False})
    output = tmp_path / "result"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review",
            "--dense",
            str(roots[0]),
            "--motion",
            str(roots[1]),
            "--dense-head",
            "head-dense",
            "--motion-head",
            "head-motion",
            "--expected-head",
            "review-head",
            "--output",
            str(output),
        ],
    )
    return output


def test_complete_pilot_reports_all_pairs_and_evidence_ceiling(tmp_path, monkeypatch):
    output = fixture_run(tmp_path, monkeypatch)
    # Both source and audit digests are checked twice; synthetic pins are equal.
    monkeypatch.setattr(
        review,
        "report",
        lambda root, audit, digest, status, worlds: (
            dict(
                source_head="head-" + root.parent.name,
                records=[
                    dict(
                        course=dict(context=c, lateral_m=s, arm=dict(receiver_learned=learned)),
                        termination="WORLD_HORIZON_COMPLETE",
                        controls=1,
                        native_steps=10,
                        result=dict(handoff_executed=True),
                        artifacts=dict(initial={"hash": "same"}),
                    )
                    for c in (2, 3, 5, 7)
                    for s in (-0.35, 0.35)
                    for learned in (False, True)
                ],
            ),
            {},
            "audit-hash",
        ),
    )
    review.main()
    result = json.loads((output / "complete.json").read_text())
    assert result["conditions"] == 8
    assert not result["all_seeds_or_blocks"] and not result["independent_fresh_contexts"]
    assert all(
        result["summary"][name]["strict_both_clean"] == 8
        for name in ("baseline", "dense", "motion")
    )


def test_baseline_mismatch_fails_without_result(tmp_path, monkeypatch):
    output = fixture_run(tmp_path, monkeypatch, same_baseline=False)
    with pytest.raises(ValueError, match="baseline or paired physical start"):
        review.main()
    assert not output.exists()
