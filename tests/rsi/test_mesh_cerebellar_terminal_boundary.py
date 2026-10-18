"""Ensure unfinished jobs cannot be mistaken for terminal audit evidence."""

import sys

import pytest

from scripts import rsi_read_mesh_cerebellar_study as reader


@pytest.mark.parametrize(
    "status",
    [
        "MainPID=23\nActiveState=active\nResult=success\nLoadState=loaded\n",
        "MainPID=0\nActiveState=failed\nResult=exit-code\nLoadState=loaded\n",
        "MainPID=0\nActiveState=inactive\nResult=signal\nLoadState=loaded\n",
        "MainPID=0\nActiveState=inactive\nResult=success\nLoadState=not-found\n",
        "MainPID=0\nActiveState=inactive\nResult=success\n",
    ],
)
def test_nonterminal_or_failed_workers_rejected_before_reading_inputs(
    monkeypatch, tmp_path, status
):
    output = tmp_path / "not-created.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reader",
            "--config",
            str(tmp_path / "absent.json"),
            "--unit",
            "synthetic.service",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(reader.subprocess, "check_output", lambda *args, **kwargs: status)
    with pytest.raises(ValueError, match="terminal"):
        reader.main()
    assert not output.exists()


def test_existing_audit_is_never_overwritten(monkeypatch, tmp_path):
    output = tmp_path / "preserved.json"
    output.write_text("original")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reader",
            "--config",
            str(tmp_path / "absent.json"),
            "--unit",
            "synthetic.service",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        reader.subprocess,
        "check_output",
        lambda *args, **kwargs: "MainPID=0\nActiveState=inactive\nResult=success\n",
    )
    with pytest.raises(ValueError, match="unused"):
        reader.main()
    assert output.read_text() == "original"


def test_incomplete_prior_audit_is_not_implicitly_retried(monkeypatch, tmp_path):
    output = tmp_path / "result.json"
    started = output.with_suffix(".started.json")
    started.write_text("preserved failed/unfinished audit")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reader",
            "--config",
            str(tmp_path / "absent.json"),
            "--unit",
            "synthetic.service",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        reader.subprocess,
        "check_output",
        lambda *args, **kwargs: "MainPID=0\nActiveState=inactive\nResult=success\n",
    )
    with pytest.raises(ValueError, match="unused"):
        reader.main()
    assert not output.exists()
    assert started.read_text() == "preserved failed/unfinished audit"


def test_optimizer_audit_counts_actual_attempts_not_budget():
    receipts = [
        dict(
            requested_optimizer_steps=4,
            attempted_optimizer_steps=t,
            accepted_joint_optimizer_steps=k,
        )
        for t, k in [(1, 0), (3, 2), (4, 4)]
    ]
    result = reader.optimizer_audit_accounting(receipts)
    assert result == dict(
        accepted_training_optimizer_steps=6,
        additional_audit_accepted_optimizer_step_recomputations=6,
        additional_audit_optimizer_step_attempts=8,
        additional_audit_optimizer_step_attempt_budget=12,
    )


@pytest.mark.parametrize("counts", [(4, 2, 3), (4, 5, 4), (True, 1, 0), (4, 1, -1)])
def test_optimizer_audit_rejects_invalid_counts(counts):
    receipt = dict(
        zip(
            (
                "requested_optimizer_steps",
                "attempted_optimizer_steps",
                "accepted_joint_optimizer_steps",
            ),
            counts,
            strict=True,
        )
    )
    with pytest.raises(ValueError, match="accounting"):
        reader.optimizer_audit_accounting([receipt] * 3)


def test_optimizer_audit_requires_all_seeds():
    with pytest.raises(ValueError, match="three"):
        reader.optimizer_audit_accounting([])
