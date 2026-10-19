"""Durable pre-action payloads only, no physical skill or policy qualification."""

import json

import numpy as np
import pytest

from rosclaw_soccer.rsi import early_impact_evidence as mod
from rosclaw_soccer.rsi.early_impact_controller import EarlyImpactCandidate, EarlyImpactCommand
from tests.rsi.test_early_feedback_calibration import raw


def command(frame=0, *, candidate=False):
    report = dict(
        frame=frame,
        requested_offset_nm=[0.0] * 29,
        future_source_frame_or_target_used=False,
        full_h300_gain_proven=False,
        neural_policy_trained=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    out, _, _ = raw()
    rows = (EarlyImpactCandidate("baseline", np.zeros(29), out, {"proxy_cost": 1.0}),)
    return EarlyImpactCommand(np.zeros(29), report, rows if candidate else ())


def test_complete_candidate_and_all_three_neural_payloads_persist_before_return(tmp_path):
    root = tmp_path / "evidence"
    writer = mod.EarlyImpactEvidence(root)
    writer.record(0, command(candidate=True))
    record = json.loads((root / "frame-000.json").read_text())
    assert record["report_hash"] == mod.hash_json(
        {k: v for k, v in record.items() if k != "report_hash"}
    )
    assert record["persisted_before_current_native_substeps"] and not writer.failed
    witness = record["raw_witnesses"][0]
    file = root / "frame-000-candidate-00.npz"
    assert witness["sha256"] == mod.hash_bytes(file.read_bytes())
    with np.load(file, allow_pickle=False) as a:
        for j in range(1, 4):
            assert a[f"next_{j}_joint_target_rad"].shape == (29,)
            assert a[f"neural_next_{j}_a"].dtype == np.float32
    writer.record(1, command(1))
    assert writer.next_frame == 2


def test_fsync_failure_is_terminal_and_cannot_be_retried(monkeypatch, tmp_path):
    writer = mod.EarlyImpactEvidence(tmp_path / "evidence")

    def fail(fd):
        raise OSError("synthetic disk write failure")

    monkeypatch.setattr(mod.os, "fsync", fail)
    with pytest.raises(OSError):
        writer.record(0, command())
    assert writer.failed and writer.next_frame == 0
    with pytest.raises(ValueError, match="contiguous"):
        writer.record(0, command())


@pytest.mark.parametrize("bad", ["frame", "permission", "offset", "skip"])
def test_inconsistent_or_non_sim_records_fail_closed(tmp_path, bad):
    writer = mod.EarlyImpactEvidence(tmp_path / "evidence")
    c = command()
    frame = 0
    if bad == "frame":
        c.report["frame"] = 1
    elif bad == "permission":
        c.report["hardware_authorized"] = True
    elif bad == "offset":
        c.offset_nm[0] = 1
    else:
        frame = 1
    with pytest.raises(ValueError):
        writer.record(frame, c)
    assert writer.failed


def test_existing_output_is_never_reused(tmp_path):
    root = tmp_path / "evidence"
    mod.EarlyImpactEvidence(root)
    with pytest.raises(ValueError, match="unused"):
        mod.EarlyImpactEvidence(root)
