"""Durable distinct-profile postcontact captures with synthetic forecasts."""

import json
from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.rsi.postcontact_exit_controller import PostContactExitCommand
from rosclaw_soccer.rsi.postcontact_exit_evidence import PostContactExitEvidence
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from tests.rsi.test_postcontact_exit_controller import bank


def command(frame=0):
    offset = np.zeros(29)
    r = dict(
        schema="soccer.rsi.postcontact80ms_exit_plan.v1",
        frame=frame,
        requested_offset_nm=offset.tolist(),
        future_source_frame_or_target_used=False,
        full_h300_gain_proven=False,
        neural_policy_trained=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return PostContactExitCommand(offset, r, tuple(bank()))


def test_distinct_whole_raw_bank_with_three_neural_and_target_captures(tmp_path):
    root = tmp_path / "exit"
    w = PostContactExitEvidence(root)
    w.record(0, command())
    d = json.loads((root / "frame-000.json").read_text())
    assert d["schema"] == "soccer.rsi.preaction_postcontact80ms_exit_decision.v1"
    assert d["report_hash"] == hash_json({k: v for k, v in d.items() if k != "report_hash"})
    assert d["persisted_before_current_native_substeps"] and len(d["raw_witnesses"]) == 3
    for witness in d["raw_witnesses"]:
        from pathlib import Path

        file = Path(witness["path"])
        assert witness["sha256"] == hash_bytes(file.read_bytes())
        with np.load(file, allow_pickle=False) as a:
            assert all(f"neural_next_{i}_token" in a and f"next_{i}_target" in a for i in (1, 2, 3))
    with pytest.raises(ValueError):
        w.record(0, command())
    assert w.failed
    with pytest.raises(ValueError, match="cannot retry"):
        w.record(1, command(1))


@pytest.mark.parametrize("bad", ["schema", "promoted", "offset", "frame", "captures"])
def test_wrong_profile_authorization_or_capture_is_terminal(tmp_path, bad):
    w = PostContactExitEvidence(tmp_path / "exit")
    c = command()
    if bad == "schema":
        c.report["schema"] = "soccer.rsi.prospective_early80ms_impact_plan.v1"
    elif bad == "promoted":
        c.report["teacher_qualified"] = True
    elif bad == "offset":
        c.offset_nm[0] = 0.1
    elif bad == "frame":
        c.report["frame"] = 1
    else:
        candidate = c.candidates[0]
        candidate = replace(candidate, forecast=replace(candidate.forecast, next_target_records=()))
        c = replace(c, candidates=(candidate, *c.candidates[1:]))
    with pytest.raises(ValueError):
        w.record(0, c)
    assert w.failed and not (w.root / "frame-000.json").exists()


def test_existing_root_cannot_be_overwritten(tmp_path):
    PostContactExitEvidence(tmp_path / "exit")
    with pytest.raises(ValueError):
        PostContactExitEvidence(tmp_path / "exit")
