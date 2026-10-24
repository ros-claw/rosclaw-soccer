import json
import sys

import pytest

from rosclaw_soccer.rsi.refinement_onpolicy_checkpoint import file_hash
from scripts import rsi_chain_prepare_parallel_refinement as chain
from scripts import rsi_prepare_refinement_continual as preparer


def test_exact96_producer_hash_forwarded_after_closure(tmp_path, monkeypatch):
    d = dict(
        status="GEN1_PARALLEL_OWN_POLICY_BATCH_ALL_NATIVE_LAW_MATCH",
        source_head="h",
        independent_whole_world_replay=True,
        sampling_law_audit_passed=True,
        config=dict(seeds_per_initial=32),
    )
    path = tmp_path / "complete.json"
    path.write_text(json.dumps(d))
    argv = [
        "chain",
        "--new-own-root",
        str(tmp_path),
        "--new-own-producer-head",
        "h",
        "--state-id",
        "25",
    ]
    calls = []
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(preparer, "main", lambda: calls.append(list(sys.argv)))
    chain.main()
    assert calls[0][-2:] == ["--new-own-hash", file_hash(path)] and "--parallel-batch" in calls[0]
    assert "--new-own-producer-head" not in calls[0]


@pytest.mark.parametrize("fault", ["head", "status", "native", "law", "small", "failed"])
def test_invalid_producer_never_enters_preparation(tmp_path, monkeypatch, fault):
    d = dict(
        status="GEN1_PARALLEL_OWN_POLICY_BATCH_ALL_NATIVE_LAW_MATCH",
        source_head="h",
        independent_whole_world_replay=True,
        sampling_law_audit_passed=True,
        config=dict(seeds_per_initial=32),
    )
    if fault == "head":
        d["source_head"] = "other"
    elif fault == "status":
        d["status"] = "COLLECTED"
    elif fault == "native":
        d["independent_whole_world_replay"] = False
    elif fault == "law":
        d["sampling_law_audit_passed"] = False
    elif fault == "small":
        d["config"]["seeds_per_initial"] = 4
    else:
        (tmp_path / "failed.json").write_text("{}")
    (tmp_path / "complete.json").write_text(json.dumps(d))
    monkeypatch.setattr(
        sys, "argv", ["chain", "--new-own-root", str(tmp_path), "--new-own-producer-head", "h"]
    )
    calls = []
    monkeypatch.setattr(preparer, "main", lambda: calls.append(1))
    with pytest.raises(ValueError):
        chain.main()
    assert not calls
