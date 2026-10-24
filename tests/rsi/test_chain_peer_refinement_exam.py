import json
import sys

import pytest

from rosclaw_soccer.rsi.refinement_onpolicy_checkpoint import file_hash
from rosclaw_soccer.rsi.refinement_onpolicy_learning import PEER_KIND, STATUS
from scripts import rsi_chain_peer_refinement_exam as chain
from scripts import rsi_run_refinement_paired_exam as downstream


def test_exact_closed_peer_producer_hash_is_forwarded_not_guessed(tmp_path, monkeypatch):
    cp = tmp_path / "complete.json"
    cp.write_text(
        json.dumps(
            dict(
                source_head="declared", checkpoint_kind=PEER_KIND, status=STATUS, new_actor_steps=4
            )
        )
    )
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chain",
            "--training-root",
            str(tmp_path),
            "--training-producer-head",
            "declared",
            "--state-id",
            "25",
            "--output",
            "unused",
        ],
    )
    monkeypatch.setattr(downstream, "main", lambda: calls.append(list(sys.argv)))
    chain.main()
    assert len(calls) == 1
    assert calls[0][-2:] == ["--training-hash", file_hash(cp)]
    assert "--training-producer-head" not in calls[0]


def test_generation1_successor_needs_explicit_continual_mode(tmp_path, monkeypatch):
    from rosclaw_soccer.rsi import refinement_continual_checkpoint as continual

    (tmp_path / "complete.json").write_text(
        json.dumps(
            dict(
                source_head="declared",
                checkpoint_kind=continual.KIND,
                status=continual.STATUS,
                new_actor_steps=4,
            )
        )
    )
    argv = ["chain", "--training-root", str(tmp_path), "--training-producer-head", "declared"]
    calls = []
    monkeypatch.setattr(downstream, "main", lambda: calls.append(list(sys.argv)))
    monkeypatch.setattr(sys, "argv", argv.copy())
    with pytest.raises(ValueError):
        chain.main()
    monkeypatch.setattr(sys, "argv", [*argv, "--continual"])
    chain.main()
    assert len(calls) == 1 and "--continual" not in calls[0]


@pytest.mark.parametrize("fault", ["head", "kind", "status", "zero", "failed"])
def test_ineligible_producer_never_enters_exam(tmp_path, monkeypatch, fault):
    d = dict(source_head="declared", checkpoint_kind=PEER_KIND, status=STATUS, new_actor_steps=4)
    if fault == "head":
        d["source_head"] = "other"
    elif fault == "kind":
        d["checkpoint_kind"] = "refinement-actor-critic-v1"
    elif fault == "status":
        d["status"] = "RUNNING"
    elif fault == "zero":
        d["new_actor_steps"] = 0
    else:
        (tmp_path / "failed.json").write_text("{}")
    (tmp_path / "complete.json").write_text(json.dumps(d))
    monkeypatch.setattr(
        sys,
        "argv",
        ["chain", "--training-root", str(tmp_path), "--training-producer-head", "declared"],
    )
    calls = []
    monkeypatch.setattr(downstream, "main", lambda: calls.append(1))
    with pytest.raises(ValueError):
        chain.main()
    assert not calls
