"""Runner fixture tests; no physics, model allocation or success evidence."""

import copy
import sys
from pathlib import Path

import pytest

from scripts import rsi_probe_contact_geometry as runner


def rows():
    return [dict(law_index=i, score=dict(qualified=False, cost=None)) for i in (0, 3, 4)]


def test_uncertified_nominal_fallback_and_no_input_mutation():
    data = rows()
    previous = copy.deepcopy(data)
    assert runner.choose_candidate(data) == 0
    assert previous == data


def test_same_gate_cost_selection_and_stable_tie():
    data = rows()
    data[1]["score"] = dict(qualified=True, cost=0.2)
    data.extend([dict(law_index=i, score=dict(qualified=True, cost=0.1)) for i in (0, 3, 4)])
    assert runner.choose_candidate(data) == 3


@pytest.mark.parametrize("cost", [None, float("nan"), float("inf"), True])
def test_invalid_qualified_cost_rejected(cost):
    data = rows()
    data[0]["score"] = dict(qualified=True, cost=cost)
    with pytest.raises(ValueError, match="finite qualified cost"):
        runner.choose_candidate(data)


def test_repeated_proposal_law_rejected():
    data = rows() + [dict(law_index=0, score=dict(qualified=False, cost=None))] * 2
    with pytest.raises(ValueError, match="one new proposal"):
        runner.choose_candidate(data)


@pytest.mark.parametrize("kind", ["short", "order", "budget"])
def test_fixed_laws_and_budget(kind):
    data = rows()
    if kind == "short":
        data.pop()
    elif kind == "order":
        data[1]["law_index"] = 1
    else:
        data += copy.deepcopy(data) + [data[0]]
    with pytest.raises(ValueError, match="at most three"):
        runner.choose_candidate(data)


def arguments(tmp_path):
    result = ["probe"]
    for key in (
        "inventory",
        "bank",
        "actor-source",
        "world",
        "motion-csv",
        "model-root",
        "core-source",
        "output",
        "previous-producer",
    ):
        result += ["--" + key, str(tmp_path / key)]
    for key in ("expected-head", "expected-actor-head", "expected-inventory-hash"):
        result += ["--" + key, "expected"]
    return result


def test_wrong_head_before_any_inventory_or_physics(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", arguments(tmp_path))
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "wrong\n")
    with pytest.raises(ValueError, match="sealed contact geometry source"):
        runner.main()


def test_dirty_bootstrap_before_helpers(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", arguments(tmp_path))
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda cmd, **k: "expected\n" if cmd[1] == "rev-parse" else b"changed",
    )
    with pytest.raises(ValueError, match="bootstrap differs"):
        runner.main()


@pytest.mark.parametrize("extra", [[], ["--preflight-only", "--cluster", "0"]])
def test_zero_actor_and_worker_stage_separation(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", arguments(tmp_path) + extra)
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda cmd, **k: (
            "expected\n" if cmd[1] == "rev-parse" else Path(cmd[-1].split(":", 1)[1]).read_bytes()
        ),
    )
    with pytest.raises(ValueError, match="cannot allocate|needs full bank"):
        runner.main()
