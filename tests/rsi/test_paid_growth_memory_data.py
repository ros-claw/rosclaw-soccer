import copy

import numpy as np
import pytest
import torch

from rosclaw_soccer.rsi.paid_growth_memory_data import (
    aggregate_behaviors,
    choose_behaviors,
    protect_historical_anchors,
)


def anchors():
    x = np.zeros((2, 288), np.float32)
    x[1, 0] = 1
    return dict(
        features=x,
        target=np.zeros((2, 29), np.float32),
        parent_delta=np.zeros((2, 29), np.float32),
        teacher=np.zeros(2, bool),
        active=np.zeros(2, bool),
        group=np.zeros(2, np.int32),
    )


def test_historical_rebase_protects_current_learned_motor_not_older_af46():
    from types import SimpleNamespace

    old = anchors()
    old["target"][:] = 0.1
    old["teacher"][:] = True
    old["active"][:] = True
    before = {k: v.copy() for k, v in old.items()}
    parent = SimpleNamespace(residual=lambda x: torch.full((len(x), 29), 0.1))
    new, error = protect_historical_anchors(parent, old)
    assert (
        error == 0
        and not new["target"].any()
        and not new["active"].any()
        and not new["teacher"].any()
    )
    assert np.all(new["parent_delta"] == np.float32(0.1))
    for k in old:
        assert np.array_equal(old[k], before[k])
    parent.residual = lambda x: torch.zeros(len(x), 29)
    with pytest.raises(ValueError, match="historical"):
        protect_historical_anchors(parent, old)


@pytest.mark.parametrize("conflict", [False, True])
def test_aggregation_preserves_all_old_anchors_and_rejects_different_labels(conflict):
    old = anchors()
    before = {k: v.copy() for k, v in old.items()}
    x = old["features"].copy()
    x[1, 0] = 2
    y = np.zeros((2, 29), np.float32)
    y[1] = 0.1
    if conflict:
        y[0] = 0.02
    arrays = dict(features=x, target=y, parent_delta=np.zeros((2, 29), np.float32))
    if conflict:
        with pytest.raises(ValueError, match="conflict"):
            aggregate_behaviors(old, [(arrays, dict(kind="executed_current_state_recovery"))])
    else:
        z, meta = aggregate_behaviors(old, [(arrays, dict(kind="executed_current_state_recovery"))])
        assert len(z["features"]) == 3 and z["active"].tolist() == [False, False, True]
        assert meta[0]["source_frames"] == [1] and meta[0]["duplicate_rows_skipped"] == 1
        for k in old:
            assert np.array_equal(z[k][:2], old[k])
    for k in old:
        assert np.array_equal(old[k], before[k])


def selection_sources():
    from test_paid_canonical_behavior import repeat

    prefix = dict(
        records=[dict(case=i, healthy_success=i == 2) for i in range(32) for _ in range(2)]
    )
    sources = {
        "handoff" + str(c): dict(
            records=[dict(repeat(1.0), case=i) for i in range(32) for _ in range(2)]
        )
        for c in (80, 96)
    }
    for cut, lateral in ((80, 0.4), (96, 0.2)):
        for j in (0, 1):
            sources["handoff" + str(cut)]["records"][j] = dict(repeat(lateral), case=0)
    for j in (2, 3):
        sources["handoff80"]["records"][j] = dict(repeat(0.5), case=1)
    return prefix, sources


def test_selects_whole_repeated_success_and_never_changes_current_success():
    prefix, sources = selection_sources()
    before = copy.deepcopy(sources)
    selected = choose_behaviors(prefix, sources)
    assert selected[0]["selected"]["cut"] == 96
    assert selected[1]["selected"]["cut"] == 80
    assert selected[2]["selected"]["kind"] == "protected_current_success"
    assert selected[3]["selected"] is None
    assert sources == before
    sources["handoff96"]["records"][1]["stand"]["healthy"] = False
    assert choose_behaviors(prefix, sources)[0]["selected"]["cut"] == 80
    prefix["records"][0]["healthy_success"] = True
    with pytest.raises(ValueError, match="agree"):
        choose_behaviors(prefix, sources)


def test_dirty_growth_curator_cannot_allocate_or_publish(tmp_path, monkeypatch):
    import sys

    from scripts import rsi_curate_paid_growth_memory as runner

    args = ["run", "--expected-head", "frozen", "--output", str(tmp_path / "out")]
    for name in ("handoff80", "audit80", "handoff96", "audit96"):
        args += ["--" + name + "-root", str(tmp_path), "--" + name + "-hash", "hash"]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: b"dirty")
    with pytest.raises(ValueError, match="frozen"):
        runner.main()
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("arm", ["learned_memory", "frozen_parent"])
def test_incomplete_or_foreign_role_candidate_never_allocates(tmp_path, monkeypatch, arm):
    import sys

    from rosclaw_soccer.rsi import paid_neighborhood_sources as sources
    from scripts import rsi_run_paid_neighborhood as runner

    args = [
        "run",
        "--plan-root",
        str(tmp_path),
        "--plan-hash",
        "plan",
        "--arm",
        arm,
        "--candidate-root",
        str(tmp_path),
        "--expected-head",
        "frozen",
        "--output",
        str(tmp_path / "out"),
    ]
    if arm == "frozen_parent":
        args += ["--candidate-hash", "hash"]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda args, **k: "frozen\n" if "rev-parse" in args else b"",
    )
    monkeypatch.setattr(sources, "load_neighborhood_plan", lambda *args: ({}, dict(config={}), {}))
    with pytest.raises(ValueError, match="growing candidate arguments"):
        runner.main()
    assert not (tmp_path / "out").exists()
