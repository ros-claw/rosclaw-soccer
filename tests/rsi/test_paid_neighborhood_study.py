import copy
import itertools

import pytest

from rosclaw_soccer.rsi.paid_neighborhood_study import (
    contact_neighborhood_cases,
    neighborhood_config,
)


def sources():
    return [
        dict(
            case=i,
            source="soccer-standard-003_left.npz",
            profile="cached-standard",
            ball_mass_kg=0.43,
            ball_world_initial=[0.8, 0.15, 0.11],
            velocity_world_initial=[0.2 if i == 4 else -0.2, 0.01],
            goal_world=[6.0, 0.0, 0.11],
        )
        for i in (4, 5)
    ]


def test_every_factorial_context_retained_and_sources_immutable():
    source = sources()
    before = copy.deepcopy(source)
    cases = contact_neighborhood_cases(source)
    assert source == before
    assert len(cases) == 32 and [c["case"] for c in cases] == list(range(32))
    for case_id in (4, 5):
        group = [c for c in cases if c["source_case"] == case_id]
        assert [tuple(c["signs"]) for c in group] == list(itertools.product((-1, 1), repeat=4))
        for row in group:
            a, b, c, d = row["signs"]
            assert row["ball_world_initial"] == [0.8 + 0.02 * a, 0.15 + 0.02 * b, 0.11]
            assert row["velocity_world_initial"] == [
                (0.2 if case_id == 4 else -0.2) + 0.04 * c,
                0.01 + 0.04 * d,
            ]
            assert row["not_fresh"] and row["bank"] == "development_not_fresh"
    config = neighborhood_config()
    assert config["maximum_episodes"] == len(cases) * config["repeats"] * len(config["arms"])
    assert config["maximum_native_steps"] == config["maximum_episodes"] * 3890
    assert config["optimizer_updates"] == 0 and config["Fresh_rows_read"] == 0


@pytest.mark.parametrize(
    "fault", ["source", "nonfinite", "velocity", "profile", "height", "duplicate"]
)
def test_invalid_bank_is_rejected_whole_not_resampled_or_clipped(fault):
    source = sources()
    if fault == "source":
        source[1]["source"] = "different.npz"
    elif fault == "nonfinite":
        source[0]["goal_world"][0] = float("nan")
    elif fault == "velocity":
        source[0]["velocity_world_initial"] = [0.59, 0]
    elif fault == "height":
        source[0]["ball_world_initial"][2] = 0.115
    elif fault == "duplicate":
        source[1]["velocity_world_initial"] = source[0]["velocity_world_initial"].copy()
    else:
        source[0]["profile"] = "other"
    with pytest.raises(ValueError):
        contact_neighborhood_cases(source)


@pytest.mark.parametrize("name", ["declare", "run", "audit"])
def test_dirty_neighborhood_runners_cannot_allocate_or_write(tmp_path, monkeypatch, name):
    import importlib
    import sys

    module = importlib.import_module("scripts.rsi_" + name + "_paid_neighborhood")
    args = ["run", "--expected-head", "frozen", "--output", str(tmp_path / "out")]
    if name == "declare":
        for key in ("data", "pipeline"):
            args += ["--" + key + "-root", str(tmp_path), "--" + key + "-hash", "hash"]
    elif name == "run":
        args += ["--plan-root", str(tmp_path), "--plan-hash", "hash", "--arm", "learned_memory"]
    else:
        args += ["--root", str(tmp_path), "--producer-head", "producer"]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: b"dirty")
    with pytest.raises(ValueError, match="frozen"):
        module.main()
    assert not (tmp_path / "out").exists()
