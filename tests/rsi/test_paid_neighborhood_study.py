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


@pytest.mark.parametrize(
    "fault", [None, "audit_hash", "status", "count", "body", "case_order", "claim"]
)
def test_all_arm_reader_requires_closed_matched_own_evidence(tmp_path, monkeypatch, fault):
    import json

    from rosclaw_soccer.rsi import paid_neighborhood_sources as module

    config = neighborhood_config()
    plan = dict(
        config=config,
        body_scene_hash="body",
        specifications=dict(models={k: dict(hash="model") for k in config["arms"]}),
    )
    monkeypatch.setattr(module, "load_neighborhood_plan", lambda *a: ({}, plan, {}))
    roots, audits = {}, {}
    for arm in config["arms"]:
        root, ar = tmp_path / arm, tmp_path / (arm + "-audit")
        root.mkdir()
        ar.mkdir()
        roots[arm] = root
        audits[arm] = ar
        common = dict(
            source_head="producer",
            arm=arm,
            plan_hash="plan",
            config=config,
            model_hash="model",
            body_scene_hash="body",
            hardware_authorized=False,
            promotion_authorized=False,
        )
        d = dict(**common, base_stand=True, teacher_reoptimized=False)
        dp = root / "declaration.json"
        dp.write_text(json.dumps(d))
        rows = [
            dict(case=i, repeat=j, quality=False, healthy_success=False)
            for i in range(32)
            for j in range(2)
        ]
        r = dict(
            **common,
            status="PAID_CONTACT_NEIGHBORHOOD32_ARM_CLOSED_PENDING_AUDIT",
            declaration_hash=module.digest(dp),
            records=copy.deepcopy(rows),
            episodes=64,
            repeats_exact=True,
            actual_native_steps=100,
            actual_neural_rows=10,
            optimizer_updates=0,
            Fresh_rows_read=0,
            input_file_hashes={},
        )
        if arm == "learned_memory" and fault == "body":
            r["body_scene_hash"] = "other"
        if arm == "learned_memory" and fault == "case_order":
            r["records"][0]["case"] = 31
        rp = root / "complete.json"
        rp.write_text(json.dumps(r))
        a = dict(
            source_head="producer",
            arm=arm,
            plan_hash="plan",
            source_hash=module.digest(rp),
            status="PAID_CONTACT_NEIGHBORHOOD32_OWN_NN_IK_ALL_NATIVE_AUDIT_PASS",
            counts=dict(episodes=64),
            outcomes=copy.deepcopy(rows),
            repeats_exact=True,
            own_native_steps=100,
            own_trajectory_neural_rows=10,
            hardware_authorized=False,
            promotion_authorized=False,
            optimizer_updates=0,
            Fresh_rows_read=0,
            input_file_hashes={},
        )
        if arm == "learned_memory":
            if fault == "audit_hash":
                a["source_hash"] = "other"
            elif fault == "status":
                a["status"] = "PENDING"
            elif fault == "count":
                a["own_native_steps"] = 90
            elif fault == "claim":
                a["outcomes"][0]["quality"] = True
        (ar / "complete.json").write_text(json.dumps(a))
    if fault:
        with pytest.raises(ValueError):
            module.load_audited_neighborhood_arms(tmp_path, "plan", roots, audits, "producer")
    else:
        _, r, pins = module.load_audited_neighborhood_arms(
            tmp_path, "plan", roots, audits, "producer"
        )
        assert set(r) == set(config["arms"]) and len(pins) == 9


def test_dirty_support_diagnostic_never_reads_models_or_creates_output(tmp_path, monkeypatch):
    import sys

    from scripts import rsi_diagnose_paid_neighborhood as runner

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--plan-root",
            str(tmp_path),
            "--plan-hash",
            "plan",
            "--arm-root",
            "learned_memory",
            str(tmp_path),
            "--audit-root",
            "learned_memory",
            str(tmp_path),
            "--producer-head",
            "producer",
            "--expected-head",
            "frozen",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: b"dirty")
    with pytest.raises(ValueError, match="frozen"):
        runner.main()
    assert not (tmp_path / "out").exists()


def test_current_prefix_protocol_is_explicit_without_loosening_fixed_reference():
    from rosclaw_soccer.rsi.paid_neighborhood_study import current_prefix_config

    original = neighborhood_config()
    for cut in (80, 96):
        cfg = current_prefix_config(cut)
        assert cfg["protocol"] != original["protocol"]
        assert cfg["handoff_control"] == cut and cfg["current_student_prefix"]
        assert not cfg["prefix_is_expert"] and not cfg["counterfactual_prefix_labels_admitted"]
        assert cfg["maximum_native_steps"] == original["maximum_native_steps"]
        assert cfg["optimizer_updates"] == 0
    assert neighborhood_config() == original and "handoff_control" not in original
    for invalid in (True, 70, 80.0, 97):
        with pytest.raises(ValueError):
            current_prefix_config(invalid)


@pytest.mark.parametrize("arm", ["fixed_successful_teacher", "learned_memory"])
def test_incomplete_or_wrong_arm_prefix_cannot_allocate_physics(tmp_path, monkeypatch, arm):
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
        "--handoff-control",
        "80",
        "--expected-head",
        "frozen",
        "--output",
        str(tmp_path / "out"),
    ]
    if arm == "learned_memory":
        for key in ("prefix", "prefix-audit"):
            args += ["--" + key + "-root", str(tmp_path), "--" + key + "-hash", "hash"]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda args, **kw: "frozen\n" if "rev-parse" in args else b"",
    )
    monkeypatch.setattr(
        sources,
        "load_neighborhood_plan",
        lambda *args: ({}, dict(config=neighborhood_config()), {}),
    )
    with pytest.raises(ValueError, match="all explicit"):
        runner.main()
    assert not (tmp_path / "out").exists()
