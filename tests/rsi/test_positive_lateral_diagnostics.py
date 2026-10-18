"""Synthetic pre-replay contracts, never new physics or teacher evidence."""

import json

import pytest

from rosclaw_soccer.rsi.positive_lateral_diagnostics import require_cohort
from scripts import rsi_read_mesh_contact_native_pilot as audit
from scripts.rsi_replay_first_touch_contacts import fixed_replay_scope
from tests.rsi.test_mesh_teacher_terminal_readback import expanded_fixture


@pytest.mark.parametrize(
    "tamper", [None, "partial", "order", "duplicate", "budget", "bool", "retag"]
)
def test_fixed_sixteen_dense_replays_are_not_extra_policy_trials(tamper):
    cfg = dict(
        replay_profile="positive-lateral-sixteen",
        new_replay_budget=16,
        new_substep_budget=48000,
        cases=[
            dict(case_id=f"{role}_positive_local_c{i}", receipt_hash=f"{role}-{i}")
            for i in range(8)
            for role in ("control", "teacher")
        ],
    )
    if tamper == "partial":
        cfg["cases"].pop()
    elif tamper == "order":
        cfg["cases"].reverse()
    elif tamper == "duplicate":
        cfg["cases"][1]["receipt_hash"] = cfg["cases"][0]["receipt_hash"]
    elif tamper == "budget":
        cfg["new_substep_budget"] = 96000
    elif tamper == "bool":
        cfg["new_replay_budget"] = True
    elif tamper == "retag":
        cfg["cases"][0]["case_id"] = "surface_receding_g76"
    if tamper:
        with pytest.raises(ValueError):
            fixed_replay_scope(cfg)
    else:
        assert fixed_replay_scope(cfg) == (16, 48000)


@pytest.mark.parametrize(
    "tamper",
    [
        None,
        "partial",
        "order",
        "failure",
        "world",
        "teacher",
        "training",
        "fresh",
        "bool",
        "split",
        "cluster",
        "pairing",
    ],
)
def test_cohort_requires_all_sixteen_original_failures_and_scoped_sources(
    tmp_path, monkeypatch, tamper
):
    output = expanded_fixture(tmp_path, monkeypatch, positive_lateral=True)
    audit.main()
    cfg = json.loads((tmp_path / "teacher-config.json").read_text())
    declaration = json.loads((tmp_path / "teacher/declaration.json").read_text())
    complete = json.loads((tmp_path / "teacher/complete.json").read_text())
    report = json.loads(output.read_text())
    if tamper == "partial":
        complete["results"].pop()
    elif tamper == "order":
        report["rows"].reverse()
    elif tamper == "failure":
        report["rows"][0]["outcome"]["task_pass"] = True
    elif tamper == "world":
        report["rows"][0]["world_hash"] = "other"
    elif tamper == "teacher":
        report["teacher_qualified"] = True
    elif tamper == "training":
        complete["neural_training_authorized"] = True
    elif tamper == "fresh":
        report["fresh_exam"] = True
    elif tamper == "bool":
        complete["new_optimizer_updates"] = False
    elif tamper == "split":
        report["prospective_split"]["train_source_seeds"].reverse()
    elif tamper == "cluster":
        report["source_scene_clusters"] = 8
    elif tamper == "pairing":
        report["paired_development"]["rescued_pairs"] = 99
    if tamper:
        with pytest.raises(ValueError):
            require_cohort(cfg, declaration, complete, report)
    else:
        assert require_cohort(cfg, declaration, complete, report) == complete["paired_development"]
