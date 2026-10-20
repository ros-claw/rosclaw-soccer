"""Contracts and synthetic scoring only, not real G1 trajectory evidence."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance, ball_entry_decision
from rosclaw_soccer.rsi.full_entry_forecast import (
    EntryCommandLaw,
    FullEntryForecaster,
    feedback_entry_laws,
    full_entry_laws,
    score_full_entry,
)


def context():
    q = np.zeros(43)
    q[2], q[3], q[36], q[38], q[39] = 0.793, 1, 0.5, 0.11, 1
    v = np.zeros(41)
    prior = SkillEntryAffordance((0.25, 0.0), 150, 1.0, "sha256:" + "1" * 64)
    return q, v, prior


def test_nominal_is_per_frame_feedback_and_held_law_expires():
    q, v, prior = context()
    laws = full_entry_laws(q, v, affordance=prior)
    assert len(laws) == 7 and laws[0].held_command is None
    held = laws[1].held_command
    q[0] += 0.1
    for frame in (20, 39, 40, 90, 91, 299):
        measured = ball_entry_decision(q, v, frame=frame, affordance=prior).command
        assert laws[0].command(q, v, frame=frame, affordance=prior) == measured
        assert laws[1].command(q, v, frame=frame, affordance=prior) == (
            held if frame < 40 else measured
        )
    assert laws[1].command(q, v, frame=91, affordance=prior) == (0, 0, 0)


def test_feedback_bank_reacts_to_own_measured_velocity_after_frame40():
    q, v, prior = context()
    bank = feedback_entry_laws()
    assert len(bank) == 6 and bank[0].name == "nominal_feedback"
    before = bank[3].command(q, v, frame=60, affordance=prior)
    v[0] = 0.4
    after = bank[3].command(q, v, frame=60, affordance=prior)
    assert after[0] < before[0]
    assert (
        bank[0].command(q, v, frame=60, affordance=prior)
        == ball_entry_decision(q, v, frame=60, affordance=prior).command
    )
    assert all(law.command(q, v, frame=91, affordance=prior) == (0, 0, 0) for law in bank)


@pytest.mark.parametrize("index", [-1, 6, True, 1.5])
def test_unqualified_feedback_index_rejected(index):
    with pytest.raises(ValueError, match="declared feedback"):
        EntryCommandLaw("bad", None, index)


@pytest.mark.parametrize(
    "bad", [[0.0, 0.0, 0.0], (1.0, 0.0, 0.0), (0.0, 0.0, 2.0), (np.nan, 0.0, 0.0)]
)
def test_unbounded_or_mutable_law_rejected(bad):
    with pytest.raises(ValueError, match="bounded finite"):
        EntryCommandLaw("bad", bad)


def sample():
    names = [
        "world",
        "pelvis",
        "left_ankle_roll_link",
        "left_ankle_pitch_link",
        "right_ankle_roll_link",
        "right_ankle_pitch_link",
        "arm",
        "ball",
    ]
    model = SimpleNamespace(
        nbody=8,
        body_rootid=np.array([0, 1, 1, 1, 1, 1, 1, 7]),
        body=lambda name: SimpleNamespace(id=names.index(name)),
    )
    q, _, _ = context()
    canonical = np.repeat(q[None], 280, axis=0)
    after = np.repeat(canonical[:, None], 10, axis=1)
    after[-1, -1, 36] += 1.0
    velocities = np.zeros((280, 10, 41))
    velocities[-1, -1, 35] = 1.0
    contacts = np.zeros((280, 10, 8))
    contacts[50, 5, 4] = 2.0
    return model, dict(
        canonical_qpos=canonical,
        target_body_canonical_qpos_after_substep=after,
        target_body_canonical_qvel_after_substep=velocities,
        ball_contact_force_by_body_n=contacts,
    )


def test_full_task_scores_touch_ball_direction_and_late_safety():
    model, arrays = sample()
    good = score_full_entry(model, arrays)
    assert good["qualified"] and good["first_foot_contact_global_frame"] == 70
    # A violation after entry/foot contact must veto the whole candidate.
    arrays["ball_contact_force_by_body_n"][-1, -1, 6] = 2.0
    bad = score_full_entry(model, arrays)
    assert not bad["qualified"] and bad["cost"] is None


@pytest.mark.parametrize(
    "failure", ["no_touch", "late_touch", "backward", "lateral", "height", "upright"]
)
def test_invalid_soccer_or_recovery_outcome_is_not_candidate(failure):
    model, arrays = sample()
    if failure in ("no_touch", "late_touch"):
        arrays["ball_contact_force_by_body_n"][:] = 0.0
        if failure == "late_touch":
            arrays["ball_contact_force_by_body_n"][220, 0, 4] = 2.0  # global240
    elif failure == "backward":
        arrays["target_body_canonical_qpos_after_substep"][-1, -1, 36] -= 2.0
    elif failure == "lateral":
        arrays["target_body_canonical_qpos_after_substep"][-1, -1, 37] += 0.31
    elif failure == "height":
        arrays["target_body_canonical_qpos_after_substep"][-1, -1, 2] = 0.64
    else:
        arrays["target_body_canonical_qpos_after_substep"][-1, -1, 3:7] = [0, 1, 0, 0]
    result = score_full_entry(model, arrays)
    assert not result["qualified"] and result["cost"] is None


def test_nonfinite_record_rejected_not_scored():
    model, arrays = sample()
    arrays["target_body_canonical_qvel_after_substep"][0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="complete finite"):
        score_full_entry(model, arrays)


@pytest.mark.parametrize(
    "consumed,frame,failed", [(19600, 20, False), (0, 21, False), (0, 20, True)]
)
def test_forecast_budget_alignment_or_fault_rejected_before_allocation(consumed, frame, failed):
    q, v, prior = context()
    forecaster = FullEntryForecaster(world_hash="unused")
    forecaster.attempted_native_steps = consumed
    forecaster.failed = failed
    body = SimpleNamespace(failed=False, next_frame=frame, record_all_ball_contacts=True)
    nav = SimpleNamespace(_next_frame=frame)
    with pytest.raises(ValueError, match="healthy aligned"):
        forecaster.forecast(
            nav,
            body,
            model_root=Path("unused"),
            source_poses=np.empty((0, 36)),
            source_evidence_hash="unused",
            affordance=prior,
            law=full_entry_laws(q, v, affordance=prior)[0],
        )


def test_forecast_fault_is_latched_and_donor_guards_checked(monkeypatch):
    from rosclaw_soccer.rsi import full_entry_forecast as module

    q, v, prior = context()
    body = SimpleNamespace(
        failed=False,
        next_frame=20,
        record_all_ball_contacts=True,
        model=object(),
        data=object(),
        kp=np.ones(29),
        kd=np.ones(29),
        attempted_steps=200,
        integrated_steps=200,
    )
    monkeypatch.setattr(module, "G1PDTorqueForecaster", lambda *a, **k: None)
    monkeypatch.setattr(module, "integration_hash", lambda *a: "body")
    monkeypatch.setattr(module, "navigation_state_hash", lambda *a: "nav")
    monkeypatch.setattr(module, "body_fields_hash", lambda *a: "fields")

    def fail(*args):
        raise RuntimeError("clone failed before any new step")

    monkeypatch.setattr(module, "clone_navigation", fail)
    forecaster = FullEntryForecaster(world_hash="unused")
    with pytest.raises(RuntimeError, match="clone failed"):
        forecaster.forecast(
            SimpleNamespace(_next_frame=20),
            body,
            model_root=Path("unused"),
            source_poses=np.empty((0, 36)),
            source_evidence_hash="unused",
            affordance=prior,
            law=full_entry_laws(q, v, affordance=prior)[0],
        )
    assert forecaster.failed and forecaster.integrated_native_steps == 0


@pytest.mark.parametrize(
    "terminal",
    [
        "MainPID=12\nResult=success\nSubState=start\nExecMainStatus=0\n",
        "MainPID=0\nResult=exit-code\nSubState=failed\nExecMainStatus=1\n",
    ],
)
def test_full_reader_rejects_live_or_failed_producer_before_physics(
    tmp_path, monkeypatch, terminal
):
    import sys

    from scripts import rsi_verify_full_entry_teacher as reader

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reader",
            "--producer",
            str(tmp_path / "absent"),
            "--world",
            "unused",
            "--core-source",
            "unused",
            "--output",
            str(tmp_path / "out"),
            "--expected-head",
            "expected",
            "--expected-producer-head",
            "expected-producer",
            "--expected-producer-hash",
            "sha256:" + "1" * 64,
            "--producer-unit",
            "fixture.service",
        ],
    )
    monkeypatch.setattr(
        reader.subprocess,
        "check_output",
        lambda cmd, **kwargs: "expected\n" if cmd[0] == "git" else terminal,
    )
    with pytest.raises(ValueError, match="closed successful"):
        reader.main()
    assert not (tmp_path / "out").exists()
