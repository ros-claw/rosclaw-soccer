"""Synthetic whole-exit-bank readback, no new physics or G1 skill proof."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.physics.g1_early_feedback_forecast import G1EarlyFeedbackForecast
from rosclaw_soccer.rsi.postcontact_exit_score import postcontact_exit_score
from scripts.rsi_postcontact_exit_decisions import read_exit_decisions
from scripts.rsi_postcontact_release_transfer import paired_summary
from scripts.rsi_read_early_feedback_screen_arrays import verify_postcontact_exit_prediction
from scripts.rsi_read_postcontact_transfer import check_complete
from tests.rsi.test_early_feedback_readback import fixture
from tests.rsi.test_postcontact_release_transfer import rows, scope


def protocol():
    a, t, _ = fixture()
    a["offset_nm"] = np.zeros(29)
    t.update(
        postcontact_exit_requested_offset_nm=np.zeros((300, 1, 29)),
        postcontact_exit_applied_offset_nm=np.zeros((300, 1, 10, 29)),
        postcontact_exit_forecast_steps=np.zeros((300, 1, 1), dtype=np.int64),
    )
    t["force_n"][29, 0, 0] = 2
    t["postcontact_exit_forecast_steps"][30] = 80
    prior = t["force_n"][29, 0]
    score = postcontact_exit_score(
        G1EarlyFeedbackForecast("donor", a["end_integration_state"], a, (), ()),
        np.zeros(29),
        previous_contact_forces=prior,
    )
    root = Path("/synthetic/exit")
    witnesses = [
        dict(
            name=name,
            path=str(root / f"frame-030-candidate-{i:02d}.npz"),
            sha256="fixture",
            donor_hash="donor",
            offset_nm=[0.0] * 29,
            score=deepcopy(score),
        )
        for i, name in enumerate(("baseline", "repeat_zero"))
    ]
    decisions = []
    for f in range(300):
        decisions.append(
            dict(
                schema="soccer.rsi.preaction_postcontact80ms_exit_decision.v1",
                frame=f,
                persisted_before_current_native_substeps=True,
                future_source_frame_or_target_used=False,
                full_h300_gain_proven=False,
                neural_policy_trained=False,
                teacher_qualified=False,
                hardware_authorized=False,
                promotion_authorized=False,
                requested_offset_nm=[0.0] * 29,
                forecast_substeps=80 if f == 30 else 0,
                accumulated_forecast_substeps=80 if f >= 30 else 0,
                raw_witnesses=witnesses if f == 30 else [],
                selected_index=0 if f == 30 else None,
                selected_name="baseline" if f >= 30 else None,
                status="SYNTHETIC",
            )
        )
    return a, t, root, decisions


def read(a, t, root, decisions):
    return read_exit_decisions(
        t,
        root,
        expected_bank=(("baseline", np.zeros(29)), ("repeat_zero", np.zeros(29))),
        sealed=lambda p: decisions[int(p.stem.removeprefix("frame-"))],
        arrays=lambda w: a,
    )


def test_exit_can_have_no_predicted_second_impact_but_raw_selection_and_alignment_verified():
    a, t, root, d = protocol()
    result = read(a, t, root, d)
    assert len(result) == 1
    assert result[0]["schema"] == "soccer.rsi.selected_postcontact80ms_execution_alignment.v1"
    assert result[0]["full80ms_execution_alignment_verified"]
    assert result[0]["physical_substeps_compared"] == 40
    assert result[0]["actual_neural_target_updates_compared"] == 3
    assert not result[0]["teacher_qualified"]


@pytest.mark.parametrize(
    "bad",
    [
        "schema",
        "promote",
        "offset",
        "steps",
        "accumulated",
        "ranking",
        "name",
        "missing",
        "path",
        "actual_neural",
        "prior",
        "basis",
    ],
)
def test_headers_banks_causality_and_actual_selected_neural_captures_not_trusted(bad):
    a, t, root, d = protocol()
    if bad == "schema":
        d[5]["schema"] = "old-profile"
    elif bad == "promote":
        d[30]["teacher_qualified"] = True
    elif bad == "offset":
        d[50]["requested_offset_nm"][0] = 0.1
    elif bad == "steps":
        d[30]["forecast_substeps"] = 40
    elif bad == "accumulated":
        d[-1]["accumulated_forecast_substeps"] = 0
    elif bad == "ranking":
        d[30]["selected_index"] = 1
    elif bad == "name":
        d[30]["selected_name"] = "lift_plus"
    elif bad == "missing":
        d[30]["raw_witnesses"].pop()
    elif bad == "path":
        d[30]["raw_witnesses"][0]["path"] = "other.npz"
    elif bad == "actual_neural":
        t["foundation_neural_encoder_features"][33, 0, 0] = -0.0
    elif bad == "prior":
        t["force_n"][29] = 0
    else:
        a["offset_nm"][0] = 0.1
    with pytest.raises(ValueError):
        read(a, t, root, d)


def test_actual_prior_force_validation_cannot_use_early_feasibility():
    a, t, _, d = protocol()
    m = dict(d[30]["raw_witnesses"][0], is_zero=True)
    verify_postcontact_exit_prediction(a, m, t["force_n"][29, 0])
    for force in (np.zeros(6), np.ones(5), np.array([2.0, 0, 0, 0, 0, np.nan])):
        with pytest.raises(ValueError):
            verify_postcontact_exit_prediction(a, m, force)


def cohort(adaptive):
    r = rows()
    cfg = dict(scope(), output_root="/synthetic")
    if adaptive:
        cfg.update(
            profile="postcontact-exit-proxy-transfer18-v1",
            action_name="CURRENT_BODY_EXIT_PROXY",
            forecast_step_budget=10080,
            postcontact_exit_enabled=True,
            maximum_forecasts_per_episode=14,
        )
    kind = "postcontact_exit_transfer" if adaptive else "postcontact_release_transfer"
    d = dict(
        config=cfg,
        schema=f"soccer.rsi.{kind}_declaration.v1",
        report_hash="declaration",
        jobs=[{}] * 18,
    )
    s = paired_summary(r)
    if adaptive:
        s["local_action_selection_promising"] = s.pop("fixed_action_locally_promising")
    c = dict(
        schema=f"soccer.rsi.{kind}_complete.v1",
        declaration_hash="declaration",
        rows=r,
        new_native_steps=54000,
        executed_control_foundation_pairs=5400,
        new_optimizer_updates=0,
        baseline_rollouts_reused=18,
        forecast_steps=0,
        hardware_authorized=False,
        **s,
    )
    return c, d


@pytest.mark.parametrize("adaptive", [False, True])
def test_complete_both_distinct_profiles(adaptive):
    c, d = cohort(adaptive)
    assert check_complete(c, d, Path("/synthetic"))[0] is adaptive


@pytest.mark.parametrize(
    "bad", ["drop", "index", "native", "bool_counter", "forecasts", "promote", "outcome"]
)
def test_whole_cohort_scope_and_primitive_summary_reject_corruption(bad):
    c, d = cohort(True)
    if bad == "drop":
        c["rows"].pop()
    elif bad == "index":
        c["rows"][-1]["index"] = 0
    elif bad == "native":
        c["new_native_steps"] = 53999
    elif bad == "bool_counter":
        c["new_optimizer_updates"] = False
    elif bad == "forecasts":
        c["forecast_steps"] = 10081
    elif bad == "promote":
        c["teacher_qualified"] = True
    else:
        c["local"]["candidate_passes"] += 1
    with pytest.raises(ValueError):
        check_complete(c, d, Path("/synthetic"))
