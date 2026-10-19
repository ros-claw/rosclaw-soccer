"""Zero-branch protocol bounds and state-signature regression, not skill gains."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.foot_heading_evidence import FootHeadingCpuReconstruction
from scripts.rsi_calibrate_feedback_branches import ik_signature, tracker_signature
from scripts.rsi_feedback_branch_inputs import require_scope
from tests.rsi.test_foot_heading_evidence import fixture as ik_fixture


def scope():
    return dict(
        mode="SIM_ONLY",
        profile="six-zero-feedback40ms-v1",
        frame_selection="RETROSPECTIVE_SOURCE_CONTACT_FRAME_DIAGNOSTIC_ONLY",
        source_cases=6,
        zero_branches_per_case=2,
        horizon_substeps=20,
        recorded_replay_substep_budget=18000,
        feedback_forecast_substep_budget=240,
        combined_substep_budget=18240,
        maximum_planner_run_calls=60,
        current_encoder_decoder_pairs=6,
        branch_encoder_decoder_pairs=12,
        original_residual_decisions=1800,
        original_recurrent_updates=1620,
        new_closed_loop_policy_trials=0,
        new_optimizer_updates=0,
        candidate_actions_optimized=0,
        hardware_authorized=False,
        promotion_authorized=False,
        private_fresh_accessed=False,
        training_authorized=False,
        teacher_qualified=False,
        actor_future_event_input_allowed=False,
    )


def test_exact_zero_feedback_protocol():
    require_scope(scope())


@pytest.mark.parametrize("key", list(scope()))
def test_no_omission_or_widening_of_protocol(key):
    cfg = scope()
    del cfg[key]
    with pytest.raises(ValueError):
        require_scope(cfg)
    cfg = scope()
    v = cfg[key]
    cfg[key] = int(v) if type(v) is bool else float(v) if type(v) is int else "REAL"
    with pytest.raises(ValueError):
        require_scope(cfg)


def test_ik_signature_handles_frozen_slots_probe_and_detects_mutable_state():
    model, data, _, _, probe = ik_fixture()
    ik = FootHeadingCpuReconstruction(model, probe)
    ik.before_step(data, 0, data.xpos, np.zeros(29))
    before = ik_signature(ik)
    assert before == ik_signature(ik)
    ik.contact_delta[0] = 0.1
    assert before != ik_signature(ik)


def test_tracker_signature_tracks_history_and_not_shared_weights():
    pytest.importorskip("torch")
    from tests.providers.test_sonic_branch_state import tracker

    t, _, _ = tracker()
    before = tracker_signature(t)
    assert before == tracker_signature(t)
    t._history[0][0][0, 0] = 0.1
    assert before != tracker_signature(t)
