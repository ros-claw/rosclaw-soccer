import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.sampling_action_opportunity import sampling_action_opportunity
from tests.rsi.test_sampling_credit_diagnostics import record


def test_complete_failures_and_contact_partition_remain_owned():
    rows = [record(0), record(1)]
    rows[1]["outcome"].update(high_quality=False, forward_60_m=0.5)
    gates = np.stack([np.ones(270), np.zeros(270)])
    latent = np.zeros((2, 270, 12))
    latent[1] = 8
    saved = copy.deepcopy(rows)
    original_gates, original_latent = gates.copy(), latent.copy()
    result = sampling_action_opportunity(rows, gates, latent, expected_episodes=2)
    assert result["aggregate"]["zero_gate_frames"] == 270
    assert result["aggregate"]["unit_gate_frames"] == 270
    assert result["aggregate"]["latent_tanh_derivative_below_0_01_coordinates"] == 270 * 12
    assert result["overlapping_outcome_strata"]["not_high_quality"]["episode_groups"] == [1]
    assert result["overlapping_outcome_strata"]["insufficient_forward"]["zero_gate_frames"] == 270
    assert sum(x["frame_samples"] for x in result["contact_time_strata"].values()) == 540
    assert result["contact_time_strata"]["before_first_contact"]["frame_samples"] == 70
    assert rows == saved
    np.testing.assert_array_equal(gates, original_gates)
    np.testing.assert_array_equal(latent, original_latent)
    for key in (
        "downstream_projection_sensitivity_measured",
        "causal_failure_explanation_proven",
        "private_fresh_accessed",
        "runtime_selection_authorized",
        "training_authorized",
        "promotion_authorized",
        "hardware_authorized",
    ):
        assert result[key] is False


def test_empty_success_and_no_contact_do_not_create_fictitious_measurements():
    row = record()
    row["outcome"].update(
        high_quality=False, clean_foot_only=False, contact_body_indices=[], first_contact_frame=None
    )
    result = sampling_action_opportunity(
        [row], np.ones((1, 270)), np.zeros((1, 270, 12)), expected_episodes=1
    )
    assert result["overlapping_outcome_strata"]["high_quality"]["minimum_gate"] is None
    assert result["contact_time_strata"]["episode_without_contact"]["frame_samples"] == 270
    assert result["aggregate"]["minimum_latent_tanh_derivative"] == 1


@pytest.mark.parametrize(
    "gates,latent",
    [
        (np.ones((1, 269)), np.zeros((1, 270, 12))),
        (np.ones((1, 270)), np.zeros((1, 270, 11))),
        (np.ones((1, 270), dtype=np.float32), np.zeros((1, 270, 12))),
        (np.ones((1, 270)), np.zeros((1, 270, 12), dtype=np.float32)),
        (np.full((1, 270), -0.01), np.zeros((1, 270, 12))),
        (np.full((1, 270), 1.01), np.zeros((1, 270, 12))),
        (np.full((1, 270), np.nan), np.zeros((1, 270, 12))),
        (np.ones((1, 270)), np.full((1, 270, 12), np.inf)),
        (np.ones((1, 270)), np.full((1, 270, 12), 1e6 + 1)),
    ],
)
def test_invalid_full_numeric_contract_is_rejected(gates, latent):
    with pytest.raises(ValueError, match="complete finite"):
        sampling_action_opportunity([record()], gates, latent, expected_episodes=1)


@pytest.mark.parametrize("groups,expected", [([0], 2), ([0, 0], 2), ([1, 0], 2)])
def test_failure_dropping_or_reordering_rejected(groups, expected):
    with pytest.raises(ValueError, match="complete"):
        sampling_action_opportunity(
            [record(g) for g in groups],
            np.ones((len(groups), 270)),
            np.zeros((len(groups), 270, 12)),
            expected_episodes=expected,
        )


@pytest.mark.parametrize("expected", [True, 0, 4097])
def test_unbounded_or_boolean_count_is_not_a_declared_bank(expected):
    with pytest.raises(ValueError, match="bounded"):
        sampling_action_opportunity([], [], [], expected_episodes=expected)
