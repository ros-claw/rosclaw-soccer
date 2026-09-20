from dataclasses import replace

import pytest

from rosclaw_soccer.training.receiving_oracle_local_probe import local_oracle_proposals
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def schedule(substrate="A3_sonic_residual", dimension=29):
    return ReceivingOracleSchedule("blue.finisher", substrate, 30, 20, ((0.0,) * dimension,) * 4)


def test_named_leg_only_with_unchanged_incumbent_timing_and_envelope():
    original = schedule()
    result = local_oracle_proposals(
        original,
        offsets=((("right_knee_joint", 0.25), ("right_ankle_pitch_joint", -0.25)),),
    )
    assert result[0] is original
    assert result[1].agent_id == original.agent_id
    assert result[1].start_frame == 30 and result[1].knot_frames == 20
    for row in result[1].knots:
        assert row[9:11] == (0.25, -0.25)
        assert all(v == 0 for i, v in enumerate(row) if i not in (9, 10))
    assert original.knots == ((0.0,) * 29,) * 4
    assert result[1].contract_hash != original.contract_hash


def test_partial_saturation_is_explicit_and_still_bounded():
    original = replace(schedule(), knots=((0.9,) + (0.0,) * 28, (0.0,) * 29))
    result = local_oracle_proposals(original, offsets=((("left_hip_pitch_joint", 0.25),),))
    assert result[1].knots[0][0] == 1.0
    assert result[1].knots[1][0] == 0.25


@pytest.mark.parametrize("value", [None, [], (), ((),), (([],),), ((("missing", 0.1),),)])
def test_invalid_proposals_rejected(value):
    with pytest.raises(ValueError):
        local_oracle_proposals(schedule(), offsets=value)


@pytest.mark.parametrize("delta", [True, 0, 0.251, -0.251, float("nan"), float("inf"), "0.1"])
def test_invalid_normalized_delta(delta):
    with pytest.raises(ValueError):
        local_oracle_proposals(schedule(), offsets=((("left_knee_joint", delta),),))


def test_leg_substrate_rejects_upper_body_coordinate():
    with pytest.raises(ValueError):
        local_oracle_proposals(schedule("A0_leg12", 12), offsets=((("waist_yaw_joint", 0.1),),))


def test_reject_repeated_coordinate_candidate_and_saturated_noop():
    change = (("left_knee_joint", 0.25),)
    for offsets in ((change, change), (change + change,)):
        with pytest.raises(ValueError):
            local_oracle_proposals(schedule(), offsets=offsets)
    saturated = replace(schedule(), knots=((1.0,) * 29,))
    with pytest.raises(ValueError):
        local_oracle_proposals(saturated, offsets=(change,))


def test_bounded_proposal_count_and_typed_schedule():
    with pytest.raises(ValueError):
        local_oracle_proposals(None, offsets=((("left_knee_joint", 0.1),),))
    with pytest.raises(ValueError):
        local_oracle_proposals(schedule(), offsets=((("left_knee_joint", 0.1),),) * 33)
