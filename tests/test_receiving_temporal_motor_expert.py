"""The temporal motor policy reads current state yet never bypasses A2 ownership."""

from types import SimpleNamespace

import pytest

from rosclaw_soccer.rsi.receiving_lateral_piecewise_expert import ReceivingLateralPiecewiseExpert
from rosclaw_soccer.rsi.receiving_temporal_motor_expert import (
    ReceivingTemporalMotorExpert,
    TemporalMotorWeights,
)
from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def _actor(**kwargs: object) -> ReceivingTemporalMotorExpert:
    schedule = ReceivingOracleSchedule(
        "red.finisher", "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    return ReceivingTemporalMotorExpert(
        "red.finisher",
        schedule.contract_hash,
        ReceiveContactMailbox("red.finisher"),
        (0.0,) * 8,
        (0.0,) * 12,
        (0.0,) * 12,
        policy=TemporalMotorWeights(output_bias=(0.2,) + (0.0,) * 11),
        **kwargs,
    )


def _observation(frame: int, *, foot: bool = False) -> SimpleNamespace:
    qpos = [0.0] * 43
    qvel = [0.0] * 41
    qpos[2] = 0.7
    qpos[36:39] = [0.4, 0.14, 0.1]
    qvel[35] = -1.25
    return SimpleNamespace(
        qpos=tuple(qpos),
        qvel=tuple(qvel),
        frame=frame,
        last_own_foot_contact_time_sec=0.5 if foot else None,
    )


def test_temporal_weights_and_exploration_fail_closed() -> None:
    with pytest.raises(ValueError, match="finite bounded temporal"):
        TemporalMotorWeights(output_bias=(float("nan"),) + (0.0,) * 11)
    with pytest.raises(ValueError, match="bounded SIM_ONLY temporal"):
        _actor(exploration_std=0.31)
    with pytest.raises(ValueError, match="bounded SIM_ONLY temporal"):
        _actor(exploration_correlation=1.0)


def test_temporal_policy_tracks_50hz_observations_under_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def base(
        self: ReceivingLateralPiecewiseExpert, observation: SimpleNamespace
    ) -> tuple[float, ...]:
        self._selected_side = 0
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingLateralPiecewiseExpert, "propose", base)
    actor = _actor()
    assert actor.propose(_observation(15)) == pytest.approx((0.0,) * 29)
    peak = actor.propose(_observation(20, foot=True))
    late = actor.propose(_observation(60, foot=True))
    assert peak[0] == pytest.approx(0.25 * 0.197375320224904)
    assert 0 < late[0] < peak[0]
    assert actor.observed_frames == [15, 20, 60]
    assert actor.observed_features[0][-1] == 0.0
    assert actor.observed_features[1][-1] == 1.0
    assert actor.activation_ceiling == "SIM_ONLY"


def test_fixed_noise_seed_replays_same_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    def base(
        self: ReceivingLateralPiecewiseExpert, observation: SimpleNamespace
    ) -> tuple[float, ...]:
        self._selected_side = 0
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingLateralPiecewiseExpert, "propose", base)
    first = _actor(exploration_std=0.1, exploration_seed=123)
    second = _actor(exploration_std=0.1, exploration_seed=123)
    for frame in (15, 20, 25):
        assert first.propose(_observation(frame)) == second.propose(_observation(frame))
    assert first.sampled_logits == second.sampled_logits


def test_correlated_exploration_is_replayable_and_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def base(
        self: ReceivingLateralPiecewiseExpert, observation: SimpleNamespace
    ) -> tuple[float, ...]:
        self._selected_side = 0
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingLateralPiecewiseExpert, "propose", base)
    first = _actor(exploration_std=0.1, exploration_seed=123, exploration_correlation=0.85)
    replay = _actor(exploration_std=0.1, exploration_seed=123, exploration_correlation=0.85)
    independent = _actor(exploration_std=0.1, exploration_seed=123)
    for frame in (15, 20, 25):
        first.propose(_observation(frame))
        replay.propose(_observation(frame))
        independent.propose(_observation(frame))
    assert first.sampled_logits == replay.sampled_logits
    assert first.sampled_logits[0] == independent.sampled_logits[0]
    assert first.sampled_logits[1] != independent.sampled_logits[1]
    assert first.activation_ceiling == "SIM_ONLY"
