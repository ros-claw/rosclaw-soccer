from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.skills.team.motor_option import TeamMotorTarget
from rosclaw_soccer.training.receiving_feedback import (
    ReceivingCaptureContext,
    ReceivingFeedbackObservation,
    ReceivingFeedbackSlot,
)
from rosclaw_soccer.training.receiving_oracle_schedule import (
    ReceivingOracleCursor,
    ReceivingOracleSchedule,
)


def schedule(start=0):
    return ReceivingOracleSchedule("blue.finisher", "A0_leg12", start, 20, ((0.0,) * 12,))


def observation(frame=0):
    q = [0.0] * 43
    q[3] = q[39] = 1.0
    return ReceivingFeedbackObservation(
        "blue.finisher",
        frame,
        frame * 0.02,
        tuple(q),
        (0.0,) * 41,
        TeamMotorTarget((0.0,) * 29, (20.0,) * 29, (1.0,) * 29),
        (0.0,) * 12,
        None,
        None,
    )


class Provider:
    agent_id = "blue.finisher"
    activation_ceiling = "SIM_ONLY"
    contract_hash = "sha256:" + "a" * 64

    def __init__(self, source):
        self.schedule_hash = source.contract_hash
        self.calls = 0
        self.result = (0.1,) * 12

    def propose(self, obs):
        self.calls += 1
        return self.result


def test_preentry_observes_without_querying_provider():
    source = schedule(2)
    provider = Provider(source)
    slot = ReceivingFeedbackSlot(provider, source)
    assert slot.step(observation(0)) is None
    assert slot.step(observation(1)) is None
    assert provider.calls == 0
    assert slot.step(observation(2)) == (0.1,) * 12
    assert provider.calls == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"frame": True},
        {"time_sec": float("nan")},
        {"time_sec": 0.01},
        {"qpos": (0.0,) * 43},
        {"qvel": [0.0] * 41},
        {"qvel": (float("inf"),) * 41},
        {"native_actor_raw": (True,) * 12},
        {"last_own_contact_foot": 1},
        {"last_own_foot_contact_time_sec": 0.01, "last_own_contact_foot": 1},
    ],
)
def test_invalid_observation(changes):
    with pytest.raises(ValueError):
        replace(observation(), **changes)


@pytest.mark.parametrize(
    "result",
    [
        [0.0] * 12,
        (0.0,) * 11,
        (True,) * 12,
        (0.10001,) * 12,
        (float("nan"),) * 12,
        (float("inf"),) * 12,
    ],
)
def test_bad_proposal_fault_latches(result):
    source = schedule()
    provider = Provider(source)
    provider.result = result
    slot = ReceivingFeedbackSlot(provider, source)
    with pytest.raises(ValueError):
        slot.step(observation())
    provider.result = (0.0,) * 12
    with pytest.raises(ValueError, match="latched"):
        slot.step(observation())
    assert provider.calls == 1


@pytest.mark.parametrize("mutation", ["binding", "input"])
def test_callback_mutation_rejected_same_tick(mutation):
    source = schedule()

    class Mutator(Provider):
        def propose(self, obs):
            if mutation == "binding":
                self.contract_hash = "sha256:" + "b" * 64
            else:
                object.__setattr__(obs, "native_actor_raw", (1.0,) * 12)
            return (0.0,) * 12

    slot = ReceivingFeedbackSlot(Mutator(source), source)
    with pytest.raises(ValueError):
        slot.step(observation())
    assert slot.faulted


def test_clock_and_source_binding_fail_closed():
    source = schedule()
    provider = Provider(source)
    slot = ReceivingFeedbackSlot(provider, source)
    with pytest.raises(ValueError):
        slot.step(observation(1))
    assert provider.calls == 0
    provider.schedule_hash = "sha256:" + "b" * 64
    with pytest.raises(ValueError):
        ReceivingFeedbackSlot(provider, source)
    for substrate in ("A1_body29", "A3_sonic_residual"):
        other = replace(source, substrate=substrate, knots=((0.0,) * 29,))
        with pytest.raises(ValueError):
            ReceivingFeedbackSlot(Provider(other), other)


def test_cursor_preserves_original_filter_limits_and_copies():
    cursor = ReceivingOracleCursor(schedule())
    result = cursor.step(0, active=True, predecessor=np.zeros(12), desired_override_rad=(0.1,) * 12)
    np.testing.assert_allclose(result, 0.02)
    result[:] = 50
    result = cursor.step(1, active=True, predecessor=np.zeros(12), desired_override_rad=(0.1,) * 12)
    np.testing.assert_allclose(result, 0.04)
    result = cursor.step(
        2, active=False, predecessor=np.zeros(12), desired_override_rad=(0.1,) * 12
    )
    np.testing.assert_allclose(result, 0.03)


def test_feedback_cannot_mix_with_phase_or_preentry():
    for source, options in ((schedule(1), {}), (schedule(), {"reference_frame": 0})):
        cursor = ReceivingOracleCursor(source)
        with pytest.raises(ValueError):
            cursor.step(
                0,
                active=True,
                predecessor=np.zeros(12),
                desired_override_rad=(0.1,) * 12,
                **options,
            )
        assert cursor.faulted


def test_unbound_provider_rejected_before_loading_assets():
    from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
    from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

    with pytest.raises(ValueError, match="feedback must bind"):
        simulate_r0_receiving_course(
            asset_root=Path("must-not-load"),
            reference_policy_path=Path("must-not-load"),
            course=ReceivingCourse("blue.finisher", 1, 0.75, -0.08),
            scenario_id="feedback.preflight",
            feedback_provider=Provider(schedule()),
        )


def test_capture_event_clock_is_not_latest_contact_clock():
    capture = ReceivingCaptureContext(0.1, 0.6, 1, (1.0, 0.0))
    obs = replace(
        observation(20),
        capture_context=capture,
        last_own_foot_contact_time_sec=0.39,
        last_own_contact_foot=1,
    )
    assert obs.time_sec - capture.start_time_sec == pytest.approx(0.3)
    assert obs.time_sec - obs.last_own_foot_contact_time_sec == pytest.approx(0.01)
    assert obs.observation_hash != observation(20).observation_hash
    for changes in (
        {"time_sec": 0.08, "frame": 4},
        {"time_sec": 0.72, "frame": 36},
        {"committed_receive": 1},
        {"capture_context": {}},
    ):
        with pytest.raises(ValueError):
            replace(obs, **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"foot": True},
        {"foot": 3},
        {"duration_sec": 1.01},
        {"start_time_sec": -0.01},
        {"direction_xy": (0.0, 0.0)},
        {"direction_xy": (float("nan"), 0.0)},
        {"direction_xy": [1.0, 0.0]},
    ],
)
def test_invalid_capture_context_rejected(changes):
    with pytest.raises(ValueError):
        replace(ReceivingCaptureContext(0.1, 0.6, 1, (1.0, 0.0)), **changes)
