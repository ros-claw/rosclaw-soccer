from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.skills.team.motor_option import TeamMotorTarget
from rosclaw_soccer.training.receiving_feedback import (
    ReceivingCaptureContext,
    ReceivingFeedbackObservation,
    ReceivingFeedbackSlot,
    ReceivingLocomotionContext,
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


def locomotion(frame=0):
    from rosclaw_soccer.providers.g1.locomotion_memory import LocomotionMemory

    return ReceivingLocomotionContext(
        frame,
        LocomotionMemory("sha256:" + "a" * 64, bytes(1024), bytes(1024)),
        "sha256:" + "b" * 64,
        (0.0,) * 29,
        (0.0,) * 3,
        False,
    )


def test_memory_is_opt_in_bound_and_hashable():
    from dataclasses import asdict

    from rosclaw_soccer.sim.contracts import hash_json

    old = observation()
    old_fields = asdict(old)
    old_fields.pop("locomotion")
    old_fields.pop("action_substrate")
    old_fields.pop("previous_body_residual_rad")
    assert old.observation_hash == hash_json(old_fields)
    new = replace(old, locomotion=locomotion())
    assert new.observation_hash != old.observation_hash
    source = schedule()
    provider = Provider(source)
    provider.requires_locomotion_memory = True
    assert ReceivingFeedbackSlot(provider, source).step(new) == provider.result
    slot = ReceivingFeedbackSlot(provider, source)
    with pytest.raises(ValueError):
        slot.step(old)
    assert slot.faulted and provider.calls == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"frame": 1},
        {"memory": None},
        {"reflected": 1},
        {"raw_action": (float("nan"),) * 29},
        {"world_command": [0.0] * 3},
        {"configuration_hash": "unknown"},
    ],
)
def test_bad_or_stale_locomotion_context(changes):
    with pytest.raises(ValueError):
        replace(observation(), locomotion=replace(locomotion(), **changes))


def test_memory_requirement_cannot_change_during_proposal():
    source = schedule()
    provider = Provider(source)
    provider.requires_locomotion_memory = True

    def mutate(obs):
        provider.requires_locomotion_memory = 1
        return provider.result

    provider.propose = mutate
    slot = ReceivingFeedbackSlot(provider, source)
    with pytest.raises(ValueError):
        slot.step(replace(observation(), locomotion=locomotion()))
    assert slot.faulted


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


def body_schedule(start=0):
    return replace(schedule(start), substrate="A1_body29", knots=((0.0,) * 29,))


def body_observation(frame=0, previous=(0.0,) * 29):
    return replace(
        observation(frame),
        action_substrate="A1_body29",
        previous_filtered_residual_rad=previous[:12],
        previous_body_residual_rad=previous,
    )


def body_provider(source):
    provider = Provider(source)
    provider.action_substrate = "A1_body29"
    provider.result = (0.1,) * 29
    return provider


def test_whole_body_feedback_requires_explicit_scope_and_complete_memory():
    source = body_schedule()
    provider = body_provider(source)
    slot = ReceivingFeedbackSlot(provider, source)
    assert slot.step(body_observation()) == provider.result
    assert slot.action_dimension == 29
    assert body_observation().observation_hash != observation().observation_hash
    changed = (0.0,) * 28 + (0.01,)
    assert (
        body_observation(previous=changed).observation_hash != body_observation().observation_hash
    )
    for bad in (observation(), body_observation(1)):
        rejected = ReceivingFeedbackSlot(provider, source)
        with pytest.raises(ValueError):
            rejected.step(bad)
        assert rejected.faulted


@pytest.mark.parametrize(
    "previous",
    [
        None,
        (0.0,) * 12,
        [0.0] * 29,
        (True,) * 29,
        (float("nan"),) * 29,
        (0.100001,) * 29,
        (0.01,) * 29,
    ],
)
def test_body_filter_requires_finite_bounded_state_and_matching_legs(previous):
    with pytest.raises(ValueError):
        replace(observation(), action_substrate="A1_body29", previous_body_residual_rad=previous)


@pytest.mark.parametrize(
    "result",
    [(0.0,) * 12, (0.0,) * 28, (float("inf"),) * 29, (0.100001,) * 29, (True,) * 29, [0.0] * 29],
)
def test_body_proposal_is_not_implicitly_padded_or_clipped(result):
    source = body_schedule()
    provider = body_provider(source)
    provider.result = result
    slot = ReceivingFeedbackSlot(provider, source)
    with pytest.raises(ValueError):
        slot.step(body_observation())
    assert slot.faulted


@pytest.mark.parametrize("mutation", ["before", "during"])
def test_action_substrate_binding_is_immutable(mutation):
    source = body_schedule()
    provider = body_provider(source)
    slot = ReceivingFeedbackSlot(provider, source)
    if mutation == "before":
        provider.action_substrate = "A0_leg12"
    else:

        def mutate(obs):
            provider.action_substrate = "A0_leg12"
            return (0.1,) * 29

        provider.propose = mutate
    with pytest.raises(ValueError):
        slot.step(body_observation())
    assert slot.faulted


def test_whole_body_cursor_matches_zero_extended_leg_feedback_and_decays_arms():
    leg = ReceivingOracleCursor(schedule(2))
    body = ReceivingOracleCursor(body_schedule(2))
    old = np.linspace(-0.02, 0.02, 12)
    for frame in range(30):
        desired = tuple(float(v) for v in 0.1 * np.sin(np.arange(12) + frame))
        active = frame % 3 != 0
        a = leg.step(
            frame,
            active=active,
            predecessor=old,
            desired_override_rad=None if frame < 2 else desired,
        )
        b = body.step(
            frame,
            active=active,
            predecessor=old,
            desired_override_rad=None if frame < 2 else desired + (0.0,) * 17,
        )
        if a is None:
            assert b is None
        else:
            np.testing.assert_array_equal(a, b[:12])
            np.testing.assert_array_equal(b[12:], np.zeros(17))
            old = a.copy()
    cursor = ReceivingOracleCursor(body_schedule())
    result = cursor.step(
        0, active=True, predecessor=np.zeros(12), desired_override_rad=(0.0,) * 12 + (0.1,) * 17
    )
    np.testing.assert_allclose(result[12:], 0.02)
    result = cursor.step(
        1, active=False, predecessor=np.zeros(12), desired_override_rad=(0.0,) * 12 + (0.1,) * 17
    )
    np.testing.assert_allclose(result[12:], 0.015)


def test_sonic_feedback_stays_rejected_even_with_explicit_opt_in():
    source = replace(body_schedule(), substrate="A3_sonic_residual")
    provider = body_provider(source)
    provider.action_substrate = source.substrate
    with pytest.raises(ValueError):
        ReceivingFeedbackSlot(provider, source)
    cursor = ReceivingOracleCursor(source)
    with pytest.raises(ValueError):
        cursor.step(0, active=True, predecessor=np.zeros(12), desired_override_rad=(0.0,) * 29)
    assert cursor.faulted


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


@pytest.mark.parametrize(
    "changes",
    [
        {"residual_admitted": 1},
        {"previous_filtered_residual_rad": (True,) * 12},
        {"previous_filtered_residual_rad": (float("nan"),) * 12},
        {"previous_filtered_residual_rad": (0.100001,) * 12},
        {"previous_filtered_residual_rad": (0.0,) * 11},
        {"previous_filtered_residual_rad": [0.0] * 12},
    ],
)
def test_invalid_prior_filter_observation(changes):
    with pytest.raises(ValueError):
        replace(observation(), **changes)
