from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw_soccer.skills.team.motor_option import TeamMotorPhysicsObservation
from rosclaw_soccer.skills.team.physics_evidence import PhysicsEvidenceSlot


def observation():
    return TeamMotorPhysicsObservation(
        0.002,
        (0.0,) * 43,
        (0.0,) * 41,
        True,
        0.0,
        0.0,
        observer_agent_id="red.a",
        contacts_complete=True,
    )


def consumer():
    rows = []
    return SimpleNamespace(
        agent_id="red.a", contract_hash="sha256:" + "a" * 64, observe_physics=rows.append
    ), rows


def test_observer_needs_no_motor_method_and_values_are_preserved():
    c, rows = consumer()
    slot = PhysicsEvidenceSlot(c)
    value = observation()
    assert not hasattr(slot, "propose") and not hasattr(c, "propose")
    slot.observe_physics(value)
    slot.observe_physics(replace(value, time_sec=0.004))
    assert rows == [value, replace(value, time_sec=0.004)] and not slot.faulted
    with pytest.raises(AttributeError):
        slot.agent_id = "blue.b"


@pytest.mark.parametrize("time", [0.002, 0.001, 0.006])
def test_clock_fault_latches(time):
    c, rows = consumer()
    slot = PhysicsEvidenceSlot(c)
    value = observation()
    slot.observe_physics(value)
    slot.observe_physics(replace(value, time_sec=time))
    slot.observe_physics(replace(value, time_sec=0.004))
    assert slot.faulted and rows == [value]


@pytest.mark.parametrize("fault", ["partial", "foreign", "untyped"])
def test_invalid_observation_never_reaches_consumer(fault):
    c, rows = consumer()
    slot = PhysicsEvidenceSlot(c)
    value = observation()
    if fault == "partial":
        value = replace(value, contacts_complete=False, observer_agent_id=None)
    if fault == "foreign":
        value = replace(value, observer_agent_id="blue.b")
    if fault == "untyped":
        value = None
    slot.observe_physics(value)
    assert slot.faulted and not rows


@pytest.mark.parametrize("when", ["before", "during"])
def test_registration_mutation_is_rejected(when):
    c, rows = consumer()
    slot = PhysicsEvidenceSlot(c)
    if when == "before":
        c.contract_hash = "sha256:" + "b" * 64
    else:

        def mutate(value):
            c.agent_id = "blue.b"

        c.observe_physics = mutate
    slot.observe_physics(observation())
    assert slot.faulted and slot.fault_reason == "ValueError"


def test_consumer_runtime_error_does_not_escape_or_retry():
    c, rows = consumer()
    count = []

    def fail(value):
        count.append(value)
        raise RuntimeError("bad consumer")

    c.observe_physics = fail
    slot = PhysicsEvidenceSlot(c)
    slot.observe_physics(observation())
    slot.observe_physics(observation())
    assert len(count) == 1 and slot.fault_reason == "RuntimeError"


def test_returned_action_is_rejected():
    c, _ = consumer()
    c.observe_physics = lambda value: (0.0,) * 29
    slot = PhysicsEvidenceSlot(c)
    slot.observe_physics(observation())
    assert slot.faulted


def test_initial_missing_step_cannot_claim_complete_episode():
    c, rows = consumer()
    slot = PhysicsEvidenceSlot(c)
    slot.observe_physics(replace(observation(), time_sec=0.004))
    assert slot.faulted and not rows


@pytest.mark.parametrize("fault", ["foreign", "alias_motor", "not_mapping"])
def test_world_rejects_invalid_observers_before_asset_loading(fault):
    from rosclaw_soccer.skills.team.independent_team_world import simulate_independent_team_world

    c, _ = consumer()
    mapping = {"red.a": c}
    motors = {}
    if fault == "foreign":
        mapping = {"blue.b": c}
    if fault == "alias_motor":
        motors = {"red.a": c}
    if fault == "not_mapping":
        mapping = []
    with pytest.raises(ValueError, match="physics evidence"):
        simulate_independent_team_world(
            asset_root=Path("must-not-be-opened"),
            roster=SimpleNamespace(agents=[SimpleNamespace(agent_id="red.a")]),
            cells=(),
            players=(),
            scenario=None,
            goal=None,
            motor_options=motors,
            physics_evidence_consumers=mapping,
        )


@pytest.mark.parametrize(
    "key,value",
    [("agent_id", "../x"), ("agent_id", None), ("contract_hash", "bad"), ("contract_hash", 0)],
)
def test_invalid_binding(key, value):
    c, _ = consumer()
    setattr(c, key, value)
    with pytest.raises(ValueError):
        PhysicsEvidenceSlot(c)
