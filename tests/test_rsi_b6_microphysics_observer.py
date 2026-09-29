"""Read-only 500 Hz local receiver-contact capture contract."""

from __future__ import annotations

import numpy as np
import pytest

from rosclaw_soccer.rsi.b6_microphysics_observer import B6MicrophysicsObserver
from rosclaw_soccer.skills.team.motor_option import TeamBallContact, TeamMotorPhysicsObservation
from rosclaw_soccer.skills.team.physics_evidence import PhysicsEvidenceSlot


def _observation(step: int, *, contact: bool = False, ball_speed: float = 0.8):
    qpos = [0.0] * 43
    qvel = [0.0] * 41
    qpos[2] = 0.8
    qpos[38] = 0.115
    qvel[35] = ball_speed
    contacts = (
        (
            TeamBallContact(
                geometry_id=1,
                agent_id="red.finisher",
                effector="right_foot",
                normal_force_n=12.0,
                contact_position_world_m=(1.0, 0.0, 0.1),
                normal_ball_to_counterpart_world=(1.0, 0.0, 0.0),
                counterpart_minus_ball_velocity_world_mps=(-0.8, 0.0, 0.0),
            ),
        )
        if contact
        else ()
    )
    return TeamMotorPhysicsObservation(
        time_sec=(step + 1) * 0.002,
        qpos=tuple(qpos),
        qvel=tuple(qvel),
        world_bodies_safe=True,
        foot_normal_force_n=12.0 if contact else 0.0,
        other_non_ground_normal_force_n=0.0,
        observer_agent_id="red.finisher",
        contacts_complete=True,
        ball_contacts=contacts,
    )


def test_dynamic_foot_collision_captures_consecutive_read_only_window():
    observer = B6MicrophysicsObserver("red.finisher")
    slot = PhysicsEvidenceSlot(observer)
    assert slot.needs_contact_velocity is True
    for step in range(326):
        slot.observe_physics(_observation(step, contact=step == 75))
    assert not slot.faulted
    assert observer.complete
    assert observer.first_foot_time_sec == pytest.approx(0.152)
    assert observer.incoming_ball_speed_mps == pytest.approx(0.8)
    arrays = observer.arrays()
    assert arrays["qpos"].shape == (326, 43)
    assert arrays["qvel"].shape == (326, 41)
    assert np.allclose(np.diff(arrays["time_sec"]), 0.002)
    assert np.flatnonzero(arrays["own_foot_normal_force_n"] > 0).tolist() == [75]
    assert arrays["contact_geometry_complete"][75]
    assert np.array_equal(arrays["ball_to_foot_normal_world"][75], [1.0, 0.0, 0.0])


def test_stationary_ball_contact_is_not_a_dynamic_receive():
    observer = B6MicrophysicsObserver("red.finisher", pre_steps=25, post_steps=100)
    slot = PhysicsEvidenceSlot(observer)
    for step in range(127):
        slot.observe_physics(_observation(step, contact=step == 25, ball_speed=0.05))
    assert not slot.faulted
    assert not observer.complete
    with pytest.raises(ValueError, match="complete measured dynamic receiver window"):
        observer.arrays()


@pytest.mark.parametrize("pre,post", [(24, 100), (101, 250), (75, 99), (75, 301)])
def test_unbounded_microphysics_window_is_rejected(pre: int, post: int):
    with pytest.raises(ValueError, match="bounded receiver microphysics observer"):
        B6MicrophysicsObserver("red.finisher", pre_steps=pre, post_steps=post)
