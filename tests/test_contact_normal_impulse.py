from dataclasses import replace

import pytest

from rosclaw_soccer.skills.team.motor_option import TeamBallContact, TeamMotorPhysicsObservation
from rosclaw_soccer.training.contact_normal_impulse import ContactNormalImpulseRecorder

IDS = ("blue.defender", "red.playmaker")


def contact(owner="blue.defender", effector="left_foot", force=10.0, normal=(-1.0, 0.0, 0.0)):
    return TeamBallContact(1, owner, effector, force, (0.0, 0.0, 0.0), normal)


def observation(index, contacts=(), safe=True):
    own = max(
        (c.normal_force_n for c in contacts if c.agent_id == IDS[0] and c.is_foot), default=0.0
    )
    other = max(
        (c.normal_force_n for c in contacts if not (c.agent_id == IDS[0] and c.is_foot)),
        default=0.0,
    )
    return TeamMotorPhysicsObservation(
        index * 0.002, (0.0,) * 43, (0.0,) * 41, safe, own, other, IDS[0], True, tuple(contacts)
    )


def test_normal_impulse_sign_ten_substep_integral_and_reset():
    rec = ContactNormalImpulseRecorder(IDS[0], IDS)
    for step in range(1, 21):
        rec.observe_physics(observation(step, (contact(),) if step <= 10 else ()))
    first, second = rec.verified_frames(control_frames=2)
    assert first.foot_normal_impulse_ns[0][1:] == pytest.approx((0.2, 0, 0))
    assert first.foot_contact_substeps == ((IDS[0], 10), (IDS[1], 0))
    assert not first.ambiguous_agents
    assert second.foot_normal_impulse_ns[0][1:] == (0, 0, 0)


def test_multiple_geometries_count_each_force_but_only_one_substep():
    rec = ContactNormalImpulseRecorder(IDS[0], IDS)
    for step in range(1, 11):
        contacts = (contact(force=10), contact(effector="right_foot", force=5)) if step == 1 else ()
        rec.observe_physics(observation(step, contacts, safe=step != 5))
    frame = rec.verified_frames(control_frames=1)[0]
    assert frame.foot_normal_impulse_ns[0][1] == pytest.approx(0.03)
    assert frame.foot_contact_substeps[0][1] == 1
    assert frame.world_bodies_safe is False
    assert frame.ambiguous_agents == ()


def test_other_foot_or_body_marks_ambiguous():
    rec = ContactNormalImpulseRecorder(IDS[0], IDS)
    for step in range(1, 11):
        contacts = (contact(), contact(IDS[1], "right_foot", 3)) if step == 1 else ()
        rec.observe_physics(observation(step, contacts))
    assert rec.verified_frames(control_frames=1)[0].ambiguous_agents == IDS


@pytest.mark.parametrize("other", [contact(IDS[1], "right_foot", 3), contact(IDS[0], "body", 3)])
def test_other_contact_at_a_different_substep_censors_the_whole_frame(other):
    rec = ContactNormalImpulseRecorder(IDS[0], IDS)
    for step in range(1, 11):
        contacts = (contact(),) if step == 1 else ((other,) if step == 8 else ())
        rec.observe_physics(observation(step, contacts))
    frame = rec.verified_frames(control_frames=1)[0]
    assert IDS[0] in frame.ambiguous_agents
    assert frame.foot_normal_impulse_ns[0][1] == pytest.approx(0.02)


def test_other_contact_in_next_frame_does_not_contaminate_preceding_frame():
    rec = ContactNormalImpulseRecorder(IDS[0], IDS)
    for step in range(1, 21):
        contacts = (
            (contact(),) if step == 10 else ((contact(IDS[0], "body", 3),) if step == 11 else ())
        )
        rec.observe_physics(observation(step, contacts))
    first, second = rec.verified_frames(control_frames=2)
    assert first.ambiguous_agents == ()
    assert second.ambiguous_agents == ()


def test_missing_normals_and_invalid_stream_fail_closed():
    rec = ContactNormalImpulseRecorder(IDS[0], IDS)
    with pytest.raises(ValueError, match="normal required"):
        rec.observe_physics(
            observation(
                1,
                (
                    replace(
                        contact(),
                        contact_position_world_m=None,
                        normal_ball_to_counterpart_world=None,
                    ),
                ),
            )
        )
    with pytest.raises(ValueError, match="contiguous"):
        rec.observe_physics(observation(2))
    with pytest.raises(ValueError, match="entire control trace"):
        rec.verified_frames(control_frames=1)
