import math

import pytest

from rosclaw_soccer.training.reception_phase import advance_reception_phase


def step(frame=0, start_frame=None, **kw):
    args = dict(ball_offset_xy=(1.0, 0.0), relative_velocity_xy=(-0.5, 0.0))
    args.update(kw)
    return advance_reception_phase(frame=frame, start_frame=start_frame, **args)


def test_observed_entry_latches_and_does_not_restart():
    assert step(ball_offset_xy=(2.0, 0.0)).start_frame is None
    entered = step(frame=20)
    assert entered.start_frame == 20 and entered.fraction == 0
    later = step(frame=40, start_frame=20, relative_velocity_xy=(1.0, 0.0))
    assert later.start_frame == 20 and later.fraction == 0.2
    assert step(frame=200, start_frame=20).fraction == 0.99


def test_tangent_stationary_and_receding_do_not_start():
    for velocity in ((0.0, 1.0), (0.0, 0.0), (1.0, 0.0)):
        assert step(relative_velocity_xy=velocity).start_frame is None


def test_rotation_and_absolute_time_translation():
    first = step(frame=10)
    rotated = step(frame=10, ball_offset_xy=(0.0, 1.0), relative_velocity_xy=(0.0, -0.5))
    assert first == rotated
    assert step(frame=140, start_frame=120).fraction == step(frame=40, start_frame=20).fraction


@pytest.mark.parametrize(
    "kw",
    [
        dict(frame=True),
        dict(frame=-1),
        dict(start_frame=1),
        dict(start_frame=False),
        dict(phase_frames=0),
        dict(phase_frames=10.0),
        dict(approach_radius_m=math.nan),
        dict(ball_offset_xy=(math.inf, 0.0)),
        dict(relative_velocity_xy=(0.0,)),
        dict(ball_offset_xy=(True, 0.0)),
        dict(relative_velocity_xy=[0.0, 0.0]),
    ],
)
def test_invalid_clock_and_physics_rejected(kw):
    with pytest.raises(ValueError):
        step(**kw)
