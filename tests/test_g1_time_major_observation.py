from dataclasses import replace

import pytest

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.providers.g1.time_major_observation import (
    G1ProprioceptiveFrame,
    G1TimeMajorObservation,
)


def frame(command: float = 0.4) -> G1ProprioceptiveFrame:
    return G1ProprioceptiveFrame(
        (0.1, 0.2, 0.3), (0.0, 0.0, -1.0), (command, 0.0, 0.0), (0.5,) * 29, (0.2,) * 29
    )


def history() -> G1TimeMajorObservation:
    return G1TimeMajorObservation(
        joint_names=G1_DDS_JOINT_NAMES, default_position=(0.1,) * 29, default_velocity=(0.05,) * 29
    )


def test_initial_command_backfill_and_causal_previous_action() -> None:
    h = history()
    first = h.prepare(tick=0, frame=frame())
    assert len(first) == 384
    assert first == first[:96] * 4
    assert first[6:9] == (0.4, 0.0, 0.0)
    assert first[9:38] == pytest.approx((0.4,) * 29)
    assert first[38:67] == pytest.approx((0.15,) * 29)
    assert first[67:96] == (0.0,) * 29
    h.commit_action((0.25,) * 29)
    second = h.prepare(tick=1, frame=frame(0.2))
    assert second[:288] == first[:288]
    assert second[-90:-87] == (0.2, 0.0, 0.0)
    assert second[-29:] == (0.25,) * 29


def test_missing_action_faults_and_prevents_reuse() -> None:
    h = history()
    h.prepare(tick=0, frame=frame())
    with pytest.raises(ValueError):
        h.prepare(tick=1, frame=frame())
    with pytest.raises(RuntimeError):
        h.commit_action((0.0,) * 29)


def test_invalid_action_and_wrong_tick_latch_fault() -> None:
    h = history()
    h.prepare(tick=0, frame=frame())
    with pytest.raises(ValueError):
        h.commit_action((float("nan"),) * 29)
    with pytest.raises(RuntimeError):
        h.prepare(tick=1, frame=frame())
    other = history()
    with pytest.raises(ValueError):
        other.prepare(tick=1, frame=frame())
    with pytest.raises(RuntimeError):
        other.prepare(tick=0, frame=frame())


def test_bad_inputs_and_unapproved_joint_order_fail_closed() -> None:
    with pytest.raises(ValueError):
        replace(frame(), projected_gravity=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        replace(frame(), command=(float("nan"), 0.0, 0.0))
    with pytest.raises(ValueError):
        G1TimeMajorObservation(
            joint_names=tuple(reversed(G1_DDS_JOINT_NAMES)),
            default_position=(0.0,) * 29,
            default_velocity=(0.0,) * 29,
        )
