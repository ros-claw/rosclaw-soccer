import numpy as np
import pytest
from test_admitted_receiver import incoming
from test_receiving_rollout import contact, trace
from test_s368_recurrent_receiver import CONFIG, POLICY, motor

from rosclaw_soccer.providers.g1.admitted_receiver import AdmittedRecurrentReceiver
from rosclaw_soccer.providers.g1.recurrent_receiver import G1RecurrentReceiver
from rosclaw_soccer.sim.contracts import hash_bytes
from rosclaw_soccer.training.receiving_rollout import receiving_window


def extended(tmp_path, frames):
    path = tmp_path / "receiver.npz"
    return G1RecurrentReceiver(
        path,
        agent_id="blue.playmaker",
        expected_actor_hash=hash_bytes(path.read_bytes()),
        foundation_hash=POLICY,
        foundation_config_hash=CONFIG,
        episode_frames=frames,
    )


def test_extended_lifetime_preserves_first_100_actions_and_has_no_implicit_rearm(tmp_path):
    original = motor(tmp_path, nonzero=True)
    longer = extended(tmp_path, 125)
    assert original.contract_hash == extended(tmp_path, 100).contract_hash
    assert original.contract_hash != longer.contract_hash
    a, b = AdmittedRecurrentReceiver(original), AdmittedRecurrentReceiver(longer)
    for frame in range(415, 515):
        assert a.propose(incoming(frame)) == b.propose(incoming(frame))
        np.testing.assert_array_equal(original.last_observation, longer.last_observation)
    assert a.propose(incoming(515)) is None
    for frame in range(515, 540):
        assert b.propose(incoming(frame)) is not None
        assert not b.readiness(frame=frame, time_sec=frame * 0.02).ball_action_ready
    assert b.propose(incoming(540)) is None
    assert b.completed and b.start_frame == 415
    for invalid in (99, 251, True, 125.0):
        with pytest.raises(ValueError):
            extended(tmp_path, invalid)


def test_longer_exam_needs_actual_additional_evidence_not_a_relabelled_short_window():
    value = trace()
    contact(value, 99)
    kwargs = dict(agent_ids=("blue.other", "red.receiver"), agent_id="red.receiver", start=1)
    assert not receiving_window(value, frames=100, **kwargs)[1]["controlled_reception"]
    assert not receiving_window(value, frames=100, required_frames=125, **kwargs)[1][
        "controlled_reception"
    ]
    with pytest.raises(ValueError):
        receiving_window(value, frames=125, required_frames=125, **kwargs)
    longer = {k: np.concatenate([v, np.repeat(v[-1:], 25, axis=0)]) for k, v in value.items()}
    longer["time"] = np.arange(1, 127) * 0.02
    _, outcome = receiving_window(longer, frames=125, required_frames=125, **kwargs)
    assert outcome["controlled_reception"]
    assert outcome["schema"] == "soccer.receiving_window.duration.v2"
    assert outcome["required_frames"] == 125
    assert not outcome["promotion_eligible"]
    longer["ball_velocity"][-1, 0] = 1
    assert not receiving_window(longer, frames=125, required_frames=125, **kwargs)[1][
        "controlled_reception"
    ]
