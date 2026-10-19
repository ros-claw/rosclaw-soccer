"""Toy body/NN parity with SYNTHETIC contact/Jacobian, not G1 contact proof."""

import numpy as np
import pytest

from rosclaw_soccer.rsi import born_postcontact_release as born
from rosclaw_soccer.rsi.full_task_branch import FullTaskBranch, integration_hash
from tests.rsi.test_full_task_branch import (  # noqa: F401
    build,
    candidate,
    current,
    imitation_parent,
    model,
    smooth_parent,
)
from tests.rsi.test_impact_torque_screen import sample


def test_nonzero_future_and_distinct_zero_match_actual_owned_control(monkeypatch, request):
    class Sampler:
        def __init__(self, compiled_model, **kwargs):
            self.model = compiled_model

        def sample(self, data):
            assert data.model is self.model
            return sample()

    monkeypatch.setattr(born, "G1JointTaskKinematicSampler", Sampler)
    parent, wh = build(
        monkeypatch, request.getfixturevalue("imitation_parent"), born_release_capture=True
    )
    prefix = []
    for f in range(31):
        prefix.append(parent.step(f))
    assert "born_release_requested_offset_nm" in prefix[0].arrays
    assert not np.any(prefix[0].arrays["born_release_requested_offset_nm"])
    with pytest.raises(ValueError, match="measured-foot"):
        parent.install_born_release(world_hash=wh, action_name="forward_minus")
    # Synthetic fixture explicitly supplies a completed foot event, not a
    # claim that the previous native foot-force trace contains this event.
    parent.ik.contact = 30
    parent.previous_forces[0] = 2.0
    donor = integration_hash(parent.model, parent.data)
    plain = FullTaskBranch(parent, world_hash=wh, action_name="baseline")
    zero = FullTaskBranch(parent, world_hash=wh, action_name="born_release:baseline")
    nonzero = FullTaskBranch(parent, world_hash=wh, action_name="born_release:forward_minus")
    expected = []
    for _ in range(269):
        a, b = plain.step(), zero.step()
        assert a.arrays.keys() == b.arrays.keys()
        for k in a.arrays:
            assert a.arrays[k].tobytes() == b.arrays[k].tobytes(), k
        expected.append(nonzero.step())
    for branch in (plain, zero, nonzero):
        r = branch.review()
        assert r["new_native_integrated_steps"] == 2690
        if r["born_postcontact_release"] is not None:
            assert r["born_postcontact_release"]["observed_command_frames"] == 269
            assert r["born_postcontact_release"]["commands_before_origin_observed"] == 0
    assert integration_hash(parent.model, parent.data) == donor and parent.release_control is None
    parent.install_born_release(world_hash=wh, action_name="forward_minus")
    for f, predicted in enumerate(expected, 31):
        actual = parent.step(f)
        assert actual.arrays.keys() == predicted.arrays.keys()
        for k in actual.arrays:
            assert actual.arrays[k].tobytes() == predicted.arrays[k].tobytes(), k
    review = parent.review()
    assert review["native_integrated_steps"] == 3000
    assert "born_postcontact_release" in review and "postcontact_release" not in review
    assert review["born_postcontact_release"]["observed_command_frames"] == 269
