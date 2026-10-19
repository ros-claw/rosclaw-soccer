"""Synthetic current-state forks; not qualified G1 feedback prediction."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.feedback_branch_control import fork_ik, fork_motor
from rosclaw_soccer.rsi.fixed_recurrent_reference_audit import _numeric_graph_hash
from rosclaw_soccer.rsi.foot_heading_evidence import FootHeadingCpuReconstruction
from rosclaw_soccer.rsi.recurrent_sampling_motor import (
    CompiledRecurrentSamplingMotor,
    make_preview,
    make_sampling_view,
)
from rosclaw_soccer.rsi.recurrent_success_motor import make_model
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_foot_heading_evidence import fixture as ik_fixture
from tests.rsi.test_imitation_proposal_motor import imitation_parent  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_recurrent_sampling_motor import long_body
from tests.rsi.test_recurrent_success_motor import boundary
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_original_ik_pending_snapshot_owns_side_contact_release():
    world, data, _, _, probe = ik_fixture()
    original = FootHeadingCpuReconstruction(world, probe)
    for frame in range(31):
        original.before_step(data, frame, data.xpos, np.zeros(29))
        if frame < 30:
            original.after_step(frame, np.zeros(6))
    branch = fork_ik(original, frame=30)
    assert branch.model is original.model
    assert branch.pending[1] is not original.pending[1]
    forces = np.array([10.0, 0, 0, 0, 0, 0])
    branch.after_step(30, forces)
    expected = branch.before_step(data, 31, data.xpos, np.zeros(29))
    assert original.contact is None and original.next_frame == 30
    original.after_step(30, forces)
    np.testing.assert_array_equal(expected, original.before_step(data, 31, data.xpos, np.zeros(29)))
    branch.pending[1][:] = 9
    assert not np.any(original.pending[1] == 9)


def test_recurrent_current_state_deeply_owned_and_same_draw(imitation_parent):  # noqa: F811
    policy = make_preview(make_sampling_view(make_model(imitation_parent), seed=901))
    original = CompiledRecurrentSamplingMotor(policy)
    observed, previous = long_body(), np.zeros(12)
    for frame in range(31):
        previous = original.delta_at_frame(policy, observed, **boundary(frame, previous))
    before = _numeric_graph_hash(original)
    branch = fork_motor(original, next_frame=31)
    assert branch._recurrent is not original._recurrent
    assert branch._memory is not original._memory
    assert branch._parent is not original._parent
    predicted = branch.delta_at_frame(policy, observed, **boundary(31, previous))
    assert _numeric_graph_hash(original) == before
    actual = original.delta_at_frame(policy, observed, **boundary(31, previous))
    np.testing.assert_array_equal(predicted, actual)
    np.testing.assert_array_equal(branch.hidden_state, original.hidden_state)
    assert _numeric_graph_hash(branch) == _numeric_graph_hash(original)


@pytest.mark.parametrize("next_frame", [30, 32, True, 31.0])
def test_wrong_recurrent_boundary_rejected(imitation_parent, next_frame):  # noqa: F811
    policy = make_preview(make_sampling_view(make_model(imitation_parent), seed=901))
    decoder = CompiledRecurrentSamplingMotor(policy)
    for frame in range(31):
        decoder.delta_at_frame(policy, long_body(), **boundary(frame, np.zeros(12)))
    with pytest.raises(ValueError):
        fork_motor(decoder, next_frame=next_frame)
