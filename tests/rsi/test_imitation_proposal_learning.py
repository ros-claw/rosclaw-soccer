"""Synthetic data identities, not authentication of successful teachers."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.imitation_proposal_learning import prepare_imitation_batch
from rosclaw_soccer.rsi.proposal_memory_motor import initial_model
from tests.rsi.test_correlated_gradient_numerical_parity import conditional_fixture_batch
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


@pytest.fixture
def source_batch(request):
    parent = initial_model(request.getfixturevalue("current")[0], maximum_mean_kl=0.05)
    arrays = conditional_fixture_batch(request.getfixturevalue("smooth_parent"))
    return parent, arrays


def prepare(parent, arrays, selected=None):
    return prepare_imitation_batch(
        parent,
        arrays,
        selected_teacher_groups=[0, 2, 4, 6] if selected is None else selected,
        behavior_model_hash=parent["model_hash"],
        teacher_selection_hash="sha256:" + "a" * 64,
    )


def test_ordered_source_density_and_whole_episode_training_weights(source_batch):
    parent, arrays = source_batch
    before = copy.deepcopy(arrays)
    result = prepare(parent, arrays)
    assert result["context"].shape == (2160, 135)
    assert result["baseline"].shape == result["targets"].shape == (2160, 12)
    assert int((result["training_weights"] > 0).sum()) == 1080
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], before[key])
    corrupted = copy.deepcopy(arrays)
    corrupted["old_log_probability"][0] += 0.01
    with pytest.raises(ValueError, match="density"):
        prepare(parent, corrupted)


@pytest.mark.parametrize(
    "fault", ["duplicate", "order", "partial_return", "scale", "authority_identity"]
)
def test_bad_selection_or_source_rejected(source_batch, fault):
    parent, arrays = source_batch
    selected = [0, 2, 4, 6]
    if fault == "duplicate":
        selected = [0, 0, 2, 4]
    elif fault == "order":
        arrays["trajectory_index"][0] = 1
    elif fault == "partial_return":
        arrays["terminal_return"][0] += 1
    elif fault == "scale":
        arrays["std_raw"][0] = 0.15
    else:
        with pytest.raises(ValueError, match="immediate"):
            prepare_imitation_batch(
                parent,
                arrays,
                selected_teacher_groups=selected,
                behavior_model_hash="sha256:" + "b" * 64,
                teacher_selection_hash="sha256:" + "a" * 64,
            )
        return
    with pytest.raises(ValueError):
        prepare(parent, arrays, selected)
