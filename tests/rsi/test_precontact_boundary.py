from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("mjlab")

from scripts.rsi_mjlab_motion_backend import BodyBoundBalancedMotionCommand  # noqa: E402
from scripts.rsi_mjlab_precontact_backend import (  # noqa: E402
    PrecontactMotionCommand,
    curriculum_window_exhausted,
)


def test_in_step_reset_does_not_skip_its_restored_reference_frame(monkeypatch):
    obj = PrecontactMotionCommand.__new__(PrecontactMotionCommand)
    restored = []
    obj._env = SimpleNamespace(
        _course_just_reset=torch.tensor([True, False]),
        _restore_initial_solver_seed=lambda ids: restored.append(ids.tolist()),
    )
    obj.time_steps = torch.tensor([100, 50])
    obj._compute_relative_transforms = lambda: None

    def advance(self):
        self.time_steps += 1

    monkeypatch.setattr(BodyBoundBalancedMotionCommand, "_update_command", advance)
    obj._update_command()
    assert obj.time_steps.tolist() == [100, 51]
    assert obj._env._course_just_reset.tolist() == [False, False]
    assert restored == [[0]]
    obj._update_command()
    assert obj.time_steps.tolist() == [101, 52]
    assert restored == [[0]]


def test_window_termination_is_elapsed_not_absolute_source_phase():
    env = SimpleNamespace(
        episode_length_buf=torch.tensor([200, 196, 220]),
        _course_start_frame=torch.tensor([200, 100, 125]),
    )
    assert curriculum_window_exhausted(env).tolist() == [False, True, False]


def test_explicit_environment_joint_cartesian_indices_do_not_touch_other_worlds():
    target = torch.zeros(8, 29)
    worlds = torch.tensor([1, 4, 7])
    joints = torch.arange(29)
    measured_previous_target = torch.arange(87, dtype=torch.float32).reshape(3, 29)
    target[worlds[:, None], joints] = measured_previous_target
    assert torch.equal(target[worlds], measured_previous_target)
    assert torch.count_nonzero(target[torch.tensor([0, 2, 3, 5, 6])]) == 0
