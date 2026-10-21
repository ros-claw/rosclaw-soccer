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
