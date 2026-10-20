"""Run exact adapter methods on CPU tensors; no MJLab/physics qualification."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def adapter_class():
    path = Path(__file__).parents[2] / "scripts/rsi_mjlab_motion_backend.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ExplicitBoundaryMotionCommand"
    )
    cls.bases = [ast.Name(id="FakeBase", ctx=ast.Load())]

    class FakeBase:
        def _resample_command(self, ids):
            self.resamples += 1
            self._env.action_manager._terms["joint_pos"]._offset[ids] = 5.0

        def _compute_soccer_ball_positions(self, ids):
            self.soccer_ball_pos[ids] = 0.1

    ns = {"FakeBase": FakeBase, "torch": torch}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"
        ),
        ns,
    )
    return ns["ExplicitBoundaryMotionCommand"]


def test_reference_exhaustion_does_not_teleport_or_resample_inside_step():
    c = adapter_class()()
    c.resamples = 0
    c.time_steps = torch.tensor([3, 4])
    c.motion_length = torch.tensor([5, 5])
    called = []
    c.kick_contact_tracker = SimpleNamespace(
        begin_step=lambda: called.append("begin"),
        get_contact_awarded=lambda: torch.tensor([False, True]),
    )
    c._update_target_points_from_sim = lambda: called.append("observed_ball")
    c._compute_relative_transforms = lambda: called.append("relative")
    c.target_point_pos = torch.ones((2, 3))
    c.initial_target_point_pos = torch.zeros((2, 3))
    c.cfg = SimpleNamespace(sampling_mode="adaptive", adaptive_alpha=0.4)
    c.bin_failed_count = torch.ones(2)
    c._current_bin_failed = torch.zeros(2)
    for _ in range(60):
        c._update_command()
    assert c.resamples == 0
    assert c.time_steps.tolist() == [4, 4]
    assert len(called) == 180
    assert c.initial_target_point_pos.tolist() == [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]


def test_explicit_reset_preserves_fixed_canonical_action_offset():
    c = adapter_class()()
    c.resamples = 0
    offset = torch.full((3, 29), 9.0)
    defaults = torch.arange(29).repeat(3, 1).to(torch.float32)
    c._env = SimpleNamespace(
        action_manager=SimpleNamespace(_terms={"joint_pos": SimpleNamespace(_offset=offset)})
    )
    c.robot = SimpleNamespace(data=SimpleNamespace(default_joint_pos=defaults))
    c._resample_command(torch.tensor([0, 2]))
    assert c.resamples == 1
    torch.testing.assert_close(offset[0], defaults[0])
    torch.testing.assert_close(offset[2], defaults[2])
    torch.testing.assert_close(offset[1], torch.full((29,), 9.0))


def test_new_ball_radius_is_reflected_in_reset_height_only_for_selected_worlds():
    c = adapter_class()()
    c.soccer_ball_pos = torch.zeros((3, 3))
    c._compute_soccer_ball_positions(torch.tensor([1, 2]))
    torch.testing.assert_close(c.soccer_ball_pos[:, 2], torch.tensor([0.0, 0.11, 0.11]))
