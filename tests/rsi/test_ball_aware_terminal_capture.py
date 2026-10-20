"""Synthetic reset lifecycle checks. No fake fixture is physical evidence."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")


def module(monkeypatch):
    class Base:
        def _reset_idx(self, ids=None):
            if ids is None:
                ids = torch.arange(self.num_envs)
            for value in (
                self.sim.data.qpos,
                self.sim.data.qvel,
                self.sim.data.actuator_force,
                self.force,
                self.target,
                self.reference_end,
            ):
                value[ids] = 0

        def step(self, action):
            self.sim.data.qpos += 2
            self.sim.data.qvel += 3
            self.sim.data.actuator_force += 4
            self.force += 15
            self.target += 5
            self.reference_end[:] = torch.tensor([True, False])
            self._reset_idx(torch.tensor([0]))
            return "synthetic_return"

    base = ModuleType("scripts.rsi_mjlab_motion_backend")
    base.ExplicitBoundaryMotionEnv = Base
    factory = ModuleType("scripts.rsi_mjlab_velocity_prior_backend")
    factory.make_velocity_prior_training_cfg = lambda **kw: None
    monkeypatch.setitem(sys.modules, base.__name__, base)
    monkeypatch.setitem(sys.modules, factory.__name__, factory)
    path = Path(__file__).resolve().parents[2] / "scripts/rsi_mjlab_ball_aware_backend.py"
    spec = importlib.util.spec_from_file_location("synthetic_terminal_backend_test", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    monkeypatch.setattr(value, "actual_foot_force_history", lambda env: env.force)
    return value


def test_terminal_state_and_contact_are_copied_before_reset(monkeypatch):
    m = module(monkeypatch)
    env = m.BallAwareMotionEnv()
    env.num_envs = 2
    env.device = "cpu"
    env.sim = NS(
        data=NS(qpos=torch.zeros(2, 43), qvel=torch.zeros(2, 41), actuator_force=torch.zeros(2, 29))
    )
    env.force = torch.zeros(2, 10, 2)
    env.target = torch.zeros(2, 29)
    env.reference_end = torch.zeros(2, dtype=torch.bool)
    env.action_manager = NS(get_term=lambda name: NS(_processed_actions=env.target))
    env.command_manager = NS(
        get_term=lambda name: NS(robot=NS(data=NS(joint_pos_target=env.target)))
    )
    env.termination_manager = NS(get_term=lambda name: env.reference_end)
    assert env.step(None) == "synthetic_return"
    assert env.sim.data.qpos[0].sum() == 0
    result = env.physical_step_data()
    assert result["terminal_capture_mask"].tolist() == [True, False]
    assert torch.all(result["qpos_after"] == 2)
    assert torch.all(result["qvel_after"] == 3)
    assert torch.all(result["foot_force_history"] == 15)
    assert result["reference_exhausted"].tolist() == [True, False]
    assert not env._capture_terminal
    result["qpos_after"].zero_()
    assert env.terminal_capture["qpos_after"].sum() > 0
    env.step(None)
    assert env.terminal_capture["qpos_after"].sum() == 2 * 43
    assert env.physical_step_data()["qpos_after"][1].sum() == 4 * 43
