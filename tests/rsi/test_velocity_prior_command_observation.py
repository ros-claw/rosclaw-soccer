"""Synthetic tensor plumbing, not a physical motion qualification."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES  # noqa: E402


def backend(monkeypatch):
    # Isolate external observation functions; this test allocates no simulator.
    obs = ModuleType("src.tasks.soccer.mdp.shooter_obs")
    obs.motion_anchor_ori_b = lambda env, name: torch.zeros((1, 6))
    obs.motion_anchor_pos_b = lambda env, name: torch.zeros((1, 3))
    factory = ModuleType("scripts.rsi_mjlab_motion_backend")
    factory.make_motion_training_cfg = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, obs.__name__, obs)
    monkeypatch.setitem(sys.modules, factory.__name__, factory)
    path = Path(__file__).resolve().parents[2] / "scripts/rsi_mjlab_velocity_prior_backend.py"
    spec = importlib.util.spec_from_file_location("synthetic_velocity_backend_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment():
    robot = NS(
        joint_names=G1_DDS_JOINT_NAMES,
        data=NS(
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            root_link_pos_w=torch.zeros((1, 3)),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
            joint_pos=torch.zeros((1, 29)),
            joint_vel=torch.zeros((1, 29)),
            indexing=NS(free_joint_v_adr=torch.arange(6)),
        ),
    )
    motion = NS(
        robot=robot,
        cfg=NS(body_names=("torso_link", "pelvis")),
        body_pos_w=torch.tensor([[[0.7, 0.4, 0.0], [0.2, 0.1, 0.0]]]),
        body_lin_vel_w=torch.tensor([[[0.6, 0.1, 0.0], [0.2, 0.2, 0.0]]]),
        body_ang_vel_w=torch.tensor([[[0.0, 0.0, 3.0], [0.0, 0.0, 0.25]]]),
        anchor_pos_w=torch.tensor([[0.7, 0.4, 0.0]]),
        anchor_lin_vel_w=torch.tensor([[0.6, 0.1, 0.0]]),
        anchor_ang_vel_w=torch.tensor([[0.0, 0.0, 3.0]]),
        robot_anchor_pos_w=torch.tensor([[0.1, 0.2, 0.0]]),
        robot_anchor_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        joint_pos=torch.zeros((1, 29)),
        joint_vel=torch.zeros((1, 29)),
    )
    env = NS(
        command_manager=NS(get_term=lambda name: motion),
        action_manager=NS(
            get_term=lambda name: NS(target_names=G1_DDS_JOINT_NAMES),
            action=torch.zeros((1, 29)),
        ),
        sim=NS(data=NS(qvel=torch.tensor([[0.0, 0.0, 0.0, 4.0, 5.0, 6.0]]))),
        num_envs=1,
        device="cpu",
        episode_length_buf=torch.zeros(1, dtype=torch.long),
    )
    return env


@pytest.mark.parametrize(
    "point,mode,expected",
    [
        ("pelvis", "reference_position_velocity", [0.4, 0.3, 0.25]),
        ("pelvis", "reference_velocity", [0.2, 0.2, 0.25]),
        ("torso", "reference_position_velocity", [1.0, 0.3, 1.0]),
        ("torso", "reference_velocity", [0.6, 0.1, 1.0]),
    ],
)
def test_actual_backend_selects_anatomical_point_not_just_rotation(
    monkeypatch, point, mode, expected
):
    module = backend(monkeypatch)
    env = environment()
    result = module.velocity_reference_observation(
        env, defaults=[0.0] * 29, command_mode=mode, command_point=point
    )
    assert result.shape == (1, 168)
    torch.testing.assert_close(result[0, 6:9], torch.tensor(expected))
    torch.testing.assert_close(result[0, :3], torch.tensor([4.0, 5.0, 6.0]))
    # The reference70 still refers to the torso even with pelvis commands.
    torch.testing.assert_close(result[0, -3:], torch.tensor([0.6, 0.1, 0.0]))
