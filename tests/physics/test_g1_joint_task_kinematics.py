import mujoco
import pytest

from rosclaw_soccer.physics.g1_joint_task_kinematics import G1JointTaskKinematicSampler


def test_non_g1_model_rejected_before_hull_or_sampler_allocation():
    with pytest.raises(ValueError, match="canonical"):
        G1JointTaskKinematicSampler(
            mujoco.MjModel.from_xml_string("<mujoco/>"), expected_world_hash="sha256:" + "a" * 64
        )


def test_fake_topology_is_not_a_native_model():
    class Fake:
        nq, nv, nu = 43, 41, 29

    with pytest.raises(ValueError, match="canonical"):
        G1JointTaskKinematicSampler(Fake(), expected_world_hash="sha256:" + "a" * 64)
