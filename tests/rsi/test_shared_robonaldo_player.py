from types import SimpleNamespace

import numpy as np
import pytest
from test_shared_native_motor_world import source_xml

from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters, SharedNativeMotorWorld
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer


class Session:
    def get_inputs(self):
        return [SimpleNamespace(name="obs", shape=[1, 547])]

    def get_outputs(self):
        return [SimpleNamespace(name="actions", shape=[1, 29])]


class Policy:
    def __init__(self):
        self.runtime_mode = "sim"
        self.use_body_frame_ball = False
        self.time_step = 0
        self.WARMUP_STEPS = 10
        self.kps, self.kds = np.full(29, 20.0), np.ones(29)
        self.state_cmd, self.policy_output = SimpleNamespace(), SimpleNamespace()
        self.ort_session = Session()
        self.measured = []

    def enter(self):
        self.entry_position = self.state_cmd.pelvis_pos_w.copy()

    def run(self):
        self.measured.append(self.state_cmd.q.copy())
        self.policy_output.actions = self.state_cmd.q + 0.001
        self.policy_output.kps, self.policy_output.kds = self.kps.copy(), self.kds.copy()
        self.time_step += 1


def setup():
    import mujoco

    xml = source_xml().replace('name="link0"', 'name="torso_link"')
    m = mujoco.MjModel.from_xml_string(shared_native_scene(xml, prefixes=("a/", "b/")))
    d = mujoco.MjData(m)
    d.qpos[m.joint("b/root").qposadr[0]] = 3
    mujoco.mj_forward(m, d)
    param = NativePDParameters(np.full(29, 20.0), np.ones(29), np.full(29, 25.0))
    w = SharedNativeMotorWorld(m, d, players={"a/": param, "b/": param}, ball_body_name="ball")
    return w


def player(w, prefix="a/", policy=None):
    return SharedRoboNaldoPlayer(w, prefix, policy or Policy(), goal=np.array([5.0, 0.0, 0.11]))


def test_two_private_players_observe_and_commit_one_shared_tick():
    w = setup()
    a, b = player(w), player(w, "b/")
    assert a.policy.state_cmd is not b.policy.state_cmd
    assert a.policy.entry_position[0] == 0 and b.policy.entry_position[0] == 3
    before = w.data.qpos.copy()
    at, ar = a.propose()
    bt, br = b.propose()
    np.testing.assert_array_equal(before, w.data.qpos)
    assert w.native_steps == 0
    np.testing.assert_array_equal(ar["canonical_qpos"][36:], br["canonical_qpos"][36:])
    w.advance({"a/": at, "b/": bt})
    a.commit()
    b.commit()
    assert w.native_steps == 10 and a.commits == b.commits == 1
    a.propose()
    with pytest.raises(ValueError, match="one fresh proposal"):
        a.propose()


def test_reused_policy_state_and_player_are_rejected():
    w = setup()
    a = player(w)
    with pytest.raises(ValueError, match="one private"):
        player(w, "b/", a.policy)
    with pytest.raises(ValueError, match="one private"):
        player(w)


def test_commit_requires_actual_control_and_unmodified_target():
    w = setup()
    a = player(w)
    at, _ = a.propose()
    with pytest.raises(ValueError, match="one shared physical"):
        a.commit()
    w.advance({"a/": at + 0.0001, "b/": np.zeros(29)})
    with pytest.raises(ValueError, match="executed target differs"):
        a.commit()
    assert a.failed


@pytest.mark.parametrize("defect", ["nan", "gain", "tick"])
def test_invalid_policy_output_poisoned_without_physics(defect):
    w = setup()
    a = player(w)
    original = a.policy.run

    def bad_run():
        original()
        if defect == "nan":
            a.policy.policy_output.actions[0] = np.nan
        elif defect == "gain":
            a.policy.policy_output.kps[0] += 1
        else:
            a.policy.time_step += 1

    a.policy.run = bad_run
    with pytest.raises(ValueError, match="finite HG29"):
        a.propose()
    assert a.failed and w.native_steps == 0
    with pytest.raises(ValueError, match="one fresh proposal"):
        a.propose()


def test_foreign_goal_and_external_physics_mutation_rejected():
    w = setup()
    a = player(w)
    a.policy.target_pos_w[0] = 9
    with pytest.raises(ValueError, match="declared goal"):
        a.propose()
    w = setup()
    a = player(w)
    w.data.qpos[0] += 1
    with pytest.raises(ValueError, match="unchanged healthy"):
        a.propose()


@pytest.mark.parametrize("defect", ["real", "shape", "gain"])
def test_preflight_rejects_wrong_contract(defect):
    w, p = setup(), Policy()
    if defect == "real":
        p.runtime_mode = "real"
    elif defect == "shape":
        p.ort_session.get_inputs = lambda: [SimpleNamespace(name="obs", shape=[1, 160])]
    else:
        p.kps[0] += 1
    with pytest.raises(ValueError):
        player(w, policy=p)
    assert w.native_steps == 0
