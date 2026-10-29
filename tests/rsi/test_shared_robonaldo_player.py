from types import SimpleNamespace

import numpy as np
import pytest
from test_shared_native_motor_world import source_xml

from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters, SharedNativeMotorWorld
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer

ISAAC_TO_MUJOCO = np.arange(29)


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
        self.default_q_mj = np.zeros(29)
        self.action_scale_mj = np.ones(29)
        self.action_clip_lo_il = np.full(29, -1.0)
        self.action_clip_hi_il = np.full(29, 1.0)
        self.last_action_il = np.zeros(29, dtype=np.float32)

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


def test_shadow_commit_encodes_the_actual_executed_target_without_modifying_physics():
    w = setup()
    a = player(w)
    a.propose()
    target = np.full(29, 0.125)
    w.advance({"a/": target, "b/": np.zeros(29)})
    q = w.data.qpos.copy()
    receipt = a.commit_observed_target()
    assert not receipt["original_proposal_executed"]
    np.testing.assert_array_equal(a.policy.last_action_il, target.astype(np.float32))
    np.testing.assert_array_equal(w.data.qpos, q)
    a.propose()
    assert a.commits == 1


def test_shadow_commit_rejects_unrepresentable_actual_action_and_latches():
    w = setup()
    a = player(w)
    a.propose()
    w.advance({"a/": np.full(29, 1.5), "b/": np.zeros(29)})
    with pytest.raises(ValueError, match="outside kick action codec"):
        a.commit_observed_target()
    assert a.failed
    with pytest.raises(ValueError):
        a.propose()


def test_shadow_commit_respects_joint_permutation(monkeypatch):
    import sys

    permutation = np.arange(28, -1, -1)
    monkeypatch.setattr(sys.modules[__name__], "ISAAC_TO_MUJOCO", permutation)
    w = setup()
    a = player(w)
    a.propose()
    target = np.linspace(-0.5, 0.5, 29)
    w.advance({"a/": target, "b/": np.zeros(29)})
    result = a.commit_observed_target()
    np.testing.assert_array_equal(a.policy.last_action_il, target[permutation].astype(np.float32))
    assert result["maximum_decode_error_rad"] < 5e-7


def test_retirement_releases_only_unused_owner_without_body_or_pd_change():
    w = setup()
    a, b = player(w), player(w, "b/")
    before = (w.data.qpos.copy(), w.data.qvel.copy(), w.data.ctrl.copy(), w.data.time)
    params = dict(w.parameters)
    receipt = a.retire_unexecuted_entry()
    assert receipt["adapter_terminal"] and not receipt["hardware_authorized"]
    assert "a/" not in w._private_policy_owners
    assert "b/" in w._private_policy_owners and not b.retired
    assert w.control_steps == w.native_steps == 0
    for observed, expected in zip((w.data.qpos, w.data.qvel, w.data.ctrl), before[:3], strict=True):
        np.testing.assert_array_equal(observed, expected)
    assert w.data.time == before[3]
    assert all(w.parameters[k] is v for k, v in params.items())
    with pytest.raises(ValueError, match="one fresh proposal"):
        a.propose()
    with pytest.raises(ValueError, match="one shared physical"):
        a.commit()
    with pytest.raises(ValueError, match="unexecuted"):
        a.retire_unexecuted_entry()
    fresh = player(w)
    assert fresh.policy is not a.policy and not fresh.retired
    fresh.propose()


@pytest.mark.parametrize("defect", ["failed", "pending", "committed", "owner", "stale"])
def test_retirement_cannot_clear_failure_or_discard_an_executed_proposal(defect):
    w = setup()
    a = player(w)
    if defect == "failed":
        a.failed = True
    elif defect == "pending":
        a.propose()
    elif defect == "committed":
        target, _ = a.propose()
        w.advance({"a/": target, "b/": np.zeros(29)})
        a.commit()
    elif defect == "owner":
        w._private_policy_owners["a/"] = (1, 2, 3)
    else:
        w.advance({"a/": np.zeros(29), "b/": np.zeros(29)})
    owner = w._private_policy_owners["a/"]
    with pytest.raises(ValueError, match="unexecuted"):
        a.retire_unexecuted_entry()
    assert w._private_policy_owners["a/"] == owner and not a.retired


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
