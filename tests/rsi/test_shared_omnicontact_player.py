from types import SimpleNamespace

import numpy as np
import pytest
from test_shared_robonaldo_player import setup

from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer


class Session:
    def get_inputs(self):
        return [
            SimpleNamespace(name="obs", shape=[1, 1244]),
            SimpleNamespace(name="time_step", shape=[1, 1]),
        ]

    def get_outputs(self):
        return [SimpleNamespace(name="actions", shape=[1, 29])]

    def get_providers(self):
        return ["CPUExecutionProvider"]


class Policy:
    def __init__(self):
        self.task, self.reference_source = "kickball", "CFgen"
        self.counter_step, self.async_stage_thread = 0, None
        self.num_obs, self.single_obs_dim, self.history_len = 1244, 141, 5
        self.future_frames = [0, 1, 2, 3, 4, 8, 12, 16, 24, 32, 50]
        self.lab2mj = self.mj2lab = np.arange(29)
        self.kps_lab, self.kds_lab = np.full(29, 20.0), np.ones(29)
        self.state_cmd, self.policy_output = SimpleNamespace(), SimpleNamespace()
        self.ort_session = Session()
        self.observed = []

    def enter(self):
        self.entry_ball = self.state_cmd.obj_pos.copy()

    def run(self):
        self.observed.append(self.state_cmd.q.copy())
        self.policy_output.actions = self.state_cmd.q + 0.001
        self.policy_output.kps, self.policy_output.kds = self.kps_lab, self.kds_lab
        self.counter_step += 1


def player(world, prefix="a/", policy=None):
    return SharedOmniContactPlayer(
        world,
        prefix,
        policy or Policy(),
        expected_policy_type=Policy,
        goal=np.array([5.0, 0.0, 0.11]),
    )


def test_private_taskspace_players_share_physics_but_not_history():
    w = setup()
    a, b = player(w), player(w, "b/")
    assert a.policy.state_cmd is not b.policy.state_cmd
    assert a.policy.observed is not b.policy.observed
    old = w.data.qpos.copy()
    ta, ra = a.propose()
    tb, rb = b.propose()
    np.testing.assert_array_equal(w.data.qpos, old)
    np.testing.assert_array_equal(ra["canonical_qpos"][36:], rb["canonical_qpos"][36:])
    assert w.native_steps == 0
    w.advance({"a/": ta, "b/": tb})
    a.commit()
    b.commit()
    assert a.commits == b.commits == 1 and w.native_steps == 10
    a.propose()
    with pytest.raises(ValueError, match="one fresh"):
        a.propose()


@pytest.mark.parametrize("defect", ["type", "codec", "gain", "async", "permutation"])
def test_unqualified_taskspace_policy_is_rejected_before_physics(defect):
    w, p = setup(), Policy()
    if defect == "type":
        with pytest.raises(ValueError, match="pinned"):
            SharedOmniContactPlayer(
                w, "a/", p, expected_policy_type=SimpleNamespace, goal=np.array([5.0, 0.0, 0.11])
            )
        return
    if defect == "codec":
        p.num_obs = 547
    elif defect == "gain":
        p.kps_lab[0] += 1
    elif defect == "async":
        p.async_stage_thread = object()
    else:
        p.lab2mj = np.zeros(29, dtype=int)
    with pytest.raises(ValueError):
        player(w, policy=p)
    assert w.native_steps == 0 and not w._private_policy_owners


def test_taskspace_ownership_and_target_commitment_are_enforced():
    w = setup()
    p = Policy()
    a = player(w, policy=p)
    with pytest.raises(ValueError, match="private"):
        player(w, "b/", p)
    target, _ = a.propose()
    with pytest.raises(ValueError, match="physical advance"):
        a.commit()
    w.advance({"a/": target + 0.001, "b/": np.zeros(29)})
    with pytest.raises(ValueError, match="executed target differs"):
        a.commit()
    assert a.failed


def test_npz_requires_explicit_admission_and_stays_bound_to_its_source():
    w, p = setup(), Policy()
    p.reference_source = "NPZmotion"
    with pytest.raises(ValueError, match="pinned"):
        player(w, policy=p)
    assert not w._private_policy_owners
    a = SharedOmniContactPlayer(
        w,
        "a/",
        p,
        expected_policy_type=Policy,
        goal=np.array([5.0, 0.0, 0.11]),
        reference_source="NPZmotion",
    )
    assert w.native_steps == 0
    p.reference_source = "CFgen"
    with pytest.raises(ValueError, match="reference source changed"):
        a.propose()
    assert a.failed and not p.observed and w.native_steps == 0
