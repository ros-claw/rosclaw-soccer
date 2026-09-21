from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.locomotion_replica import FrozenLocomotionReplica, _vector

HASH = "sha256:" + "a" * 64


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), -(2**63)])
def test_bad_vector_values(bad):
    with pytest.raises(ValueError):
        _vector([bad] * 3, 3, 100.0)


def test_vector_is_owned():
    source = np.zeros(3, dtype=np.float32)
    value = _vector(source, 3, 100.0)
    value[0] = 1
    assert source[0] == 0 and value.dtype == source.dtype


@pytest.mark.parametrize("changes", [{"policy_hash": "unknown"}, {"correct_mirrored_yaw": 1}])
def test_invalid_binding_before_load(changes):
    args = dict(
        policy_hash=HASH,
        configuration_hash=HASH,
        adapter_hash=HASH,
        synchronize_action_frame=False,
        correct_mirrored_yaw=True,
    )
    args.update(changes)
    with pytest.raises(ValueError):
        FrozenLocomotionReplica(Path("must-not-load-assets"), **args)


def test_content_mismatch_rejected(tmp_path):
    path = tmp_path / "artifact"
    path.write_bytes(b"wrong content")
    with pytest.raises(ValueError, match="differs"):
        FrozenLocomotionReplica._verify_files({path: HASH})


def stub_replica():
    torch = pytest.importorskip("torch")
    replica = object.__new__(FrozenLocomotionReplica)
    replica.policy_hash = HASH
    replica._restored = True
    replica._correct_yaw = True
    replica._synchronize = False
    policy = SimpleNamespace(
        range_velx=np.array([-1.0, 1.0]),
        range_vely=np.array([-1.0, 1.0]),
        range_velz=np.array([-1.0, 1.0]),
        obs=np.zeros(96),
        action=np.zeros(29),
        policy=SimpleNamespace(
            hidden_state=torch.zeros(1, 1, 256), cell_state=torch.ones(1, 1, 256)
        ),
    )
    replica._controller = SimpleNamespace(
        policy=policy, state=SimpleNamespace(), output=SimpleNamespace(actions=np.zeros(29))
    )
    return replica


def qpos():
    q = np.zeros(43)
    q[3] = q[39] = 1
    return q


def test_outputs_do_not_alias_private_state(monkeypatch):
    import rosclaw_soccer.skills.team.independent_team_world as world

    replica = stub_replica()
    monkeypatch.setattr(world, "_run_locomotion", lambda *a, **k: None)
    target = replica.step(qpos(), np.zeros(41), np.zeros(3))
    target[:] = 1
    replica.observation[:] = 2
    replica.raw_action[:] = 3
    assert not replica._controller.output.actions.any()
    assert not replica.observation.any() and not replica.raw_action.any()


def test_failed_inference_requires_restore(monkeypatch):
    import rosclaw_soccer.skills.team.independent_team_world as world

    replica = stub_replica()
    memory = replica.memory

    def fail(*args, **kwargs):
        raise RuntimeError("private inference failed")

    monkeypatch.setattr(world, "_run_locomotion", fail)
    with pytest.raises(RuntimeError):
        replica.step(qpos(), np.zeros(41), np.zeros(3))
    with pytest.raises(ValueError, match="restore"):
        replica.step(qpos(), np.zeros(41), np.zeros(3))
    replica.restore(memory, np.zeros(29), reflected=False)
    assert replica._restored


def test_bad_pose_rejected_before_private_inference():
    replica = stub_replica()
    with pytest.raises(ValueError, match="quaternion"):
        replica.step(np.zeros(43), np.zeros(41), np.zeros(3))
    assert replica._restored
