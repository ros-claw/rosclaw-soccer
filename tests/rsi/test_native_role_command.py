import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.native_role_command import (
    ROLE_COMMAND_SCHEMA,
    NativeRoleCommandPolicy,
    role_command_observation,
)
from rosclaw_soccer.rsi.native_team_intent import ROLES


def model_file(tmp_path, **changes):
    value = dict(
        schema=ROLE_COMMAND_SCHEMA,
        binding_hash="test-binding",
        observation_dim=118,
        action_semantics="ABSOLUTE_BODY_VELOCITY_COMMAND_BEFORE_CLEARANCE",
        hardware_authorized=False,
        optimizer_updates=512,
        training_kind="PUBLIC_COURSE_COACHED_BEHAVIOR_CLONING",
        tensors={"mean": np.zeros(118).tolist(), "std": np.ones(118).tolist()},
    )
    for i, shape in enumerate(((128, 118), (128, 128), (3, 128))):
        value["tensors"][f"w{i}"] = np.zeros(shape).tolist()
        value["tensors"][f"b{i}"] = np.zeros(shape[0]).tolist()
    value.update(changes)
    payload = json.dumps(value).encode()
    path = tmp_path / "policy.json"
    path.write_bytes(payload)
    return path, "sha256:" + hashlib.sha256(payload).hexdigest()


def test_portable_policy_is_bounded_owned_and_not_an_executor(tmp_path):
    path, digest = model_file(tmp_path)
    policy = NativeRoleCommandPolicy(path, expected_hash=digest, expected_binding="test-binding")
    obs = np.zeros(118, np.float32)
    command = policy.propose(obs)
    np.testing.assert_array_equal(command, [0.25, 0, 0])
    command[:] = 100
    np.testing.assert_array_equal(policy.propose(obs), [0.25, 0, 0])
    assert not policy.arrays["w0"].flags.writeable
    with pytest.raises(ValueError):
        policy.propose(np.full(118, np.nan))


@pytest.mark.parametrize(
    "change",
    [
        dict(hardware_authorized=True),
        dict(observation_dim=98),
        dict(action_semantics="TORQUE"),
        dict(optimizer_updates=0),
    ],
)
def test_model_semantics_cannot_be_relabelled(tmp_path, change):
    path, digest = model_file(tmp_path, **change)
    with pytest.raises(ValueError):
        NativeRoleCommandPolicy(path, expected_hash=digest, expected_binding="test-binding")


def test_model_binding_and_bytes_are_both_checked(tmp_path):
    path, digest = model_file(tmp_path)
    with pytest.raises(ValueError):
        NativeRoleCommandPolicy(path, expected_hash=digest, expected_binding="another-body")
    with pytest.raises(ValueError):
        NativeRoleCommandPolicy(
            path, expected_hash="sha256:" + "0" * 64, expected_binding="test-binding"
        )


def test_dagger_continuation_requires_the_explicit_update_kind_pair(tmp_path):
    path, digest = model_file(
        tmp_path, optimizer_updates=1024, training_kind="PUBLIC_COURSE_DAGGER_CONTINUATION"
    )
    policy = NativeRoleCommandPolicy(path, expected_hash=digest, expected_binding="test-binding")
    assert policy.metadata["optimizer_updates"] == 1024
    path, digest = model_file(tmp_path, optimizer_updates=1024)
    with pytest.raises(ValueError):
        NativeRoleCommandPolicy(path, expected_hash=digest, expected_binding="test-binding")


def test_observation_uses_previous_command_but_never_current_control_label():
    q, v = np.zeros(295), np.zeros(286)
    layouts, body_names = {}, {}
    for i, (team, role) in enumerate((t, r) for t in ("red", "blue") for r in ROLES):
        prefix = team + "_" + role + "/"
        rq, rv = 7 + i * 36, 6 + i * 35
        q[rq : rq + 3] = [i, 0, 0.8]
        layouts[prefix] = SimpleNamespace(
            rq=rq,
            rv=rv,
            bq=0,
            bv=0,
            pelvis=i + 1,
            qi=np.arange(rq + 7, rq + 36),
            vi=np.arange(rv + 6, rv + 35),
        )
        for j, side in enumerate(("left", "right")):
            body_names[prefix + side + "_ankle_roll_link"] = 16 + 2 * i + j
    model = SimpleNamespace(body=lambda name: SimpleNamespace(id=body_names[name]))
    for lo in layouts.values():
        lo.model = model
    metric = SimpleNamespace(xpos=np.zeros((32, 3)), xmat=np.tile(np.eye(3).reshape(1, 9), (32, 1)))
    data = SimpleNamespace(qpos=q, qvel=v, ctrl=np.zeros(232))
    previous = np.array([0.1, 0.2, 0.3])

    def observe():
        return role_command_observation(
            data, metric, layouts, team="red", role="finisher", previous_body_command=previous
        )

    first = observe()
    assert first.shape == (118,)
    np.testing.assert_allclose(first[87:90], previous)
    data.ctrl[:] = 100  # Current label/control is deliberately not an input.
    np.testing.assert_array_equal(first, observe())
    first[:] = 100
    assert not np.array_equal(first, observe())
