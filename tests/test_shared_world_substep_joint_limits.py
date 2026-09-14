import ast
import dataclasses
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.skills.team import shared_world


def test_result_version_distinguishes_native_sampling_without_inventing_evidence():
    fields = {field.name: field for field in dataclasses.fields(shared_world.G1SharedWorldResult)}
    assert fields["schema_version"].default == "rosclaw_soccer.g1_shared_world_result.v27"
    assert fields["joint_limit_sampling_period_sec"].default is None
    assert fields["joint_limit_samples_per_role"].default is None


def fixture():
    model = SimpleNamespace(
        jnt_range=np.array([[-1.0, 1.0], [-0.52, 0.52], [0.0, 0.0]]),
        jnt_limited=np.array([True, True, False]),
    )
    data = SimpleNamespace(qpos=np.array([0.0, 0.0, 42.0]))
    robots = (
        SimpleNamespace(role="passer", joint_ids=np.array([0]), joint_qpos=np.array([0])),
        SimpleNamespace(role="shooter", joint_ids=np.array([1, 2]), joint_qpos=np.array([1, 2])),
    )
    return dict(
        model=model, data=data, robots=robots, violations={"passer": False, "shooter": False}
    )


def test_substep_excursion_remains_failed_after_policy_endpoint_recovers():
    args = fixture()
    for substep in range(10):
        args["data"].qpos[1] = 0.52136 if substep == 7 else 0.50
        shared_world._latch_joint_limit_violations(**args)
    assert args["data"].qpos[1] < 0.52
    assert args["violations"] == {"passer": False, "shooter": True}


def test_does_not_mutate_simulator_arrays():
    args = fixture()
    before = args["data"].qpos.copy()
    ranges = args["model"].jnt_range.copy()
    assert not shared_world._latch_joint_limit_violations(**args)
    np.testing.assert_array_equal(args["data"].qpos, before)
    np.testing.assert_array_equal(args["model"].jnt_range, ranges)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_original_tolerance_is_preserved(sign):
    args = fixture()
    args["data"].qpos[0] = sign * (1.0 + 0.5e-5)
    assert not shared_world._latch_joint_limit_violations(**args)
    args["data"].qpos[0] = sign * (1.0 + 2e-5)
    assert shared_world._latch_joint_limit_violations(**args)
    assert args["violations"] == {"passer": True, "shooter": False}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("index", [0, 2])
def test_nonfinite_is_fail_closed_even_for_unlimited_joint(value, index):
    args = fixture()
    args["data"].qpos[index] = value
    assert shared_world._latch_joint_limit_violations(**args)


def test_empty_roles_preserve_an_existing_failure():
    args = fixture()
    args["robots"] = []
    args["violations"]["passer"] = True
    assert shared_world._latch_joint_limit_violations(**args)


def test_production_call_is_in_native_substep_loop_after_step_before_contacts():
    tree = ast.parse(inspect.getsource(shared_world._simulate_shared_world))
    loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "substep_index"
    )
    calls = [node for node in ast.walk(loop) if isinstance(node, ast.Call)]
    checks = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "_latch_joint_limit_violations"
    ]
    steps = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "mj_step"
    ]
    contacts = [
        node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "_contacts"
    ]
    assert len(checks) == len(steps) == 1
    assert steps[0].lineno < checks[0].lineno < min(node.lineno for node in contacts)
