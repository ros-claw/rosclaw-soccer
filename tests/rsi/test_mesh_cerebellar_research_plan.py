"""Synthetic study scheduling/feature tests; no physical learning claims."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.cerebellar_torque_policy import initialize
from rosclaw_soccer.rsi.mesh_cerebellar_research_plan import (
    SEEDS,
    build_jobs,
    same_input_torque_drift,
    zero_head_anchor_features,
)
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.rsi.public_physics_option_planning import planning_contexts
from rosclaw_soccer.sim.contracts import hash_json

WORLD = "sha256:" + "a" * 64


@pytest.fixture(scope="module")
def bank():
    plan = PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), WORLD, 18)
    return planning_contexts(plan), {
        seed: initialize(seed=seed, world_hash=WORLD, time_aware=True) for seed in SEEDS
    }


def test_all_controls_collections_and_frozen_evaluations_are_declared(bank):
    jobs = build_jobs(*bank)
    assert len(jobs) == len({j["job"] for j in jobs}) == 44
    assert [j["kind"] for j in jobs[:8]] == ["zero_control"] * 8
    assert [j["kind"] for j in jobs[8:20]] == ["collection"] * 12
    assert {j["context_index"] for j in jobs[8:20]} == {0, 2, 4, 6}
    assert {j["context_index"] for j in jobs if j["kind"] == "evaluation"} == {1, 3, 5, 7}
    assert len({j["sampling_seed"] for j in jobs[8:20]}) == 12
    assert all(j["sampling_seed"] is None for j in jobs[:8] + jobs[20:])
    assert [j["group"] for j in jobs[:4]] == [4, 8, 32, 56]
    for seed in SEEDS:
        assert len([j for j in jobs if j["kind"] == "collection" and j["actor_seed"] == seed]) == 4
        retained = [j for j in jobs if j["kind"] == "retention" and j["actor_seed"] == seed]
        assert [j["group"] for j in retained] == [4, 8, 32, 56]
        assert all(j["model_hash"] is None for j in retained)
    assert jobs == build_jobs(*bank)


@pytest.mark.parametrize("fault", ["reorder", "old_schema", "different_world", "short_contexts"])
def test_invalid_actor_or_context_bank_rejected(bank, fault):
    contexts, actors = copy.deepcopy(bank)
    if fault == "reorder":
        actors = dict(reversed(list(actors.items())))
    elif fault == "old_schema":
        actors[SEEDS[0]] = initialize(seed=SEEDS[0], world_hash=WORLD)
    elif fault == "different_world":
        actors[SEEDS[0]] = initialize(
            seed=SEEDS[0], world_hash="sha256:" + "b" * 64, time_aware=True
        )
    else:
        contexts = contexts[:-1]
    with pytest.raises(ValueError):
        build_jobs(contexts, actors)


def trace():
    q = np.zeros((300, 1, 43), dtype=np.float64)
    q[:, :, 3] = 1
    return dict(
        canonical_qpos=q,
        canonical_qvel=np.zeros((300, 1, 41)),
        joint_target_rad=np.zeros((300, 1, 29)),
    )


def test_frame_start_anchors_preserve_clock_and_zero_previous_applied_offset():
    result = zero_head_anchor_features([trace(), trace()])
    assert result.shape == (600, 127)
    np.testing.assert_array_equal(result[:300, -1], (3000 - np.arange(300) * 10) / 3000)
    np.testing.assert_array_equal(result[:300], result[300:])
    assert not result[:, 68:97].any()


@pytest.mark.parametrize("fault", ["single", "nan", "shape", "learned_offset", "quaternion"])
def test_incomplete_or_nonoriginal_anchor_traces_rejected(fault):
    traces = [trace(), trace()]
    if fault == "single":
        traces.pop()
    elif fault == "nan":
        traces[0]["canonical_qvel"][0, 0, 0] = np.nan
    elif fault == "shape":
        traces[0]["canonical_qpos"] = traces[0]["canonical_qpos"][:299]
    elif fault == "quaternion":
        traces[0]["canonical_qpos"][0, 0, 3] = 0
    else:
        traces[0]["applied_cerebellar_torque_offset_nm"] = np.zeros((3000, 29))
    with pytest.raises(ValueError):
        zero_head_anchor_features(traces)


def test_actual_same_input_torque_drift_is_zero_for_identity_and_nonzero_for_changed_head(bank):
    actor = bank[1][SEEDS[0]]
    x = zero_head_anchor_features([trace(), trace()])
    assert same_input_torque_drift(actor, actor, x)["maximum_vector_drift_nm"] == 0
    changed = copy.deepcopy(actor)
    changed["layers"][-1]["bias"][0] = 0.01
    changed["model_hash"] = hash_json({k: v for k, v in changed.items() if k != "model_hash"})
    measured = same_input_torque_drift(actor, changed, x)
    assert measured["maximum_absolute_joint_drift_nm"] > 0
    assert not measured["closed_loop_improvement_claimed"]
    # A model cannot claim to be a fresh zero head by leaving optimizer_steps=0.
    actors = dict(bank[1])
    actors[SEEDS[0]] = changed
    with pytest.raises(ValueError):
        build_jobs(bank[0], actors)
