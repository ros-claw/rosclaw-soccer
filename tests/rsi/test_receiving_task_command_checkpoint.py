import copy

import pytest
import torch
from test_receiving_task_command_learning import fixture as parent_fixture  # noqa: F401

from rosclaw_soccer.rsi.receiving_task_command_actor import ReceivingTaskCommandActor
from rosclaw_soccer.rsi.receiving_task_command_checkpoint import load_task_command
from rosclaw_soccer.rsi.receiving_task_command_learning import continue_task_command


@pytest.fixture(scope="module")
def checkpoint(request):
    p, x, y, tr, va, own, advice = request.getfixturevalue("parent_fixture")
    fitted = continue_task_command(p, x, y, tr, va, own, advice, own[:, 112] > 0, steps=400)
    a = fitted["actor"]
    state = dict(
        schema="soccer.receiving_task_command_dagger.v2",
        actor=a.state_dict(),
        optimizer=fitted["optimizer"],
        batch_rng=fitted["batch_rng"],
        seed=123,
        optimizer_updates=400,
        cumulative_prior_steps=800,
        bank_hash="synthetic-bank",
        parent_bc_hash="synthetic-parent-bc",
        parent_checkpoint_hash="synthetic-parent-cp",
        policy_domain=ReceivingTaskCommandActor.policy_domain,
    )
    return (
        state,
        dict(bank_hash="synthetic-bank", parent_bc_hash="synthetic-parent-bc"),
        dict(seed=123, parent_checkpoint_hash="synthetic-parent-cp"),
        p,
        own,
        a,
    )


def test_exact_seven_tensor_continuation_reload_is_frozen_and_same_function(checkpoint):
    saved, report, candidate, parent, x, actor = checkpoint
    loaded = load_task_command(saved, report, candidate, parent)
    with torch.no_grad():
        assert torch.equal(loaded(x), actor(x))
    assert all(not p.requires_grad for p in loaded.parameters())


@pytest.mark.parametrize("fault", ["reset", "seed", "bank", "parent", "scale", "envelope"])
def test_other_parent_reinterpreted_inputs_or_reset_optimizer_refused(checkpoint, fault):
    saved, report, candidate, parent, _, _ = copy.deepcopy(checkpoint)
    if fault == "reset":
        saved["optimizer"]["state"][0]["step"].fill_(400)
    elif fault == "seed":
        saved["seed"] = 124
    elif fault == "bank":
        saved["bank_hash"] = "other"
    elif fault == "parent":
        saved["parent_checkpoint_hash"] = "other"
    elif fault == "scale":
        saved["actor"]["prior.observation_scale"][0] *= 2
    else:
        saved["actor"]["prior.command_radius"][0] *= 2
    with pytest.raises(ValueError):
        load_task_command(saved, report, candidate, parent)


@pytest.fixture(scope="module")
def second_checkpoint(checkpoint, request):
    first, _, _, _, _, _ = checkpoint
    _, x, y, tr, va, own, advice = request.getfixturevalue("parent_fixture")
    result = continue_task_command(
        first, x, y, tr, va, own, advice, own[:, 112] > 0, steps=400, continuation_round=2
    )
    state = dict(
        schema="soccer.receiving_task_command_dagger.v3",
        actor=result["actor"].state_dict(),
        optimizer=result["optimizer"],
        batch_rng=result["batch_rng"],
        seed=123,
        optimizer_updates=400,
        cumulative_prior_steps=1200,
        bank_hash="second-bank",
        parent_bc_hash=first["parent_bc_hash"],
        parent_training_hash="task1-receipt",
        parent_checkpoint_hash="task1-cp",
        policy_domain=ReceivingTaskCommandActor.policy_domain,
    )
    return (
        state,
        dict(
            bank_hash="second-bank",
            parent_bc_hash=first["parent_bc_hash"],
            parent_training_hash="task1-receipt",
        ),
        dict(seed=123, parent_checkpoint_hash="task1-cp"),
        first,
        own,
        result,
    )


def test_second_round_carries_all_seven_adam_states_task_normalizer_and_rng(second_checkpoint):
    saved, report, candidate, parent, x, result = second_checkpoint
    loaded = load_task_command(saved, report, candidate, parent)
    assert torch.equal(loaded(x), result["actor"](x))
    assert result["optimizer_parameter_steps"] == [1200] * 6 + [800]
    assert torch.equal(result["initial_batch_rng"], parent["batch_rng"])
    assert result["exact_parent_function_before_updates"] is True
    assert len(result["changed_parameter_tensors"]) == 7
    for key in ("task_mean", "task_scale", "prior.observation_mean", "prior.observation_scale"):
        assert torch.equal(saved["actor"][key], parent["actor"][key])
    assert [float(parent["optimizer"]["state"][i]["step"]) for i in range(7)] == [800] * 6 + [400]


@pytest.mark.parametrize(
    "fault",
    ["task_scale", "task_mean", "parent_round", "parent_receipt", "reset_task", "reset_prior"],
)
def test_second_round_checkpoint_cannot_reset_or_reinterpret_parent(second_checkpoint, fault):
    saved, report, candidate, parent, _, _ = copy.deepcopy(second_checkpoint)
    if fault in ("task_scale", "task_mean"):
        saved["actor"][fault][0] += 0.1
    elif fault == "parent_round":
        parent["schema"] = "soccer.receiving_task_command_dagger.v3"
    elif fault == "parent_receipt":
        saved["parent_training_hash"] = "other"
    else:
        saved["optimizer"]["state"][6 if fault == "reset_task" else 0]["step"].fill_(0)
    with pytest.raises(ValueError):
        load_task_command(saved, report, candidate, parent)


@pytest.mark.parametrize("fault", ["round", "old_schema", "reset", "round3"])
def test_second_fitter_rejects_unbounded_or_reset_parent(checkpoint, request, fault):
    parent = copy.deepcopy(checkpoint[0])
    _, x, y, tr, va, own, advice = request.getfixturevalue("parent_fixture")
    if fault == "round":
        parent["cumulative_prior_steps"] = 1200
    elif fault == "old_schema":
        parent["schema"] = "soccer.receiving_command_bc.v1"
    elif fault == "reset":
        parent["optimizer"]["state"][6]["step"].fill_(0)
    progress = []
    with pytest.raises(ValueError):
        continue_task_command(
            parent,
            x,
            y,
            tr,
            va,
            own,
            advice,
            own[:, 112] > 0,
            steps=1,
            continuation_round=3 if fault == "round3" else 2,
            on_step=lambda *args: progress.append(args),
        )
    assert progress == []
