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
