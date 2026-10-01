import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi import step_motor_ppo as ppo
from rosclaw_soccer.rsi.online_step_execution import delta_at_frame, make_preview
from rosclaw_soccer.rsi.stochastic_step_execution import latent_sample, make_sampling_view
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_step_motor_execution import body
from tests.rsi.test_step_motor_network import model  # noqa: F401


def batch(warm_model):
    rng = np.random.default_rng(317)
    observations = rng.normal(size=(120, 134))
    view = make_sampling_view(warm_model, seed=317, std=0.1)
    samples = [latent_sample(view, x, 30 + i) for i, x in enumerate(observations)]
    return dict(
        observation=observations,
        latent_action=np.stack([s[0] for s in samples]),
        old_log_probability=np.asarray([s[1] for s in samples]),
        terminal_return=np.repeat([0.0, 1.0, 2.0, 3.0], 30),
        std_raw=np.full(120, 0.1),
        trajectory_index=np.repeat(np.arange(4), 30),
    )


def test_policy_gradient_update_and_reload_are_explicitly_unqualified(model):  # noqa: F811
    arrays = batch(model)
    updated = ppo.fit_update(
        model,
        arrays,
        batch_hash=hash_json({"fixture_only": True}),
        teacher_features=arrays["observation"],
        epochs=1,
    )
    ppo.validate_model(updated)
    assert updated["warm_start_model"] == model
    assert updated["actor"] != model["actor"]
    assert updated["critic"] != model["critic"]
    assert updated["learning_receipt"]["completed_optimizer_steps"] == 1
    assert updated["learning_receipt"]["exact_mean_latent_kl"] <= 0.02
    assert updated["learning_receipt"]["physical_rollout_count"] == 4
    for flag in ppo.AUTHORITY_FLAGS:
        assert updated[flag] is False
    policy = make_preview(updated)
    proposal = delta_at_frame(
        policy,
        body(),
        frame=30,
        nominal_target=np.zeros(29),
        baseline=np.zeros(12),
        limits=np.tile([-1.0, 1.0], (12, 1)),
        previous=np.zeros(12),
        previous_contact_forces=np.zeros(6),
    )
    assert np.max(np.abs(proposal)) <= 0.012
    forged = copy.deepcopy(updated)
    forged["hardware_authorized"] = True
    forged["model_hash"] = hash_json({k: v for k, v in forged.items() if k != "model_hash"})
    with pytest.raises(ValueError):
        ppo.validate_model(forged)


def test_forged_latent_density_cannot_be_used_for_policy_gradient(model):  # noqa: F811
    arrays = batch(model)
    arrays["old_log_probability"][0] += 1
    with pytest.raises(ValueError, match="likelihood"):
        ppo.fit_update(
            model,
            arrays,
            batch_hash=hash_json({"fixture_only": True}),
            teacher_features=arrays["observation"],
            epochs=1,
        )
