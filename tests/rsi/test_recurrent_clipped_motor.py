"""Numeric policy-gradient artifacts, not native G1 qualification."""

import copy

import numpy as np
import pytest
from rosclaw.growth.causal_residual_memory import CausalResidualMemory
from rosclaw.growth.recurrent_clipped_actor_critic import (
    RecurrentClippedActorCriticConfig,
    fit_recurrent_clipped_actor_critic,
    initial_value_parameters,
)

from rosclaw_soccer.rsi.recurrent_clipped_motor import (
    CompiledRecurrentClippedMotor,
    make_model,
    make_preview,
    validate_model,
)
from rosclaw_soccer.rsi.recurrent_sampling_motor import (
    CompiledRecurrentSamplingMotor,
    make_sampling_view,
)
from rosclaw_soccer.rsi.recurrent_sampling_motor import make_preview as sampling_preview
from rosclaw_soccer.rsi.recurrent_success_motor import make_model as imitation_model
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_imitation_proposal_motor import imitation_parent  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_recurrent_success_motor import boundary
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_execution import body
from tests.rsi.test_step_motor_network import model  # noqa: F401


def fit(origin, critic):
    x = np.random.default_rng(771).normal(scale=0.1, size=(4, 270, 135))
    base, gates = np.zeros((4, 270, 12)), np.ones((4, 270))
    mean = base.copy()
    for episode in range(4):
        memory = CausalResidualMemory(origin["parameters"])
        for frame in range(270):
            mean[episode, frame] += 0.2 * memory.step(x[episode, frame], index=frame)
    sigma = np.full((4, 270), 0.1 * np.sqrt(1 - 0.9**2))
    sigma[:, 0] = 0.1
    noise = np.random.default_rng(772).normal(size=mean.shape)
    action = mean.copy()
    offset = np.zeros((4, 12))
    for frame in range(270):
        offset = 0.9 * offset + sigma[:, frame, None] * noise[:, frame]
        action[:, frame] += offset
    logp = np.sum(-0.5 * noise**2 - np.log(sigma[:, :, None]) - 0.5 * np.log(2 * np.pi), axis=2)
    reward = np.broadcast_to(np.array([-1.5, -0.5, 0.5, 1.5])[:, None], (4, 270)).copy()
    return fit_recurrent_clipped_actor_critic(
        actor_parameters=origin["parameters"],
        critic_parameters=critic,
        context=x,
        baseline=base,
        gates=gates,
        latent_actions=action,
        behavior_log_probabilities=logp,
        advantages=reward,
        returns=reward,
        config=RecurrentClippedActorCriticConfig(steps=2, seed=771),
    )


@pytest.fixture
def learned(imitation_parent):  # noqa: F811
    origin = imitation_model(imitation_parent)
    critic = initial_value_parameters(135, seed=771)
    result = fit(origin, critic)
    return origin, critic, result, make_model(origin, result, initial_critic=critic)


def reseal(value):
    value.pop("model_hash")
    value["model_hash"] = hash_json(value)
    return value


def test_actual_gradient_parameters_and_continuation_flattening(learned):
    origin, critic, result, first = learned
    assert first["parameters"] != origin["parameters"]
    assert first["critic_parameters"] != critic
    assert result["negative_advantage_rows"] == 540
    assert result["positive_advantage_rows"] == 540
    result2 = fit(first, first["critic_parameters"])
    second = make_model(first, result2)
    assert second["previous_model_hash"] == first["model_hash"]
    assert second["previous_actor_parameters"] == first["parameters"]
    assert second["previous_critic_parameters"] == first["critic_parameters"]
    assert second["frozen_parent"] == origin["frozen_parent"]
    assert "previous_model" not in second
    assert set(first) == set(second)
    assert "actor_parameters" not in second["learning_receipt"]
    for field in ("promotion_authorized", "runtime_execution_authorized", "hardware_authorized"):
        assert second[field] is False
    with pytest.raises(ValueError, match="preserve"):
        make_model(first, result2, initial_critic=critic)
    with pytest.raises(ValueError):
        make_model(origin, result2, initial_critic=critic)


def test_current_policy_execution_sampling_and_critic_not_motor_input(learned):
    artifact = learned[-1]
    policy = make_preview(artifact)
    first, second = [CompiledRecurrentClippedMotor(policy) for _ in range(2)]
    changed = copy.deepcopy(artifact)
    changed["critic_parameters"]["bias_1"][0] += 0.1
    changed["learning_receipt"]["fitted_critic_parameters_hash"] = hash_json(
        changed["critic_parameters"]
    )
    changed = reseal(changed)
    other_policy = make_preview(changed)
    critic_changed = CompiledRecurrentClippedMotor(other_policy)
    sample_policy = sampling_preview(make_sampling_view(artifact, seed=771))
    sampler = CompiledRecurrentSamplingMotor(sample_policy)
    obs = {k: np.repeat(v[:1], 300, axis=0) for k, v in body().items()}
    previous, sampled_previous = np.zeros(12), np.zeros(12)
    for frame in range(300):
        delta = first.delta_at_frame(policy, obs, **boundary(frame, previous))
        np.testing.assert_array_equal(
            delta, second.delta_at_frame(policy, obs, **boundary(frame, previous))
        )
        np.testing.assert_array_equal(
            delta, critic_changed.delta_at_frame(other_policy, obs, **boundary(frame, previous))
        )
        sampled_previous = sampler.delta_at_frame(
            sample_policy, obs, **boundary(frame, sampled_previous)
        )
        np.testing.assert_array_equal(first.hidden_state, second.hidden_state)
        np.testing.assert_array_equal(first.hidden_state, critic_changed.hidden_state)
        assert np.max(np.abs(delta)) <= 0.16
        assert np.max(np.abs(sampled_previous)) <= 0.16
        if frame >= 30:
            assert first._recurrent.next_index == sampler._recurrent.next_index == frame - 29
            assert sampler.sampled_transition["recurrent_sampling_performed"][0]
        previous = delta
    artifact["parameters"]["head_bias"][0] = 123
    assert first._recurrent.parameters()["head_bias"][0] != 123


def test_resealed_numeric_receipt_source_and_authority_faults(learned):
    good = learned[-1]
    faults = [
        ("model", "promotion_authorized", True),
        ("model", "hardware_authorized", 0),
        ("model", "runtime_execution_authorized", True),
        ("model", "fresh_holdout_open_authorized", True),
        ("model", "protected_memory_changed", True),
        ("model", "physical_action_bounds_changed", True),
        ("model", "source_hash", "sha256:" + "f" * 64),
        ("model", "learner_source_hash", "sha256:" + "f" * 64),
        ("model", "parent_model_hash", "sha256:" + "f" * 64),
        ("model", "previous_model_hash", "invalid"),
        ("receipt", "original_actor_parameters_hash", "sha256:" + "f" * 64),
        ("receipt", "original_critic_parameters_hash", "sha256:" + "f" * 64),
        ("receipt", "fitted_actor_parameters_hash", "sha256:" + "f" * 64),
        ("receipt", "fitted_critic_parameters_hash", "sha256:" + "f" * 64),
        ("receipt", "all_rows_behavior_density_validated", 1),
        ("receipt", "on_policy_collection_provenance_verified", True),
        ("receipt", "physical_batch_verified", True),
        ("receipt", "td_bootstrapping", True),
        ("receipt", "full_batch_conditional_mean_kl", 0.2),
        ("receipt", "negative_advantage_rows", True),
        ("receipt", "behavior_density_max_abs_error", 1e-7),
        ("receipt", "accepted_joint_optimizer_steps", 0),
        ("receipt", "accepted_update_history", []),
        ("receipt", "episode_horizon", 269),
    ]
    for section, key, value in faults:
        bad = copy.deepcopy(good)
        (bad if section == "model" else bad["learning_receipt"])[key] = value
        with pytest.raises(ValueError):
            validate_model(reseal(bad))
    for name in (
        "parameters",
        "critic_parameters",
        "previous_actor_parameters",
        "previous_critic_parameters",
    ):
        bad = copy.deepcopy(good)
        key = "head_bias" if "actor" in name or name == "parameters" else "bias_1"
        bad[name][key][0] += 0.1
        with pytest.raises(ValueError):
            validate_model(reseal(bad))
