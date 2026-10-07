"""Exact preview contracts only; synthetic data is not G1 physics evidence."""

import copy

import pytest

from rosclaw_soccer.rsi.owned_recurrent_sampling_preview import OwnedRecurrentSamplingPreview
from rosclaw_soccer.rsi.recurrent_sampling_motor import make_preview, make_sampling_view
from rosclaw_soccer.rsi.recurrent_success_motor import make_model
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_imitation_proposal_motor import imitation_parent  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_recurrent_clipped_motor import learned  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def reseal(value):
    value.pop("model_hash", None)
    value["model_hash"] = hash_json(value)
    return value


@pytest.mark.parametrize("value", [None, [], True, 0, "model"])
def test_non_mapping_allocation_rejected(value):
    with pytest.raises(ValueError, match="dictionary"):
        OwnedRecurrentSamplingPreview(value)


def test_original_imitation_and_learned_actor_critic_preview_parity(learned):  # noqa: F811
    for mean in (learned[0], learned[-1]):
        compiler = OwnedRecurrentSamplingPreview(mean)
        for seed in (0, 778, 2**32 - 1):
            view = make_sampling_view(mean, seed=seed)
            assert compiler.sampling_view(seed=seed) == view
            original = make_preview(view)
            actual = compiler.preview(view)
            assert hash_json(actual) == hash_json(original)
            assert actual == original
            assert compiler.validate_preview(actual) == view
            if "critic_parameters" in mean:
                assert (
                    actual["step_motor_proof"]["model"]["mean_model"]["critic_parameters"]
                    == mean["critic_parameters"]
                )
            actual["step_motor_proof"]["model"]["mean_model"]["parameters"]["head_bias"][0] = 123
            assert compiler.preview(view) == original
            assert view["mean_model"]["parameters"]["head_bias"][0] != 123


@pytest.mark.parametrize(
    "key,value",
    [
        ("std_raw", 0.15),
        ("rho", 0.8),
        ("seed", True),
        ("seed", -1),
        ("seed", 2**32),
        ("seed", 0.0),
        ("training_only", 1),
        ("hardware_authorized", True),
        ("hardware_authorized", 0),
        ("promotion_authorized", True),
        ("runtime_execution_authorized", True),
        ("fresh_holdout_open_authorized", True),
        ("activation_ceiling", "REAL"),
        ("source_hash", "sha256:" + "f" * 64),
        ("exploration_source_hash", "sha256:" + "f" * 64),
        ("extra", True),
    ],
)
def test_resealed_sampling_or_authority_change_rejected(imitation_parent, key, value):  # noqa: F811
    mean = make_model(imitation_parent)
    compiler = OwnedRecurrentSamplingPreview(mean)
    view = make_sampling_view(mean, seed=778)
    view[key] = value
    with pytest.raises(ValueError):
        compiler.preview(reseal(view))


def test_complete_mean_mutation_and_preview_tamper_rejected(learned):  # noqa: F811
    mean = learned[-1]
    compiler = OwnedRecurrentSamplingPreview(mean)
    original = make_sampling_view(mean, seed=778)
    for key in (
        "parameters",
        "critic_parameters",
        "previous_actor_parameters",
        "previous_critic_parameters",
    ):
        changed = copy.deepcopy(original)
        parameters = changed["mean_model"][key]
        field = "head_bias" if "head_bias" in parameters else "bias_1"
        parameters[field][0] += 0.01
        reseal(changed["mean_model"])
        with pytest.raises(ValueError, match="actor/critic/parent"):
            compiler.preview(reseal(changed))
    policy = compiler.preview(original)
    policy["step_motor_proof"]["decision_start_frame"] = 29
    policy.pop("policy_hash")
    policy["policy_hash"] = hash_json(policy)
    with pytest.raises(ValueError, match="integrity"):
        compiler.validate_preview(policy)
    policy = compiler.preview(original)
    policy["step_motor_proof"]["model"]["mean_model"]["parameters"]["head_bias"][0] += 0.01
    with pytest.raises(ValueError, match="actor/critic/parent"):
        compiler.validate_preview(policy)


def test_cached_complete_bytes_and_source_drift_fail_closed(imitation_parent, monkeypatch):  # noqa: F811
    mean = make_model(imitation_parent)
    compiler = OwnedRecurrentSamplingPreview(mean)
    view = make_sampling_view(mean, seed=778)
    object.__setattr__(compiler._mean, "_data", b"{}")
    with pytest.raises(ValueError):
        compiler.preview(view)
    compiler = OwnedRecurrentSamplingPreview(mean)
    path = next(iter(compiler._pins))
    monkeypatch.setitem(compiler._pins, path, "sha256:" + "f" * 64)
    with pytest.raises(ValueError, match="dependency"):
        compiler.preview(view)
