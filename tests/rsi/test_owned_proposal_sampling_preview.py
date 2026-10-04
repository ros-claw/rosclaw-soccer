"""Canonical reference equivalence and rejection, not physics qualification."""

import copy

import pytest
from rosclaw.growth.frozen_payload_field import FrozenPayloadField

from rosclaw_soccer.rsi.owned_proposal_sampling_preview import OwnedProposalSamplingPreview
from rosclaw_soccer.rsi.proposal_memory_learning import fit_update
from rosclaw_soccer.rsi.proposal_memory_motor import initial_model
from rosclaw_soccer.rsi.proposal_sampling_motor import make_preview, make_sampling_view
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_phase_balanced_proposal_motor import imbalanced_complete_batch
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_private_envelope_restoration_does_not_skip_complete_preview_validation(current):  # noqa: F811
    mean = initial_model(current[0], maximum_mean_kl=0.05)
    owned = OwnedProposalSamplingPreview(mean)
    view = make_sampling_view(mean, seed=17)
    cache = FrozenPayloadField(mean)
    fields = {k: v for k, v in view.items() if k != "mean_model"}
    envelope = cache.envelope(fields, "mean_model")
    restored = owned.restore_envelope(envelope)
    assert hash_json(restored) == hash_json(view)
    assert hash_json(owned.preview(restored)) == hash_json(make_preview(view))
    restored["mean_model"]["hardware_authorized"] = True
    with pytest.raises(ValueError):
        owned.preview(restored)
    assert owned.restore_envelope(envelope)["mean_model"]["hardware_authorized"] is False
    wrong = FrozenPayloadField(dict(mean, hardware_authorized=True)).envelope(fields, "mean_model")
    with pytest.raises(ValueError):
        owned.restore_envelope(wrong)
    unsafe = dict(fields, hardware_authorized=0)
    unsafe.pop("model_hash")
    unsafe["model_hash"] = cache.document_hash(unsafe, "mean_model")
    bad = cache.envelope(unsafe, "mean_model")
    with pytest.raises(ValueError):
        owned.preview(owned.restore_envelope(bad))


def test_wider_profile_is_explicit_and_cannot_be_resealed_to_arbitrary_noise(current):  # noqa: F811
    mean = initial_model(current[0], maximum_mean_kl=0.05)
    owned = OwnedProposalSamplingPreview(mean)
    view = make_sampling_view(mean, seed=19, exploration_profile="wider_bounded")
    assert view["std_raw"] == 0.15
    assert hash_json(owned.preview(view)) == hash_json(make_preview(view))
    for key, value in (
        ("exploration_profile", "nominal"),
        ("exploration_profile", True),
        ("exploration_profile", "unbounded"),
        ("std_raw", 0.150001),
        ("std_raw", 0.25),
        ("rho", 0.99),
    ):
        bad = copy.deepcopy(view)
        bad[key] = value
        bad["model_hash"] = hash_json({k: v for k, v in bad.items() if k != "model_hash"})
        for preview in (owned.preview, make_preview):
            with pytest.raises(ValueError):
                preview(bad)
    with pytest.raises(ValueError):
        make_sampling_view(mean, seed=19, exploration_profile="unbounded")


@pytest.mark.parametrize("learned", [False, True])
def test_owned_preview_exact_and_resealed_faults_rejected(
    current,  # noqa: F811
    smooth_parent,  # noqa: F811
    learned,
    tmp_path,
):
    mean = initial_model(current[0], maximum_mean_kl=0.05)
    if learned:
        mean = fit_update(
            mean,
            imbalanced_complete_batch(smooth_parent),
            batch_hash="sha256:" + "b" * 64,
            numeric_implementation="bounded_snapshot",
        )
    owned = OwnedProposalSamplingPreview(mean)
    for seed in (0, 17, 2**32 - 1):
        view = make_sampling_view(mean, seed=seed)
        expected, actual = make_preview(view), owned.preview(view)
        assert hash_json(expected) == hash_json(actual)
        assert hash_json(owned.validate_preview(actual)) == hash_json(view)
        actual["step_motor_proof"]["model"]["mean_model"]["hardware_authorized"] = True
        assert view["mean_model"]["hardware_authorized"] is False
        with pytest.raises(ValueError):
            owned.validate_preview(actual)
    view = make_sampling_view(mean, seed=17)
    for key, value in (
        ("hardware_authorized", 0),
        ("promotion_authorized", True),
        ("training_only", 1),
        ("std_raw", 0.11),
        ("rho", 0.8),
        ("seed", True),
        ("seed", -1),
        ("seed", 2**32),
        ("source_hash", "wrong"),
        ("extra", 0),
    ):
        bad = copy.deepcopy(view)
        bad[key] = value
        bad["model_hash"] = hash_json({k: v for k, v in bad.items() if k != "model_hash"})
        with pytest.raises(ValueError):
            owned.preview(bad)
    bad = copy.deepcopy(view)
    bad["mean_model"]["hardware_authorized"] = 0
    bad["model_hash"] = hash_json({k: v for k, v in bad.items() if k != "model_hash"})
    with pytest.raises(ValueError, match="canonical mean"):
        owned.preview(bad)
    bad_policy = make_preview(view)
    bad_policy["proposal_sampling_motor_proof"]["training_only"] = 1
    bad_policy["policy_hash"] = hash_json(
        {k: v for k, v in bad_policy.items() if k != "policy_hash"}
    )
    with pytest.raises(ValueError, match="integrity"):
        owned.validate_preview(bad_policy)
    mean["hardware_authorized"] = True
    assert hash_json(owned.preview(view)) == hash_json(make_preview(view))
    path = tmp_path / "dependency.py"
    path.write_text("original")
    owned._pins[str(path)] = hash_bytes(path.read_bytes())
    path.write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        owned.preview(view)
