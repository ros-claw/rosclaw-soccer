"""Synthetic actual Torch arithmetic/history, NOT real model qualification."""

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
from rosclaw_soccer.rsi.early_foundation_audit import audit_early_foundation
from tests.rsi.test_foundation_torch_capture_audit import sample  # noqa: F401


@pytest.fixture
def trace_and_auditor(sample):  # noqa: F811
    auditor, original = sample
    default = np.asarray(G1SonicRunupController.default_angles)
    q, v = np.zeros((300, 1, 43)), np.zeros((300, 1, 41))
    q[:, :, 3] = q[:, :, 39] = 1
    q[:, :, 7:36] = default
    token = np.broadcast_to(original["foundation_neural_latent_token"][0, 0], (300, 1, 64)).copy()
    decoder = np.zeros((300, 1, 994), dtype=np.float32)
    decoder[:, :, :64] = token
    decoder[:, :, -30:] = np.tile([0, 0, -1], 10)
    trace = dict(
        canonical_qpos=q,
        canonical_qvel=v,
        foundation_navigation_joint_target_rad=np.broadcast_to(default, (300, 1, 29)).copy(),
        foundation_neural_encoder_features=np.zeros((300, 1, 640), dtype=np.float32),
        foundation_neural_latent_token=token,
        foundation_neural_decoder_input=decoder,
        foundation_neural_raw_action_isaac=np.zeros((300, 1, 29), dtype=np.float32),
        foundation_neural_joint_target_mujoco_rad=np.broadcast_to(
            default.astype(np.float32), (300, 1, 29)
        ).copy(),
    )
    return trace, auditor


def test_all300_calls_and_history_with_original_onnx_differences_preserved(trace_and_auditor):
    trace, auditor = trace_and_auditor
    r = audit_early_foundation(trace, auditor)
    assert r["actual_neural_calls_reconstructed"] == 300
    assert r["measured_decoder_history_reconstructed"]
    assert len(r["arithmetic"]["original_token_differences"]) == 300
    assert not r["planner_reference_reconstructed"]
    assert not r["encoder_feature_construction_reconstructed"]
    assert not r["full_causal_policy_qualified"] and not r["teacher_qualified"]
    assert r["new_physical_steps"] == r["new_optimizer_updates"] == 0


@pytest.mark.parametrize(
    "bad", ["history", "target", "action", "token", "nonfinite", "frames", "dtype"]
)
def test_broken_links_and_incomplete_neural_data_rejected(trace_and_auditor, bad):
    t, a = trace_and_auditor
    if bad == "history":
        t["foundation_neural_decoder_input"][150, 0, -1] += 0.1
    elif bad == "target":
        t["foundation_navigation_joint_target_rad"][150, 0, 0] += 0.1
    elif bad == "action":
        t["foundation_neural_raw_action_isaac"][150, 0, 0] += 0.1
    elif bad == "token":
        t["foundation_neural_latent_token"][150, 0, 0] += 0.1
    elif bad == "nonfinite":
        t["foundation_neural_encoder_features"][150, 0, 0] = np.nan
    elif bad == "frames":
        t["foundation_neural_encoder_features"] = t["foundation_neural_encoder_features"][:-1]
    else:
        t["foundation_neural_encoder_features"] = t["foundation_neural_encoder_features"].astype(
            np.float64
        )
    with pytest.raises(ValueError):
        audit_early_foundation(t, a)


def test_caller_cannot_substitute_a_passing_callback(trace_and_auditor):
    trace, _ = trace_and_auditor
    with pytest.raises(ValueError, match="separately constructed"):
        audit_early_foundation(trace, object())
