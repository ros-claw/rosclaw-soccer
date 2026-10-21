"""New native course weights cannot inherit old checkpoint admission."""

from pathlib import Path

import pytest

from rosclaw_soccer.rsi import native_contact_checkpoint as module


@pytest.mark.parametrize(
    "state",
    [
        {"schema": "legacy", "stable_prior_rehearsal": True},
        {
            "schema": "soccer.rsi.native_contact24_ppo.v1",
            "protocol": "native-contact24-v1",
            "episode_count": 8,
        },
    ],
)
def test_checkpoint_reader_rejects_old_intent_or_posthoc_checkpoint(monkeypatch, state):
    monkeypatch.setattr(module, "file_hash", lambda p: "sha256:test")

    def load(*args, **kwargs):
        assert kwargs["weights_only"] is True
        return state

    monkeypatch.setattr(module.torch, "load", load)
    with pytest.raises(ValueError, match="fixed final20"):
        module.NativeContactSession(
            Path("actor.onnx"),
            Path("params.yaml"),
            expected_model_hash="sha256:onnx",
            expected_config_hash="sha256:yaml",
            checkpoint=Path("native.pt"),
            expected_checkpoint_hash="sha256:test",
            parent_checkpoint=Path("parent.pt"),
            expected_parent_hash="sha256:test",
            expected_native_model_hash="sha256:body",
            arm="final",
        )
