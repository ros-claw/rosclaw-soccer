"""Batch export is fail-closed against the frozen SONIC reference."""

from __future__ import annotations

from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from scripts.rsi_sonic_batch_export import (  # noqa: E402
    batch_dynamic_decoder,
    batch_dynamic_encoder,
    validate_batch_parity,
)

MODEL_ROOT = Path("/code/rosclaw/rosclaw_football/datasets/GEAR-SONIC/low_latency")


@pytest.mark.skipif(not MODEL_ROOT.is_dir(), reason="optional SONIC assets not present")
def test_decoder_batch_matches_frozen_onnx() -> None:
    original = MODEL_ROOT / "model_decoder.onnx"
    batch_model, changes = batch_dynamic_decoder(onnx.load(str(original)))
    assert changes == 0
    assert validate_batch_parity(original, batch_model, count=8, width=994) < 1e-4


@pytest.mark.skipif(not MODEL_ROOT.is_dir(), reason="optional SONIC assets not present")
def test_encoder_shape_only_rewrite_is_rejected() -> None:
    original = MODEL_ROOT / "model_encoder.onnx"
    batch_model, changes = batch_dynamic_encoder(onnx.load(str(original)))
    assert changes >= 10
    with pytest.raises(ValueError, match="parity failed"):
        validate_batch_parity(original, batch_model, count=8, width=1247)
