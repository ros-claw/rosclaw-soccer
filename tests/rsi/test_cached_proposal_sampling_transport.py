"""Lossless shared representation and RAM transport, not simulator evidence."""

import json
import tempfile
from pathlib import Path

import pytest
from rosclaw.growth.frozen_payload_field import FrozenPayloadField
from rosclaw.growth.shared_proof_payload import restore_payload

from rosclaw_soccer.rsi.proposal_memory_motor import initial_model
from rosclaw_soccer.rsi.proposal_sampling_motor import make_sampling_view
from rosclaw_soccer.rsi.sampling_model_io import load_sampling_model
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_shared_logical_views_and_original_ram_loader_are_exact(current):  # noqa: F811
    if not Path("/dev/shm").is_dir():
        pytest.skip("Linux anonymous RAM transport unavailable")
    actor = initial_model(current[0], maximum_mean_kl=0.05)
    cache = FrozenPayloadField(actor)
    template = make_sampling_view(actor, seed=0)
    fields = {k: v for k, v in template.items() if k not in ("model_hash", "mean_model")}
    for seed in (0, 17, 2**32 - 1):
        unsigned = {**fields, "seed": seed}
        signed = {**unsigned, "model_hash": cache.document_hash(unsigned, "mean_model")}
        envelope = cache.envelope(signed, "mean_model")
        restored = restore_payload(envelope, actor)
        assert restored == make_sampling_view(actor, seed=seed)
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8", dir="/dev/shm") as stream:
            json.dump(restored, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            stream.seek(0)
            assert load_sampling_model(Path(f"/proc/self/fd/{stream.fileno()}")) == restored
