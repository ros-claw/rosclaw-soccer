"""Toy matrix capture hydration, not official SONIC or G1 task qualification."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rosclaw_soccer.providers.g1.sonic_causal_capture_state import (  # noqa: E402
    restore_current_tracker,
)
from tests.providers.test_sonic_branch_state import tracker as make_tracker  # noqa: E402


def captured():
    t, q, v = make_tracker()
    t.reference = torch.cat((t.reference, t.reference[:, -1:].repeat(1, 64, 1)), dim=1)
    reference = t.reference[0].numpy().copy()
    values = dict(canonical_qpos=[q.copy()], canonical_qvel=[v.copy()])
    for k, a in t.neural_observation().items():
        values["foundation_neural_" + k] = [a.numpy().copy()]
    for frame in range(1, 31):
        v = v.copy()
        v[:, 6:35] = frame * 0.001
        t.observe(q, v)
        t.update(frame, q, v)
        values["canonical_qpos"].append(q.copy())
        values["canonical_qvel"].append(v.copy())
        for k, a in t.neural_observation().items():
            values["foundation_neural_" + k].append(a.numpy().copy())
    return t, reference, {k: np.asarray(v) for k, v in values.items()}


def test_current_tracker_restored_from_past_state_action_history():
    source, reference, prefix = captured()
    restored = restore_current_tracker(
        source.model, cached_reference=reference, prefix=prefix, frame=30
    )
    assert restored.model is source.model
    assert restored._next_frame == 31 and restored._pending_observation
    for k, v in restored.neural_observation().items():
        assert torch.equal(v, source.neural_observation()[k])
    q, v = prefix["canonical_qpos"][-1], prefix["canonical_qvel"][-1]
    source.observe(q, v)
    restored.observe(q, v)
    assert torch.equal(source.update(31, q, v), restored.update(31, q, v))


@pytest.mark.parametrize(
    "bad", ["future_rows", "past_action", "current_encoder", "current_action", "past_state"]
)
def test_no_future_input_or_corrupt_capture_is_admitted(bad):
    t, reference, prefix = captured()
    if bad == "future_rows":
        prefix = {k: np.concatenate((a, a[-1:])) for k, a in prefix.items()}
    elif bad == "past_action":
        prefix["foundation_neural_raw_action_isaac"][10, 0, 0] += 0.1
    elif bad == "current_encoder":
        prefix["foundation_neural_encoder_features"][30, 0, 0] += 0.1
    elif bad == "current_action":
        prefix["foundation_neural_raw_action_isaac"][30, 0, 0] += 0.1
    else:
        prefix["canonical_qvel"][20, 0, 6] += 0.1
    with pytest.raises(ValueError):
        restore_current_tracker(t.model, cached_reference=reference, prefix=prefix, frame=30)
