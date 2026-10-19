"""Toy matrix histories only; no official SONIC model or G1 performance claim."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rosclaw_soccer.providers.g1.sonic_branch_state import fork_sonic_tracker  # noqa: E402
from rosclaw_soccer.providers.g1.sonic_torch import FrozenSonicG1Torch  # noqa: E402
from rosclaw_soccer.providers.g1.sonic_vector import BatchedSonicTracker  # noqa: E402


def tracker():
    # Deliberately synthetic weights, not qualified source assets.
    model = object.__new__(FrozenSonicG1Torch)
    model._torch, model.device, model.variant = torch, torch.device("cpu"), "low_latency"
    model.qualification = SimpleNamespace(reference_stride=5, heading_normalized=True)
    model._encoder = [(torch.zeros((640, 64)), torch.zeros(64))]
    model._quant = [torch.zeros(64), torch.ones(64), torch.zeros(64), torch.ones(64)]
    model._decoder = [(torch.full((994, 29), 0.001), torch.zeros(29))]
    reference = np.zeros((1, 64, 36), dtype=np.float32)
    reference[:, :, 3] = 1
    t = BatchedSonicTracker(
        model, reference, low_latency_legacy_encoder_layout=True, capture_neural_observation=True
    )
    q, v = np.zeros((1, 43)), np.zeros((1, 41))
    q[:, 3] = 1
    t.reset(q, v)
    t.update(0, q, v)
    return t, q, v


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if type(value) in (list, tuple):
        return [t for v in value for t in tensors(v)]
    if type(value) is dict:
        return [t for v in value.values() for t in tensors(v)]
    return []


def test_weights_only_shared_all_history_and_capture_tensors_owned():
    original, _, _ = tracker()
    branch = fork_sonic_tracker(original)
    assert branch.model is original.model and branch._torch is original._torch
    for key, value in vars(original).items():
        if key in ("model", "_torch"):
            continue
        a, b = tensors(value), tensors(getattr(branch, key))
        assert len(a) == len(b)
        for old, new in zip(a, b, strict=True):
            assert old.data_ptr() != new.data_ptr()
            assert torch.equal(old, new)
    branch._history[0][0].fill_(123)
    branch.reference.zero_()
    branch._neural_observation["raw_action_isaac"].fill_(123)
    assert not torch.any(original._history[0][0] == 123)
    assert torch.all(original.reference[:, :, 3] == 1)
    assert not torch.any(original.neural_observation()["raw_action_isaac"] == 123)


def test_fork_rollout_keeps_live_clock_and_action_then_zero_branch_is_exact():
    original, q, v = tracker()
    branch = fork_sonic_tracker(original)
    changed = fork_sonic_tracker(original)
    old_action = original.action.clone()
    branch.observe(q, v)
    baseline = branch.update(1, q, v)
    cv = v.copy()
    cv[:, 6:35] = 0.1
    changed.observe(q, cv)
    altered = changed.update(1, q, cv)
    assert original._next_frame == 1 and original._pending_observation
    assert torch.equal(original.action, old_action)
    assert not torch.equal(altered, baseline)
    original.observe(q, v)
    actual = original.update(1, q, v)
    assert torch.equal(baseline, actual)
    for key, value in branch.neural_observation().items():
        assert torch.equal(value, original.neural_observation()[key])


@pytest.mark.parametrize(
    "bad",
    [
        "extra",
        "not_pending",
        "clock",
        "batch",
        "history",
        "nan",
        "grad",
        "unknown_state",
        "too_large",
        "history_shape",
        "action_dtype",
    ],
)
def test_unreviewed_or_wrong_boundary_fails_closed(bad):
    t, _, _ = tracker()
    if bad == "extra":
        t.unreviewed_state = []
    elif bad == "not_pending":
        t._pending_observation = False
    elif bad == "clock":
        t._next_frame = 300
    elif bad == "batch":
        t.count = 2
    elif bad == "history":
        t._history.pop()
    elif bad == "nan":
        t.action[0, 0] = float("nan")
    elif bad == "grad":
        t.action.requires_grad_(True)
    elif bad == "unknown_state":
        t._neural_observation = {"unexpected": object()}
    elif bad == "history_shape":
        t._history[0] = tuple(torch.zeros((1, 1)) for _ in range(5))
    elif bad == "action_dtype":
        t.action = t.action.double()
    else:
        t.reference = torch.zeros((1, 100000, 36))
    with pytest.raises(ValueError):
        fork_sonic_tracker(t)
