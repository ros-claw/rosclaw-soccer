"""Pure codec and recipient fixture checks, no NN/native physics claims."""

import collections
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_external_history import (
    attach_external_history,
    equivalent_sonic_action,
    prepare_external_history,
)
from rosclaw_soccer.providers.g1.sonic_pose_reference import SonicPoseReference
from rosclaw_soccer.providers.g1.sonic_runup import (
    G1SonicRunupConfig,
    G1SonicRunupController,
    _sonic_control_parameters,
)


def fixture():
    c = object.__new__(G1SonicRunupController)
    c.config = G1SonicRunupConfig()
    c._history = collections.deque(maxlen=10)
    c.reference = np.empty((0, 36))
    c._history_handoff_binding = None
    c._kp, c._kd, c._action_scale = _sonic_control_parameters(1.0, (1.0,) * 29)
    q = np.zeros((10, 43))
    q[:, 2] = 0.793
    q[:, 3] = q[:, 39] = 1.0
    q[:, 7:36] = c.default_angles + np.arange(29) * 0.001
    v = np.arange(410).reshape(10, 41).astype(np.float64) * 0.001
    target = q[:, 7:36] + np.arange(10)[:, None] * 0.002
    h = prepare_external_history(q, v, target, defaults=c.default_angles, scale=c._action_scale)
    ref = SonicPoseReference(
        tuple(tuple(float(x) for x in q[-1, :36]) for _ in range(10)), "sha256:" + "0" * 64
    )
    return c, q, v, target, h, ref


def test_actual_hg_target_inverse_permutation_not_foreign_raw_action():
    c, q, v, target, h, ref = fixture()
    for i in range(10):
        action, residual = equivalent_sonic_action(target[i], c.default_angles, c._action_scale)
        assert action.dtype == np.float32
        assert residual < 2e-7
        expected = c._history_entry(SimpleNamespace(qpos=q[i], qvel=v[i]), action)
        for a, b in zip(expected, h.samples[i], strict=True):
            np.testing.assert_array_equal(a, b)
    before_q, before_v = q.copy(), v.copy()
    kp, kd = c.kp.copy(), c.kd.copy()
    r = attach_external_history(
        c, h, ref, current_qpos=q[-1], current_qvel=v[-1], padded_reference_frames=309
    )
    np.testing.assert_array_equal(q, before_q)
    np.testing.assert_array_equal(v, before_v)
    np.testing.assert_array_equal(c.kp, kp)
    np.testing.assert_array_equal(c.kd, kd)
    np.testing.assert_array_equal(c.target, target[-1])
    assert len(c._history) == 10
    assert not r["physical_gain_compatibility_qualified"]
    assert not r["native_trace_provenance_qualified"]
    assert c.reference.shape == (309, 36)
    with pytest.raises(ValueError):
        attach_external_history(
            c, h, ref, current_qpos=q[-1], current_qvel=v[-1], padded_reference_frames=309
        )


@pytest.mark.parametrize("kind", ["position", "velocity", "defaults", "scale", "used", "reference"])
def test_wrong_boundary_or_used_recipient_fails_without_partial_assignment(kind):
    c, q, v, target, h, ref = fixture()
    if kind == "position":
        q[-1, 0] += 0.01
    elif kind == "velocity":
        v[-1, 0] += 0.01
    elif kind == "defaults":
        c.default_angles = c.default_angles.copy() + 0.01
    elif kind == "scale":
        c._action_scale = c._action_scale * 2
    elif kind == "used":
        c._history.append((np.zeros(3),))
    else:
        ref = SonicPoseReference(
            tuple(
                tuple(float(x) for x in q[0, :36] + np.array([0.01] + [0.0] * 35))
                for _ in range(10)
            ),
            "sha256:" + "0" * 64,
        )
    old_len = len(c._history)
    with pytest.raises(ValueError):
        attach_external_history(
            c, h, ref, current_qpos=q[-1], current_qvel=v[-1], padded_reference_frames=309
        )
    assert len(c._history) == old_len
    assert c.reference.shape == (0, 36)
    assert c._history_handoff_binding is None


def test_exactly_ten_real_samples_and_owned_immutable_values():
    c, q, v, target, h, ref = fixture()
    with pytest.raises(ValueError):
        prepare_external_history(
            q[:9], v[:9], target[:9], defaults=c.default_angles, scale=c._action_scale
        )
    old = h.contract_hash
    q[:] = 0
    v[:] = 0
    target[:] = 0
    assert h.contract_hash == old


@pytest.mark.parametrize("kind", ["nonfinite", "scale", "huge", "dtype"])
def test_invalid_inverse_not_clipped(kind):
    c, q, v, target, h, ref = fixture()
    t, d, s = target[-1].copy(), c.default_angles.copy(), c._action_scale.copy()
    if kind == "nonfinite":
        t[0] = np.nan
    elif kind == "scale":
        s[0] = 0
    elif kind == "huge":
        t[0] = 100
    else:
        t = t.astype(np.float32)
    with pytest.raises(ValueError):
        equivalent_sonic_action(t, d, s)


def test_corrupt_immutable_history_rejected():
    c, q, v, target, h, ref = fixture()
    with pytest.raises(ValueError):
        replace(h, trace_hash="not-a-hash")
    with pytest.raises(ValueError):
        replace(h, maximum_target_roundtrip_error_rad=float("nan"))
    row = list(h.samples[-1])
    row[0] = (1.0, 2.0, 3.0)
    changed = replace(h, samples=(*h.samples[:-1], tuple(row)))
    with pytest.raises(ValueError, match="proprioception"):
        attach_external_history(
            c, changed, ref, current_qpos=q[-1], current_qvel=v[-1], padded_reference_frames=309
        )
