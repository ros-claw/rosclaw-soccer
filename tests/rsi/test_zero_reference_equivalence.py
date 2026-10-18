"""Synthetic zero-head algebra/decoder tests, not new physical outcomes."""

import numpy as np
import pytest
from rosclaw.growth.causal_residual_memory import initial_parameters

from rosclaw_soccer.rsi.reference_student_decoder import ReferenceStudentDecoder
from rosclaw_soccer.rsi.reference_student_execution import make_model
from rosclaw_soccer.rsi.zero_reference_equivalence import require_zero_reference_equivalence
from tests.rsi.test_pre_parent_reference_features import measured_trace


def model(mean=None, scale=None, parameters=None, **extra):
    return make_model(
        initial_parameters(135, 12, hidden_dimension=64, seed=2026101902)
        if parameters is None
        else parameters,
        mean=np.zeros(135) if mean is None else mean,
        scale=np.ones(135) if scale is None else scale,
        world_hash=extra.get("world_hash", "sha256:" + "a" * 64),
        parent_model_hash=extra.get("parent_model_hash", "sha256:" + "b" * 64),
    )


def test_changed_norm_changes_hidden_states_but_all_300_navigation_outputs_bit_exact():
    mean, scale = np.full(135, 123.0), np.full(135, 0.05)
    mean[-1], scale[-1] = 0, 1
    a, b = model(), model(mean, scale)
    proof = require_zero_reference_equivalence(a, b)
    assert proof["hidden_states_may_differ"]
    assert not proof["nonzero_student_behavior_equivalent"]
    assert not proof["source_authentication_performed_here"]
    decoders = [
        ReferenceStudentDecoder(
            m["parameters"],
            mean=np.asarray(m["mean"]),
            scale=np.asarray(m["scale"]),
            world_hash=m["world_hash"],
            parent_model_hash=m["parent_model_hash"],
        )
        for m in (a, b)
    ]
    base = np.zeros(29)
    base[::2] = -0.0
    bounds = np.tile([-2.0, 2.0], (29, 1))
    trace = measured_trace()
    differs = False
    for frame in range(300):
        records = [
            d.advance(trace, frame=frame, navigation_target=base, joint_limits=bounds)
            for d in decoders
        ]
        differs |= not np.array_equal(
            records[0]["recurrent_hidden_state"], records[1]["recurrent_hidden_state"]
        )
        for r in records:
            assert r["pre_parent_joint_target_rad"].tobytes() == base.tobytes()
            assert np.all(r["requested_reference_increment_rad"] == 0)
            assert np.all(r["applied_reference_increment_rad"] == 0)
    assert differs


@pytest.mark.parametrize("tamper", ["head", "hidden", "seed", "world", "parent", "seal"])
def test_nonzero_or_different_abi_control_cannot_inherit_nav_receipts(tamper):
    a, b = model(), model()
    if tamper in ("head", "hidden"):
        p = initial_parameters(135, 12, hidden_dimension=64, seed=2026101902)
        p["head_bias" if tamper == "head" else "bias_ih"][0] += 1e-8
        b = model(parameters=p)
    elif tamper == "seed":
        b = model(parameters=initial_parameters(135, 12, hidden_dimension=64, seed=7))
    elif tamper in ("world", "parent"):
        b = model(
            **{tamper + ("_hash" if tamper == "world" else "_model_hash"): "sha256:" + "c" * 64}
        )
    else:
        b["mean"][0] = 1
    with pytest.raises(ValueError):
        require_zero_reference_equivalence(a, b)
