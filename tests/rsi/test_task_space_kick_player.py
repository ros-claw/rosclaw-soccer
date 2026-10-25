from types import SimpleNamespace

import numpy as np

from rosclaw_soccer.rsi.task_space_kick_player import encode_modified_target


def policy():
    return SimpleNamespace(
        last_action_il=np.linspace(-0.3, 0.3, 29, dtype=np.float32),
        default_q_mj=np.linspace(-0.2, 0.2, 29, dtype=np.float32),
        action_scale_mj=np.full(29, 0.25, np.float32),
        action_clip_lo_il=np.full(29, -2.0, np.float32),
        action_clip_hi_il=np.full(29, 2.0, np.float32),
    )


def test_composite_history_quantizes_only_changed_legs_and_does_not_commit_early():
    p = policy()
    mapping = np.arange(29)[::-1]
    old = p.last_action_il.copy()
    nominal = (p.default_q_mj + p.action_scale_mj * old[mapping]).astype(np.float64)
    target = nominal.copy()
    target[:12] += 0.0123
    quantized, encoded = encode_modified_target(p, target, mapping)
    np.testing.assert_array_equal(p.last_action_il, old)
    np.testing.assert_array_equal(quantized[12:], nominal[12:])
    np.testing.assert_array_equal(encoded[mapping[12:]], old[mapping[12:]])
    reconstructed = p.default_q_mj + p.action_scale_mj * encoded[mapping]
    np.testing.assert_array_equal(reconstructed[:12].astype(np.float64), quantized[:12])


def test_excessive_original_action_is_abstention_not_clip_disguised_as_command():
    p = policy()
    assert encode_modified_target(p, np.full(29, 100.0), np.arange(29)) is None
