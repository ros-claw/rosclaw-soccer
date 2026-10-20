from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.foundation_motor_codec import FoundationMotorCodec


def codec():
    feeds = []

    def run(_, feed):
        x = next(iter(feed.values())).copy()
        feeds.append(x)
        return [x[:, :29].copy()]

    backend = SimpleNamespace(
        qualification=SimpleNamespace(model_variant="low_latency", eligible=True),
        config=SimpleNamespace(onnx_graph_encoder_layout=False),
        _decoder=SimpleNamespace(run=run, get_inputs=lambda: [SimpleNamespace(name="body")]),
        _history_entry=lambda state, action: (
            state.qvel[3:6].copy(),
            state.qpos[7:36].copy(),
            state.qvel[6:35].copy(),
            action.copy(),
            np.zeros(3),
        ),
    )
    q = np.zeros(43)
    q[3] = q[39] = 1
    return FoundationMotorCodec(backend), feeds, q, np.zeros(41)


def test_pure_decode_needs_no_encoder_or_reference_and_uses_own_actual_history():
    c, feeds, q, v = codec()
    token = np.full(64, 0.25, dtype=np.float32)
    previous = np.zeros(29, dtype=np.float32)
    c.observe(q, v, previous, frame=0)
    a = c.decode_motion(token)
    c.commit_consumed(a)
    c.observe(q, v, a, frame=1)
    c.decode_motion(token)
    np.testing.assert_array_equal(feeds[-1][0, 935:964], a)
    assert np.count_nonzero(feeds[0][0, 674:964]) == 0


def test_missing_commit_or_wrong_action_latches_off():
    c, _, q, v = codec()
    c.observe(q, v, np.zeros(29, dtype=np.float32), frame=0)
    c.decode_motion(np.zeros(64, dtype=np.float32))
    with pytest.raises(ValueError, match="differs"):
        c.commit_consumed(np.ones(29, dtype=np.float32))
    assert c._faulted


def test_external_reference_actions_cannot_be_hidden_after_frame20():
    c, _, q, v = codec()
    previous = np.zeros(29, dtype=np.float32)
    for frame in range(20):
        c.observe(q, v, previous, frame=frame)
        c.commit_consumed(previous, foundation_prefix=True)
    c.observe(q, v, previous, frame=20)
    with pytest.raises(ValueError, match="prefix20"):
        c.commit_consumed(previous, foundation_prefix=True)
