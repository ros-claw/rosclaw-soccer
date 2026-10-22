import numpy as np
import pytest

from rosclaw_soccer.rsi.contact_consolidation_data import (
    contact_loss_weights,
    own_reference_sequence,
)


def test_reference_has_own_causal_history_and_exact_dual_reference():
    import torch
    from test_dual_recurrent_motor import model

    parent, learner = model()
    obs = np.random.default_rng(7).normal(size=(12, 160)).astype(np.float32)
    nominal, prior = own_reference_sequence(parent, obs)
    state = learner.initial_state()
    with torch.no_grad():
        for i, value in enumerate(obs):
            _, state, evidence = learner.forward_with_evidence(
                torch.from_numpy(value).reshape(1, 1, 160), state
            )
            expected = nominal[i] + prior[i] / parent.action_scale.numpy()
            assert np.array_equal(expected, evidence["reference_mean"][0, 0].numpy())
    changed = obs.copy()
    changed[6:] += 10
    n2, p2 = own_reference_sequence(parent, changed)
    assert np.array_equal(nominal[:6], n2[:6])
    assert np.array_equal(prior[:6], p2[:6])


def test_prefix_impact_and_neural_recovery_keep_different_explicit_weights():
    w = contact_loss_weights(289, 100, 1035)
    assert w.shape == (389,) and w.dtype == np.float32
    assert (w[:79] == 20).all()
    assert (w[79:124] == 100).all()
    assert (w[124:289] == 1).all()
    assert (w[289:] == 20).all()


def test_missing_contact_never_gets_a_fabricated_contact_window():
    w = contact_loss_weights(100, 0, None)
    assert (w[:80] == 20).all() and (w[80:] == 1).all()
    assert not (w == 100).any()


@pytest.mark.parametrize(
    "args", [(0, 100, None), (301, 100, None), (289, 101, None), (289, 100, -1), (289, 100, 3890)]
)
def test_no_padding_or_outside_contact_label(args):
    with pytest.raises(ValueError):
        contact_loss_weights(*args)
