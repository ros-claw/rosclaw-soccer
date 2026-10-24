import numpy as np
import pytest
import torch

from rosclaw_soccer.rsi.motor_sampling_law import history_motor_logp
from rosclaw_soccer.rsi.shared_motor_sampling import SharedMotorSampling


def test_independent_streams_reproduce_draws_rng_and_normalized_density():
    scale = np.full(29, 0.5, np.float32)
    a, b = [SharedMotorSampling(scale, seed=17, profile="contact-v1") for _ in range(2)]
    means, actions, densities = [], [], []
    for k in range(12):
        m = np.linspace(-0.39, 0.39, 29, dtype=np.float32) + np.float32(k * 0.0001)
        reference = np.zeros(29, np.float32)
        x, tx = a.propose(m, reference, control=k, standing=False)
        y, ty = b.propose(m, reference, control=k, standing=False)
        np.testing.assert_array_equal(x, y)
        assert tx.keys() == ty.keys()
        for key in tx:
            np.testing.assert_array_equal(tx[key], ty[key])
        assert tx["active"] and np.max(np.abs(x * scale)) <= 0.200001
        assert not np.array_equal(tx["rng_before"], tx["rng_after"])
        means.append(m)
        actions.append(x)
        densities.append(tx["logp"])
    actual = history_motor_logp(
        "box-truncated-ar1",
        torch.from_numpy(np.stack(means)),
        torch.from_numpy(np.stack(actions)),
        torch.zeros(12, 29),
        torch.from_numpy(scale),
        a.log_std,
        rho=0.9,
    )
    np.testing.assert_allclose(actual.numpy(), np.stack(densities), atol=1e-4, rtol=1e-6)


def test_neural_recovery_is_deterministic_and_never_reset_into_motion():
    sampler = SharedMotorSampling(np.full(29, 0.5, np.float32), seed=19, profile="contact-v1")
    m = np.zeros(29, np.float32)
    sampler.propose(m, m, control=0, standing=False)
    m.fill(0.15)
    action, trace = sampler.propose(m, np.zeros(29, np.float32), control=1, standing=True)
    np.testing.assert_array_equal(action, m)
    assert not trace["active"] and trace["logp"] == 0
    np.testing.assert_array_equal(trace["rng_before"], trace["rng_after"])
    meta = sampler.metadata(reference_controls=1, recovery_controls=1)
    assert meta["stochastic_controls"] == 1 and meta["total_observed_controls"] == 2
    assert meta["PPO_actor_rows"] == 0 and meta["requires_whole_native_and_law_audit"]
    with pytest.raises(ValueError):
        sampler.propose(m, m, control=2, standing=False)


def test_full_recovery_continues_same_private_history_all389_rows_density_match():
    scale = np.full(29, 0.5, np.float32)
    legacy = SharedMotorSampling(scale, seed=19, profile="contact-v1")
    full, own = [
        SharedMotorSampling(scale, seed=19, profile="contact-v1", include_recovery=True)
        for _ in range(2)
    ]
    means, actions, densities = [], [], []
    m, r = np.zeros(29, np.float32), np.zeros(29, np.float32)
    last_rng = None
    for k in range(389):
        m.fill(0.05 if k >= 289 else 0.01)
        x, tx = full.propose(m, r, control=k, standing=k >= 289)
        y, ty = own.propose(m, r, control=k, standing=k >= 289)
        old, old_trace = legacy.propose(m, r, control=k, standing=k >= 289)
        np.testing.assert_array_equal(x, y)
        for key in tx:
            np.testing.assert_array_equal(tx[key], ty[key])
        assert tx["active"]
        assert not np.array_equal(tx["rng_before"], tx["rng_after"])
        if last_rng is not None:
            np.testing.assert_array_equal(tx["rng_before"], last_rng)
        last_rng = tx["rng_after"]
        if k < 289:
            np.testing.assert_array_equal(x, old)
        else:
            assert not old_trace["active"] and not np.array_equal(x, old)
        means.append(m.copy())
        actions.append(x)
        densities.append(tx["logp"])
    logp = history_motor_logp(
        "box-truncated-ar1",
        torch.from_numpy(np.stack(means)),
        torch.from_numpy(np.stack(actions)),
        torch.zeros(389, 29),
        torch.from_numpy(scale),
        full.log_std,
        rho=0.9,
    )
    np.testing.assert_allclose(logp.numpy(), densities, atol=1e-4, rtol=1e-6)
    meta = full.metadata(reference_controls=289, recovery_controls=100)
    assert meta["stochastic_controls"] == meta["total_observed_controls"] == 389
    assert meta["controller_kind"] == "pure-refinement-full-recovery-stochastic-native-v2"
    assert meta["deterministic_recovery_excluded_from_actor_rows"] is False
    with pytest.raises(ValueError):
        full.propose(m, r, control=389, standing=True)
    with pytest.raises(ValueError):
        full.propose(m, r, control=389, standing=False)


@pytest.mark.parametrize("value", [None, 1, "yes"])
def test_coverage_mode_must_be_an_explicit_boolean(value):
    with pytest.raises(ValueError):
        SharedMotorSampling(
            np.ones(29, np.float32), seed=1, profile="contact-v1", include_recovery=value
        )


@pytest.mark.parametrize("seed", [True, -1, 2**31, 1.0])
def test_bad_seed_before_sampling(seed):
    with pytest.raises(ValueError):
        SharedMotorSampling(np.ones(29, np.float32), seed=seed, profile="contact-v1")


def test_private_history_does_not_alias_caller_arrays_and_different_seeds_explore():
    scale = np.full(29, 0.5, np.float32)
    a = SharedMotorSampling(scale, seed=1, profile="contact-v1")
    b = SharedMotorSampling(scale, seed=2, profile="contact-v1")
    mean = np.zeros(29, np.float32)
    x, _ = a.propose(mean, mean, control=0, standing=False)
    y, _ = b.propose(mean, mean, control=0, standing=False)
    assert not np.array_equal(x, y)
    mean.fill(2)
    scale.fill(3)
    assert not a.previous_mean.count_nonzero()
    np.testing.assert_array_equal(a.scale.numpy(), np.full(29, 0.5, np.float32))


@pytest.mark.parametrize("fault", ["clock", "nan", "dtype", "metadata"])
def test_bad_observations_or_clock_not_silently_repaired(fault):
    sampler = SharedMotorSampling(np.ones(29, np.float32), seed=1, profile="micro-v1")
    mean = np.zeros(29, np.float32)
    if fault == "metadata":
        with pytest.raises(ValueError):
            sampler.metadata(reference_controls=1, recovery_controls=1)
        return
    if fault == "nan":
        mean[0] = np.nan
    if fault == "dtype":
        mean = mean.astype(np.float64)
    with pytest.raises(ValueError):
        sampler.propose(
            mean, np.zeros(29, np.float32), control=1 if fault == "clock" else 0, standing=False
        )


@pytest.mark.parametrize(
    "module_name", ["rsi_pass_transfer_trial", "rsi_pass_transfer_replay_trial"]
)
def test_stochastic_stacked_teacher_rejected_before_world_allocation(module_name):
    import importlib
    from types import SimpleNamespace

    module = importlib.import_module("scripts." + module_name)
    setup = SimpleNamespace(prepare=lambda *a: pytest.fail("invalid sampling cannot allocate"))
    args = dict(
        output=None,
        index=0,
        counts={},
        sampling_seed=1,
        sampling_profile="contact-v1",
        feedback_variant=1,
    )
    with pytest.raises(ValueError):
        if module_name.endswith("replay_trial"):
            module.replay_trial(setup, None, dict(assisted=True), {}, raw_root=None, **args)
        else:
            module.collect_trial(setup, None, dict(assisted=True), **args)
