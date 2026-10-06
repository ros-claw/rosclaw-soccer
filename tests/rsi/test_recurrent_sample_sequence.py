"""Synthetic sequence reconstruction, never physical provenance evidence."""

import copy

import numpy as np
import pytest
from rosclaw.growth.causal_residual_memory import CausalResidualMemory

from rosclaw_soccer.rsi.recurrent_sample_sequence import extract_sequence
from rosclaw_soccer.rsi.recurrent_sampling_motor import (
    TRACE_FIELDS,
    CompiledRecurrentSamplingMotor,
    make_preview,
    make_sampling_view,
)
from rosclaw_soccer.rsi.recurrent_success_motor import STATE_FIELD
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_imitation_proposal_motor import imitation_parent  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_recurrent_clipped_motor import learned  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_execution import body
from tests.rsi.test_step_motor_network import model  # noqa: F401


@pytest.fixture
def sample(learned):  # noqa: F811
    fitted = learned[-1]
    view = make_sampling_view(fitted, seed=778)
    policy = make_preview(view)
    decoder = CompiledRecurrentSamplingMotor(policy)
    trace = {k: np.repeat(v[:1], 300, axis=0) for k, v in body().items()}
    trace.update(
        pre_motor_joint_target_rad=np.zeros((300, 1, 29)),
        motor_delta_rad=np.zeros((300, 1, 12)),
        force_n=np.zeros((300, 1, 6)),
        **{STATE_FIELD: np.zeros((300, 1, 64))},
    )
    trace["force_n"][75, 0, 0] = 6
    for k, d in zip(TRACE_FIELDS, (12, 12, 1, 1), strict=True):
        trace[k] = np.zeros((300, 1, d), dtype=np.bool_ if k == TRACE_FIELDS[-1] else np.float64)
    limits, previous = np.tile([-1.0, 1.0], (12, 1)), np.zeros(12)
    for frame in range(300):
        delta = decoder.delta_at_frame(
            policy,
            trace,
            frame=frame,
            nominal_target=trace["pre_motor_joint_target_rad"][frame, 0],
            baseline=np.zeros(12),
            limits=limits,
            previous=previous,
            previous_contact_forces=trace["force_n"][frame - 1, 0] if frame else np.zeros(6),
        )
        trace["motor_delta_rad"][frame, 0] = delta
        trace[STATE_FIELD][frame, 0] = decoder.hidden_state
        for key, value in decoder.sampled_transition.items():
            trace[key][frame, 0] = value
        previous = delta
    return view, trace, limits


def test_all_actual_student_inputs_states_mean_and_density_reconstruct(sample):
    view, trace, limits = sample
    before = copy.deepcopy(trace)
    data = extract_sequence(view, trace, motor_limits=limits, terminal_mc_return=-123.0)
    assert set(data) == {
        "context",
        "baseline",
        "gates",
        "latent_actions",
        "behavior_log_probabilities",
        "returns",
    }
    assert data["context"].shape == (270, 135)
    assert data["baseline"].shape == (270, 12)
    assert np.all(data["returns"] == -123.0)
    assert set(data["context"][:, -1]) == {0, 1, 2}
    memory = CausalResidualMemory(view["mean_model"]["parameters"])
    means = np.empty((270, 12))
    for frame in range(270):
        means[frame] = data["baseline"][frame] + 0.2 * data["gates"][frame] * memory.step(
            data["context"][frame], index=frame
        )
        np.testing.assert_array_equal(memory.state, trace[STATE_FIELD][frame + 30, 0])
    np.testing.assert_array_equal(means, trace[TRACE_FIELDS[0]][30:, 0])
    conditional = means.copy()
    conditional[1:] += 0.9 * (data["latent_actions"][:-1] - means[:-1])
    sigma = np.full(270, 0.1 * np.sqrt(1 - 0.9**2))
    sigma[0] = 0.1
    density = np.sum(
        -0.5 * ((data["latent_actions"] - conditional) / sigma[:, None]) ** 2
        - np.log(sigma[:, None])
        - 0.5 * np.log(2 * np.pi),
        axis=1,
    )
    np.testing.assert_allclose(density, data["behavior_log_probabilities"], atol=1e-12, rtol=0)
    for key in trace:
        np.testing.assert_array_equal(trace[key], before[key])
    data["latent_actions"][:] = 123
    assert np.max(np.abs(trace[TRACE_FIELDS[1]])) < 123


def test_reward_is_label_not_actor_input_and_failure_sequence_is_retained(sample):
    view, trace, limits = sample
    failed = extract_sequence(view, trace, motor_limits=limits, terminal_mc_return=-123.0)
    successful = extract_sequence(view, trace, motor_limits=limits, terminal_mc_return=12.0)
    for key in failed.keys() - {"returns"}:
        np.testing.assert_array_equal(failed[key], successful[key])
    assert failed["context"].shape == successful["context"].shape == (270, 135)


def test_changed_actual_state_action_draw_or_density_is_rejected(sample):
    view, trace, limits = sample
    for key in (STATE_FIELD, "motor_delta_rad", *TRACE_FIELDS[:3]):
        bad = copy.deepcopy(trace)
        bad[key][35, 0, 0] += 0.001
        with pytest.raises(ValueError, match="reconstruct"):
            extract_sequence(view, bad, motor_limits=limits, terminal_mc_return=-123.0)


def test_incomplete_nonfinite_malformed_or_ambiguous_trace_rejected(sample):
    view, trace, limits = sample
    for kind in (
        "missing",
        "nonfinite",
        "short",
        "float-mask",
        "extra",
        "prestart-mask",
        "negative-force",
    ):
        bad = copy.deepcopy(trace)
        if kind == "missing":
            del bad[STATE_FIELD]
        elif kind == "nonfinite":
            bad["joint_velocity_rad_s"][35, 0, 0] = np.nan
        elif kind == "short":
            bad["motor_delta_rad"] = bad["motor_delta_rad"][:-1]
        elif kind == "float-mask":
            bad[TRACE_FIELDS[-1]] = bad[TRACE_FIELDS[-1]].astype(np.float64)
        elif kind == "extra":
            bad["recurrent_sampling_unknown"] = np.zeros((300, 1, 1))
        elif kind == "prestart-mask":
            bad[TRACE_FIELDS[-1]][29] = True
        else:
            bad["force_n"][35, 0, 0] = -1
        with pytest.raises(ValueError):
            extract_sequence(view, bad, motor_limits=limits, terminal_mc_return=-123.0)
    for label in (True, np.nan, np.inf, 1e7):
        with pytest.raises(ValueError):
            extract_sequence(view, trace, motor_limits=limits, terminal_mc_return=label)
    for envelope in (
        np.zeros((12, 2)),
        np.zeros((29, 2)),
        np.full((12, 2), np.iinfo(np.int64).min),
    ):
        with pytest.raises(ValueError):
            extract_sequence(view, trace, motor_limits=envelope, terminal_mc_return=-123.0)
