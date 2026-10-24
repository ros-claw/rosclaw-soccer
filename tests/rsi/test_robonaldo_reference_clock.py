import subprocess
import sys

import numpy as np
import pytest

from rosclaw_soccer.rsi.robonaldo_reference_clock import (
    declared_clock_starts,
    reference_policy_index,
    retimed_reference_velocity,
)


def test_rate_one_is_exact_identity_and_does_not_alias_assets():
    assert [reference_policy_index(i, warmup=10, rate=1.0) for i in range(652)] == list(range(652))
    value = np.linspace(-5, 5, 87, dtype=np.float32).reshape(3, 29)
    result = retimed_reference_velocity(value, rate=1.0)
    np.testing.assert_array_equal(result, value)
    assert result.dtype == value.dtype and not np.shares_memory(result, value)


def test_reference_accelerates_not_warmup_or_physical_calls():
    clock = [reference_policy_index(i, warmup=10, rate=1.12) for i in range(652)]
    assert clock[:10] == list(range(10))
    assert len(clock) == 652
    assert clock[236] == 263
    assert set(np.diff(clock)).issubset({1, 2})
    value = np.ones((4, 29), np.float32)
    np.testing.assert_array_equal(
        retimed_reference_velocity(value, rate=1.12), np.full_like(value, np.float32(1.12))
    )
    assert np.all(value == 1)


@pytest.mark.parametrize("rate", [0.0, 0.9, 1.21, np.nan, np.inf])
def test_invalid_reference_rate_rejected(rate):
    with pytest.raises(ValueError):
        reference_policy_index(0, warmup=10, rate=rate)
    with pytest.raises(ValueError):
        retimed_reference_velocity(np.zeros((2, 29)), rate=rate)


def test_expanded_domain_preserves_original_identity_anchors():
    original = [[x, y] for x in (0.97, 1.03) for y in (-0.03, 0.03)]
    extra = [[x, y] for x in (0.94, 1.0, 1.06) for y in (-0.06, -0.02, 0.02, 0.06)]
    assert declared_clock_starts({"starts_m": original}) == [tuple(p) for p in original]
    assert len(declared_clock_starts({"starts_m": original + extra})) == 16
    for bad in [
        original[::-1],
        original + extra[:-1],
        original + extra[:-1] + [[1.07, 0.0]],
        original + extra[:-1] + [original[0]],
        original + extra[:-1] + [[1.0, np.nan]],
    ]:
        with pytest.raises(ValueError):
            declared_clock_starts({"starts_m": bad})


@pytest.mark.parametrize(
    "options",
    [
        ["--reference-rate=1.12"],
        ["--clock-plan=missing", "--clock-plan-hash=unused"],
        ["--clock-authority"],
        ["--clock-authority", "--reference-rate=nan"],
        ["--clock-authority", "--reference-rate=1.21"],
        ["--clock-authority", "--torque-teacher"],
    ],
)
def test_clock_cli_rejects_undeclared_changes_before_physics(tmp_path, options):
    output = tmp_path / "must-not-exist"
    args = [
        sys.executable,
        "-m",
        "scripts.rsi_probe_live_neural_transition",
        "--source=missing",
        "--own=missing",
        "--output=" + str(output),
        "--source-hash=unused",
        "--own-hash=unused",
        "--expected-head=unused",
        *options,
    ]
    result = subprocess.run(args, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "clock authority" in result.stderr
    assert not output.exists()


def test_clock_candidate_cannot_reuse_unbound_slow_parent(tmp_path):
    output = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_probe_live_neural_transition",
            "--source=missing",
            "--source-hash=unused",
            "--own=missing",
            "--own-hash=unused",
            "--expected-head=unused",
            "--output=" + str(output),
            "--clock-authority",
            "--clock-plan=missing",
            "--clock-plan-hash=unused",
            "--single-player-only",
            "--contact-release",
            "--core-source-root=missing",
            "--core-expected-head=unused",
            "--torque-candidate-root=missing",
            "--torque-candidate-hash=unused",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "exact independent fast parent pair" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    "extra", [[], ["--torque-candidate-root=missing", "--torque-candidate-hash=unused"]]
)
def test_residual_zero_flag_cannot_silently_reinterpret_old_policy(tmp_path, extra):
    output = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_probe_live_neural_transition",
            "--source=missing",
            "--source-hash=unused",
            "--own=missing",
            "--own-hash=unused",
            "--expected-head=unused",
            "--output=" + str(output),
            "--parent-residual-zero-head",
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "zero residual head requires" in result.stderr
    assert not output.exists()
