"""The foot teacher cannot silently compose with other research interventions."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "conflict",
    [
        ["--receiver-lateral-authority"],
        ["--receiver-follow-study"],
        ["--receiver-whole-identity"],
        ["--passer-residual-pair"],
        ["--precontact-fullbody-pair"],
        ["--reference-bridge-controls", "5"],
    ],
)
def test_foot_teacher_pair_rejects_mixed_experimental_protocol(conflict):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_collect_neural_relay_chain",
            "--frontier-root",
            "/not-read",
            "--frontier-head",
            "unused",
            "--frontier-hash",
            "unused",
            "--expected-head",
            "unused",
            "--output",
            "/not-written",
            "--new-context-block",
            "0",
            "--receiver-foot-teacher-pair",
            *conflict,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "foot teacher pair requires one isolated development block" in result.stderr


def test_predictive_teacher_requires_explicit_paired_study():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_collect_neural_relay_chain",
            "--frontier-root",
            "/not-read",
            "--frontier-head",
            "unused",
            "--frontier-hash",
            "unused",
            "--expected-head",
            "unused",
            "--output",
            "/not-written",
            "--receiver-predictive-foot-shin",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "predictive foot/shin mode requires the isolated foot teacher pair" in result.stderr


def test_axis_local_fallback_requires_predictive_study():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_collect_neural_relay_chain",
            "--frontier-root",
            "/not-read",
            "--frontier-head",
            "unused",
            "--frontier-hash",
            "unused",
            "--expected-head",
            "unused",
            "--output",
            "/not-written",
            "--receiver-axis-local-fallback",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "axis-local fallback requires the predictive foot/shin study" in result.stderr


def test_early_intercept_requires_isolated_predictive_axis_study():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_collect_neural_relay_chain",
            "--frontier-root",
            "/not-read",
            "--frontier-head",
            "unused",
            "--frontier-hash",
            "unused",
            "--expected-head",
            "unused",
            "--output",
            "/not-written",
            "--receiver-early-foot-intercept",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "early intercept requires the predictive axis-local foot teacher" in result.stderr


@pytest.mark.parametrize(
    "conflict", ["--receiver-lateral-authority", "--receiver-foot-teacher-pair"]
)
def test_leg_projection_rejects_mixed_interventions(conflict):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_collect_neural_relay_chain",
            "--frontier-root",
            "/not-read",
            "--frontier-head",
            "unused",
            "--frontier-hash",
            "unused",
            "--expected-head",
            "unused",
            "--output",
            "/not-written",
            "--new-context-block",
            "0",
            "--receiver-leg-limit-pair",
            conflict,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "leg-limit projection requires an isolated development block" in result.stderr


def test_projected_memory_requires_explicit_projection_pair():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_collect_neural_relay_chain",
            "--frontier-root",
            "/not-read",
            "--frontier-head",
            "unused",
            "--frontier-hash",
            "unused",
            "--expected-head",
            "unused",
            "--output",
            "/not-written",
            "--receiver-projection-executed-memory",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "projected executed-action memory requires the isolated leg-limit pair" in result.stderr
