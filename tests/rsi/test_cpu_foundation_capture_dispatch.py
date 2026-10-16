import pytest

from rosclaw_soccer.rsi import foundation_observation_capture as capture
from rosclaw_soccer.rsi.cpu_motor_transfer_evidence import validate_cpu_foundation_capture


def test_regular_captured_neural_policy_does_not_require_foundation_ablation(monkeypatch):
    calls = []
    monkeypatch.setattr(capture, "validate_capture", lambda *a, **kw: calls.append("capture"))
    monkeypatch.setattr(capture, "validate_measured_history", lambda *a: calls.append("history"))
    validate_cpu_foundation_capture(
        {"foundation_observation_capture": {}},
        {"foundation_observation_capture": {}},
        {capture.PREFIX + "encoder_features": object()},
    )
    assert calls == ["capture", "history"]


def test_absent_capture_without_tensors_remains_legal():
    validate_cpu_foundation_capture({}, {}, {})


def test_undeclared_capture_is_rejected():
    with pytest.raises(ValueError, match="explicit commitment"):
        validate_cpu_foundation_capture({}, {}, {capture.PREFIX + "encoder_features": object()})


@pytest.mark.parametrize(
    "report,commitment",
    [({"foundation_observation_capture": {}}, {}), ({}, {"foundation_observation_capture": {}})],
)
def test_capture_presence_must_match_commitment(report, commitment):
    with pytest.raises(ValueError, match="differs from commitment"):
        validate_cpu_foundation_capture(report, commitment, {})


def test_malformed_capture_still_rejected():
    with pytest.raises(ValueError, match="contract/source"):
        validate_cpu_foundation_capture(
            {"foundation_observation_capture": {}}, {"foundation_observation_capture": {}}, {}
        )


def test_optional_ablation_does_not_bypass_pure_policy_requirements(monkeypatch):
    monkeypatch.setattr(capture, "validate_capture", lambda *a, **kw: None)
    monkeypatch.setattr(capture, "validate_measured_history", lambda *a: None)
    with pytest.raises(ValueError, match="pure foundation"):
        validate_cpu_foundation_capture(
            {"foundation_observation_capture": {}},
            {"foundation_observation_capture": {}, "foundation_approach_ablation": {}},
            {},
        )
