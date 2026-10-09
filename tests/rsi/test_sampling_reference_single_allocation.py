"""Early validation and ownership contracts; not native physics qualification."""

import copy

import pytest

import rosclaw_soccer.rsi.cpu_motor_transfer_evidence as audit_module
from rosclaw_soccer.rsi.sampling_reference_audit import SamplingReferenceAuditCompiler


@pytest.mark.parametrize("option", [True, 1, None, "yes"])
def test_requires_explicit_compiler_before_report_read(tmp_path, monkeypatch, option):
    def unopened(*args, **kwargs):
        raise AssertionError("invalid option must reject before reading evidence")

    monkeypatch.setattr(audit_module, "_sealed", unopened)
    with pytest.raises(ValueError, match="single reference allocation"):
        audit_module.audit_cpu_transfer(
            tmp_path, tmp_path / "source.py", allocate_sampling_reference_once=option
        )


@pytest.mark.parametrize("single", [False, True])
def test_full_validation_precedes_physics_and_episode_is_local(tmp_path, monkeypatch, single):
    # Exact trusted class, with constructor bypassed only for this orchestration
    # unit test. The full compiler's numeric/ownership tests run separately.
    compiler = object.__new__(SamplingReferenceAuditCompiler)
    policy = {
        "recurrent_sampling_motor_proof": {},
        "step_motor_proof": {"model": {"model_hash": "bound-model"}},
    }
    report = {"executed_motor_policy": policy, "step_model_hash": "bound-model"}
    monkeypatch.setattr(audit_module, "_sealed", lambda *args, **kwargs: copy.deepcopy(report))
    episodes = []
    calls = []

    def verify(self, value):
        assert self is compiler and value == policy
        calls.append("verify")

    def allocate(self, value):
        assert self is compiler and value == policy
        calls.append("allocate")
        episode = object()
        episodes.append(episode)
        return episode

    def before_physics(_value):
        assert calls[-1] == ("allocate" if single else "verify")
        raise RuntimeError("stopped at source verification before model or physics")

    monkeypatch.setattr(SamplingReferenceAuditCompiler, "verify_policy", verify)
    monkeypatch.setattr(SamplingReferenceAuditCompiler, "new_episode", allocate)
    monkeypatch.setattr(audit_module, "hash_bytes", before_physics)
    source = tmp_path / "source.py"
    source.write_text("# unit fixture\n")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="before model or physics"):
            audit_module.audit_cpu_transfer(
                tmp_path,
                source,
                sampling_reference_compiler=compiler,
                allocate_sampling_reference_once=single,
            )
    assert calls == ["allocate" if single else "verify"] * 2
    assert len(episodes) == (2 if single else 0)
    if single:
        assert episodes[0] is not episodes[1]


def test_rejected_full_policy_cannot_reach_physics(tmp_path, monkeypatch):
    compiler = object.__new__(SamplingReferenceAuditCompiler)
    report = {
        "executed_motor_policy": {
            "recurrent_sampling_motor_proof": {},
            "step_motor_proof": {"model": {"model_hash": "bound"}},
        },
        "step_model_hash": "bound",
    }
    monkeypatch.setattr(audit_module, "_sealed", lambda *args, **kwargs: report)

    def reject(*args):
        raise ValueError("full policy rejected")

    def unopened(*args):
        raise AssertionError("rejected policy must not reach physical source verification")

    monkeypatch.setattr(SamplingReferenceAuditCompiler, "new_episode", reject)
    monkeypatch.setattr(audit_module, "hash_bytes", unopened)
    with pytest.raises(ValueError, match="full policy rejected"):
        audit_module.audit_cpu_transfer(
            tmp_path,
            tmp_path / "unopened.py",
            sampling_reference_compiler=compiler,
            allocate_sampling_reference_once=True,
        )
