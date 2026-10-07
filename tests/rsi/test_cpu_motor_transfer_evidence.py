import json
from pathlib import Path

import pytest

from rosclaw_soccer.rsi.cpu_motor_transfer_evidence import audit_cpu_transfer
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once


@pytest.mark.parametrize("mutation", ["authority", "trace", "source"])
@pytest.mark.parametrize("compressed", [False, True])
def test_cpu_replay_rejects_untrusted_evidence_before_parsing_model(tmp_path, mutation, compressed):
    mujoco = pytest.importorskip("mujoco")
    source = Path(__file__)
    snapshot = tmp_path / "compiled_model.mjb"
    trace = tmp_path / "physical_trace.npz"
    snapshot.write_bytes(b"never parse this untrusted model")
    trace.write_bytes(b"untrusted trace")
    report = dict(
        source_hash=hash_bytes(source.read_bytes()),
        physical_trace_hash=hash_bytes(trace.read_bytes()),
        compiled_model_hash=hash_bytes(snapshot.read_bytes()),
        activation_ceiling="SIM_ONLY",
        promotion_authorized=False,
        hardware_authorized=False,
        physics=dict(engine="MuJoCo", device="cpu", version=mujoco.__version__),
    )
    if mutation == "authority":
        report["hardware_authorized"] = True
    elif mutation == "trace":
        report["physical_trace_hash"] = hash_bytes(b"other")
    else:
        report["source_hash"] = hash_bytes(b"other")
    report["report_hash"] = hash_json(report)
    write_once(tmp_path / ("report.json.gz" if compressed else "report.json"), report)
    with pytest.raises(ValueError, match="authority contract"):
        audit_cpu_transfer(tmp_path, source)


def test_cpu_replay_rejects_ambiguous_plain_and_gzip_report(tmp_path):
    pytest.importorskip("mujoco")
    report = {"report_hash": hash_json({})}
    (tmp_path / "report.json").write_text(json.dumps(report))
    write_once(tmp_path / "report.json.gz", report)
    with pytest.raises(ValueError, match="exactly one complete physical report"):
        audit_cpu_transfer(tmp_path, Path(__file__))


@pytest.mark.parametrize("factory", [object(), lambda _: None])
def test_cpu_audit_does_not_accept_arbitrary_decoder_callback(tmp_path, monkeypatch, factory):
    pytest.importorskip("mujoco")
    from rosclaw_soccer.rsi import cpu_motor_transfer_evidence as module

    monkeypatch.setattr(module, "_sealed", lambda _: {"executed_motor_policy": {}})
    with pytest.raises(ValueError, match="verified smooth sampling factory"):
        audit_cpu_transfer(tmp_path, Path(__file__), sampling_decoder_factory=factory)


@pytest.mark.parametrize("factory", [object(), lambda: None])
def test_cpu_audit_rejects_arbitrary_mean_decoder_callback(tmp_path, monkeypatch, factory):
    pytest.importorskip("mujoco")
    from rosclaw_soccer.rsi import cpu_motor_transfer_evidence as module

    monkeypatch.setattr(module, "_sealed", lambda _: {"executed_motor_policy": {}})
    with pytest.raises(ValueError, match="private fixed proposal mean factory"):
        audit_cpu_transfer(tmp_path, Path(__file__), mean_decoder_factory=factory)


@pytest.mark.parametrize(
    "fault", ["null", "compiler", "extra-report-null", "family", "malformed-family", "typed-drift"]
)
def test_owned_compilation_evidence_rejected_before_loading_native_model(tmp_path, fault):
    mujoco = pytest.importorskip("mujoco")
    from rosclaw_soccer.rsi.proposal_decoder_selection import compilation_contract

    source = Path(__file__)
    snapshot, trace = tmp_path / "compiled_model.mjb", tmp_path / "physical_trace.npz"
    snapshot.write_bytes(b"must not load untrusted model")
    trace.write_bytes(b"untrusted trace")
    contract = compilation_contract("owned_snapshot")
    if fault == "null":
        contract = None
    elif fault == "compiler":
        contract["compiler_source_hash"] = "sha256:" + "f" * 64
    commitment = dict(
        source_hash=hash_bytes(source.read_bytes()),
        physical_trace_hash=hash_bytes(trace.read_bytes()),
        compiled_model_hash=hash_bytes(snapshot.read_bytes()),
        activation_ceiling="SIM_ONLY",
        promotion_authorized=False,
        hardware_authorized=False,
        physics=dict(engine="MuJoCo", device="cpu", version=mujoco.__version__),
        numeric_compilation=contract,
    )
    if fault == "extra-report-null":
        commitment.pop("numeric_compilation")
    write_once(tmp_path / "commitment.json", commitment)
    report = dict(**commitment, commitment_hash=hash_json(commitment), executed_motor_policy={})
    if fault == "extra-report-null":
        report["numeric_compilation"] = None
    elif fault == "typed-drift":
        report["numeric_compilation"] = dict(contract, actor_weights_changed=0)
    elif fault == "malformed-family":
        report["executed_motor_policy"] = {
            "proposal_memory_motor_proof": {},
            "step_motor_proof": None,
        }
    report["report_hash"] = hash_json(report)
    write_once(tmp_path / "report.json.gz", report)
    with pytest.raises(ValueError, match="compilation|proposal motor family"):
        audit_cpu_transfer(tmp_path, source)


@pytest.mark.parametrize("fault", ["null", "extra-report-null", "mixed", "typed-report-drift"])
def test_recurrent_sampling_declaration_rejected_before_loading_native_model(
    tmp_path, fault, monkeypatch
):
    mujoco = pytest.importorskip("mujoco")
    source = Path(__file__)
    snapshot, trace = tmp_path / "compiled_model.mjb", tmp_path / "physical_trace.npz"
    snapshot.write_bytes(b"must not load untrusted model")
    trace.write_bytes(b"untrusted trace")
    commitment = dict(
        source_hash=hash_bytes(source.read_bytes()),
        physical_trace_hash=hash_bytes(trace.read_bytes()),
        compiled_model_hash=hash_bytes(snapshot.read_bytes()),
        activation_ceiling="SIM_ONLY",
        promotion_authorized=False,
        hardware_authorized=False,
        physics=dict(engine="MuJoCo", device="cpu", version=mujoco.__version__),
        recurrent_sampling_factory=None,
    )
    if fault == "extra-report-null":
        commitment.pop("recurrent_sampling_factory")
    elif fault == "mixed":
        commitment["numeric_compilation"] = None
    elif fault == "typed-report-drift":
        commitment["recurrent_sampling_factory"] = {"hardware_authorized": False}
    write_once(tmp_path / "commitment.json", commitment)
    report = dict(**commitment, commitment_hash=hash_json(commitment), executed_motor_policy={})
    if fault == "extra-report-null":
        report["recurrent_sampling_factory"] = None
    elif fault == "typed-report-drift":
        report["recurrent_sampling_factory"] = {"hardware_authorized": 0}
        from rosclaw_soccer.rsi import recurrent_sampling_episode_factory as factory_module

        def must_not_validate_identity(*args):
            raise AssertionError("typed report/commitment drift must be rejected first")

        monkeypatch.setattr(
            factory_module, "validate_compilation_contract", must_not_validate_identity
        )
    report["report_hash"] = hash_json(report)
    write_once(tmp_path / "report.json.gz", report)
    with pytest.raises(ValueError, match="identity|presence|reference review|declaration differs"):
        audit_cpu_transfer(tmp_path, source)
