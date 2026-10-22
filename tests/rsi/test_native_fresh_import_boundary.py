import runpy

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "scripts.rsi_seal_native_fresh",
        "scripts.rsi_native_fresh_shard",
        "scripts.rsi_close_native_fresh",
        "scripts.rsi_qualify_shared_native_motor",
        "scripts.rsi_audit_shared_native_motor",
        "scripts.rsi_collect_native_receiving_roles",
        "scripts.rsi_audit_native_receiving_roles",
        "scripts.rsi_review_receiving_handoff_windows",
        "scripts.rsi_collect_neural_relay_chain",
        "scripts.rsi_audit_neural_relay_chain",
        "scripts.rsi_review_neural_relay_reference",
        "scripts.rsi_qualify_continuous_velocity",
        "scripts.rsi_search_targeted_pass",
        "scripts.rsi_audit_targeted_pass",
        "scripts.rsi_refine_targeted_pass",
        "scripts.rsi_audit_pass_response",
        "scripts.rsi_collect_pass_transfer",
        "scripts.rsi_audit_pass_transfer",
        "scripts.rsi_pass_transfer_trial",
        "scripts.rsi_qualify_pass_perturbations",
    ],
)
def test_import_does_not_allocate_workers_or_write_physical_outputs(module, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runpy.run_module(module, run_name="__mp_main__")
    assert list(tmp_path.iterdir()) == []
